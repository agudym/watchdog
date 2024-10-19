# Module for NN-detector wrappers:
# Input - list of images(numpy-array);
# Processing - torch-reshape (bi-linear), batch construction, inference;
# Output - list of `DetectResult` objects, an auxiliary container for
# category-ID, confidence, bounding box storage with drawing methods.
# NOTE torch and all necessary NN-routines should be imported only here
# in order to minimize memory allocations out of the main (detection) process

from typing import Union, Tuple, List

import sys, os, logging

import numpy as np

import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

from .utils import DetectResult

class Detector():
    """ Wrapper for NN detector (should be extended for new models support) """

    @torch.inference_mode()
    def __init__(self,
                 yolo_lib_path:str,
                 checkpoint_path:str,
                 img_width_height_channels:Tuple[int,int,int],
                 device_switch:str,
                 is_model_fp16:bool):
        """
        Parameters:
        ----------
        yolo_lib_path
            Path to dir with Yolo's root path with __init__.py, If it's possible install Yolo with pip.
        checkpoint_path
            Path to the checkpoint file (model weights, inference only) for the detector model.
        img_width_height_channels
            Size of the detector model input (original images are resized if necessary).
        device_switch
            Values are 'auto', 'cpu', 'cuda' to pick the right computational unit.
        is_model_fp16
            Used by GPU, truncated operations for memory conservation.
        """
        self.img_whc = img_width_height_channels
        self._is_model_fp16 = is_model_fp16

        self._device = None
        if device_switch in ("cuda", "auto") and torch.cuda.is_available():
            self._device = torch.device("cuda")
            torch.cuda.empty_cache()
        elif device_switch in ("cpu", "auto") :
            self._device = torch.device("cpu")
            if self._is_model_fp16:
                raise ValueError("FP16 YOLO-model is not supported on CPU and must be turned off in the config.")
        elif device_switch == "cuda": # but no cuda...
            raise ValueError("GPU-CUDA acceleration is not available!")
        else:
            raise ValueError(f"Unknown device {device_switch}. Must be either 'auto', 'cpu', 'cuda'!")

        if not os.path.exists(checkpoint_path) :
            raise ValueError(f"{checkpoint_path}.pt not found!")
        else:
            # Yolo is required for complete model load (`weights_only` = False)
            sys.path.insert(1, yolo_lib_path)

            checkpoint = torch.load(checkpoint_path, map_location=self._device)# , weights_only=False # torch 1.10 support
            self._yolo = checkpoint["model"].float().fuse().eval()
            if self._is_model_fp16:
                self._yolo = self._yolo.half()
            model_stride = 32

            w, h, c = self.img_whc
            assert c == 3, "Only 3-channel images supported"

            if (w % model_stride) or (h % model_stride):
                raise ValueError(
                    f"Loaded model doesn't support {self.img_whc} size, padding (mod {model_stride}) is required!" )

            self._resizer = Resize((h, w), interpolation=InterpolationMode.BILINEAR, antialias=False)

    @torch.inference_mode()
    def detect(self,
            images_raw : List[np.ndarray],
            confidence_threshold : float,
            category_ids : Union[List[int], np.ndarray]) :
        """
        Detect object of specific classes in the input 8-bit images list with confidence above the threshold.

        Parameters:
        -----------
        images_raw
            List of input 8-bit images.
        confidence_threshold
            Minimal probability for the detected object category to be accepted.
        category_ids
            List of integer category-indices to detect. Index is in range from 0 to Num of categories - 1.
        """
        w_model, h_model, c_model = self.img_whc
        imgs_batch = []
        for img_raw in images_raw:
            assert issubclass(img_raw.dtype.type, np.integer), "Invalid input type, must be 8-bit RGB images!"

            img = torch.permute(
                self._adapt_float_type(torch.from_numpy(img_raw)).to(self._device),
                (2, 0, 1)).contiguous() # contiguous() is required to get faster operations on the Tensor
            img /= 255.  # 0 - 255 to 0.0 - 1.0

            c_raw, h_raw, w_raw = img.shape
            assert c_raw == c_model, f"Invalid(!= {c_model}) channels number!"
            if h_raw != h_model or w_raw != w_model:
                if np.abs(w_raw / h_raw - w_model / h_model) > 1e-2:
                    logging.warning(f"Different aspect (w/h) between raw ({w_raw/h_raw:.2f}) and processed ({w_model/h_model:.2f}) images!"
                                    " Adjust detector resolution or camera streaming settings.")
                img = self._resizer(img)
            imgs_batch.append(img)

        detect_tensor = self._yolo(torch.stack(imgs_batch))[0].permute(0, 2, 1)

        detect_results = []
        for _, prediction_tensor in enumerate(detect_tensor):
            detect_results.append( self._unpack_predictions(prediction_tensor, confidence_threshold, category_ids) )
        return detect_results

    def _unpack_predictions(
            self,
            prediction_tensor: torch.Tensor,
            confidence_threshold: float,
            category_ids:Union[List[int], np.ndarray]) :
        assert prediction_tensor.ndim == 2
        assert prediction_tensor.shape[-1] > 4 + np.max(category_ids)

        category_ids = torch.Tensor(category_ids).to(prediction_tensor.device)

        class_probs_all = prediction_tensor[:, 4:]
        category_ids_pred = torch.argmax(class_probs_all, dim=-1)
        class_confidences = torch.take_along_dim(class_probs_all, category_ids_pred[:, None], dim=-1).flatten()

        mask = (class_confidences > confidence_threshold) & torch.isin(category_ids_pred, category_ids)

        xy_bounds = prediction_tensor[mask,:4]
        obj_bboxes = torch.empty(xy_bounds.shape) # ((top left x, top left y), (bottom right x, bottom right y))
        obj_bboxes[:, 0] = xy_bounds[:, 0] - xy_bounds[:, 2] / 2  # top left x
        obj_bboxes[:, 1] = xy_bounds[:, 1] - xy_bounds[:, 3] / 2  # top left y
        obj_bboxes[:, 2] = xy_bounds[:, 0] + xy_bounds[:, 2] / 2  # bottom right x
        obj_bboxes[:, 3] = xy_bounds[:, 1] + xy_bounds[:, 3] / 2  # bottom right y

        w, h, _ = self.img_whc
        obj_bboxes[:,(0,2)] /= w
        obj_bboxes[:,(1,3)] /= h

        return DetectResult(
            category_ids_pred[mask].cpu().detach().numpy(),
            class_confidences[mask].cpu().detach().numpy(),
            obj_bboxes.cpu().detach().numpy() )

    def _adapt_float_type(self, img: torch.Tensor) :
        return img.half() if self._is_model_fp16 else img.float()  # uint8 to fp16/32

    def get_device_memory_available(self):
        """ Amount of free GPU memory in (GiB) and free/total ratio """
        if "cuda" in str(self._device):
            mem_total = torch.cuda.get_device_properties(self._device).total_memory
            mem_allocated = torch.cuda.max_memory_allocated(self._device)
            mem_free = mem_total - mem_allocated
            return mem_free  / 2**30, mem_free / mem_total
        else:
            return 0, 0

if __name__ == "__main__":
    import time, argparse
    import cv2
    
    from .camera import FolderImgReader
    from .utils import setup_logger, load_config, generate_rgb8_colors

    setup_logger(verbose_level="INFO")

    parser = argparse.ArgumentParser(description="Check Yolo detector.")
    parser.add_argument("config_json_path", type=str, help="Path to the main config file.")
    parser.add_argument("input_images_path", type=str, help="Path to the directory with images for processing.")
    parser.add_argument("output_images_path", type=str, help="If specific objects are found, images with bounding boxes are saved here.")
    args = parser.parse_args()

    config_dict = load_config(args.config_json_path)
    
    detector = Detector(**config_dict["Detector"]["Init"])
    
    category_names_all = config_dict["Detector"]["categories_all"].split(",")
    category_names_notify = config_dict["Detector"]["categories_notify"].split(",")
    category_ids_notify = DetectResult.get_ids(category_names_all, category_names_notify)
    category_colors = generate_rgb8_colors(len(category_names_all))
    
    # Lower confidence for the visualization
    confidence_threshold = config_dict["Detector"]["confidence_threshold"] / 2
    bbox_merge_dist = config_dict["Detector"]["bbox_merge_dist"]

    # Dry run, just in case test resolution is 2x bigger
    w, h, c = detector.img_whc
    timg_batch = np.random.randint(0, 255, (4, h * 2, w * 2, c), dtype=np.uint8)
    start_time = time.time()
    _ = detector.detect(timg_batch, 0.01, category_ids_notify)[0].merge(bbox_merge_dist)
    proc_time = time.time() - start_time
    print(f"Dry run for {timg_batch.shape} in {proc_time:.2f} sec.")

    # Process dir
    if not os.path.exists(args.output_images_path):
        os.makedirs(args.output_images_path)

    for img_id, (status, img) in enumerate(FolderImgReader(args.input_images_path)):
        start_time = time.time()
        detect_result: DetectResult = detector.detect([img], confidence_threshold, category_ids_notify)[0]
        proc_time = time.time() - start_time
        print(f"For image #{img_id} found {detect_result.num_objects} objects in {proc_time:.2f} sec.")
        if detect_result.num_objects > 0:
            start_time = time.time()
            detect_result = detect_result.merge(bbox_merge_dist)
            proc_time = time.time() - start_time
            print(f"{detect_result.num_objects} objects left after merge ({proc_time:.2f} sec): {detect_result.describe(category_names_all)}")
            cv2.imwrite(
                os.path.join(args.output_images_path, f"img_{img_id}.jpg"),
                detect_result.draw(img, category_names_all, category_colors))

    mem_gpu_gb, mem_gpu_rel = detector.get_device_memory_available()
    print(f"Cuda memory available: {mem_gpu_gb:.2f} GiB ({mem_gpu_rel:.2f})")
