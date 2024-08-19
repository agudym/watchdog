# Module for NN-detector wrappers:
# Input - list of images(numpy-array);
# Processing - torch-reshape (bi-linear), batch construction, inference;
# Output - list of `DetectResult` objects, an auxiliary container for
# category-ID, confidence, bounding box storage with drawing methods.
# NOTE torch and all necessary NN-routines should be imported only here.

from typing import Union, Tuple, List

import sys, os
import logging

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN # should go before OpenCV due to some bug...
import cv2

def _get_category_colors(num_classes:int):
    rs = RandomState(MT19937(SeedSequence(0)))
    cmap = plt.get_cmap("hsv", num_classes) # cm.get_cmap
    colors = []
    for i in rs.permutation(np.arange(num_classes)):
        color = cmap(i)
        colors.append([int(color[j] * 255) for j in range(3)])
    return colors

class DetectResult():
    """
    List of detected objects (class/object category IDs), confidences, bounding boxes. With auxiliary image-drawing routines.

    Each individual image must have a specific `DetectResult`.
    """

    # List of all detectable 80 classes from COCO 2017 dataset http://cocodataset.org
    category_names_coco = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush' ]

    # Randomized individual class colors
    category_colors = _get_category_colors(len(category_names_coco))
    
    def __init__(self,
                 category_ids : Union[np.ndarray, List[int]] = [],
                 confidence : Union[np.ndarray, List[float]] = [],
                 bboxes : np.ndarray = np.empty((0, 4)) ) :
        """
        Parameters:
        -----------
        category_ids
            Detected object category identifier in [0, N], N is the num all possible category, shape (M,), M - num found objects.
        confidence
            Probability [0, 1] that the detected object belongs to the particular class, shape (M,).
        bboxes
            Axis-aligned bounding box (x_lower, y_lower, x_upper, y_upper), coordinates normalized to [0,1], shape (M,4).
        """
        assert len(bboxes) == len(category_ids) and len(bboxes) == len(confidence)
        assert bboxes.shape[1] == 4

        self.category_ids = np.array(category_ids)
        self.confidence = np.array(confidence)
        self.bboxes = bboxes

    @property
    def num_objects(self):
        return len(self.category_ids)

    def merge(self, bbox_merge_dist: float):
        """
        Merge very close Bounding Boxes (4-vector), likely for the same object, using DBSCAN.

        Bounding boxes are normalized, i.e. clustering coordinates space is [0,1]^4.
        
        Parameters:
        -----------
        bbox_merge_dist
            Maximal distance (1 - whole image 1D size!) between identical bounding boxes, e.g.
            belonging to the same object.
        
        Return:
        -------
        `DetectResult` with merged bounding boxes belonging to the same object (likely).
        """
        if bbox_merge_dist == 0.:
            return self

        dbscan = DBSCAN(eps=bbox_merge_dist, min_samples=2)
        
        category_ids = []
        confidence = []
        bboxes = []
        for category_id in set(self.category_ids) :
            mask = self.category_ids == category_id
            if np.sum(mask) == 0 :
                continue
            bboxes_masked = self.bboxes[mask]
            confidence_masked = self.confidence[mask]
            dbscan.fit(bboxes_masked)

            match_clusters = set(dbscan.labels_)
            for cluster_id in match_clusters :
                mask_cluster = cluster_id == dbscan.labels_

                if cluster_id < 0: # -1 - each instance added individually
                    for obj_id in np.where(mask_cluster)[0] :
                        category_ids.append(category_id)
                        confidence.append(confidence_masked[obj_id])
                        bboxes.append(bboxes_masked[obj_id])
                else:
                    category_ids.append(category_id)
                    confidence.append(np.mean( confidence_masked[mask_cluster]) )
                    bboxes.append(np.mean( bboxes_masked[mask_cluster], axis=0 ))

        return DetectResult(category_ids, confidence, np.array(bboxes).reshape((-1, 4)))

    def draw(self, image_orig: np.ndarray) :
        """
        Draw found object's bounding boxes in the image (return modified copy).
        """
        image = image_orig.copy()
        h, w, _ = image.shape
        
        bboxes_pix = (self.bboxes * (w,h,w,h)).astype(int)
        for category_id, conf, bbox in zip(self.category_ids, self.confidence, bboxes_pix) :
            c = self.category_colors[category_id]
            text = f"{self.category_names_coco[category_id]} {conf:.2f}"
            cv2.putText(image, text, tuple(bbox[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, c, thickness=2)
            cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), c, 1)
            # For the history - work with matplotlib
            # w, h = bbox[2:] - (x,y)
            # ax.add_patch(patches.Rectangle(
            #     (x,y), w, h, linewidth=1, edgecolor=c, facecolor="none"))
            # ax.text(x, y, f"{model.class_names[class_id]}\np={prob:.2f}", weight="bold", va="top", color=c)
        return image
    
    def describe(self):
        """ Short description of the detection results """
        description = ""
        for i, (category_id, conf) in enumerate(zip(self.category_ids, self.confidence)) :
            if i > 0:
                description += "; "
            description += f"{self.category_names_coco[category_id]} {conf:.2f}"
        return description + "."

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

            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
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
            categories : List[str]) :
        """
        Detect object of specific classes in the input 8-bit images list with confidence above the threshold.

        Parameters:
        -----------
        images_raw
            List of input 8-bit images.
        confidence_threshold
            Minimal probability for the detected object category to be accepted.
        categories
            List of possible categories to detect.
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
        for img_id, prediction_tensor in enumerate(detect_tensor):
            detect_results.append( self._unpack_predictions(prediction_tensor, confidence_threshold, categories) )
        return detect_results

    def get_device_memory_available(self):
        """ Amount of free GPU memory in (GiB) and free/total ratio """
        if "cuda" in str(self._device):
            mem_total = torch.cuda.get_device_properties(self._device).total_memory
            mem_allocated = torch.cuda.max_memory_allocated(self._device)
            mem_free = mem_total - mem_allocated
            return mem_free  / 2**30, mem_free / mem_total
        else:
            return 0, 0

    def _unpack_predictions(
            self,
            prediction_tensor: torch.Tensor,
            confidence_threshold: float,
            categories:List[str]) :
        assert prediction_tensor.ndim == 2
        assert prediction_tensor.shape[-1] == 4 + len(DetectResult.category_names_coco)

        category_ids_subset = np.where(np.isin(DetectResult.category_names_coco, categories))[0]
        category_ids_subset = torch.Tensor(category_ids_subset).to(prediction_tensor.device)

        class_probs_all = prediction_tensor[:, 4:]
        category_ids = torch.argmax(class_probs_all, dim=-1)
        class_confidences = torch.take_along_dim(class_probs_all, category_ids[:, None], dim=-1).flatten()

        mask = (class_confidences > confidence_threshold) & torch.isin(category_ids, category_ids_subset)

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
            category_ids[mask].cpu().detach().numpy(),
            class_confidences[mask].cpu().detach().numpy(),
            obj_bboxes.cpu().detach().numpy() )

    def _adapt_float_type(self, img: torch.Tensor) :
        return img.half() if self._is_model_fp16 else img.float()  # uint8 to fp16/32

if __name__ == "__main__":
    import time, argparse, json
    from camera import FolderImgReader

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Check Yolo detector.")
    parser.add_argument("config_json_path", type=str, help="Path to the main config file.")
    parser.add_argument("input_images_path", type=str, help="Path to the directory with images for processing.")
    parser.add_argument("output_images_path", type=str, help="If specific objects are found, images with bounding boxes are saved here.")
    args = parser.parse_args()

    with open(args.config_json_path, "r") as file:
        config_dict = json.load(file)
    
    detector = Detector(**config_dict["Detector"]["Init"])
    categories = config_dict["Detector"]["categories"]
    confidence_threshold = config_dict["Detector"]["confidence_threshold"]
    bbox_merge_dist = config_dict["Detector"]["bbox_merge_dist"]

    # Dry run
    w, h, c = detector.img_whc
    timg_batch = np.random.randint(0, 255, (4, h * 2, w * 2, c), dtype=np.uint8)
    start_time = time.time()
    _ = detector.detect(timg_batch, 0.01, categories)[0].merge(bbox_merge_dist)
    proc_time = time.time() - start_time
    print(f"Dry run for {timg_batch.shape} in {proc_time:.2f} sec.")

    # Process dir
    if not os.path.exists(args.output_images_path):
        os.makedirs(args.output_images_path)

    for img_id, (status, img) in enumerate(FolderImgReader(args.input_images_path)):
        start_time = time.time()
        detect_result = detector.detect([img], confidence_threshold, categories)[0]
        proc_time = time.time() - start_time
        print(f"For image #{img_id} found {detect_result.num_objects} objects in {proc_time:.2f} sec.")
        if detect_result.num_objects > 0:
            start_time = time.time()
            detect_result = detect_result.merge(bbox_merge_dist)
            proc_time = time.time() - start_time
            print(f"{detect_result.num_objects} objects left after merge ({proc_time:.2f} sec): {detect_result.describe()}")
            cv2.imwrite(
                os.path.join(args.output_images_path, f"img_{img_id}.jpg"), detect_result.draw(img))

    mem_gpu_gb, mem_gpu_rel = detector.get_device_memory_available()
    print(f"Cuda memory available: {mem_gpu_gb:.2f} GiB ({mem_gpu_rel:.2f})")
