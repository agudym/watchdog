import os, time, pathlib, re, logging

from typing import List, Union, Tuple, Optional, Dict

import numpy as np

import multiprocessing
import concurrent.futures

import cv2

class CameraConfig():
    """ Main reading settings """
    def __init__(self, uri:Union[int, str], name:str, fps:float, timeout_err: float, count_err:int):
        if fps <=0 :
            raise ValueError(f"Unsupported FPS {fps} value!")
        if timeout_err <=0 :
            raise ValueError(f"Unsupported timeout {timeout_err} value!")
        if isinstance(uri, str) and len(uri) == 0:
            raise ValueError(f"Camera URI must be set!")
        if isinstance(name, str) and len(name) == 0:
            raise ValueError(f"Camera name must be set!")

        # main camera address (USB ID, RTSP IP address, GStreamer pipeline etc.)
        self.uri = uri

        # camera logging alias
        self.name = name

        # minimal read(poll) delay
        self.timeout_read = 1 / fps

        # timeout for init/read/release error report
        self.timeout_err = timeout_err

        # amount of failed frames before reset
        self.count_err = count_err
    
    def __str__(self):
        return f"uri: {self.uri}, name: {self.name}, fps: {1 / self.timeout_read}, timeout_err: {self.timeout_err}"

    @staticmethod
    def parse(config_dict:Dict):
        cam_configs = []
        for cam_dist in config_dict:
            uri = cam_dist["uri"]
            uri = int(uri) if isinstance(uri, int) or uri.isdigit() else uri
            cam_configs.append(CameraConfig(
                uri,
                str(cam_dist["name"]),
                float(cam_dist["fps"]),
                float(cam_dist["timeout_err"]),
                int(cam_dist["count_err"]),
            ))
        return cam_configs

FrameDataType = Tuple[np.ndarray, float]
class TimeoutCamerasReader():
    """
    Multi-Treaded multi-cameras polling (sync-reading)

    TIMEOUT-controlled communication with the camera (init, capture, release operations).
    """
    def __init__(self, cams_conf: List[CameraConfig]):
        # Necessary to allow RTSP cameras capture properly with OpenCV VideoCapture
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        os.environ["OPENCV_LOG_LEVEL"] = "OFF"
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "OFF"

        self._cams_conf = cams_conf
        self._num_cams = len(self._cams_conf)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._num_cams)
        self._cams_cap = [None] * self._num_cams
        self._init_cams_cap()

    def release(self, cams_ids2reset:List[int]):
        def release_capturer_thread(cam_id: int):
            if self._cams_cap[cam_id] is not None:
                self._cams_cap[cam_id].release()
                self._cams_cap[cam_id] = None

        futures_with_id = []
        for cam_id in cams_ids2reset:
            future = self._executor.submit(release_capturer_thread, cam_id)
            futures_with_id.append((future, cam_id))

        for future, cam_id in futures_with_id:
            future.result(timeout=self._cams_conf[cam_id].timeout_err)

    def reset(self, cams_ids2reset:List[int]):
        self.release(cams_ids2reset)
        return self._init_cams_cap()

    def read_frames(self):
        def read_frame_thread(cam_id: int):
            read_status, img_raw = self._cams_cap[cam_id].read()
            return img_raw if read_status else None

        futures_with_id = []
        for cam_id in range(self._num_cams):
            if self._cams_cap[cam_id] is None:
                continue
            future = self._executor.submit(read_frame_thread, cam_id)
            futures_with_id.append((future, cam_id))

        imgs_read = [None] * self._num_cams
        for future, cam_id in futures_with_id:
            imgs_read[cam_id] = future.result(timeout=self._cams_conf[cam_id].timeout_err)
        return imgs_read

    def _init_cams_cap(self):
        def create_capturer_thread(cam_id:int):
            uri = self._cams_conf[cam_id].uri
            if isinstance(uri, str) and os.path.isdir(uri):
                capturer = FolderImgReader(uri)
            else:
                capturer = cv2.VideoCapture(uri)
            if not capturer.isOpened():
                raise RuntimeError(f"Failed to open OpenCV VideoCapture for camera '{self._cams_conf[cam_id].name}'")
            return capturer

        futures_with_id = []
        for cam_id in range(self._num_cams):
            if self._cams_cap[cam_id] is not None:
                continue
            future = self._executor.submit(create_capturer_thread, cam_id)
            futures_with_id.append((future, cam_id))

        for future, cam_id in futures_with_id:
            self._cams_cap[cam_id] = future.result(timeout=self._cams_conf[cam_id].timeout_err)

class CamMultiprocReader():
    """
    Multi-Process multi-cameras polling (non-blocking).
     
    Non-blocking poll: empty buffer is returned if no data is ready
    """
    def __init__(self, cams_conf: List[CameraConfig]):
        self._cams_conf = cams_conf
        self._num_cams = len(self._cams_conf)
        self._timeout_read_all = np.max([cam_conf.timeout_read for cam_conf in self._cams_conf])

        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self._num_cams)

        self._manager = multiprocessing.Manager()
        self._images_shared = self._manager.list( [None] * self._num_cams )
        self._is_running_shared = self._manager.Value(bool, True)
        
        # Start capturing process per camera
        self._capturing_futures = [None] * self._num_cams
        self._restart()

    def read_frames(self):
        time.sleep(self._timeout_read_all)
        # restart dead processes
        self._restart()
        images = self._images_shared[:]
        self._images_shared[:] = [None] * self._num_cams
        return images

    def stop(self):
        self._is_running_shared.value = False
        concurrent.futures.wait([future for future in self._capturing_futures if future is not None])

    def _restart(self):
        for cam_id, cam_conf in enumerate(self._cams_conf):
            if self._capturing_futures[cam_id] is not None and not self._capturing_futures[cam_id].done():
                continue

            logging.info(f"New process for camera '{cam_conf.name}' starting...")
            self._capturing_futures[cam_id] = self._executor.submit(
                self._run_capture_process,
                cam_id,
                cam_conf,
                self._images_shared,
                self._is_running_shared)

    @staticmethod
    def _run_capture_process(
        cam_id:int,
        cam_conf: CameraConfig,
        images_shared,
        is_running_shared):
        try:
            mreader = None
            mreader = TimeoutCamerasReader([cam_conf])
            logging.info(f"Camera '{cam_conf.name}' initialized.")

            error_reads_count = 0
            while is_running_shared.value :

                start_time = time.time()
                image = mreader.read_frames()[0]
                img_read_time = time.time() - start_time

                if img_read_time < cam_conf.timeout_read :
                    time.sleep(cam_conf.timeout_read - img_read_time)

                images_shared[cam_id] = image

                if image is not None:
                    error_reads_count = 0
                else:
                    logging.warning(f"Camera '{cam_conf.name}' read-error!")
                    error_reads_count += 1

                    if error_reads_count >= cam_conf.count_err:
                        raise RuntimeError(f"Too many ({error_reads_count}) read errors!")
        except Exception as e:
            logging.error(f"Exception for camera '{cam_conf.name}': {repr(e)}.")
        finally:
            if mreader is not None:
                try:
                    mreader.release([0])
                except Exception as e2:
                    logging.critical(f"Reset error for camera '{cam_conf.name}': {repr(e2)}.")
            logging.info(f"Process stop for camera '{cam_conf.name}'.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class FolderImgReader():
    def __init__(self, root_path:str):
        self.images = []
        for ext in ("png", "jpg", "jpeg", "tiff"):
            for filepath in pathlib.Path(root_path).glob(f"**/*.{ext}"):
                self.images.append(str(filepath))

        def natural_keys(text:str):
            atoi = lambda s: int(s) if s.isdigit() else s
            return [ atoi(c) for c in re.split('(\d+)', text) ]
        self.images = sorted(self.images, key=natural_keys)

        self.it = -1
    
    def __iter__(self) :
        return self
    
    def __next__(self) :
        if len(self.images) == self.it + 1:
            raise StopIteration
        else:
            return self.read()

    def read(self):
        self.it += 1
        if len(self.images) <= self.it:
            raise RuntimeError("All images are read.")
        img_pathname = self.images[self.it]
        img = cv2.imread(img_pathname)
        return img is not None, img

    def release(self):
        self.images = []

    def isOpened(self):
        return True

def setup_logger(verbose:bool, filepath:Optional[str] = None) :
    logging.basicConfig(
        filename=filepath,
        # "w"-mode might not work, for root-log file-writing (e.g. collision with other loggers or some bug?)
        filemode="a",
        format="%(asctime)s %(process)6d %(levelname)8s %(message)s",
        datefmt="%Y.%m.%d %H:%M:%S",
        #encoding="utf-8", # not supported in 3.6
        level=logging.INFO if verbose else logging.ERROR)

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Record test data from cameras")
    parser.add_argument("config_json_path", type=str, help="Path to the main config file")
    args = parser.parse_args()

    with open(args.config_json_path, "r") as file:
        config_dict = json.load(file)
    
    setup_logger(config_dict["Watchdog"]["verbose"])

    save_path = config_dict["Watchdog"]["output_path"]
    if not os.path.exists(save_path):
        raise ValueError(f"Output path '{save_path}' doesn't exit")

    cams_conf = CameraConfig.parse(config_dict["Cameras"])
    with CamMultiprocReader( cams_conf ) as mreader:
        frame_group_id, num_frames2capture = 0, 10
        while True:
            time.sleep(1)
            images = mreader.read_frames()

            if np.sum([(v is not None) for v in images]) != len(cams_conf):
                continue

            frame_group_id += 1
            if frame_group_id > num_frames2capture:
                break

            for cam_id, (cam_conf, img) in enumerate(zip(cams_conf, images)) :
                save_path_cam = os.path.join(save_path, f"cam_{cam_conf.name}")
                if not os.path.exists(save_path_cam):
                    os.makedirs(save_path_cam)

                time_str = time.strftime(f"%Y%m%d_%H%M%S")
                filename = f"{time_str}_frame{frame_group_id:05d}.jpg"
                # Some elementary processing
                print(f"{filename} avg colors: {np.average(img, axis=(0,1))}.")

                cv2.imwrite(os.path.join(save_path_cam, filename), img)

