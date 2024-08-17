from typing import Dict, List, Optional

import os, time, logging, shutil
import concurrent.futures, multiprocessing
import multiprocessing.managers
import numpy as np

from .detector import DetectResult, Detector # should go before OpenCV due to some bug...
from .camera import CameraConfig, CamMultiprocReader, setup_logger
from .bot import CameraTeleBotComm

import cv2

class MultiCamWatchdog():
    """ Director of multi-camera(multi-process) capture and nn-model detection """

    def __init__( self, config_dict : Dict ):
        """ Parse config-file settings """
        try:
            self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            self._manager = multiprocessing.Manager()
            self._ns = self._manager.Namespace() # multi-process interchange
            self._jobs = []

            self._output_path = config_dict["Watchdog"]["output_path"]
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

            setup_logger(
                config_dict["Watchdog"]["verbose"],
                filepath=os.path.join(self._output_path, "log.txt"))

            # Init Yolo-detector
            self._detector = Detector(**config_dict["Detector"]["Init"])

            # Key Detection parameters
            self._ns.detect_categories = config_dict["Detector"]["categories"]
            self._ns.detect_conf_thr = config_dict["Detector"]["confidence_threshold"]
            self._ns.bbox_merge_dist = config_dict["Detector"]["bbox_merge_dist"]

            # Extract cameras settings
            self._cams_config = CameraConfig.parse(config_dict["Cameras"])

            # Log images with given frequency
            self._ns.img_log_timeout = config_dict["Watchdog"]["img_log_timeout"] # seconds
            self._ns.img_log_time = 0

            # Init Telegram bot
            token = config_dict["Bot"]["token"]
            chat_id_str = config_dict["Bot"]["chat_id"]

            # How often bot notifications can appear
            self._ns.bot_warning_timeout = config_dict["Bot"]["bot_warning_timeout"] # seconds
            self._ns.bot_warning_time = 0

            self._ns.bot = None
            if token != "" and chat_id_str != "":
                self._ns.bot = CameraTeleBotComm(
                    self._ns.bot_warning_timeout,
                    self._ns.detect_conf_thr,
                    self._ns.detect_categories,
                    token, int(chat_id_str) )

            self._run_pipeline()
        except Exception as e:
            logging.critical(repr(e))
            raise

    def _run_pipeline(self):
        with CamMultiprocReader(self._cams_config) as cams_reader:
            detections_total = [0] * len(self._cams_config)
            while True: # Capture till the End of Time...
                timestamp_str, self._ns.start_time = time.strftime(f"%Y%m%d_%H%M%S"), time.time()

                img_root_path = os.path.join(self._output_path, timestamp_str[:8])# new folder everyday!
                if not os.path.exists(img_root_path):
                    os.makedirs(img_root_path)

                img_cams_all = cams_reader.read_frames()
                img_cams_act = [img for img in img_cams_all if img is not None]
                if len(img_cams_act) > 0:
                    detections_act = self._detector.detect(
                        img_cams_act, self._ns.detect_conf_thr, self._ns.detect_categories)
                else:
                    detections_act = []

                cam_ids_act = []
                for cam_id, image in enumerate(img_cams_all) :
                    if image is None:
                        continue
                    
                    detection = detections_act[len(cam_ids_act)]
                    cam_ids_act.append(cam_id)
                    if detection.num_objects == 0:
                        continue

                    detections_total[cam_id] += 1
                    # In a separate process finalize (merge) detection results, save images and send message
                    self._run_process(
                        self._warning_process,
                        self._ns,
                        detection,
                        image,
                        f"#{cam_id+1} {self._cams_config[cam_id].name}",
                        os.path.join(img_root_path, timestamp_str + "_cam" + str(cam_id+1)) )

                inference_time = time.time() - self._ns.start_time
                statistics_msg = f"Inference({len(img_cams_act):3} cams) {inference_time:5.3f} sec. " \
                                 f"Free space: {self._desc_mem(img_root_path)}. " \
                                 f"Detections by camera: {detections_total}."
                logging.info(statistics_msg)

                # Save images and send log message (if bot-status request happened)
                self._run_process(
                    self._log_process,
                    self._ns,
                    img_cams_act,
                    cam_ids_act,
                    os.path.join(img_root_path, timestamp_str),
                    statistics_msg )

    @staticmethod
    def _log_process(
        ns: multiprocessing.managers.Namespace,
        img_cams_act: List[np.ndarray],
        cam_ids: List[int],
        img_path_prefix:str,
        statistics_msg:str
    ) :
        try :
            # Flag for regular saving of captured images (logging)
            is_img_log_active = ns.start_time - ns.img_log_time > ns.img_log_timeout

            bot_status_request = False
            if ns.bot is not None :
                bot = ns.bot
                if bot.parse() :
                    bot_status_request, bot.status = bot.status, bot_status_request

                    ns.bot_warning_timeout = bot.warning_timeout
                    ns.detect_conf_thr = bot.detect_conf_thr
                    ns.detect_categories = bot.detect_categories
                ns.bot = bot # trigger bot update for all processes

            if bot_status_request or is_img_log_active :
                if bot_status_request:
                    _ = ns.bot.send_message(statistics_msg)

                for cam_id, image in zip(cam_ids, img_cams_act):
                    img_path = img_path_prefix + f"_cam{cam_id+1}_status.jpg"
                    cv2.imwrite(img_path, image)

                    if bot_status_request :
                        with open(img_path, "rb") as img_file:
                            _ = ns.bot.send_image(img_file, os.path.split(img_path)[1])
                if len(img_cams_act) > 0: # if there are new images then update log timer
                    ns.img_log_time = time.time()
        except Exception as e:
            logging.error(f"Notifier exception: {repr(e)}")

    @staticmethod
    def _warning_process(
        ns: multiprocessing.managers.Namespace,
        detection: DetectResult,
        image:np.ndarray,
        cam_desc:str,
        img_path_prefix:str
    ) :
        try :
            # Flag allows bot notifications when something is detected
            is_bot_warn_active = ns.start_time - ns.bot_warning_time > ns.bot_warning_timeout

            detection_m = detection.merge(ns.bbox_merge_dist)
            description = f"{cam_desc}: {detection_m.describe()}"
            logging.info(description)

            img_path = img_path_prefix + "_detect.jpg"
            cv2.imwrite(img_path, detection_m.draw(image))

            if is_bot_warn_active and ns.bot is not None:
                is_ok1 = ns.bot.send_message(description)
                with open(img_path, "rb") as img_file:
                    is_ok2 = ns.bot.send_image(img_file, os.path.split(img_path)[1])
                if is_ok1 and is_ok2:
                    ns.bot_warning_time = time.time()

            # Save original image for debugging/training etc.
            cv2.imwrite(img_path_prefix + "_detect_raw.jpg", image)
        except Exception as e:
            logging.error(f"Notifier exception: {repr(e)}")

    def _run_process(self, function, *args) :
        self._jobs = [job for job in self._jobs if not job.done()]
        logging.debug(f"Active notify jobs {len(self._jobs)}.")
        self._jobs.append(self._executor.submit(function, *args))

    def _desc_mem(self, path:str) :
        mem_gpu_GiB, mem_gpu_rel = self._detector.get_device_memory_available()
        mem_disk_total, _, mem_disk_free = shutil.disk_usage(path)
        mem_disk_rel = mem_disk_free / mem_disk_total
        mem_disk_GiB = mem_disk_free / 2**30
        return f"VRAM {mem_gpu_GiB:3.1f}GiB({mem_gpu_rel:.2f}); DISK {mem_disk_GiB:5.1f}GiB({mem_disk_rel:.2f})"
