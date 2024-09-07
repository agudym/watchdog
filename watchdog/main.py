from typing import List, Optional
import os, time, datetime, logging, argparse
import multiprocessing, multiprocessing.managers

import numpy as np

# sklearn (import from .utils) should go before cv2 due to some bug...
from .utils import load_config, setup_logger, DetectResult, desc_system_resources
import cv2

from .camera import CameraConfig, CamMultiprocReader
from .bot import CameraTeleBotComm

class ProcessIO :
    """
    Auxiliary settings/data structure, with the 2 kind of fields:
    1. local process data, originally shared via COPY between processes 
    2. global exchangeable data between processes
    """
    def __init__(self, config_json_path:str):
        config_dict = load_config(config_json_path)

        self.loc_output_path = config_dict["Watchdog"]["output_path"]
        if not os.path.exists(self.loc_output_path):
            os.makedirs(self.loc_output_path)

        setup_logger(
            config_dict["Watchdog"]["verbose"],
            filepath=os.path.join(self.loc_output_path, "log.txt"))

        self._manager = multiprocessing.Manager()
        self.glob = self._manager.Namespace() # multi-process interchange
        # Signalling conditional variables
        self.glob_condition_notify = multiprocessing.Condition()
        self.glob_condition_stats = multiprocessing.Condition()

        self.loc_stop_file = os.path.join(self.loc_output_path, "stop")
        if os.path.exists(self.loc_stop_file):
            raise RuntimeError(f"Can't start, because {self.loc_stop_file} file already exists, delete it first.")
        self.glob.is_stopped = False

        # NN-detector configuration
        self.loc_detector_config = config_dict["Detector"]["Init"]

        # Key Detection parameters
        self.loc_category_names_all = config_dict["Detector"]["categories_all"].split(",")
        self.loc_bbox_merge_dist = config_dict["Detector"]["bbox_merge_dist"]
        category_names_notify = config_dict["Detector"]["categories_notify"].split(",")
        # Bot-updated values
        self.glob.category_ids_notify = DetectResult.get_ids(self.loc_category_names_all, category_names_notify)
        self.glob.detect_conf_thr = config_dict["Detector"]["confidence_threshold"]

        # Extract cameras settings
        self.loc_cams_config = CameraConfig.parse(config_dict["Cameras"])
        self.glob.detections_total = [0] * len(self.loc_cams_config)

        # Log images with given frequency
        self.loc_img_log_timeout = config_dict["Watchdog"]["img_log_timeout"] # seconds
        self.loc_img_log_time = 0.

        # Init Telegram bot
        token = config_dict["Bot"]["token"]
        chat_id_str = config_dict["Bot"]["chat_id"]

        # How often bot notifications can appear
        self.glob.bot_warning_timeout = config_dict["Bot"]["bot_warning_timeout"] # seconds
        self.loc_bot_warning_time = 0.

        self.loc_bot = None
        if token != "" and chat_id_str != "":
            self.loc_bot = CameraTeleBotComm(
                self.glob.bot_warning_timeout,
                self.glob.detect_conf_thr,
                self.loc_category_names_all,
                category_names_notify,
                token, int(chat_id_str) )

    def add_cameras_image(
            self,
            img_cams_all: List[Optional[np.ndarray]],
            detections_all: List[DetectResult],
            timestamp: float,
            inference_time: float ) :
        self.glob.img_data = (img_cams_all, detections_all, timestamp, inference_time)

    def get_cameras_image(self) :
        return self.glob.img_data

    def get_img_path(self, timestamp:float) :
        timestamp_str = datetime.datetime.fromtimestamp(timestamp).strftime(f"%Y%m%d_%H%M%S")
        img_root_path = os.path.join(self.loc_output_path, timestamp_str[:8])# new folder everyday!
        return img_root_path, timestamp_str

    def stop(self) :
        self.glob.is_stopped = True
        with self.glob_condition_notify :
            self.glob_condition_notify.notify()
        with self.glob_condition_stats :
            self.glob_condition_stats.notify()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def detect_process(proc_io: ProcessIO):
    """ Read batch of images and run detection, fire conditionals for saving/notification """
    try:
        num_cams = len(proc_io.loc_cams_config)
        cams_reader = None
        cams_reader = CamMultiprocReader(proc_io.loc_cams_config)

        # Imports Pytorch, AFTER camera processes start to preserve memory
        from .detector import Detector
        detector = Detector(**proc_io.loc_detector_config)

        inference_time = 0
        while not proc_io.glob.is_stopped :
            timestamp = time.time()

            if os.path.exists(proc_io.loc_stop_file):
                logging.info(f"The Watchdog has been stopped with {proc_io.loc_stop_file}.")
                break

            img_cams_all = cams_reader.read_frames()

            detections_total = proc_io.glob.detections_total[:]
            detection_trigger = False
            detections_all = [DetectResult()] * num_cams
            img_cams_active = [img for img in img_cams_all if img is not None]
            if len(img_cams_active) > 0:
                detections_active = detector.detect(
                    img_cams_active,
                    proc_io.glob.detect_conf_thr,
                    proc_io.glob.category_ids_notify)
                j = 0
                for cam_id, img in enumerate(img_cams_all):
                    if img is not None:
                        detections_all[cam_id] = detections_active[j]
                        j += 1
                        if detections_all[cam_id].num_objects > 0 :
                            detection_trigger = True
                            detections_total[cam_id] += 1

            if detection_trigger:
                with proc_io.glob_condition_notify :
                    proc_io.glob.detections_total = detections_total
                    proc_io.add_cameras_image(img_cams_all, detections_all, timestamp, inference_time )
                    proc_io.glob_condition_notify.notify()

            with proc_io.glob_condition_stats :
                proc_io.add_cameras_image(img_cams_all, detections_all, timestamp, inference_time )
                proc_io.glob_condition_stats.notify()
            # Previous value is logged
            inference_time = time.time() - timestamp

    except Exception as e:
        logging.critical(f"Detection process exception: {repr(e)}")
    finally:
        logging.debug("Detection process exits.")
        proc_io.stop()
        if cams_reader is not None:
            cams_reader.stop()

def save_image(img_path:str, img:np.ndarray) :
    cv2.imwrite(img_path, img)
    if os.path.exists(img_path) :
        logging.info(f"Saved {img_path}")
        return True
    else:
        logging.error(f"Failed to save {img_path}")
        return False

def notify_process(proc_io: ProcessIO) :
    """ Save images with detections, notify user via bot """
    try :
        bot = proc_io.loc_bot
        while True :
            with proc_io.glob_condition_notify :
                if not proc_io.glob.is_stopped:
                    proc_io.glob_condition_notify.wait()
                else:
                    break
                img_cams_all, detections_all, timestamp, inference_time = proc_io.get_cameras_image()
                img_root_path, timestamp_str = proc_io.get_img_path(timestamp)

            for cam_id, (img, detection) in enumerate(zip(img_cams_all, detections_all)) :
                if img is None or detection.num_objects == 0:
                    continue
                detection_m: DetectResult = detection.merge(proc_io.loc_bbox_merge_dist)

                cam_name = proc_io.loc_cams_config[cam_id].name
                description = f"#{cam_id+1} {cam_name}: {detection_m.describe(proc_io.loc_category_names_all)}"
                logging.info(description)

                img_path = os.path.join(img_root_path, f"{timestamp_str}_cam{cam_id+1}_detect.jpg")
                is_img_saved = save_image(img_path, detection_m.draw(img, proc_io.loc_category_names_all))

                # Flag allows bot notifications when something is detected
                is_bot_warn_active = timestamp - proc_io.loc_bot_warning_time > proc_io.glob.bot_warning_timeout
                if bot is not None and is_bot_warn_active and bot.send_message(description) and is_img_saved:
                    with open(img_path, "rb") as img_file:
                        if bot.send_image(img_file, os.path.split(img_path)[1]) :
                            proc_io.loc_bot_warning_time = time.time()

                # Save original image for debugging/training etc.
                img_path = os.path.join(img_root_path, f"{timestamp_str}_cam{cam_id+1}_detect_raw.jpg")
                save_image(img_path, img)
    except Exception as e:
        logging.critical(f"Notify process exception: {repr(e)}")
    finally:
        logging.debug("Notify process exits.")
        proc_io.stop()

def stats_process(proc_io: ProcessIO) :
    """ Save all images regularly, parse user input """
    try :
        bot = proc_io.loc_bot
        while True :
            with proc_io.glob_condition_stats :
                if not proc_io.glob.is_stopped :
                    proc_io.glob_condition_stats.wait()
                else:
                    break
                img_cams_all, detections_all, timestamp, inference_time = proc_io.get_cameras_image()
                detections_total = proc_io.glob.detections_total[:]
                img_root_path, timestamp_str = proc_io.get_img_path(timestamp)

            if not os.path.exists(img_root_path):
                os.makedirs(img_root_path)

            num_cams_active = 0
            for img in img_cams_all:
                num_cams_active += int(img is not None)

            statistics_msg = f"Inference({num_cams_active:3} cams) {inference_time:5.3f} sec. Detections: {detections_total}. " \
                             f"{desc_system_resources(img_root_path)}"
            logging.info(statistics_msg)

            # Flag for regular saving of captured images (logging)
            is_img_log_active = timestamp - proc_io.loc_img_log_time > proc_io.loc_img_log_timeout

            bot_status_request = False
            if bot is not None and bot.parse():
                bot_status_request, bot.status = bot.status, bot_status_request

                proc_io.glob.bot_warning_timeout = bot.warning_timeout
                proc_io.glob.detect_conf_thr = bot.detect_conf_thr
                proc_io.glob.category_ids_notify = DetectResult.get_ids(proc_io.loc_category_names_all, bot.categories_notify)

                if bot.exit:
                    logging.info(f"The Watchdog has been stopped with the bot.")
                    _ = bot.send_message("Goodbye!")
                    break

            if bot_status_request or is_img_log_active :
                if bot_status_request:
                    _ = bot.send_message(statistics_msg)

                for cam_id, img in enumerate(img_cams_all):
                    if img is None:
                        continue
                    img_path = os.path.join(img_root_path, f"{timestamp_str}_cam{cam_id+1}_status.jpg")
                    is_img_saved = save_image(img_path, img)

                    if bot_status_request and is_img_saved:
                        with open(img_path, "rb") as img_file:
                            _ = bot.send_image(img_file, os.path.split(img_path)[1])

                if num_cams_active > 0: # if there are new images then update log timer
                    proc_io.loc_img_log_time = time.time()
    except Exception as e:
        logging.critical(f"Stats process exception: {repr(e)}")
    finally:
        logging.debug("Stats process exits.")
        proc_io.stop()

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Run watchdog: Initialize detector, cameras and communication bot, according to the configuration. "
                        "Create empty file 'stop' in the output dir to interrupt the process. See ReadMe.md for help!")
        parser.add_argument("config_json_path", type=str, help="Path to the main config file")
        args = parser.parse_args()

        with ProcessIO(args.config_json_path) as proc_io:
            # Starting 3 main processes
            processes = []

            processes.append( multiprocessing.Process(target=detect_process, args=(proc_io,)) )
            processes.append( multiprocessing.Process(target=notify_process, args=(proc_io,)) )
            processes.append( multiprocessing.Process(target=stats_process, args=(proc_io,)) )

            for p in processes:
                p.start()
            for p in processes:
                p.join()

        logging.info("Watchdog stops. Goodbye!")

    except Exception as e:
        logging.critical(f"Main process exception: {repr(e)}")

