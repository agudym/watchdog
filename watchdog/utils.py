from typing import Optional, Union, List, Dict

import sys, shutil, logging, json
import numpy as np

import psutil

# sklearn should go before cv2 due to some bug...
from sklearn.cluster import DBSCAN
import cv2

class DetectResult():
    """
    List of detected objects (class/object category IDs), confidences, bounding boxes. With auxiliary image-drawing routines.

    Each individual image must have a specific `DetectResult`.
    """

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
    
    @staticmethod
    def get_ids(category_names_all:List[str], category_names_subset:List[str]) :
        """ String names to indices in the main list `category_names_all` """
        return np.where(np.isin(category_names_all, category_names_subset))[0]

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

    def draw(self, image_orig: np.ndarray, category_names_all: List[str], category_colors: Optional[np.ndarray]=None) :
        """
        Draw found object's bounding boxes in the image (return modified copy).
        """
        image = image_orig.copy()
        h, w, _ = image.shape
        
        bboxes_pix = (self.bboxes * (w,h,w,h)).astype(int)
        for category_id, conf, bbox in zip(self.category_ids, self.confidence, bboxes_pix) :
            c = category_colors[category_id] if category_colors is not None else (0, 255, 0)
            text = f"{category_names_all[category_id]} {conf:.2f}"
            cv2.putText(image, text, tuple(bbox[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, c, thickness=2)
            cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), c, 1)
            # For the history - work with matplotlib
            # w, h = bbox[2:] - (x,y)
            # ax.add_patch(patches.Rectangle(
            #     (x,y), w, h, linewidth=1, edgecolor=c, facecolor="none"))
            # ax.text(x, y, f"{model.class_names[class_id]}\np={prob:.2f}", weight="bold", va="top", color=c)
        return image
    
    def describe(self, category_names_all: List[str]):
        """ Short description of the detection results """
        description = ""
        for i, (category_id, conf) in enumerate(zip(self.category_ids, self.confidence)) :
            if i > 0:
                description += "; "
            description += f"{category_names_all[category_id]} {conf:.2f}"
        return description + "."

def setup_logger(verbose:bool, filepath:Optional[str] = None) :
    class StreamToLogger(object):
        """
        Redirect STDOUT and STDERR to the root-log
        originates from 
        https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
        """
        def __init__(self, logger:logging.Logger, level:int):
            self.logger = logger
            self.level = level
            self.linebuf = ''
        def write(self, buf:str):
            for line in buf.splitlines():
                line = line.rstrip()
                if len(line) > 0:
                    self.logger.log(self.level, line)
        def flush(self):
            pass

    logging.basicConfig(
        format="%(asctime)s %(process)6d %(levelname)8s %(message)s",
        handlers=[logging.FileHandler(filepath, encoding="utf-8")] if filepath is not None else None,
        level=logging.INFO if verbose else logging.ERROR)

    root_log = logging.getLogger("")
    sys.stdout = StreamToLogger(root_log, logging.INFO)
    sys.stderr = StreamToLogger(root_log, logging.ERROR)

def load_config(config_json_path: str):
    with open(config_json_path, "r") as file:
        return json.load(file)

def save_config(config_json_path: str, config_dict: Dict):
    with open(config_json_path, "w") as file:
        file.write(json.dumps(config_dict, indent=4))

def generate_rgb8_colors(num_classes:int, cmap_name:str="hsv"):
    # Note some imports on the first run, not globally for the sake of multi-process memory preservation
    from numpy.random import MT19937, RandomState, SeedSequence
    import matplotlib.pyplot as plt

    rs = RandomState(MT19937(SeedSequence(0)))
    cmap = plt.get_cmap(cmap_name, num_classes) # cm.get_cmap
    colors = []
    for i in rs.permutation(np.arange(num_classes)):
        color = cmap(i)
        colors.append([int(color[j] * 255) for j in range(3)])
    return colors

def desc_system_resources(path:str) :
    """ Describe CPU load, RAM, DISK memory available """
    mem_ram_total, mem_ram_free = psutil.virtual_memory().total, psutil.virtual_memory().available
    mem_ram_rel = mem_ram_free / mem_ram_total
    mem_ram_GiB = mem_ram_free / 2**30
    mem_ram_desc = f"{mem_ram_GiB:5.2f}GiB({mem_ram_rel:.2f})"

    mem_disk_total, _, mem_disk_free = shutil.disk_usage(path)
    mem_disk_rel = mem_disk_free / mem_disk_total
    mem_disk_GiB = mem_disk_free / 2**30
    mem_disk_desc = f"{mem_disk_GiB:5.1f}GiB({mem_disk_rel:.2f})"

    cpu_usage = ", ".join([f"{v/100:.2f}" for v in psutil.cpu_percent(percpu=True)])

    return f"Free space: RAM {mem_ram_desc}; DISK {mem_disk_desc}. CPU usage: [{cpu_usage}]."
