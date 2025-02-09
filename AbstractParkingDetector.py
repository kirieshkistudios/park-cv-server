from abc import ABC, abstractmethod
from ultralytics import YOLO
import os
import numpy as np

MODEL_DIR = "./models"
# --------------------- New Detector Classes ---------------------
class AbstractParkingDetector(ABC):
    def __init__(self, model_size='s', conf_thres=0.45, iou_thres=0.4):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.parking_areas = []  # List of parking spot polygons (as numpy arrays)
        self.model = None

    def load_model(self, model):
        self.model = YOLO(os.path.join(MODEL_DIR, model))

    def set_parking_areas(self, polygons):
        self.parking_areas = [np.array(poly, np.int32) for poly in polygons]

    @abstractmethod
    def detect_and_get_detections(self, frame):
        """Run detection and return occupancy info and detections."""
        pass

    @abstractmethod
    def visualize_results(self, frame, occupied_spots, detections=None):
        """Visualize the parking areas and detections."""
        pass