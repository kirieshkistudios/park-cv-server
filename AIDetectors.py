from AbstractParkingDetector import AbstractParkingDetector, MODEL_DIR
import cv2
import os 
import PIL
import numpy as np
from ultralytics import YOLO

class BBoxParkingDetector(AbstractParkingDetector):
    def detect_and_get_detections(self, frame):
        results = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres,
                                     classes=[2, 5, 7], verbose=False)
        occupied_spots = set()
        boxes = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            boxes.append((x1, y1, x2, y2))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            for idx, area in enumerate(self.parking_areas):
                if cv2.pointPolygonTest(area, (cx, cy), False) >= 0:
                    occupied_spots.add(idx)
                    break
        return occupied_spots, boxes

    def visualize_results(self, frame, occupied_spots, detections=None):
        for idx, area in enumerate(self.parking_areas):
            color = (0, 0, 255) if idx in occupied_spots else (0, 255, 0)
            cv2.polylines(frame, [area], True, color, 2)
            cv2.putText(frame, f"Spot {idx+1}", tuple(area[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if detections:
            for (x1, y1, x2, y2) in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame


class SegmentationParkingDetector(AbstractParkingDetector):
    def __init__(self, model_size='s', conf_thres=0.45, iou_thres=0.4):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.parking_areas = []  # List of parking spot polygons (as numpy arrays)
        self.model_path = os.path.join(MODEL_DIR, f'yolo11{model_size}-seg.pt')
        self._download_model_if_needed(model_size)
        self.model = YOLO(self.model_path)

    def detect_and_get_detections(self, frame):
        results = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres,
                                     classes=[2, 5, 7], verbose=False)
        occupied_spots = set()
        # Check that segmentation masks are available
        if not hasattr(results[0], 'masks') or results[0].masks is None:
            raise ValueError("Segmentation masks not available in the model output.")
        masks = results[0].masks.data.cpu().numpy()  # Assume shape (N, H, W)
        # Check each parking area for sufficient overlap with any detection mask
        for idx, area in enumerate(self.parking_areas):
            poly_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(poly_mask, [area], 255)
            poly_area = cv2.countNonZero(poly_mask)
            for seg_mask in masks:
                seg_mask_bin = (seg_mask > 0.5).astype(np.uint8) * 255
                intersection = cv2.bitwise_and(poly_mask, seg_mask_bin)
                inter_area = cv2.countNonZero(intersection)
                if inter_area > 0.1 * poly_area:  # More than 10% overlap
                    occupied_spots.add(idx)
                    break
        # For visualization, extract contours from segmentation masks
        detection_contours = []
        for seg_mask in masks:
            seg_mask_bin = (seg_mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(seg_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                detection_contours.append(c)
        return occupied_spots, detection_contours

    def visualize_results(self, frame, occupied_spots, detections=None):
        for idx, area in enumerate(self.parking_areas):
            color = (0, 0, 255) if idx in occupied_spots else (0, 255, 0)
            cv2.polylines(frame, [area], True, color, 2)
            cv2.putText(frame, f"Spot {idx+1}", tuple(area[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if detections:
            cv2.drawContours(frame, detections, -1, (255, 0, 0), 2)
        return frame