from ultralytics import YOLO
import os
import cv2
import numpy as np

MODEL_DIR = "./models"


class ParkingLotDetector:
    def __init__(self, model_size='yolo11s.pt', conf_thres=0.45, iou_thres=0.4, occl_thres=0.3, classes=None):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.occl_threshold = occl_thres
        self.classes = classes if classes else [2, 5, 7]
        self.parking_areas = []
        self.model = YOLO(os.path.join(MODEL_DIR, model_size))

    def set_parking_areas(self, polygons):
        self.parking_areas = [np.array(poly, np.int32) for poly in polygons]

    def _calculate_overlap(self, boxes, area, frame_shape):
        """
        Рассчитывает процент перекрытия парковочного места (area) объединёнными объектами из списка boxes.
        :param boxes: список прямоугольников (x1, y1, x2, y2) обнаруженных объектов.
        :param area: np.array – контур парковочного места.
        :param frame_shape: tuple – (высота, ширина) изображения для создания маски.
        :return: float – процент закрытости парковочного места (от 0.0 до 1.0).
        """
        # Создаём маску для парковочного места
        parking_mask = np.zeros(frame_shape, dtype=np.uint8)
        cv2.fillPoly(parking_mask, [area], 1)

        # Создаём маску для объединения всех обнаруженных объектов
        detection_mask = np.zeros(frame_shape, dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(detection_mask, (x1, y1), (x2, y2), 1, thickness=-1)

        # Вычисляем пересечение масок
        intersection_mask = cv2.bitwise_and(parking_mask, detection_mask)
        intersection_area = cv2.countNonZero(intersection_mask)
        parking_area_pixels = cv2.countNonZero(parking_mask)

        if parking_area_pixels > 0:
            return intersection_area / parking_area_pixels
        return 0.0

    def detect_and_get_detections(self, frame):
        results = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres,
                                     classes=self.classes, verbose=False)

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        occupied_spots = set()
        frame_shape = frame.shape[:2]

        # Для каждого парковочного места вычисляем объединённую степень перекрытия
        for idx, area in enumerate(self.parking_areas):
            occlusion = self._calculate_overlap(boxes, area, frame_shape)
            if occlusion >= self.occl_threshold:
                occupied_spots.add(idx)

        return occupied_spots, boxes.tolist()

    def visualize_results(self, frame, occupied_spots, detections=None):
        detections = detections or []
        frame_shape = frame.shape[:2]

        for idx, area in enumerate(self.parking_areas):
            # Вычисляем процент занятости парковочного места, используя объединённую маску обнаруженных объектов
            occlusion = self._calculate_overlap(detections, area, frame_shape)
            occlusion_percentage = min(int(occlusion * 100), 100)

            # Определяем цвет: красный (занято) или зелёный (свободно)
            color = (0, 0, 255) if idx in occupied_spots else (0, 255, 0)

            # Определяем центр парковочного места
            moments = cv2.moments(area)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = area.mean(axis=0).astype(int)

            # Отрисовываем контур парковочного места и процент покрытия в его центре
            cv2.polylines(frame, [area], True, color, 2)
            cv2.putText(frame, f"{occlusion_percentage}%", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Отрисовка обнаруженных объектов (прямоугольники)
        if detections:
            for (x1, y1, x2, y2) in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return frame
