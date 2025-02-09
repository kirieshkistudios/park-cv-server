import os
import cv2
import numpy as np
from ultralytics import YOLO


class ParkingLotDetector:
    """
    Класс для детекции автомобилей на парковке с использованием модели YOLO.
    Позволяет анализировать изображение, определять занятость парковочных мест и визуализировать результаты.
    """

    def __init__(self, model_dir="./models", model_size='yolo11s.pt', conf_thres=0.45, iou_thres=0.4, occl_thres=0.3,
                 classes=None):
        """
        Инициализация детектора парковки.

        :param model_dir: str - путь к директории с моделью.
        :param model_size: str - название модели YOLO.
        :param conf_thres: float - порог уверенности детекции.
        :param iou_thres: float - порог NMS.
        :param occl_thres: float - минимальный процент перекрытия, при котором место считается занятым.
        :param classes: list - список классов YOLO для детекции (по умолчанию: [2, 5, 7]).
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.occl_threshold = occl_thres
        self.classes = classes if classes else [2, 5, 7]
        self.parking_areas = []
        self.model = YOLO(os.path.join(model_dir, model_size))

    def set_parking_areas(self, polygons):
        """
        Устанавливает координаты парковочных мест.

        :param polygons: list - список полигонов парковочных мест.
        """
        self.parking_areas = [np.array(poly, np.int32) for poly in polygons]

    def _calculate_overlap(self, boxes, area, frame_shape):
        """
        Рассчитывает процент перекрытия парковочного места (area) объектами (boxes).

        :param boxes: list - список координат (x1, y1, x2, y2) обнаруженных объектов.
        :param area: np.array - контур парковочного места.
        :param frame_shape: tuple - размер изображения (высота, ширина).
        :return: float - процент закрытости парковочного места (0.0 - 1.0).
        """
        parking_mask = np.zeros(frame_shape, dtype=np.uint8)
        cv2.fillPoly(parking_mask, [area], 1)

        detection_mask = np.zeros(frame_shape, dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(detection_mask, (x1, y1), (x2, y2), 1, thickness=-1)

        intersection_mask = cv2.bitwise_and(parking_mask, detection_mask)
        intersection_area = cv2.countNonZero(intersection_mask)
        parking_area_pixels = cv2.countNonZero(parking_mask)

        return intersection_area / parking_area_pixels if parking_area_pixels > 0 else 0.0

    def detect_and_get_detections(self, frame):
        """
        Выполняет детекцию автомобилей и определяет занятые и свободные парковочные места.

        :param frame: np.array - изображение OpenCV.
        :return: tuple - множество занятых мест, список координат обнаруженных автомобилей, количество занятых и свободных мест.
        """
        results = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres, classes=self.classes,
                                     verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        occupied_spots = set()
        frame_shape = frame.shape[:2]

        for idx, area in enumerate(self.parking_areas):
            occlusion = self._calculate_overlap(boxes, area, frame_shape)
            if occlusion >= self.occl_threshold:
                occupied_spots.add(idx)

        total_spots = len(self.parking_areas)
        free_spots = total_spots - len(occupied_spots)

        return occupied_spots, boxes.tolist(), len(occupied_spots), free_spots

    def visualize_results(self, frame, occupied_spots, detections=None):
        """
        Визуализирует парковочные места и обнаруженные автомобили.

        :param frame: np.array - изображение OpenCV.
        :param occupied_spots: set - индексы занятых парковочных мест.
        :param detections: list - список координат обнаруженных автомобилей.
        :return: np.array - изображение с аннотациями.
        """
        detections = detections or []
        frame_shape = frame.shape[:2]

        for idx, area in enumerate(self.parking_areas):
            occlusion = self._calculate_overlap(detections, area, frame_shape)
            occlusion_percentage = min(int(occlusion * 100), 100)
            color = (0, 0, 255) if idx in occupied_spots else (0, 255, 0)

            moments = cv2.moments(area)
            cx, cy = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])) if moments[
                                                                                                         "m00"] != 0 else area.mean(
                axis=0).astype(int)

            cv2.polylines(frame, [area], True, color, 2)
            cv2.putText(frame, f"{occlusion_percentage}%", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return frame
