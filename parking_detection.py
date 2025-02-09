import cv2
import json
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

class ParkingMonitor:
    def __init__(self):
        self.points = []
        self.model = YOLO('yolov8n.pt')
        self.warp_size = (300, 150)  # Initial warp size (width, height)
        self.perspective_matrix = None
        self.current_img = None
        self.last_key = None
        self.setup_gui()
        self.load_config()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def mouse_handler(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            img = params[0].copy()
            self.draw_ui_elements(img)
            cv2.imshow('Parking Space Definition', img)

    def draw_ui_elements(self, img):
        for i, (x, y) in enumerate(self.points):
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(img, self.points[i - 1], (x, y), (0, 255, 0), 2)
        if len(self.points) == 4:
            cv2.polylines(img, [np.array(self.points)], True, (0, 255, 0), 2)

    def calculate_perspective(self):
        src = np.array(self.points, dtype=np.float32)
        dst = np.array([[0, 0], [self.warp_size[0], 0],
                        [self.warp_size[0], self.warp_size[1]], [0, self.warp_size[1]]],
                       dtype=np.float32)
        self.perspective_matrix = cv2.getPerspectiveTransform(src, dst)
        self.save_config()

    def save_config(self):
        config = {
            'points': self.points,
            'warp_size': self.warp_size,
            'perspective_matrix': self.perspective_matrix.tolist() if self.perspective_matrix is not None else None
        }
        with open('parking_config.json', 'w') as f:
            json.dump(config, f)

    def load_config(self):
        try:
            with open('parking_config.json', 'r') as f:
                config = json.load(f)
                self.points = [tuple(p) for p in config['points']]
                self.warp_size = tuple(config['warp_size'])
                if config['perspective_matrix']:
                    self.perspective_matrix = np.array(config['perspective_matrix'])
                return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False

    def define_parking_space(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return False

        self.points = []
        cv2.namedWindow('Parking Space Definition')
        cv2.setMouseCallback('Parking Space Definition', self.mouse_handler, [img])

        while True:
            display_img = img.copy()
            self.draw_ui_elements(display_img)
            cv2.imshow('Parking Space Definition', display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and len(self.points) == 4:
                self.calculate_perspective()
                break

        cv2.destroyAllWindows()
        return len(self.points) == 4

    def process_image(self, img):
        if self.perspective_matrix is None:
            return

        # Warp the parking space
        warped = cv2.warpPerspective(img, self.perspective_matrix, self.warp_size)

        # Detect vehicles in warped space
        results = self.model(warped, verbose=False)

        # Clear previous detections
        display_img = img.copy()
        warped_display = warped.copy()

        occupied = False
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 2:  # Car class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    occupied = True

                    # Draw in warped view (thicker lines)
                    cv2.rectangle(warped_display, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Transform coordinates with boundary checking
                    try:
                        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                        transformed_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2),
                                                                  np.linalg.inv(self.perspective_matrix))
                        cv2.polylines(display_img, [np.int32(transformed_pts)], True, (0, 255, 0), 3)
                    except Exception as e:
                        print(f"Transform error: {e}")

        # Draw parking space overlay (thicker lines)
        color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.polylines(display_img, [np.array(self.points, dtype=np.int32)], True, color, 3)

        # Show results
        cv2.imshow('Parking Monitor', display_img)
        cv2.imshow('Warped View', cv2.resize(warped_display, (600, 300)))

    def run(self):
        if not self.load_config():
            image_path = filedialog.askopenfilename(title="Select Reference Image")
            if image_path and not self.define_parking_space(image_path):
                return

        cv2.namedWindow('Parking Monitor')
        cv2.resizeWindow('Parking Monitor', 800, 600)
        cv2.createTrackbar('Warp Width', 'Parking Monitor', self.warp_size[0], 600, lambda x: None)
        cv2.createTrackbar('Warp Height', 'Parking Monitor', self.warp_size[1], 600, lambda x: None)
        cv2.namedWindow('Warped View')
        cv2.resizeWindow('Warped View', 600, 300)

        while True:
            # Force focus on main window
            cv2.setWindowProperty('Parking Monitor', cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty('Parking Monitor', cv2.WND_PROP_TOPMOST, 0)

            # Update warp size
            new_width = cv2.getTrackbarPos('Warp Width', 'Parking Monitor')
            new_height = cv2.getTrackbarPos('Warp Height', 'Parking Monitor')

            if (new_width, new_height) != self.warp_size and len(self.points) == 4:
                self.warp_size = (new_width, new_height)
                self.calculate_perspective()
                if self.current_img is not None:
                    self.process_image(self.current_img.copy())

            # Key handling with longer wait time
            key = cv2.waitKey(100) & 0xFF
            if key != 255:
                self.last_key = key
                print(f"Key pressed: {chr(key)}")  # Debug output

            if self.last_key == ord('o'):
                image_path = filedialog.askopenfilename(title="Select Test Image")
                if image_path:
                    self.current_img = cv2.imread(image_path)
                    if self.current_img is not None:
                        self.process_image(self.current_img.copy())
                self.last_key = None

            elif self.last_key == ord('r'):
                image_path = filedialog.askopenfilename(title="Select New Reference Image")
                if image_path:
                    self.define_parking_space(image_path)
                self.last_key = None

            elif self.last_key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = ParkingMonitor()
    monitor.run()