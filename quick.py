import os
import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

# --------------------- Modified ParkingSpotDetector Class ---------------------

# Directory for saving models
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure the models folder exists

class ParkingSpotDetector:
    def __init__(self, model_size='s', conf_thres=0.45, iou_thres=0.4):
        """
        :param model_size: n= nano, s= small, m= medium, l= large
        """
        self.model_path = os.path.join(MODEL_DIR, f'yolo11{model_size}.pt')
        self._download_model_if_needed(model_size)  # Ensure model is available
        self.model = YOLO(self.model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.parking_areas = []  # List of parking spot polygons (as numpy arrays)

    def _download_model_if_needed(self, model_size):
        """Check if model exists; if not, download it."""
        if not os.path.exists(self.model_path):
            print(f"Downloading model {self.model_path}...")
            try:
                # Instantiating a YOLO model with the given model name will download it if needed.
                YOLO(f'yolo11{model_size}.pt')
                print(f"Model saved to {self.model_path}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise

    def set_parking_areas(self, polygons):
        """Define parking spot regions using polygon coordinates."""
        self.parking_areas = [np.array(poly, np.int32) for poly in polygons]

    def detect_occupancy(self, frame):
        """Run detection and return a set of occupied parking spot indices."""
        results = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres, 
                                     classes=[2, 5, 7], verbose=False, augment=True)
        occupied_spots = set()
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            for idx, area in enumerate(self.parking_areas):
                if cv2.pointPolygonTest(area, (cx, cy), False) >= 0:
                    occupied_spots.add(idx)
                    break
        return occupied_spots

    def detect_and_get_detections(self, frame, scale=1.0):
        """
        Run detection and return both the occupancy info and bounding boxes of vehicles.
        The 'scale' parameter indicates that the image has been upscaled by that factor.
        The returned bounding boxes are scaled back to the original image coordinates.
        
        :returns: (occupied_spots, boxes)
                  occupied_spots: set of parking area indices that are occupied.
                  boxes: list of tuples (x1, y1, x2, y2) for detected vehicles.
        """
        results = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres, 
                                     classes=[2, 5, 7], verbose=False)
        
        occupied_spots = set()
        boxes = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            # Scale the coordinates back to the original image coordinate system.
            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
            boxes.append((x1, y1, x2, y2))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            for idx, area in enumerate(self.parking_areas):
                if cv2.pointPolygonTest(area, (cx, cy), False) >= 0:
                    occupied_spots.add(idx)
                    break
        return occupied_spots, boxes

    def visualize_results(self, frame, occupied_spots, boxes=None):
        """
        Draw parking spot regions and (optionally) vehicle bounding boxes on the frame.
        Parking areas are drawn in red (if occupied) or green (if free). Vehicle bounding boxes are drawn in blue.
        """
        for idx, area in enumerate(self.parking_areas):
            color = (0, 0, 255) if idx in occupied_spots else (0, 255, 0)
            cv2.polylines(frame, [area], True, color, 2)
            cv2.putText(frame, f"Spot {idx+1}", tuple(area[0]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if boxes:
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame

# -------------------------- Tkinter GUI Application --------------------------

class ParkingLotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Parking Lot Detector App")
        self.geometry("900x700")

        # Latest evaluated image is stored here for re-evaluation.
        self.latest_frame = None

        # Create Notebook with two tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        self.create_markdown_tab()
        self.create_evaluation_tab()

    # ----- Tab 1: Parking Lot Markdown (Define parking areas) -----
    def create_markdown_tab(self):
        self.markdown_frame = tk.Frame(self.notebook)
        self.notebook.add(self.markdown_frame, text="Parking Lot Markdown")

        # Top buttons frame
        top_frame = tk.Frame(self.markdown_frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        load_btn = tk.Button(top_frame, text="Load Sample Image", command=self.load_markdown_image)
        load_btn.pack(side=tk.LEFT, padx=5)

        finish_btn = tk.Button(top_frame, text="Finish Polygon", command=self.finish_polygon)
        finish_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(top_frame, text="Clear Current Polygon", command=self.clear_current_polygon)
        clear_btn.pack(side=tk.LEFT, padx=5)

        save_btn = tk.Button(top_frame, text="Save Config", command=self.save_config)
        save_btn.pack(side=tk.LEFT, padx=5)

        # Canvas to display the sample image and draw polygons
        self.markdown_canvas = tk.Canvas(self.markdown_frame, bg="grey")
        self.markdown_canvas.pack(fill='both', expand=True)
        self.markdown_canvas.bind("<Button-1>", self.on_markdown_canvas_click)

        # Variables to hold image and polygon data
        self.markdown_image = None   # Original image (cv2 format)
        self.markdown_photo = None   # Image for Tkinter display
        self.current_polygon = []    # Points of the polygon currently being defined
        self.polygons = []           # Finalized list of polygons (each as list of (x, y))
        self.markup_drawings = []    # IDs of canvas objects for the current polygon

    def load_markdown_image(self):
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        if not filename:
            return
        self.markdown_image = cv2.imread(filename)
        if self.markdown_image is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
        # Convert BGR (OpenCV) image to RGB and then to PIL Image
        image_rgb = cv2.cvtColor(self.markdown_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        self.markdown_photo = ImageTk.PhotoImage(pil_image)
        self.markdown_canvas.config(width=self.markdown_photo.width(), height=self.markdown_photo.height())
        self.markdown_canvas.create_image(0, 0, image=self.markdown_photo, anchor=tk.NW)

    def on_markdown_canvas_click(self, event):
        # Record the click coordinate relative to the canvas
        x, y = event.x, event.y
        self.current_polygon.append((x, y))
        r = 3  # radius for the point marker
        dot = self.markdown_canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")
        self.markup_drawings.append(dot)
        # If there is a previous point, draw a line from that point to the current one
        if len(self.current_polygon) > 1:
            x_prev, y_prev = self.current_polygon[-2]
            line = self.markdown_canvas.create_line(x_prev, y_prev, x, y, fill="yellow", width=2)
            self.markup_drawings.append(line)

    def finish_polygon(self):
        if len(self.current_polygon) < 3:
            messagebox.showerror("Error", "A polygon must have at least 3 points.")
            return
        # Draw a closing line (last point to first point)
        x_first, y_first = self.current_polygon[0]
        x_last, y_last = self.current_polygon[-1]
        line = self.markdown_canvas.create_line(x_last, y_last, x_first, y_first, fill="yellow", width=2)
        self.markup_drawings.append(line)
        # Save the finalized polygon and display its ID near the first point
        self.polygons.append(self.current_polygon.copy())
        label = self.markdown_canvas.create_text(x_first, y_first - 10, text=f"ID {len(self.polygons)}", fill="blue")
        self.markup_drawings.append(label)
        # Clear the current polygon data for the next one (but keep the drawings on canvas)
        self.current_polygon.clear()
        self.markup_drawings.clear()

    def clear_current_polygon(self):
        # Remove the current (unfinished) polygon drawings from the canvas
        for item in self.markup_drawings:
            self.markdown_canvas.delete(item)
        self.markup_drawings.clear()
        self.current_polygon.clear()

    def save_config(self):
        if not self.polygons:
            messagebox.showerror("Error", "No polygons have been defined!")
            return
        filename = filedialog.asksaveasfilename(
            title="Save Config",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if not filename:
            return
        # Save the polygons as a list of lists of [x, y] points
        with open(filename, 'w') as f:
            json.dump(self.polygons, f)
        messagebox.showinfo("Success", "Config saved successfully!")

    # ----- Tab 2: Evaluation (Load model, config, and run evaluation) -----
    def create_evaluation_tab(self):
        self.evaluation_frame = tk.Frame(self.notebook)
        self.notebook.add(self.evaluation_frame, text="Evaluation")

        # Controls frame at the top of the evaluation tab
        controls_frame = tk.Frame(self.evaluation_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Model size selection
        tk.Label(controls_frame, text="Model Size:").pack(side=tk.LEFT, padx=5)
        self.model_size_var = tk.StringVar(value='s')
        model_options = ['n', 's', 'm', 'l']
        self.model_menu = ttk.Combobox(
            controls_frame, textvariable=self.model_size_var,
            values=model_options, state="readonly", width=5
        )
        self.model_menu.pack(side=tk.LEFT, padx=5)

        # Confidence threshold slider
        tk.Label(controls_frame, text="Conf Thres:").pack(side=tk.LEFT, padx=5)
        self.conf_slider = tk.Scale(controls_frame, from_=0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL)
        self.conf_slider.set(0.45)
        self.conf_slider.pack(side=tk.LEFT, padx=5)

        # IOU threshold slider
        tk.Label(controls_frame, text="IOU Thres:").pack(side=tk.LEFT, padx=5)
        self.iou_slider = tk.Scale(controls_frame, from_=0.1, to=1.0, resolution=0.01, orient=tk.HORIZONTAL)
        self.iou_slider.set(0.4)
        self.iou_slider.pack(side=tk.LEFT, padx=5)

        # Image Scale Factor slider (for upscaling wide images)
        tk.Label(controls_frame, text="Scale Factor:").pack(side=tk.LEFT, padx=5)
        self.scale_slider = tk.Scale(controls_frame, from_=1.0, to=15.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.scale_slider.set(1.0)
        self.scale_slider.pack(side=tk.LEFT, padx=5)

        # Button to load the model
        load_model_btn = tk.Button(controls_frame, text="Load Model", command=self.load_model)
        load_model_btn.pack(side=tk.LEFT, padx=5)

        # Button to load the parking areas config
        load_config_btn = tk.Button(controls_frame, text="Load Config", command=self.load_config)
        load_config_btn.pack(side=tk.LEFT, padx=5)

        # Button to evaluate an image
        eval_image_btn = tk.Button(controls_frame, text="Evaluate Image", command=self.evaluate_image)
        eval_image_btn.pack(side=tk.LEFT, padx=5)

        # New Button to reevaluate the latest image with updated settings
        reevaluate_btn = tk.Button(controls_frame, text="Reevaluate Latest Image", command=self.reevaluate_latest_image)
        reevaluate_btn.pack(side=tk.LEFT, padx=5)

        # Canvas to display the evaluation image
        self.evaluation_canvas = tk.Canvas(self.evaluation_frame, bg="grey")
        self.evaluation_canvas.pack(fill='both', expand=True)

        # Detector instance (will be created when the model is loaded)
        self.detector = None

    def load_model(self):
        model_size = self.model_size_var.get()
        try:
            self.detector = ParkingSpotDetector(model_size=model_size,
                                                conf_thres=self.conf_slider.get(),
                                                iou_thres=self.iou_slider.get())
            messagebox.showinfo("Success", f"Model loaded with size '{model_size}'.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def load_config(self):
        if not self.detector:
            messagebox.showerror("Error", "Please load the model first!")
            return
        filename = filedialog.askopenfilename(
            title="Select Config JSON",
            filetypes=[("JSON Files", "*.json")]
        )
        if not filename:
            return
        try:
            with open(filename, 'r') as f:
                polygons = json.load(f)
            # Convert each polygon to a list of (x,y) tuples
            polygons_converted = [ [tuple(point) for point in poly] for poly in polygons ]
            self.detector.set_parking_areas(polygons_converted)
            messagebox.showinfo("Success", "Config loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")

    def evaluate_image(self):
        """Load an image via dialog, run detection, and display results."""
        if not self.detector:
            messagebox.showerror("Error", "Please load the model first!")
            return

        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        if not filename:
            return
        frame = cv2.imread(filename)
        if frame is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        # Save the latest evaluated frame for possible re-evaluation.
        self.latest_frame = frame.copy()

        self._process_and_display(frame)

    def reevaluate_latest_image(self):
        """Re-run detection on the latest image using updated slider settings."""
        if self.latest_frame is None:
            messagebox.showerror("Error", "No image has been evaluated yet!")
            return
        self._process_and_display(self.latest_frame)

    def _process_and_display(self, frame):
        """
        Upscale the image if needed, run detection (with debug prints), and update the display.
        """
        # Update thresholds from sliders
        self.detector.conf_thres = self.conf_slider.get()
        self.detector.iou_thres = self.iou_slider.get()
        scale_factor = self.scale_slider.get()

        # Debug prints to help differentiate GUI issues from AI evaluation issues
        print("DEBUG: Evaluating image...")
        print(f"DEBUG: Confidence Threshold = {self.detector.conf_thres}")
        print(f"DEBUG: IOU Threshold = {self.detector.iou_thres}")
        print(f"DEBUG: Scale Factor = {scale_factor}")

        # If scaling is requested, upscale the image for detection
        if scale_factor != 1.0:
            scaled_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        else:
            scaled_frame = frame

        # Run detection and get occupancy and vehicle bounding boxes.
        occupied, boxes = self.detector.detect_and_get_detections(scaled_frame, scale=scale_factor)
        print(f"DEBUG: Detected {len(boxes)} vehicle bounding boxes.")
        print(f"DEBUG: Boxes: {boxes}")
        print(f"DEBUG: Occupied Spots: {occupied}")

        # Draw results on the original frame
        annotated_frame = self.detector.visualize_results(frame.copy(), occupied, boxes)
        # Convert the annotated frame from BGR to RGB and then to a PIL image for Tkinter
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(annotated_rgb)
        self.eval_photo = ImageTk.PhotoImage(pil_image)
        self.evaluation_canvas.config(width=self.eval_photo.width(), height=self.eval_photo.height())
        self.evaluation_canvas.create_image(0, 0, image=self.eval_photo, anchor=tk.NW)

# -------------------------- Main --------------------------
if __name__ == '__main__':
    app = ParkingLotApp()
    app.mainloop()
