import os
import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import AbstractParkingDetector
from AIDetectors import SegmentationParkingDetector, BBoxParkingDetector
from detector2 import ParkingLotDetector


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
        self.model_size_var = tk.StringVar(value='yolo11s.pt')
        model_options = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'custom.pt']
        self.model_menu = ttk.Combobox(
            controls_frame, textvariable=self.model_size_var,
            values=model_options, state="readonly", width=5
        )
        self.model_menu.pack(side=tk.LEFT, padx=5)

        # NEW: Detection Method selection dropdown
        tk.Label(controls_frame, text="Detection Method:").pack(side=tk.LEFT, padx=5)
        self.detector_type_var = tk.StringVar(value="Regular")
        detection_options = ["Regular", "Segmentation"]
        self.detector_type_menu = ttk.Combobox(
            controls_frame, textvariable=self.detector_type_var,
            values=detection_options, state="readonly", width=12
        )
        self.detector_type_menu.pack(side=tk.LEFT, padx=5)

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

        # Occlusion Threshold slider
        tk.Label(controls_frame, text="Occlusion %:").pack(side=tk.LEFT, padx=5)
        self.occl_slider = tk.Scale(controls_frame, from_=0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL)
        self.occl_slider.set(0.3)
        self.occl_slider.pack(side=tk.LEFT, padx=5)

        # Class selection
        tk.Label(controls_frame, text="Classes:").pack(side=tk.LEFT, padx=5)
        self.classes_entry = tk.Entry(controls_frame, width=15)
        self.classes_entry.insert(0, "2,5,7")
        self.classes_entry.pack(side=tk.LEFT, padx=5)

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
        try:
            classes = [int(c.strip()) for c in self.classes_entry.get().split(',')]
            self.detector = ParkingLotDetector(
                model=self.model_size_var.get(),
                conf_thres=self.conf_slider.get(),
                iou_thres=self.iou_slider.get(),
                occl_thres=self.occl_slider.get(),
                classes=classes
            )
            messagebox.showinfo("Success", "Model loaded with current settings")
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
        # Update thresholds from sliders
        self.detector.conf_thres = self.conf_slider.get()
        self.detector.iou_thres = self.iou_slider.get()
        self.detector.occl_threshold = self.occl_slider.get()

        print("DEBUG: Evaluating image...")
        print(f"DEBUG: Confidence Threshold = {self.detector.conf_thres}")
        print(f"DEBUG: IOU Threshold = {self.detector.iou_thres}")

        # Run detection without any scaling
        occupied, detections = self.detector.detect_and_get_detections(frame)
        print(f"DEBUG: Occupied Spots: {occupied}")

        # Draw results on the original frame
        annotated_frame = self.detector.visualize_results(frame.copy(), occupied, detections)
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(annotated_rgb)
        self.eval_photo = ImageTk.PhotoImage(pil_image)
        self.evaluation_canvas.config(width=self.eval_photo.width(), height=self.eval_photo.height())
        self.evaluation_canvas.create_image(0, 0, image=self.eval_photo, anchor=tk.NW)


# -------------------------- Main --------------------------
if __name__ == '__main__':
    app = ParkingLotApp()
    app.mainloop()
