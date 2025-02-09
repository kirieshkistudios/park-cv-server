import os
import cv2
import numpy as np
import time
import threading
import queue
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from typing import Optional
from tempfile import NamedTemporaryFile
from detector2 import ParkingLotDetector
import logging

# Configuration
API_KEYS_FILE = "API_KEYS"
EXTERNAL_API_KEY_FILE = "EXTERNAL_KEY"
MODEL_DIR = "./models"
EXTERNAL_URL = "https://external-server.com/api/upload"
BASE_CONFIG = {
    "camera1": {
        "polygons": [
            [(100, 100), (200, 100), (200, 200), (100, 200)],
            [(250, 100), (350, 100), (350, 200), (250, 200)]
        ]
    }
}

# Load API keys
def load_api_keys(file_path: str) -> set:
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r") as f:
        return {line.strip() for line in f}

api_keys = load_api_keys(API_KEYS_FILE)
external_api_key = load_api_keys(EXTERNAL_API_KEY_FILE).pop() if os.path.exists(EXTERNAL_API_KEY_FILE) else ""

# Global processing queue
task_queue = queue.Queue()
lock = threading.Lock()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start worker thread on startup
    worker_thread = threading.Thread(target=process_images, daemon=True)
    worker_thread.start()
    yield
    # Add cleanup logic here if needed

app = FastAPI(lifespan=lifespan)

# Worker function
def process_images():
    while True:
        task = task_queue.get()
        try:
            with open(task["temp_path"], "rb") as f:
                image_data = f.read()

            frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Invalid image data")

            # Get camera configuration
            config = BASE_CONFIG.get(task["camera_id"], {})
            polygons = config.get("polygons", [])

            # Initialize detector
            detector = ParkingLotDetector(
                model_dir=MODEL_DIR,
                model_size=task["model"],
                conf_thres=task["conf_thres"],
                iou_thres=task["iou_thres"],
                occl_thres=task["occl_thres"],
                classes=task["classes"]
            )
            detector.set_parking_areas(polygons)

            # Process image
            start_time = time.time()
            occupied_spots, detections, occupied_count, free_count = detector.detect_and_get_detections(frame)
            processing_time = time.time() - start_time

            # Annotate image
            annotated_frame = detector.visualize_results(frame.copy(), occupied_spots, detections)
            _, img_encoded = cv2.imencode(".jpg", annotated_frame)
            img_bytes = img_encoded.tobytes()

            # Prepare payload
            files = {"image": ("annotated.jpg", img_bytes, "image/jpeg")}
            data = {
                "free": free_count,  # Используем free_count
                "occupied": occupied_count,  # Используем occupied_count
                "processing_time": processing_time,
                "camera_id": task["camera_id"],
                "api_key": external_api_key
            }

            # Send to external server
            response = requests.post(
                EXTERNAL_URL,
                files=files,
                data=data,
                headers={"X-API-Key": external_api_key}
            )
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
        finally:
            os.unlink(task["temp_path"])
            task_queue.task_done()

# API Endpoint
@app.post("/process-image")
async def process_image(
        api_key: str,
        camera_id: str,
        model: str = "yolov8n.pt",
        conf_thres: float = 0.45,
        iou_thres: float = 0.4,
        occl_thres: float = 0.3,
        classes: Optional[str] = None,
        image: UploadFile = File(...),
):
    # Validate API key
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Validate parameters
    if not 0 <= conf_thres <= 1 or not 0 <= iou_thres <= 1 or not 0 <= occl_thres <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Thresholds must be between 0 and 1"
        )

    # Process classes
    try:
        class_list = [int(c) for c in classes.split(",")] if classes else None
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid classes format"
        )

    # Save image to temp file
    try:
        temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
        contents = await image.read()
        temp_file.write(contents)
        temp_file.close()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error handling file: {str(e)}"
        )

    # Add task to queue
    task = {
        "temp_path": temp_file.name,
        "model": model,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "occl_thres": occl_thres,
        "classes": class_list,
        "camera_id": camera_id
    }

    with lock:
        task_queue.put(task)

    return {"message": "Image queued for processing", "queue_size": task_queue.qsize()}