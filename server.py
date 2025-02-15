import json
import os
import cv2
import numpy as np
import time
import threading
import queue
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
from typing import Optional
from tempfile import NamedTemporaryFile
from detector2 import ParkingLotDetector
import logging
from pydantic import BaseModel
from typing import Optional, List
from server_config import EXTERNAL_URL

# Configuration
TOKENS_FILE = "TOKENS"
BACKEND_TOKEN_KEY_FILE = "BACKEND_TOKEN"
MODEL_DIR = "./models"

class ConfigModel(BaseModel):
    model: str = "yolov8n.pt"
    conf_thres: float = 0.45
    iou_thres: float = 0.4
    occl_thres: float = 0.3
    classes: Optional[List[int]] = [2, 5, 7]
    polygons: List[List[List[int]]]

# Load API keys
def load_api_keys(file_path: str) -> set:
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r") as f:
        return {line.strip() for line in f}

tokens = load_api_keys(TOKENS_FILE)
backend_token = load_api_keys(BACKEND_TOKEN_KEY_FILE).pop() if os.path.exists(BACKEND_TOKEN_KEY_FILE) else ""

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


            # Initialize detector
            detector = ParkingLotDetector(
                model_dir=MODEL_DIR,
                model_size=task["model"],
                conf_thres=task["conf_thres"],
                iou_thres=task["iou_thres"],
                occl_thres=task["occl_thres"],
                classes=task["classes"],
            )
            detector.set_parking_areas(task["polygons"])

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
                "token": backend_token
            }

            # Send to external server
            response = requests.post(
                EXTERNAL_URL,
                files=files,
                data=data
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
        token: str = Form(...),
        camera_id: str = Form(...),
        config: str = Form(...),  # Получаем как строку
        image: UploadFile = File(...),
):
    logger.info(f"Received request: {token}, {camera_id}, {config}")

    try:
        # Парсим JSON-строку в модель
        config_data = ConfigModel(**json.loads(json.loads(config)))
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid config format: {str(e)}"
        )

    # Validate API key
    if token not in tokens:
        logger.error(f"Invalid token: {token}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


    # Validate parameters
    if not 0 <= config_data.conf_thres <= 1 or not 0 <= config_data.iou_thres <= 1 or not 0 <= config_data.occl_thres <= 1:
        logger.error(f"Invalid thresholds: conf_thres={config_data.conf_thres}, iou_thres={config_data.iou_thres}, occl_thres={config_data.occl_thres}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Thresholds must be between 0 and 1"
        )

    # Save image to temp file
    try:
        temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
        contents = await image.read()
        temp_file.write(contents)
        temp_file.close()
    except Exception as e:
        logger.error(f"Error handling file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error handling file: {str(e)}"
        )

    # Add task to queue
    task = {
        "temp_path": temp_file.name,
        "model": config_data.model,
        "conf_thres": config_data.conf_thres,
        "iou_thres": config_data.iou_thres,
        "occl_thres": config_data.occl_thres,
        "classes": config_data.classes,
        "camera_id": camera_id,
        "polygons": config_data.polygons
    }

    with lock:
        task_queue.put(task)

    logger.info(f"Task added to queue: camera_id={camera_id}, queue_size={task_queue.qsize()}")
    return {"message": "Image queued for processing", "queue_size": task_queue.qsize()}