"""YOLO model loading and inference logic."""

import os
import time

import cv2
import numpy as np
from ultralytics import YOLO


class VegetableDetector:
    def __init__(self):
        self.model_path = os.environ.get("MODEL_PATH", "../api/best_new.pt")
        self.conf_threshold = float(os.environ.get("CONF_THRESHOLD", "0.25"))
        self.imgsz = int(os.environ.get("IMGSZ", "640"))

        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.classes = list(self.model.names.values())
        print(f"Model loaded. Classes: {self.classes}")

    def decode_image(self, contents: bytes):
        # UploadFile gives raw bytes. OpenCV decodes those bytes into an image matrix.
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image.")
        return image

    def detect(self, contents: bytes):
        image = self.decode_image(contents)

        start = time.time()
        result = self.model(
            image,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        inference_ms = int((time.time() - start) * 1000)

        detections = self.format_boxes(result.boxes)

        return {
            "count": len(detections),
            "top": detections[0]["class"] if detections else None,
            "inference_ms": inference_ms,
            "detections": detections,
        }

    def format_boxes(self, boxes):
        detections = []

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = round(float(box.conf[0]), 4)
            class_name = self.model.names[class_id]
            x1, y1, x2, y2 = [round(float(value), 1) for value in box.xyxy[0]]

            detections.append({
                "class_id": class_id,
                "class": class_name,
                "confidence": confidence,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
            })

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections


detector = VegetableDetector()
