import glob
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class ModelService:
    """Lazy-loads YOLO model and provides a simple inference API."""

    def __init__(self, model_path: str | None = None, confidence_threshold: float = 0.25) -> None:
        # Fix for PyTorch 2.8.0+ weights_only issue
        original_load = torch.load

        def safe_load(file, **kwargs):
            return original_load(file, weights_only=False, **kwargs)

        torch.load = safe_load

        self.confidence_threshold = confidence_threshold
        self.model = self._load_model(model_path)

    def _load_model(self, explicit_path: str | None) -> YOLO:
        if explicit_path and os.path.exists(explicit_path):
            return YOLO(explicit_path)

        model_files = glob.glob("runs/detect/*/weights/best.pt")
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            return YOLO(latest_model)

        # Fallback to a small base config if no trained weights exist
        return YOLO("yolov8n.yaml")

    def run_inference(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Run YOLO inference, return annotated image and list of detections.

        Each detection: {class_id, class_name, confidence, box: [x1,y1,x2,y2]}.
        """
        results = self.model(image_bgr, conf=self.confidence_threshold)
        if len(results) == 0:
            return image_bgr, []

        result = results[0]
        annotated_bgr = result.plot()

        detections: List[Dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "box": [float(x) for x in xyxy],
                    }
                )

        return annotated_bgr, detections


