import glob
import os
import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger("webapp.model")


class ModelService:
    """Lazy-loads YOLO model and provides a simple inference API."""

    def __init__(self, model_path: str | None = None, confidence_threshold: float = 0.25) -> None:
        # Fix for PyTorch 2.8.0+ weights_only issue
        original_load = torch.load

        def safe_load(file, **kwargs):
            return original_load(file, weights_only=False, **kwargs)

        torch.load = safe_load

        # Read optional overrides from environment
        env_model_path = os.environ.get("WEB_MODEL_PATH")
        env_conf = os.environ.get("WEB_CONFIDENCE")
        if env_conf:
            try:
                confidence_threshold = float(env_conf)
            except Exception:
                logger.warning(f"Invalid WEB_CONFIDENCE='{env_conf}', using default {confidence_threshold}")

        self.confidence_threshold = confidence_threshold
        self.model_source: str = ""
        self.model = self._load_model(model_path or env_model_path)

    def _load_model(self, explicit_path: str | None) -> YOLO:
        if explicit_path and os.path.exists(explicit_path):
            logger.info(f"Loading model from explicit path: {explicit_path}")
            self.model_source = explicit_path
            return YOLO(explicit_path)

        model_files = glob.glob("runs/detect/*/weights/best.pt")
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            logger.info(f"Loading latest trained model: {latest_model}")
            self.model_source = latest_model
            return YOLO(latest_model)

        # Known project paths used in main.py fallback
        known_paths = [
            os.path.join("runs", "detect", "solar_defect_train9", "weights", "best.pt"),
            os.path.join("runs", "detect", "solar_defect_train", "weights", "best.pt"),
        ]
        for p in known_paths:
            if os.path.exists(p):
                logger.info(f"Loading model from known path: {p}")
                self.model_source = p
                return YOLO(p)

        # Prefer local lightweight weights if present to avoid downloads
        local_pt = os.path.join(os.getcwd(), "yolov8n.pt")
        if os.path.exists(local_pt):
            logger.warning(f"No trained weights found. Falling back to local weights: {local_pt}")
            self.model_source = local_pt
            return YOLO(local_pt)

        # Final fallback: base YAML config (may require network for weights)
        logger.warning("No trained weights found. Falling back to base config 'yolov8n.yaml' (untrained, may yield no detections)")
        self.model_source = "yolov8n.yaml"
        return YOLO("yolov8n.yaml")

    def run_inference(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Run YOLO inference, return annotated image and list of detections.

        Each detection: {class_id, class_name, confidence, box: [x1,y1,x2,y2]}.
        """
        logger.info(f"Running YOLO inference (conf={self.confidence_threshold})")
        results = self.model(image_bgr, conf=self.confidence_threshold)
        if len(results) == 0:
            logger.info("No results returned from model")
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

        logger.info(f"Detections parsed: {len(detections)}")

        return annotated_bgr, detections

    def get_model_source(self) -> str:
        return self.model_source


