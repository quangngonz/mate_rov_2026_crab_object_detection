"""
Crab Detector
============

Core detection class for running inference on images.
"""

from pathlib import Path
from typing import Dict
import torch
import numpy as np
from ultralytics import YOLO

from config.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
)


class CrabDetector:
    """Wrapper class for crab detection inference."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD
    ):
        """
        Initialize the crab detector.

        Args:
            model_path: Path to trained model weights (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = YOLO(str(self.model_path))

        # Determine device
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

    def predict(
        self,
        image_path: str,
        verbose: bool = False
    ) -> Dict[str, any]:
        """
        Run detection on a single image.

        Args:
            image_path: Path to input image
            verbose: Whether to print verbose output

        Returns:
            Dictionary with detection results:
            - 'boxes': np.ndarray of bounding boxes [x1, y1, x2, y2]
            - 'confidences': np.ndarray of confidence scores
            - 'class_ids': np.ndarray of class IDs
            - 'num_detections': int number of detections
        """
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=verbose
        )[0]

        # Extract detection information
        boxes = results.boxes.xyxy.cpu().numpy() if len(
            results.boxes) > 0 else np.array([])
        confidences = results.boxes.conf.cpu().numpy() if len(
            results.boxes) > 0 else np.array([])
        class_ids = results.boxes.cls.cpu().numpy().astype(
            int) if len(results.boxes) > 0 else np.array([])

        return {
            'boxes': boxes,
            'confidences': confidences,
            'class_ids': class_ids,
            'num_detections': len(boxes)
        }

    @property
    def class_names(self) -> Dict[int, str]:
        """Get class names from model."""
        return self.model.names
