from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass
class Detection:
    """Normalized detection shape used by the renderer."""

    class_name: str
    bbox: List[int]  # [x1, y1, x2, y2]
    conf: float


class BaseDetectorBackend:
    """Interface for model inference backends."""

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    def set_confidence(self, conf_threshold: float) -> None:
        """Optional runtime confidence update hook for backends."""
        _ = conf_threshold


class FunctionDetectorBackend(BaseDetectorBackend):
    """Adapter for inference functions returning list[dict]."""

    def __init__(self, infer_fn: Callable[[np.ndarray], List[Dict]]):
        self._infer_fn = infer_fn

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        raw = self._infer_fn(frame_bgr)
        detections: List[Detection] = []
        for item in raw:
            cls_name = str(item.get("class", ""))
            bbox = item.get("bbox", [0, 0, 0, 0])
            conf = float(item.get("conf", 0.0))
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            detections.append(Detection(class_name=cls_name,
                              bbox=[x1, y1, x2, y2], conf=conf))
        return detections


class UltralyticsBackend(BaseDetectorBackend):
    """Backend for direct YOLO usage via ultralytics."""

    def __init__(self, model_path: str, conf: float = 0.4, iou: float = 0.45):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        if result.boxes is None:
            return detections

        names = result.names
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, class_id in zip(boxes_xyxy, confs, classes):
            cls_idx = int(class_id)
            cls_name = str(names.get(cls_idx, cls_idx)) if isinstance(
                names, dict) else str(names[cls_idx])
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            detections.append(Detection(class_name=cls_name, bbox=[
                              x1, y1, x2, y2], conf=float(score)))

        return detections

    def set_confidence(self, conf_threshold: float) -> None:
        self.conf = float(conf_threshold)


def normalize_label(text: str) -> str:
    """Normalize class labels for resilient text matching."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return " ".join(cleaned.split())


def is_green_crab_class(detected_class: str, target_class: str) -> bool:
    """Return True when a detected label matches the configured green crab label."""
    detected_norm = normalize_label(detected_class)
    target_norm = normalize_label(target_class)

    if not detected_norm:
        return False
    if detected_norm == target_norm:
        return True
    if target_norm and (target_norm in detected_norm or detected_norm in target_norm):
        return True

    aliases = {"european green crab", "green crab", "carcinus maenas"}
    if detected_norm in aliases:
        return True

    target_tokens = set(target_norm.split())
    detected_tokens = set(detected_norm.split())
    return bool(target_tokens and target_tokens.issubset(detected_tokens))
