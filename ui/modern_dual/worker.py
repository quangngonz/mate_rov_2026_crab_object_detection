from __future__ import annotations

import threading
import time
from typing import Dict

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from .backends import BaseDetectorBackend, is_green_crab_class


class VideoInferenceThread(QThread):
    """Background worker for capture, inference, and packet emission."""

    frame_packet = pyqtSignal(object)
    stream_error = pyqtSignal(str)

    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        detector_backend: BaseDetectorBackend,
        green_class_name: str = "European Green Crab",
    ):
        super().__init__()
        self.source = source
        self.width = width
        self.height = height
        self.detector_backend = detector_backend
        self.green_class_name = green_class_name

        self._running = True
        self._lock = threading.Lock()
        self._conf_threshold = 0.40
        self._contrast = 1.0
        self._brightness = 0

    def set_controls(self, conf_threshold: float, contrast: float, brightness: int) -> None:
        with self._lock:
            self._conf_threshold = conf_threshold
            self._contrast = contrast
            self._brightness = brightness
        self.detector_backend.set_confidence(conf_threshold)

    def stop(self) -> None:
        self._running = False

    def _get_controls(self) -> tuple[float, float, int]:
        with self._lock:
            return self._conf_threshold, self._contrast, self._brightness

    def run(self) -> None:
        cap_source: object = int(self.source) if str(
            self.source).isdigit() else self.source
        cap = cv2.VideoCapture(cap_source)

        if not cap.isOpened():
            self.stream_error.emit(
                f"Unable to open video source: {self.source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        fps_time = time.time()
        frame_counter = 0
        fps = 0.0

        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                self.stream_error.emit(
                    "Video stream ended or frame could not be read.")
                break

            conf_threshold, contrast, brightness = self._get_controls()
            adjusted = cv2.convertScaleAbs(
                frame, alpha=contrast, beta=brightness)

            try:
                detections = self.detector_backend.infer(adjusted)
            except Exception as exc:
                self.stream_error.emit(f"Inference error: {exc}")
                break

            detections = [d for d in detections if d.conf >= conf_threshold]
            green_detections = [
                d for d in detections if is_green_crab_class(d.class_name, self.green_class_name)
            ]

            frame_counter += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = frame_counter / elapsed
                frame_counter = 0
                fps_time = time.time()

            packet: Dict = {
                "source": str(self.source),
                "adjusted": adjusted,
                "all_detections": detections,
                "green_detections": green_detections,
                "green_count": len(green_detections),
                "fps": fps,
            }
            self.frame_packet.emit(packet)

        cap.release()
