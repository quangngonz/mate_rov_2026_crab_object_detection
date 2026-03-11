from __future__ import annotations

from typing import Dict, List, Tuple

from PyQt6.QtWidgets import QApplication, QMessageBox

from .backends import BaseDetectorBackend, Detection, is_green_crab_class
from .rendering import draw_modern_label, draw_rounded_box
from .windows import JudgeViewWindow, PilotDebugWindow
from .worker import VideoInferenceThread


class DualWindowController:
    """Coordinates worker lifecycle and both windows."""

    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        backend: BaseDetectorBackend,
        green_class_name: str,
        camera_options: List[Tuple[int, str]],
    ):
        self.width = width
        self.height = height
        self.backend = backend
        self.green_class_name = green_class_name

        self.current_source = str(source)
        self.last_good_source = str(source)

        self.pilot_window = PilotDebugWindow()
        self.judge_window = JudgeViewWindow()
        self.pilot_window.set_camera_options(
            camera_options, self.current_source)

        self.worker: VideoInferenceThread | None = None
        self._create_and_start_worker(self.current_source)

        self.pilot_window.controls_changed.connect(self._on_controls_changed)
        self.pilot_window.camera_changed.connect(self._on_camera_changed)
        self.pilot_window.initialize_controls()

    def _create_and_start_worker(self, source: str) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)

        self.current_source = str(source)
        self.worker = VideoInferenceThread(
            source=self.current_source,
            width=self.width,
            height=self.height,
            detector_backend=self.backend,
            green_class_name=self.green_class_name,
        )
        self.worker.frame_packet.connect(self._on_frame_packet)
        self.worker.stream_error.connect(self._on_stream_error)
        self.worker.start()

    def start(self) -> None:
        self.pilot_window.show()
        self.judge_window.show()

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)

    def _on_controls_changed(self, conf: float, contrast: float, brightness: int) -> None:
        if self.worker is not None:
            self.worker.set_controls(conf, contrast, brightness)

    def _on_camera_changed(self, source: str) -> None:
        source = str(source)
        if source == self.current_source:
            return
        self._create_and_start_worker(source)

    def _on_stream_error(self, message: str) -> None:
        QMessageBox.warning(self.pilot_window, "Stream Error", message)
        if self.current_source != self.last_good_source:
            self._create_and_start_worker(self.last_good_source)

    def _on_frame_packet(self, packet: Dict) -> None:
        adjusted = packet["adjusted"]
        all_detections: List[Detection] = packet["all_detections"]
        green_detections: List[Detection] = packet["green_detections"]
        green_count = int(packet["green_count"])
        fps = float(packet["fps"])

        self.last_good_source = str(packet.get("source", self.current_source))

        pilot_frame = adjusted.copy()
        for det in all_detections:
            x1, y1, x2, y2 = det.bbox
            color = (255, 179, 71)
            if is_green_crab_class(det.class_name, self.green_class_name):
                color = (65, 214, 167)
            draw_rounded_box(pilot_frame, x1, y1, x2, y2,
                             color=color, thickness=2, radius=10)
            draw_modern_label(
                pilot_frame,
                f"{det.class_name} {det.conf:.2f}",
                (x1, y1),
                bg_color=(12, 40, 52),
            )

        judge_frame = adjusted.copy()
        for det in green_detections:
            x1, y1, x2, y2 = det.bbox
            draw_rounded_box(judge_frame, x1, y1, x2, y2, color=(
                72, 237, 189), thickness=3, radius=12)
            draw_modern_label(
                judge_frame,
                f"European Green Crab {det.conf:.2f}",
                (x1, y1),
                bg_color=(18, 66, 58),
            )

        self.pilot_window.update_video(pilot_frame)
        self.pilot_window.update_fps(fps)
        self.judge_window.update_video(judge_frame)
        self.judge_window.update_green_count(green_count)
