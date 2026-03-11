from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from .rendering import bgr_to_qpixmap
from .tabs import PilotPanel, VideoPanel


class PilotDebugWindow(QMainWindow):
    """Pilot-facing window with camera and inference controls."""

    controls_changed = pyqtSignal(float, float, int)
    camera_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pilot View")
        self.resize(1400, 900)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)

        self.pilot_panel = PilotPanel()
        root_layout.addWidget(self.pilot_panel)

        self.pilot_panel.controls_changed.connect(self.controls_changed.emit)
        self.pilot_panel.camera_changed.connect(self.camera_changed.emit)

        self.setStyleSheet("QMainWindow { background-color: #041016; }")

    def set_camera_options(self, camera_options: list[tuple[int, str]], selected_source: str) -> None:
        self.pilot_panel.set_camera_options(camera_options, selected_source)

    def update_video(self, frame_bgr: np.ndarray) -> None:
        self.pilot_panel.update_video(frame_bgr)

    def update_fps(self, fps: float) -> None:
        self.pilot_panel.update_fps(fps)

    def initialize_controls(self) -> None:
        self.pilot_panel._on_controls_changed()


class JudgeViewWindow(QMainWindow):
    """Judge-facing clean window with green-crab-only overlays."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Official Judge View")
        self.resize(1300, 860)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title_label = QLabel("Official Judge View")
        title_label.setFont(QFont("Sans Serif", 16, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #d8f6ff;")
        layout.addWidget(title_label)

        self.count_label = QLabel("Invasive Green Crabs Detected: 0")
        self.count_label.setFont(
            QFont("Sans Serif", 24, QFont.Weight.ExtraBold))
        self.count_label.setStyleSheet(
            """
            QLabel {
                color: #e8fff7;
                background-color: #12453b;
                border: 1px solid #52e9bd;
                border-radius: 10px;
                padding: 12px 18px;
            }
            """
        )
        layout.addWidget(self.count_label)

        self.video_panel = VideoPanel("Official view starting...")
        layout.addWidget(self.video_panel)

        self.setStyleSheet("QMainWindow { background-color: #020b10; }")

    def update_video(self, frame_bgr: np.ndarray) -> None:
        pixmap = bgr_to_qpixmap(frame_bgr)
        self.video_panel.set_frame_pixmap(pixmap)

    def update_green_count(self, green_count: int) -> None:
        self.count_label.setText(
            f"Invasive Green Crabs Detected: {green_count}")
