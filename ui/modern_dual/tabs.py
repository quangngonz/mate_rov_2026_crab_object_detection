from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PyQt6.QtCore import QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .rendering import bgr_to_qpixmap


class VideoPanel(QLabel):
    """Styled video panel widget."""

    def __init__(self, title: str):
        super().__init__()
        self._title = title
        self._pixmap: QPixmap | None = None
        self._corner_radius = 14

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setText(title)
        self.setStyleSheet(
            """
            QLabel {
                background-color: #06141b;
                color: #a6d4e1;
                font-size: 20px;
                font-weight: 600;
            }
            """
        )

    def set_frame_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def set_placeholder_text(self, text: str) -> None:
        self._title = text
        if self._pixmap is None:
            self.update()

    def clear_frame(self) -> None:
        self._pixmap = None
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        panel_rect = self.rect().adjusted(0, 0, -1, -1)
        rounded = QPainterPath()
        rounded.addRoundedRect(QRectF(panel_rect), self._corner_radius,
                               self._corner_radius)

        painter.fillPath(rounded, QColor("#06141b"))
        painter.setPen(QPen(QColor("#204557"), 1))
        painter.drawPath(rounded)

        content_rect = self.rect().adjusted(2, 2, -2, -2)
        if self._pixmap is not None and not self._pixmap.isNull() and content_rect.width() > 0 and content_rect.height() > 0:
            scaled = self._pixmap.scaled(
                content_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            x = content_rect.x() + (content_rect.width() - scaled.width()) // 2
            y = content_rect.y() + (content_rect.height() - scaled.height()) // 2
            draw_rect = QRect(x, y, scaled.width(), scaled.height())

            painter.save()
            painter.setClipPath(rounded)
            painter.drawPixmap(draw_rect, scaled)
            painter.restore()
        else:
            painter.setPen(QColor("#a6d4e1"))
            painter.setFont(self.font())
            painter.drawText(content_rect, Qt.AlignmentFlag.AlignCenter,
                             self._title)

        painter.end()


class PilotPanel(QWidget):
    """Pilot view with camera selector and image/inference controls."""

    controls_changed = pyqtSignal(float, float, int)
    camera_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        self.video_panel = VideoPanel("Waiting for ROV stream...")
        root.addWidget(self.video_panel, stretch=1)

        controls_row = QWidget()
        controls = QHBoxLayout(controls_row)
        controls.setContentsMargins(12, 12, 12, 12)
        controls.setSpacing(18)

        self.camera_title = QLabel("Camera")
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(320)

        self.conf_title = QLabel("Confidence")
        self.conf_value = QLabel("0.40")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(40)

        self.brightness_title = QLabel("Brightness")
        self.brightness_value = QLabel("0")
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-80, 80)
        self.brightness_slider.setValue(0)

        self.contrast_title = QLabel("Contrast")
        self.contrast_value = QLabel("1.00")
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(50, 240)
        self.contrast_slider.setValue(100)

        self.fps_label = QLabel("FPS: 0.0")

        for widget in [
            self.camera_title,
            self.conf_title,
            self.conf_value,
            self.brightness_title,
            self.brightness_value,
            self.contrast_title,
            self.contrast_value,
            self.fps_label,
        ]:
            widget.setFont(QFont("Sans Serif", 11, QFont.Weight.DemiBold))

        controls.addWidget(self.camera_title)
        controls.addWidget(self.camera_combo)
        controls.addSpacing(12)

        controls.addWidget(self.conf_title)
        controls.addWidget(self.conf_slider, stretch=1)
        controls.addWidget(self.conf_value)
        controls.addSpacing(12)

        controls.addWidget(self.brightness_title)
        controls.addWidget(self.brightness_slider, stretch=1)
        controls.addWidget(self.brightness_value)
        controls.addSpacing(12)

        controls.addWidget(self.contrast_title)
        controls.addWidget(self.contrast_slider, stretch=1)
        controls.addWidget(self.contrast_value)
        controls.addSpacing(20)

        controls.addWidget(self.fps_label)

        root.addWidget(controls_row)

        self.setStyleSheet(
            """
            QWidget { color: #d7ecf1; font-family: 'Sans Serif'; }
            QSlider::groove:horizontal {
                border: 1px solid #2f5e70;
                height: 10px;
                border-radius: 5px;
                background: #0d2530;
            }
            QSlider::handle:horizontal {
                background: #59c7d9;
                border: 1px solid #8de1ef;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QComboBox {
                color: #000000;
            }
            """
        )

        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self.conf_slider.valueChanged.connect(self._on_controls_changed)
        self.brightness_slider.valueChanged.connect(self._on_controls_changed)
        self.contrast_slider.valueChanged.connect(self._on_controls_changed)

    def set_camera_options(self, camera_options: List[Tuple[int, str]], selected_source: str) -> None:
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        selected_index = 0
        for i, (cam_idx, cam_name) in enumerate(camera_options):
            source_value = str(cam_idx)
            self.camera_combo.addItem(f"{cam_idx}: {cam_name}", source_value)
            if source_value == str(selected_source):
                selected_index = i

        self.camera_combo.setCurrentIndex(selected_index)
        self.camera_combo.blockSignals(False)

    def _on_camera_changed(self) -> None:
        source = self.camera_combo.currentData()
        if source is not None:
            self.camera_changed.emit(str(source))

    def _on_controls_changed(self) -> None:
        conf = float(self.conf_slider.value()) / 100.0
        brightness = int(self.brightness_slider.value())
        contrast = float(self.contrast_slider.value()) / 100.0
        self.conf_value.setText(f"{conf:.2f}")
        self.brightness_value.setText(str(brightness))
        self.contrast_value.setText(f"{contrast:.2f}")
        self.controls_changed.emit(conf, contrast, brightness)

    def update_video(self, frame_bgr: np.ndarray) -> None:
        pixmap = bgr_to_qpixmap(frame_bgr)
        self.video_panel.set_frame_pixmap(pixmap)

    def update_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")
