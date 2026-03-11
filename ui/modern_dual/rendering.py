from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def draw_rounded_box(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple,
    thickness: int = 2,
    radius: int = 10,
) -> None:
    """Draw a rounded rectangle using lines and quarter arcs."""
    radius = max(3, min(radius, (x2 - x1) // 3, (y2 - y1) // 3))

    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(image, (x1 + radius, y1 + radius),
                (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y1 + radius),
                (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(image, (x1 + radius, y2 - radius),
                (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y2 - radius),
                (radius, radius), 0, 0, 90, color, thickness)


def draw_modern_label(
    image: np.ndarray,
    text: str,
    anchor_xy: tuple,
    bg_color: tuple,
    text_color: tuple = (245, 245, 245),
) -> None:
    """Draw a semi-transparent text pill near detections."""
    x, y = anchor_xy
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    pad_x, pad_y = 8, 6
    rect_x1 = max(0, x)
    rect_y1 = max(0, y - th - baseline - 2 * pad_y)
    rect_x2 = min(image.shape[1] - 1, x + tw + 2 * pad_x)
    rect_y2 = min(image.shape[0] - 1, y)

    if rect_x2 <= rect_x1 or rect_y2 <= rect_y1:
        return

    overlay = image.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1),
                  (rect_x2, rect_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    cv2.putText(
        image,
        text,
        (rect_x1 + pad_x, rect_y2 - pad_y),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def bgr_to_qpixmap(frame_bgr: np.ndarray) -> QPixmap:
    """Convert OpenCV BGR frame to QPixmap for Qt labels."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)
