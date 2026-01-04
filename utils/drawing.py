"""
Drawing Utilities
=================

Functions for drawing bounding boxes and labels on images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: Dict[int, str],
    class_colors: List[Tuple[int, int, int]],
    show_conf: bool = True,
    line_width: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes with labels on an image.

    Args:
        image: Input image as RGB numpy array
        boxes: Array of bounding boxes [x1, y1, x2, y2]
        confidences: Array of confidence scores
        class_ids: Array of class IDs
        class_names: Dictionary mapping class IDs to names
        class_colors: List of RGB colors for each class
        show_conf: Whether to show confidence scores
        line_width: Thickness of box lines

    Returns:
        Image with bounding boxes drawn
    """
    annotated = image.copy()

    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        cls_id = int(cls_id)

        # Get color and name for this class
        color = class_colors[cls_id] if cls_id < len(
            class_colors) else (0, 255, 0)
        class_name = class_names.get(cls_id, f'class_{cls_id}')

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)

        # Draw label with confidence
        if show_conf:
            label = f'{class_name} {conf:.2f}'
        else:
            label = class_name

        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Draw background rectangle for text
        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            thickness,
            cv2.LINE_AA
        )

    return annotated


def draw_info_overlay(
    image: np.ndarray,
    info: Dict[str, any],
    position: Tuple[int, int] = (10, 10),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.6
) -> np.ndarray:
    """
    Draw an information overlay on the image.

    Args:
        image: Input image
        info: Dictionary of information to display
        position: Top-left position of overlay (x, y)
        bg_color: Background color (B, G, R)
        text_color: Text color (B, G, R)
        alpha: Background transparency (0-1)

    Returns:
        Image with overlay
    """
    overlay = image.copy()
    height, width = image.shape[:2]
    x, y = position

    # Calculate overlay size based on content
    line_height = 25
    num_lines = len(info)
    overlay_height = num_lines * line_height + 20
    overlay_width = 300

    # Draw semi-transparent background
    cv2.rectangle(
        overlay,
        (x, y),
        (x + overlay_width, y + overlay_height),
        bg_color,
        -1
    )
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    y_pos = y + 20

    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(
            image,
            text,
            (x + 10, y_pos),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )
        y_pos += line_height

    return image
