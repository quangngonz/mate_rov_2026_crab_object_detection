"""
Bounding Box Utilities
======================

Functions for bounding box format conversion and geometric operations.
"""

import numpy as np
from typing import Tuple, List


def bbox_to_yolo_format(
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert pixel bbox to YOLO format (normalized center coordinates).

    Args:
        bbox: Bounding box as (x_min, y_min, x_max, y_max) in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
    """
    x_min, y_min, x_max, y_max = bbox

    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    # Clip to [0, 1]
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    width = np.clip(width, 0, 1)
    height = np.clip(height, 0, 1)

    return (x_center, y_center, width, height)


def yolo_to_bbox_format(
    yolo_bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert YOLO format to pixel coordinates.

    Args:
        yolo_bbox: YOLO format (x_center, y_center, width, height) normalized
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in pixels
    """
    x_center, y_center, width, height = yolo_bbox

    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    x_min = x_center_px - width_px / 2
    y_min = y_center_px - height_px / 2
    x_max = x_center_px + width_px / 2
    y_max = y_center_px + height_px / 2

    return (x_min, y_min, x_max, y_max)


def calculate_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box as (x_min, y_min, x_max, y_max)
        box2: Second bounding box as (x_min, y_min, x_max, y_max)

    Returns:
        IoU score between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    x_i_min = max(x1_min, x2_min)
    y_i_min = max(y1_min, y2_min)
    x_i_max = min(x1_max, x2_max)
    y_i_max = min(y1_max, y2_max)

    if x_i_max <= x_i_min or y_i_max <= y_i_min:
        return 0.0

    intersection = (x_i_max - x_i_min) * (y_i_max - y_i_min)

    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def check_overlap(
    new_bbox: Tuple[float, float, float, float],
    existing_bboxes: List[Tuple[float, float, float, float]],
    threshold: float = 0.3
) -> bool:
    """
    Check if new bbox overlaps too much with existing ones.

    Args:
        new_bbox: New bounding box as (x_min, y_min, x_max, y_max)
        existing_bboxes: List of existing bounding boxes
        threshold: Maximum allowed IoU

    Returns:
        True if overlap is acceptable (below threshold), False otherwise
    """
    if not existing_bboxes:
        return True

    for bbox in existing_bboxes:
        iou = calculate_iou(new_bbox, bbox)
        if iou > threshold:
            return False

    return True
