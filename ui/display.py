"""
Display Utilities for Live Detection
====================================

Handles overlays and FPS tracking for live video feeds.
"""

import cv2
import numpy as np
import time
from typing import Dict, Tuple, List


class FPSCounter:
    """Tracks and calculates frames per second."""

    def __init__(self, update_interval: int = 30):
        """
        Initialize FPS counter.

        Args:
            update_interval: Number of frames between FPS updates
        """
        self.update_interval = update_interval
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

    def update(self) -> float:
        """
        Update frame count and calculate FPS if interval reached.

        Returns:
            Current FPS value
        """
        self.frame_count += 1

        if self.frame_count >= self.update_interval:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        return self.fps

    def reset(self) -> None:
        """Reset the FPS counter."""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0


class DisplayOverlay:
    """Draws information overlays on video frames."""

    def __init__(
        self,
        class_names: List[str],
        class_colors: List[Tuple[int, int, int]]
    ):
        """
        Initialize display overlay.

        Args:
            class_names: List of detection class names
            class_colors: List of colors for each class (RGB)
        """
        self.class_names = class_names
        self.class_colors = class_colors

    def draw_info(
        self,
        image: np.ndarray,
        fps: float,
        num_detections: int,
        class_counts: Dict[str, int],
        conf_threshold: float,
        contrast: float = 1.0
    ) -> np.ndarray:
        """
        Draw information overlay on image.

        Args:
            image: Input image
            fps: Current FPS
            num_detections: Total number of detections in frame
            class_counts: Dictionary of counts per class
            conf_threshold: Current confidence threshold
            contrast: Current contrast multiplier

        Returns:
            Image with overlay drawn
        """
        height, width = image.shape[:2]
        overlay = image.copy()

        # Draw semi-transparent background
        cv2.rectangle(overlay, (10, 10), (320, 205), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Draw text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        y_pos = 30

        # FPS
        cv2.putText(image, f"FPS: {fps:.1f}", (20, y_pos),
                    font, font_scale, color, thickness)
        y_pos += 25

        # Total detections
        cv2.putText(image, f"Total Detections: {num_detections}",
                    (20, y_pos), font, font_scale, color, thickness)
        y_pos += 25

        # Per-class detections
        for i, class_name in enumerate(self.class_names):
            count = class_counts.get(class_name, 0)
            class_color = self.class_colors[i] if i < len(
                self.class_colors) else (255, 255, 255)
            cv2.putText(image, f"{class_name}: {count}", (20,
                        y_pos), font, font_scale, class_color, thickness)
            y_pos += 25

        # Confidence threshold
        cv2.putText(image, f"Conf: {conf_threshold:.2f}",
                    (20, y_pos), font, font_scale, color, thickness)
        y_pos += 25

        # Contrast
        cv2.putText(image, f"Contrast: {contrast:.1f}",
                    (20, y_pos), font, font_scale, color, thickness)

        return image

    def draw_controls(
        self,
        image: np.ndarray,
        controls_text: str = "Q:Quit | S:Save | +/-:Conf | I/O:Contrast | F:FPS"
    ) -> np.ndarray:
        """
        Draw control instructions at bottom of image.

        Args:
            image: Input image
            controls_text: Text describing controls

        Returns:
            Image with controls drawn
        """
        height = image.shape[0]
        cv2.putText(
            image,
            controls_text,
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        return image

    def draw_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        line_width: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes with labels on image.

        Args:
            image: Input image
            boxes: Detection boxes [[x1, y1, x2, y2], ...]
            confidences: Confidence scores
            class_ids: Class IDs
            line_width: Line width for boxes

        Returns:
            Image with boxes drawn
        """
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls_id)

            # Get class name and color
            class_name = self.class_names[cls_id] if cls_id < len(
                self.class_names) else f"Class {cls_id}"
            color = self.class_colors[cls_id] if cls_id < len(
                self.class_colors) else (255, 255, 255)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)

            # Draw label
            label = f"{class_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness)

            # Background for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Text
            cv2.putText(image, label, (x1, y1 - baseline - 2),
                        font, font_scale, (0, 0, 0), thickness)

        return image
