"""
Live Crab Detector
==================

Real-time crab detection from camera feed.
"""

import cv2
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
from collections import defaultdict

from models.detector import CrabDetector
from .display import DisplayOverlay, FPSCounter
from config.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_CAMERA_ID,
    DEFAULT_CAMERA_RESOLUTION,
    CRAB_CLASSES,
    CLASS_COLORS,
    FPS_UPDATE_INTERVAL,
)


class LiveCrabDetector:
    """Real-time crab detection from camera feed."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        camera_id: int = DEFAULT_CAMERA_ID,
        resolution: Tuple[int, int] = DEFAULT_CAMERA_RESOLUTION
    ):
        """
        Initialize live crab detector.

        Args:
            model_path: Path to trained model weights
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            camera_id: Camera device ID or video file path
            resolution: Camera resolution (width, height)
        """
        self.detector = CrabDetector(model_path, conf_threshold, iou_threshold)
        self.camera_id = camera_id
        self.resolution = resolution

        # Display components
        self.fps_counter = FPSCounter(update_interval=FPS_UPDATE_INTERVAL)
        self.display = DisplayOverlay(CRAB_CLASSES, CLASS_COLORS)

        # Settings
        self.show_fps = True
        self.line_width = 2

        # Statistics
        self.total_frames = 0
        self.total_detections = 0
        self.session_class_counts = defaultdict(int)

    def initialize_camera(self) -> bool:
        """
        Initialize camera capture.

        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            return False

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        return True

    def process_frame(self, frame: cv2.Mat) -> Tuple[cv2.Mat, int, Dict[str, int]]:
        """
        Process a single frame.

        Args:
            frame: Input frame from camera

        Returns:
            Tuple of (annotated frame, num_detections, class_counts)
        """
        # Save frame temporarily for detection
        temp_path = "/tmp/temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        # Run detection
        result = self.detector.predict(temp_path)

        # Count detections per class
        class_counts = defaultdict(int)
        for cls_id in result['class_ids']:
            cls_id = int(cls_id)
            if cls_id < len(CRAB_CLASSES):
                class_counts[CRAB_CLASSES[cls_id]] += 1

        # Draw boxes
        annotated = self.display.draw_boxes(
            frame.copy(),
            result['boxes'],
            result['confidences'],
            result['class_ids'],
            self.line_width
        )

        return annotated, result['num_detections'], dict(class_counts)

    def save_frame(self, frame: cv2.Mat) -> None:
        """Save current frame to disk."""
        output_dir = Path("detections/live")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"✓ Frame saved: {filename}")

    def run(self) -> None:
        """Run live detection loop."""
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return

        print("\n" + "=" * 50)
        print("Starting live detection...")
        print("Controls:")
        print("  Q or ESC: Quit")
        print("  S: Save current frame")
        print("  +: Increase confidence threshold")
        print("  -: Decrease confidence threshold")
        print("  F: Toggle FPS display")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                # Process frame
                annotated, num_detections, class_counts = self.process_frame(
                    frame)

                # Update statistics
                self.total_frames += 1
                self.total_detections += num_detections
                for cls_name, count in class_counts.items():
                    self.session_class_counts[cls_name] += count

                # Update FPS
                fps = self.fps_counter.update()

                # Draw overlays
                if self.show_fps:
                    annotated = self.display.draw_info(
                        annotated,
                        fps,
                        num_detections,
                        class_counts,
                        self.detector.conf_threshold
                    )
                annotated = self.display.draw_controls(annotated)

                # Display
                cv2.imshow('Crab Detection - Live', annotated)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('s'):
                    self.save_frame(annotated)
                elif key == ord('+') or key == ord('='):
                    self.detector.conf_threshold = min(
                        0.95, self.detector.conf_threshold + 0.05)
                    print(
                        f"Confidence threshold: {self.detector.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.detector.conf_threshold = max(
                        0.05, self.detector.conf_threshold - 0.05)
                    print(
                        f"Confidence threshold: {self.detector.conf_threshold:.2f}")
                elif key == ord('f'):
                    self.show_fps = not self.show_fps

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources and print statistics."""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

        # Print statistics
        print("\n" + "=" * 50)
        print("Session Statistics:")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Total detections: {self.total_detections}")
        if self.total_frames > 0:
            print(
                f"  Avg detections/frame: {self.total_detections/self.total_frames:.2f}")
        print("\nDetections by Class:")
        for cls_name, count in self.session_class_counts.items():
            print(f"  {cls_name}: {count}")
        print("=" * 50)
