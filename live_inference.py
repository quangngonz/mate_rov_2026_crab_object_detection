"""
Live Camera Inference for Crab Detection
========================================

This script captures video from a camera and runs real-time crab detection
using the trained YOLOv8 model.

Features:
- Real-time detection from webcam or video stream
- Live bounding box visualization with confidence scores
- FPS counter
- Keyboard controls for configuration
- Support for multiple camera sources

Controls:
- 'q' or ESC: Quit
- 's': Save current frame
- '+': Increase confidence threshold
- '-': Decrease confidence threshold
- 'f': Toggle FPS display
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import torch
import time
from datetime import datetime


class LiveCrabDetector:
    """Real-time crab detection from camera feed."""

    def __init__(
        self,
        model_path="weights/best.pt",
        conf_threshold=0.25,
        iou_threshold=0.45,
        camera_id=0,
        resolution=(640, 480)
    ):
        """
        Initialize the live crab detector.

        Args:
            model_path: Path to trained model weights (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            camera_id: Camera device ID or video file path
            resolution: Camera resolution (width, height)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.camera_id = camera_id
        self.resolution = resolution

        # Initialize model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("✓ Model loaded successfully")

        # Check device
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        print(f"Using device: {self.device}")

        # Class names and colors
        self.class_names = [
            "European Green Crab",
            "Jonah Crab",
            "Native Rock Crab"
        ]
        self.class_colors = [
            (0, 255, 0),    # Green for European Green Crab
            (255, 165, 0),  # Orange for Jonah Crab
            (255, 0, 255)   # Magenta for Native Rock Crab
        ]

        # Display settings
        self.show_fps = True
        self.line_width = 2

        # Statistics
        self.fps = 0
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts = {name: 0 for name in self.class_names}

    def initialize_camera(self):
        """Initialize the camera capture."""
        print(f"Initializing camera (ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Get actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera initialized at {actual_width}x{actual_height}")

        return True

    def draw_boxes(self, image, boxes, confidences, class_ids):
        """
        Draw bounding boxes on image.

        Args:
            image: Input image (numpy array)
            boxes: Detection boxes [[x1, y1, x2, y2], ...]
            confidences: Confidence scores
            class_ids: Class IDs

        Returns:
            Annotated image
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
            cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_width)

            # Draw label with confidence
            label = f"{class_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                thickness
            )

        return image

    def draw_info_overlay(self, image, num_detections, fps, current_detections):
        """
        Draw information overlay on image.

        Args:
            image: Input image
            num_detections: Number of detections in current frame
            fps: Current FPS
            current_detections: Dictionary with current frame class counts

        Returns:
            Image with overlay
        """
        height, width = image.shape[:2]
        overlay = image.copy()

        # Semi-transparent background for info
        cv2.rectangle(
            overlay,
            (10, 10),
            (320, 180),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Draw text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1

        y_pos = 30
        cv2.putText(
            image,
            f"FPS: {fps:.1f}",
            (20, y_pos),
            font,
            font_scale,
            color,
            thickness
        )

        y_pos += 25
        cv2.putText(
            image,
            f"Total Detections: {num_detections}",
            (20, y_pos),
            font,
            font_scale,
            color,
            thickness
        )

        # Show per-class detections
        y_pos += 25
        for i, class_name in enumerate(self.class_names):
            count = current_detections.get(class_name, 0)
            class_color = self.class_colors[i]
            cv2.putText(
                image,
                f"{class_name}: {count}",
                (20, y_pos),
                font,
                font_scale,
                class_color,
                thickness
            )
            y_pos += 25

        cv2.putText(
            image,
            f"Conf: {self.conf_threshold:.2f}",
            (20, y_pos),
            font,
            font_scale,
            color,
            thickness
        )

        # Draw controls at bottom
        controls_text = "Q:Quit | S:Save | +/-:Conf | F:FPS"
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

    def process_frame(self, frame):
        """
        Run detection on a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Annotated frame with detections
        """
        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]

        # Get detection info
        boxes = results.boxes.xyxy.cpu().numpy() if len(
            results.boxes) > 0 else np.array([])
        confidences = results.boxes.conf.cpu().numpy() if len(
            results.boxes) > 0 else np.array([])
        class_ids = results.boxes.cls.cpu().numpy().astype(
            int) if len(results.boxes) > 0 else np.array([])
        num_detections = len(boxes)

        # Count detections per class for current frame
        current_detections = {name: 0 for name in self.class_names}
        for cls_id in class_ids:
            cls_id = int(cls_id)
            if cls_id < len(self.class_names):
                current_detections[self.class_names[cls_id]] += 1

        # Draw bounding boxes
        annotated_frame = self.draw_boxes(
            frame.copy(),
            boxes,
            confidences,
            class_ids
        )

        return annotated_frame, num_detections, current_detections

    def save_frame(self, frame):
        """Save current frame to disk."""
        output_dir = Path("detections/live")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"detection_{timestamp}.jpg"

        cv2.imwrite(str(filename), frame)
        print(f"✓ Frame saved: {filename}")

    def run(self):
        """
        Main loop for live detection.

        Run live detection from camera feed with real-time visualization.
        """
        try:
            # Initialize camera
            self.initialize_camera()

            print("\n" + "=" * 50)
            print("Starting live detection...")
            print("Controls:")
            print("  Q or ESC: Quit")
            print("  S: Save current frame")
            print("  +: Increase confidence threshold")
            print("  -: Decrease confidence threshold")
            print("  F: Toggle FPS display")
            print("=" * 50 + "\n")

            # FPS calculation variables
            fps_start_time = time.time()
            fps_frame_count = 0

            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                # Process frame
                start_time = time.time()
                annotated_frame, num_detections, current_detections = self.process_frame(
                    frame)
                process_time = time.time() - start_time

                # Update statistics
                self.frame_count += 1
                self.total_detections += num_detections
                for class_name, count in current_detections.items():
                    self.class_counts[class_name] += count
                fps_frame_count += 1

                # Calculate FPS every 30 frames
                if fps_frame_count >= 30:
                    elapsed_time = time.time() - fps_start_time
                    self.fps = fps_frame_count / elapsed_time
                    fps_frame_count = 0
                    fps_start_time = time.time()

                # Draw info overlay
                if self.show_fps:
                    annotated_frame = self.draw_info_overlay(
                        annotated_frame,
                        num_detections,
                        self.fps,
                        current_detections
                    )

                # Display frame
                cv2.imshow('Crab Detection - Live', annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nStopping detection...")
                    break
                elif key == ord('s'):  # Save frame
                    self.save_frame(annotated_frame)
                elif key == ord('+') or key == ord('='):  # Increase confidence
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"Confidence threshold: {self.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):  # Decrease confidence
                    self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                    print(f"Confidence threshold: {self.conf_threshold:.2f}")
                elif key == ord('f'):  # Toggle FPS
                    self.show_fps = not self.show_fps
                    print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during detection: {e}")
            raise
        finally:
            # Cleanup
            if hasattr(self, 'cap'):
                self.cap.release()
            cv2.destroyAllWindows()

            # Print statistics
            print("\n" + "=" * 50)
            print("Detection Statistics:")
            print(f"  Total frames processed: {self.frame_count}")
            print(f"  Total detections: {self.total_detections}")
            print(f"  Average FPS: {self.fps:.1f}")
            if self.frame_count > 0:
                print(
                    f"  Avg detections/frame: {self.total_detections/self.frame_count:.2f}")
            print("\nDetections by Class:")
            for class_name, count in self.class_counts.items():
                print(f"  {class_name}: {count}")
            print("=" * 50)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Live crab detection from camera feed"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.75,
        help="Confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (0.0-1.0)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (0 for default webcam)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Camera frame width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Camera frame height"
    )

    args = parser.parse_args()

    # Create detector
    detector = LiveCrabDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        camera_id=args.camera,
        resolution=(args.width, args.height)
    )

    # Run live detection
    detector.run()


if __name__ == "__main__":
    main()
