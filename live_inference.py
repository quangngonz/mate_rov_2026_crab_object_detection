"""
Live Inference Script
====================

CLI wrapper for real-time crab detection from camera feed.
"""

from ui import LiveCrabDetector
from pathlib import Path
import argparse


def main():
    """Main entry point for live detection."""

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
        default=0.77,
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

    # Create and run detector
    detector = LiveCrabDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        camera_id=args.camera,
        resolution=(args.width, args.height)
    )

    detector.run()


if __name__ == "__main__":
    main()
