"""
Live Inference Script
====================

CLI wrapper for real-time crab detection from camera feed.
Handles cross-platform camera enumeration and model initialization.
"""

import argparse
import sys
from typing import List, Tuple

from cv2_enumerate_cameras import enumerate_cameras

# Local imports
from ui import LiveCrabDetector


def get_camera_list() -> List[Tuple[int, str]]:
    """
    Returns a list of tuples (index, name) for available cameras.
    Uses cv2_enumerate_cameras for cross-platform camera detection.
    """
    cameras = []
    try:
        for idx, camera_info in enumerate(enumerate_cameras()):
            # Extract index and name from camera_info object
            # idx = camera_info.index
            name = camera_info.name if hasattr(
                camera_info, 'name') else f"Camera {idx}"
            cameras.append((idx, name))
    except Exception as e:
        print(f"Warning: Error enumerating cameras: {e}")
        # Fallback to index 0 if enumeration fails
        cameras = [(0, "Camera 0")]

    return cameras


def prompt_user_selection(available_cams: List[Tuple[int, str]]) -> int:
    """
    Interactively prompts the user to select a camera index.
    """
    if not available_cams:
        print("No cameras found. Defaulting to index 0.")
        return 0

    print("\nSelect Camera Source:")
    print("-" * 50)
    print(f"{'Index':<7} | {'Device Name'}")
    print("-" * 50)

    for idx, name in available_cams:
        print(f"{idx:<7} | {name}")
    print("-" * 50)

    valid_indices = {c[0] for c in available_cams}

    while True:
        choice = input("\nEnter camera index (default: 0): ").strip()

        if choice == "":
            return 0

        try:
            selection = int(choice)
            if selection in valid_indices:
                return selection
            print(f"Invalid index '{selection}'. Please choose from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    parser = argparse.ArgumentParser(
        description="Live crab detection from camera feed"
    )
    parser.add_argument(
        "--model", type=str, default="weights/best.pt", help="Path to .pt model"
    )
    parser.add_argument(
        "--conf", type=float, default=0.77, help="Confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera ID (default: 0)"
    )
    parser.add_argument(
        "--select-camera",
        action="store_true",
        help="List and select camera interactively",
    )
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int,
                        default=1080, help="Frame height")

    args = parser.parse_args()

    camera_id = args.camera

    # Handle interactive selection
    if args.select_camera:
        try:
            devices = get_camera_list()
            camera_id = prompt_user_selection(devices)
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)

    print(f"Starting detector on Camera {camera_id}...")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Model: {args.model} (Conf: {args.conf}, IoU: {args.iou})")

    try:
        detector = LiveCrabDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            camera_id=camera_id,
            resolution=(args.width, args.height),
        )
        detector.run()
    except Exception as e:
        print(f"\nError initializing detector: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
