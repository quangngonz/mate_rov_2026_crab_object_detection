"""
Live Inference Script
====================

CLI wrapper for real-time crab detection from camera feed.
Handles cross-platform camera enumeration and model initialization.
"""

import argparse
import glob
import os
import platform
import sys
from typing import List, Tuple

# Local imports
from ui import LiveCrabDetector


def _get_linux_cameras() -> List[Tuple[int, str]]:
    """
    Enumerates cameras on Linux using sysfs to avoid opening devices.
    Returns: List of (index, name).
    """
    cameras = []
    try:
        # Sort to keep video0, video1 in order
        video_devices = sorted(glob.glob("/sys/class/video4linux/video*"))
        for dev_path in video_devices:
            # Extract index from filenames like 'video0'
            try:
                idx = int(os.path.basename(dev_path).replace("video", ""))
            except ValueError:
                continue

            name_path = os.path.join(dev_path, "name")
            if os.path.exists(name_path):
                with open(name_path, "r", encoding="utf-8") as f:
                    camera_name = f.read().strip()
            else:
                camera_name = f"Camera {idx}"

            cameras.append((idx, camera_name))
    except OSError:
        pass  # Fallback will handle this if permission denied
    return cameras


def _get_windows_cameras() -> List[Tuple[int, str]]:
    """
    Enumerates cameras on Windows using pygrabber (DirectShow).
    Falls back to brute-force OpenCV checks if pygrabber is missing.
    """
    cameras = []
    try:
        from pygrabber.dshow_graph import FilterGraph

        graph = FilterGraph()
        devices = graph.get_input_devices()
        for idx, name in enumerate(devices):
            cameras.append((idx, name))

    except ImportError:
        print("\n[Info] 'pygrabber' not found. Install it for real device names.")
        print("       pip install pygrabber")
        print("[Info] Falling back to index checking...\n")
        return _get_fallback_cameras()

    return cameras


def _get_fallback_cameras(max_checks: int = 5) -> List[Tuple[int, str]]:
    """
    Brute-force checks indices 0 to max_checks using OpenCV.
    Used for macOS or as a fallback for Windows/Linux failures.
    """
    import cv2

    cameras = []
    # Suppress MSMF verbosity on Windows during fallback checks
    if os.name == "nt":
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

    for i in range(max_checks):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append((i, f"Camera Device {i}"))
            cap.release()

    return cameras


def get_camera_list() -> List[Tuple[int, str]]:
    """
    Returns a list of tuples (index, name) for available cameras.
    Dispatches logic based on Operating System.
    """
    system_os = platform.system()

    if system_os == "Linux":
        cams = _get_linux_cameras()
        # If sysfs method failed to find anything, try fallback
        return cams if cams else _get_fallback_cameras()

    if system_os == "Windows":
        return _get_windows_cameras()

    # macOS and others
    return _get_fallback_cameras()


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
