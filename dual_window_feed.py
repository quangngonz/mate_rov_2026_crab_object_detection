"""
Modern Dual-Window Crab Detection App (Modular Launcher)
=========================================================

This launcher wires the modular UI package under ui/modern_dual.
Teammates can independently extend tabs and modules without touching
this entrypoint.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
from PyQt6.QtWidgets import QApplication

from ui.modern_dual import (
    BaseDetectorBackend,
    DualWindowController,
    FunctionDetectorBackend,
    UltralyticsBackend,
    get_camera_list,
    prompt_user_selection,
)


def your_model_inference_hook(frame_bgr: np.ndarray) -> List[Dict]:
    """
    Replace this body with your existing inference call.

    Expected return format:
    [
        {"class": "European Green Crab", "bbox": [x1, y1, x2, y2], "conf": 0.85},
        ...
    ]
    """
    _ = frame_bgr
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modern dual-window crab detection UI")
    parser.add_argument("--source", default="0",
                        help="Camera index (e.g. 0) or video/stream URL")
    parser.add_argument(
        "--select-camera",
        action="store_true",
        help="Interactively select camera index; overrides --source",
    )
    parser.add_argument("--width", type=int, default=1280,
                        help="Capture width")
    parser.add_argument("--height", type=int, default=720,
                        help="Capture height")
    parser.add_argument(
        "--green-class",
        type=str,
        default="European Green Crab",
        help="Exact class name to keep in judge view",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/best_tuned.pt",
        help="Optional Ultralytics model path; if empty, uses your_model_inference_hook",
    )
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Ultralytics confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="Ultralytics IoU threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_options = get_camera_list()

    if args.select_camera:
        try:
            args.source = prompt_user_selection(camera_options)
        except KeyboardInterrupt:
            print("\nCamera selection cancelled.")
            return

    app = QApplication([])
    app.setApplicationName("ROV Dual View - Crab Detection")

    if args.model_path:
        backend: BaseDetectorBackend = UltralyticsBackend(
            model_path=args.model_path,
            conf=args.conf,
            iou=args.iou,
        )
    else:
        backend = FunctionDetectorBackend(your_model_inference_hook)

    controller = DualWindowController(
        source=str(args.source),
        width=args.width,
        height=args.height,
        backend=backend,
        green_class_name=args.green_class,
        camera_options=camera_options,
    )

    app.aboutToQuit.connect(controller.shutdown)
    controller.start()
    app.exec()


if __name__ == "__main__":
    main()
