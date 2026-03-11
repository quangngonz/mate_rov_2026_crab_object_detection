"""Modular dual-window UI package for ROV operations."""

from .backends import (
    BaseDetectorBackend,
    Detection,
    FunctionDetectorBackend,
    UltralyticsBackend,
)
from .camera import get_camera_list, prompt_user_selection
from .controller import DualWindowController

__all__ = [
    "BaseDetectorBackend",
    "Detection",
    "FunctionDetectorBackend",
    "UltralyticsBackend",
    "DualWindowController",
    "get_camera_list",
    "prompt_user_selection",
]
