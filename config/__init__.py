"""Configuration module for crab detection system."""

from .constants import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    CRAB_CLASSES,
    CLASS_COLORS,
)
from .data_config import load_data_config, create_data_config

__all__ = [
    'DEFAULT_IMAGE_SIZE',
    'DEFAULT_CONFIDENCE_THRESHOLD',
    'DEFAULT_IOU_THRESHOLD',
    'CRAB_CLASSES',
    'CLASS_COLORS',
    'load_data_config',
    'create_data_config',
]
