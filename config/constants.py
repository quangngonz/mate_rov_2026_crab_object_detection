"""
Constants and Configuration Values
==================================

Central location for all configurable constants used throughout the system.
"""

from typing import Tuple, List

# Dataset Generation Constants
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (640, 640)
MIN_CRAB_SIZE: int = 50
MAX_CRAB_SIZE: int = 400
MAX_CRABS_PER_IMAGE: int = 8
OVERLAP_THRESHOLD: float = 0.3
RANDOM_SEED: int = 42

# Training Constants
DEFAULT_MODEL_SIZE: str = 'n'  # YOLOv8 model size (n, s, m, l, x)
DEFAULT_EPOCHS: int = 150
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_PATIENCE: int = 50
DEFAULT_SAVE_PERIOD: int = 10
DEFAULT_WORKERS: int = 8
DEFAULT_LEARNING_RATE: float = 0.01

# Inference Constants
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
DEFAULT_IOU_THRESHOLD: float = 0.45
DEFAULT_LINE_WIDTH: int = 2

# Live Detection Constants
DEFAULT_CAMERA_ID: int = 0
DEFAULT_CAMERA_RESOLUTION: Tuple[int, int] = (1920, 1080)
FPS_UPDATE_INTERVAL: int = 30

# Crab Classes
CRAB_CLASSES: List[str] = [
    "European Green Crab",
    "Jonah Crab",
    "Native Rock Crab"
]

# Class Colors (RGB format)
CLASS_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0),    # Green for European Green Crab
    (255, 165, 0),  # Orange for Jonah Crab
    (255, 0, 255)   # Magenta for Native Rock Crab
]

# Paths
DEFAULT_DATASET_DIR: str = 'dataset'
DEFAULT_MODEL_PATH: str = 'weights/best.pt'
DEFAULT_REFERENCE_DIR: str = 'reference_images'
DEFAULT_OUTPUT_DIR: str = 'runs/detect'
DEFAULT_TEST_DIR: str = 'test_images'
DEFAULT_DETECTIONS_DIR: str = 'detections'

# Augmentation Parameters
ROTATION_RANGE: Tuple[float, float] = (0, 360)
SCALE_RANGE: Tuple[float, float] = (0.3, 1.5)
FLIP_PROBABILITY: float = 0.5
COLOR_AUG_PROBABILITY: float = 0.7
BLUR_PROBABILITY: float = 0.4
MOTION_BLUR_PROBABILITY: float = 0.2
PERSPECTIVE_PROBABILITY: float = 0.5
NOISE_PROBABILITY: float = 0.3

BRIGHTNESS_RANGE: Tuple[float, float] = (0.7, 1.3)
CONTRAST_RANGE: Tuple[float, float] = (0.7, 1.3)
SATURATION_RANGE: Tuple[float, float] = (0.6, 1.4)
HUE_SHIFT_RANGE: Tuple[float, float] = (-30, 30)
BLUR_RADIUS_RANGE: Tuple[float, float] = (0, 3)
MOTION_BLUR_KERNEL_RANGE: Tuple[int, int] = (3, 15)
PERSPECTIVE_SCALE_RANGE: Tuple[float, float] = (0.25, 0.45)
NOISE_SIGMA_RANGE: Tuple[float, float] = (5, 15)
