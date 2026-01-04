"""Models module for training and inference."""

from .detector import CrabDetector
from .trainer import CrabTrainer
from .inference import InferenceEngine

__all__ = [
    'CrabDetector',
    'CrabTrainer',
    'InferenceEngine',
]
