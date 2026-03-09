"""
Crab Detection Trainer
=====================

Handles training of YOLOv26 models for crab detection.
"""

from pathlib import Path
from typing import Optional
import torch
import yaml
from ultralytics import YOLO

from config.constants import (
    DEFAULT_MODEL_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PATIENCE,
    DEFAULT_SAVE_PERIOD,
    DEFAULT_WORKERS,
    DEFAULT_LEARNING_RATE,
)


class CrabTrainer:
    """Trains YOLOv26 models for crab detection."""

    def __init__(
        self,
        data_yaml: str,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = ''
    ):
        """
        Initialize the crab trainer.

        Args:
            data_yaml: Path to dataset configuration file
            model_size: YOLOv26 model size (n, s, m, l, x)
            device: Device to train on ('' for auto, 'cpu', 'cuda', 'mps')
        """
        self.data_yaml = Path(data_yaml)
        self.model_size = model_size
        self.device = self._select_device(device)

        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"Dataset configuration not found: {data_yaml}")

    def _select_device(self, device: str) -> str:
        """Select appropriate device for training."""
        if device:
            return device

        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def load_model(self, pretrained: bool = True, checkpoint_path: Optional[str] = None) -> YOLO:
        """
        Load YOLOv26 model.

        Args:
            pretrained: Whether to use pretrained weights
            checkpoint_path: Path to checkpoint to resume from (optional)

        Returns:
            YOLO model instance
        """
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            return YOLO(checkpoint_path)

        if pretrained:
            # Prefer current naming convention, fallback for compatibility.
            model_candidates = [
                f'yolo26{self.model_size}.pt',
                f'yolov26{self.model_size}.pt',
            ]
        else:
            model_candidates = [
                f'yolo26{self.model_size}.yaml',
                f'yolov26{self.model_size}.yaml',
            ]

        last_error = None
        for model_name in model_candidates:
            try:
                print(f"Loading model: {model_name}")
                return YOLO(model_name)
            except FileNotFoundError as e:
                last_error = e
                continue

        if last_error is not None:
            raise last_error

        raise RuntimeError("No valid model candidates were generated.")

    def train(
        self,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        img_size: int = DEFAULT_IMAGE_SIZE[0],
        project: str = 'runs/detect',
        name: str = 'crab_detector',
        exist_ok: bool = False,
        pretrained: bool = True,
        patience: int = DEFAULT_PATIENCE,
        save_period: int = DEFAULT_SAVE_PERIOD,
        workers: int = DEFAULT_WORKERS,
        resume_checkpoint: Optional[str] = None
    ) -> dict:
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            project: Project directory
            name: Run name
            exist_ok: Whether to overwrite existing run
            pretrained: Use pretrained weights
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            workers: Number of dataloader workers
            resume_checkpoint: Path to checkpoint to resume from

        Returns:
            Dictionary with training results
        """
        # Load model
        model = self.load_model(pretrained, resume_checkpoint)

        # Train
        results = model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=self.device,
            project=project,
            name=name,
            exist_ok=exist_ok,
            pretrained=pretrained,
            patience=patience,
            save_period=save_period,
            workers=workers,
            # Augmentation settings
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            # Optimizer settings
            optimizer='AdamW',
            lr0=DEFAULT_LEARNING_RATE,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            # Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,
            # Other settings
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=10,
            amp=False if self.device in ["mps", "cpu"] else True,
            fraction=1.0,
            profile=False,
            freeze=None,
        )

        return {
            'results': results,
            'save_dir': Path(project) / name,
            'best_model': Path(project) / name / 'weights' / 'best.pt',
            'last_model': Path(project) / name / 'weights' / 'last.pt'
        }
