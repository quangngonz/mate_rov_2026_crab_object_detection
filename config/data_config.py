"""
Data Configuration Utilities
============================

Utilities for loading and creating dataset configuration files.
"""

from pathlib import Path
from typing import Dict, List, Any
import yaml


def load_data_config(config_path: str) -> Dict[str, Any]:
    """
    Load dataset configuration from YAML file.

    Args:
        config_path: Path to data.yaml configuration file

    Returns:
        Dictionary containing dataset configuration

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If configuration file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def create_data_config(
    output_path: str,
    dataset_path: str,
    class_names: List[str],
    train_dir: str = 'images/train',
    val_dir: str = 'images/val'
) -> None:
    """
    Create a YOLO format dataset configuration file.

    Args:
        output_path: Path to save the data.yaml file
        dataset_path: Absolute path to dataset root directory
        class_names: List of class names
        train_dir: Relative path to training images from dataset root
        val_dir: Relative path to validation images from dataset root
    """
    output_path = Path(output_path)
    dataset_path = Path(dataset_path).absolute()

    config = {
        'path': str(dataset_path),
        'train': train_dir,
        'val': val_dir,
        'nc': len(class_names),
        'names': class_names
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def validate_dataset_structure(dataset_dir: str) -> bool:
    """
    Validate that dataset has the required directory structure.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        True if structure is valid, False otherwise
    """
    dataset_dir = Path(dataset_dir)

    required_dirs = [
        dataset_dir / 'images' / 'train',
        dataset_dir / 'images' / 'val',
        dataset_dir / 'labels' / 'train',
        dataset_dir / 'labels' / 'val',
    ]

    return all(d.exists() and d.is_dir() for d in required_dirs)
