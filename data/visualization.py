"""
Dataset Visualization Module
============================

Utilities for visualizing and analyzing generated datasets.
"""

import cv2
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import collections

from utils.bbox import yolo_to_bbox_format
from config.data_config import load_data_config


class DatasetVisualizer:
    """Visualizes synthetic dataset samples."""

    def __init__(self, class_names: List[str]):
        """
        Initialize the dataset visualizer.

        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.colors = ['#00FF00', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF']

    def load_annotations(
        self,
        label_path: str,
        img_width: int,
        img_height: int
    ) -> List[Tuple[float, float, float, float, int]]:
        """
        Load YOLO format annotations and convert to pixel coordinates.

        Args:
            label_path: Path to .txt annotation file
            img_width: Image width
            img_height: Image height

        Returns:
            List of bounding boxes [(x1, y1, x2, y2, class_id), ...]
        """
        label_path = Path(label_path)
        if not label_path.exists():
            return []

        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert to pixel coordinates
                x1, y1, x2, y2 = yolo_to_bbox_format(
                    (x_center, y_center, width, height),
                    img_width,
                    img_height
                )

                boxes.append((x1, y1, x2, y2, class_id))

        return boxes

    def visualize_sample(
        self,
        image_path: str,
        label_path: str
    ) -> plt.Figure:
        """
        Visualize a single image with its bounding boxes.

        Args:
            image_path: Path to image
            label_path: Path to annotation file

        Returns:
            Matplotlib figure
        """
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Load annotations
        boxes = self.load_annotations(label_path, width, height)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)

        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2, class_id = box
            color = self.colors[class_id % len(self.colors)]

            # Draw rectangle
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            # Add label
            label = self.class_names[class_id] if class_id < len(
                self.class_names) else f'Class {class_id}'
            ax.text(
                x1, y1 - 5, label,
                color='white', fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
            )

        ax.set_title(
            f'{Path(image_path).name} - {len(boxes)} object(s)', fontsize=12)
        ax.axis('off')

        return fig

    def visualize_grid(
        self,
        image_dir: str,
        label_dir: str,
        num_samples: int = 16,
        random_sample: bool = True
    ) -> Optional[plt.Figure]:
        """
        Visualize a grid of samples from the dataset.

        Args:
            image_dir: Directory containing images
            label_dir: Directory containing labels
            num_samples: Number of samples to show
            random_sample: Whether to randomly sample or take first N

        Returns:
            Matplotlib figure or None if no images found
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        # Get image files
        image_files = sorted(
            list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        )

        if not image_files:
            return None

        # Sample images
        if random_sample:
            if len(image_files) > num_samples:
                image_files = random.sample(image_files, num_samples)
        else:
            image_files = image_files[:num_samples]

        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_samples)))

        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        if grid_size == 1:
            axes = np.array([[axes]])
        axes = axes.flatten()

        for idx, image_path in enumerate(image_files):
            # Get corresponding label file
            label_path = label_dir / f"{image_path.stem}.txt"

            # Load image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]

            # Load annotations
            boxes = self.load_annotations(label_path, width, height)

            # Plot
            ax = axes[idx]
            ax.imshow(image)

            # Draw bounding boxes
            for box in boxes:
                x1, y1, x2, y2, class_id = box
                color = self.colors[class_id % len(self.colors)]

                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)

            ax.set_title(f'{len(boxes)} objects', fontsize=8)
            ax.axis('off')

        # Hide unused subplots
        for idx in range(len(image_files), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig


class DatasetAnalyzer:
    """Analyzes dataset statistics."""

    def analyze(
        self,
        dataset_dir: str,
        class_names: Optional[List[str]] = None
    ) -> None:
        """
        Analyze and print dataset statistics.

        Args:
            dataset_dir: Root directory of dataset
            class_names: List of class names (loaded from config if None)
        """
        dataset_dir = Path(dataset_dir)

        # Load class names if not provided
        if class_names is None:
            yaml_path = dataset_dir / 'data.yaml'
            if yaml_path.exists():
                config = load_data_config(str(yaml_path))
                class_names = config.get('names', [])
            else:
                class_names = []

        print("="*80)
        print("DATASET ANALYSIS")
        print("="*80)

        # Analyze each split
        for split in ['train', 'val']:
            self._analyze_split(dataset_dir, split, class_names)

        print("="*80)

    def _analyze_split(
        self,
        dataset_dir: Path,
        split: str,
        class_names: List[str]
    ) -> None:
        """Analyze a single dataset split."""
        label_dir = dataset_dir / 'labels' / split
        image_dir = dataset_dir / 'images' / split

        if not label_dir.exists():
            return

        print(f"\n{split.upper()} Split:")
        print("-" * 80)

        # Count files
        label_files = list(label_dir.glob('*.txt'))
        image_files = list(image_dir.glob('*.jpg')) + \
            list(image_dir.glob('*.png'))

        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")

        # Analyze annotations
        total_objects = 0
        objects_per_image = []
        empty_images = 0
        class_counts = collections.defaultdict(int)

        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                num_objects = len(lines)
                total_objects += num_objects
                objects_per_image.append(num_objects)
                if num_objects == 0:
                    empty_images += 1

                # Count classes
                for line in lines:
                    try:
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(class_names):
                            class_counts[class_names[class_id]] += 1
                        else:
                            class_counts[f"Class {class_id}"] += 1
                    except (ValueError, IndexError):
                        pass

        if objects_per_image:
            print(f"  Total objects: {total_objects}")
            print(
                f"  Objects per image: {np.mean(objects_per_image):.2f} +/- {np.std(objects_per_image):.2f}")
            print(f"  Min objects: {np.min(objects_per_image)}")
            print(f"  Max objects: {np.max(objects_per_image)}")
            print(f"  Empty images: {empty_images}")

            # Object count distribution
            print(f"\n  Images by Object Count:")
            for i in range(int(np.max(objects_per_image)) + 1):
                count = objects_per_image.count(i)
                if count > 0:
                    pct = count / len(objects_per_image) * 100
                    bar = '█' * int(pct / 2)
                    print(f"    {i} objects: {count:4d} ({pct:5.1f}%) {bar}")

            # Class distribution
            if class_counts:
                print(f"\n  Class Distribution:")
                sorted_counts = sorted(
                    class_counts.items(), key=lambda x: x[1], reverse=True)
                for name, count in sorted_counts:
                    pct = count / total_objects * 100
                    bar = '█' * int(pct / 2)
                    print(f"    {name}: {count:4d} ({pct:5.1f}%) {bar}")
