"""
Dataset Visualization Utility
==============================

This script helps visualize the generated synthetic dataset to verify
that augmentations and annotations are correct.

Usage:
    python visualize_dataset.py --split train --num_samples 10
"""

import cv2
import numpy as np
from pathlib import Path
import random
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml
import collections


def load_yolo_annotations(label_path, img_width, img_height):
    """
    Load YOLO format annotations and convert to pixel coordinates.

    Args:
        label_path: Path to .txt annotation file
        img_width, img_height: Image dimensions

    Returns:
        list: List of bounding boxes [(x1, y1, x2, y2, class_id), ...]
    """
    if not Path(label_path).exists():
        return []

    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            boxes.append((x1, y1, x2, y2, class_id))

    return boxes


def visualize_sample(image_path, label_path, class_names):
    """
    Visualize a single image with its bounding boxes.

    Args:
        image_path: Path to image
        label_path: Path to annotation file
        class_names: List of class names

    Returns:
        matplotlib figure
    """
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Load annotations
    boxes = load_yolo_annotations(label_path, width, height)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    # Draw bounding boxes
    colors = ['#00FF00', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF']
    for box in boxes:
        x1, y1, x2, y2, class_id = box
        color = colors[class_id % len(colors)]

        # Draw rectangle
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        label = class_names[class_id] if class_id < len(
            class_names) else f'Class {class_id}'
        ax.text(
            x1, y1 - 5, label,
            color='white', fontsize=10,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
        )

    ax.set_title(f'{image_path.name} - {len(boxes)} object(s)', fontsize=12)
    ax.axis('off')

    return fig


def visualize_grid(image_dir, label_dir, class_names, num_samples=16, random_sample=True):
    """
    Visualize a grid of samples from the dataset.

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        class_names: List of class names
        num_samples: Number of samples to show
        random_sample: Whether to randomly sample or take first N

    Returns:
        matplotlib figure
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    # Get image files
    image_files = sorted(list(image_dir.glob('*.jpg')) +
                         list(image_dir.glob('*.png')))

    if not image_files:
        print(f"No images found in {image_dir}")
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
        boxes = load_yolo_annotations(label_path, width, height)

        # Plot
        ax = axes[idx]
        ax.imshow(image)

        # Draw bounding boxes
        colors = ['#00FF00', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF']
        for box in boxes:
            x1, y1, x2, y2, class_id = box
            color = colors[class_id % len(colors)]

            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor=color, facecolor='none'
            )
            # Add label
            label = class_names[class_id] if class_id < len(
                class_names) else f'Class {class_id}'
            ax.text(
                x1, y1 - 2, label,
                color='white', fontsize=6,
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1)
            )

        ax.set_title(f'{len(boxes)} objects', fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(len(image_files), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def analyze_dataset(dataset_dir, class_names=None):
    """
    Analyze dataset statistics.

    Args:
        dataset_dir: Root directory of dataset
        class_names: List of class names (optional)
    """
    dataset_dir = Path(dataset_dir)

    # Load data.yaml if class_names not provided
    if class_names is None:
        yaml_path = dataset_dir / 'data.yaml'
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                class_names = data_config.get('names', [])
        else:
            class_names = []

    print("="*80)
    print("DATASET ANALYSIS")
    print("="*80)

    # Analyze each split
    for split in ['train', 'val']:
        label_dir = dataset_dir / 'labels' / split
        image_dir = dataset_dir / 'images' / split

        if not label_dir.exists():
            continue

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
                sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                max_len = max([len(str(n)) for n, c in sorted_counts])
                for name, count in sorted_counts:
                    pct = count / total_objects * 100
                    bar = '█' * int(pct / 2)
                    print(f"    {name:<{max_len}}: {count:4d} ({pct:5.1f}%) {bar}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize synthetic crab dataset')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to visualize')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to visualize')
    parser.add_argument('--random', action='store_true',
                        help='Randomly sample images (default: sequential)')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze dataset statistics')
    parser.add_argument('--single', type=str, default=None,
                        help='Visualize a single image by filename')
    parser.add_argument('--output', type=str, default=None,
                        help='Save visualization to file instead of showing')

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Verify dataset exists
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("Please run generate_dataset.py first.")
        return

    # Load class names
    yaml_path = dataset_dir / 'data.yaml'
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            class_names = data_config.get('names', ['crab'])
    else:
        class_names = ['crab']

    # Analyze dataset if requested
    if args.analyze:
        analyze_dataset(dataset_dir, class_names)
        return

    # Visualize single image if requested
    if args.single:
        image_path = dataset_dir / 'images' / args.split / args.single
        label_path = dataset_dir / 'labels' / \
            args.split / f"{Path(args.single).stem}.txt"

        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return

        fig = visualize_sample(image_path, label_path, class_names)

        if args.output:
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {args.output}")
        else:
            plt.show()

        return

    # Visualize grid
    image_dir = dataset_dir / 'images' / args.split
    label_dir = dataset_dir / 'labels' / args.split

    print(f"Visualizing {args.num_samples} samples from {args.split} split...")

    fig = visualize_grid(
        image_dir, label_dir, class_names,
        num_samples=args.num_samples,
        random_sample=args.random
    )

    if fig is None:
        return

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
