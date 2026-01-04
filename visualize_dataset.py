"""
Dataset Visualization Script
============================

CLI wrapper for visualizing and analyzing datasets.
"""

from config.data_config import load_data_config
from data import DatasetVisualizer, DatasetAnalyzer
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='Visualize synthetic crab dataset'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='dataset',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='Dataset split to visualize'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=16,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Randomly sample images (default: sequential)'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze dataset statistics'
    )
    parser.add_argument(
        '--single',
        type=str,
        default=None,
        help='Visualize a single image by filename'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save visualization to file instead of showing'
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Verify dataset exists
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("Please run scripts/generate_dataset.py first.")
        return

    # Load class names
    yaml_path = dataset_dir / 'data.yaml'
    if yaml_path.exists():
        config = load_data_config(str(yaml_path))
        class_names = config.get('names', ['crab'])
    else:
        class_names = ['crab']

    # Analyze dataset if requested
    if args.analyze:
        analyzer = DatasetAnalyzer()
        analyzer.analyze(str(dataset_dir), class_names)
        return

    # Create visualizer
    visualizer = DatasetVisualizer(class_names)

    # Visualize single image if requested
    if args.single:
        image_path = dataset_dir / 'images' / args.split / args.single
        label_path = dataset_dir / 'labels' / \
            args.split / f"{Path(args.single).stem}.txt"

        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return

        fig = visualizer.visualize_sample(str(image_path), str(label_path))

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

    fig = visualizer.visualize_grid(
        str(image_dir),
        str(label_dir),
        num_samples=args.num_samples,
        random_sample=args.random
    )

    if fig is None:
        print(f"No images found in {image_dir}")
        return

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
