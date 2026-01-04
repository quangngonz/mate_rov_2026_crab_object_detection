"""
Synthetic Dataset Generation Script
===================================

CLI wrapper for generating synthetic crab detection datasets.
"""

from config.constants import RANDOM_SEED
from data import SyntheticDatasetGenerator
from pathlib import Path
import random
import numpy as np


def main():
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Configuration
    reference_dir = Path('reference_images')
    output_dir = Path('dataset')

    # Get reference images
    reference_images = [
        reference_dir / 'European_Green_Crab_Image.png',
        reference_dir / 'Jonah_crab.png',
        reference_dir / 'Native_Rock_Crab.png'
    ]

    # Verify reference images exist
    for img_path in reference_images:
        if not img_path.exists():
            print(f"Error: Reference image not found: {img_path}")
            return

    # Create generator
    generator = SyntheticDatasetGenerator(output_dir, image_size=(640, 640))

    # Generate dataset
    generator.generate_dataset(
        reference_images=reference_images,
        num_train=1000,
        num_val=200
    )


if __name__ == '__main__':
    main()
