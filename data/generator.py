"""
Synthetic Dataset Generator
===========================

Generates synthetic training images with multiple crabs.
"""

from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
from typing import List, Tuple

from .segmentation import CrabSegmentor
from .augmentation import CrabAugmentor
from utils.bbox import bbox_to_yolo_format, check_overlap
from utils.image_processing import create_background
from config.constants import (
    DEFAULT_IMAGE_SIZE,
    MIN_CRAB_SIZE,
    MAX_CRAB_SIZE,
    MAX_CRABS_PER_IMAGE,
    OVERLAP_THRESHOLD,
)
from config.data_config import create_data_config


class SyntheticDatasetGenerator:
    """Generates synthetic training images with multiple crabs."""

    def __init__(
        self,
        output_dir: str,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    ):
        """
        Initialize the synthetic dataset generator.

        Args:
            output_dir: Directory to save generated dataset
            image_size: Size of generated images (width, height)
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.segmentor = CrabSegmentor()
        self.augmentor = CrabAugmentor()

        # Create directory structure
        self.train_img_dir = self.output_dir / 'images' / 'train'
        self.val_img_dir = self.output_dir / 'images' / 'val'
        self.train_label_dir = self.output_dir / 'labels' / 'train'
        self.val_label_dir = self.output_dir / 'labels' / 'val'

        for dir_path in [self.train_img_dir, self.val_img_dir,
                         self.train_label_dir, self.val_label_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def paste_crab_on_background(
        self,
        background: Image.Image,
        crab_rgba: Image.Image,
        position: Tuple[int, int]
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Paste augmented crab onto background at given position.

        Args:
            background: PIL RGB image
            crab_rgba: PIL RGBA image
            position: (x, y) top-left position

        Returns:
            Tuple of (background with crab pasted, bounding box as (x_min, y_min, x_max, y_max))
        """
        import numpy as np

        x, y = position
        background.paste(crab_rgba, (x, y), crab_rgba)

        # Calculate bounding box from actual alpha channel (non-transparent pixels)
        # This ensures bbox fits the visible crab after perspective transforms
        alpha = np.array(crab_rgba.split()[3])
        # Threshold to ignore near-transparent pixels
        coords = np.argwhere(alpha > 10)

        if len(coords) > 0:
            y_min_rel, x_min_rel = coords.min(axis=0)
            y_max_rel, x_max_rel = coords.max(axis=0)

            # Convert to absolute coordinates
            x_min = x + x_min_rel
            y_min = y + y_min_rel
            x_max = x + x_max_rel + 1
            y_max = y + y_max_rel + 1
        else:
            # Fallback to full image if no visible pixels
            x_min = x
            y_min = y
            x_max = x + crab_rgba.width
            y_max = y + crab_rgba.height

        return background, (x_min, y_min, x_max, y_max)

    def generate_single_image(
        self,
        crab_templates: List[Tuple[int, Image.Image]],
        max_crabs: int = MAX_CRABS_PER_IMAGE
    ) -> Tuple[Image.Image, List[Tuple[int, float, float, float, float]]]:
        """
        Generate one synthetic training image with multiple crabs.

        Args:
            crab_templates: List of tuples (class_id, segmented crab RGBA image)
            max_crabs: Maximum number of crabs per image

        Returns:
            Tuple of (generated image, YOLO format annotations)
            Annotations are list of (class_id, x_center, y_center, w, h)
        """
        # Create background
        background = create_background(self.image_size)

        # Randomly decide number of crabs
        num_crabs = random.randint(1, max_crabs)

        bboxes = []
        annotations = []

        max_attempts = 50
        for _ in range(num_crabs):
            # Select random crab template with its class ID
            class_id, crab_template = random.choice(crab_templates)

            # Augment the crab
            augmented_crab = self.augmentor.augment_crab(
                crab_template,
                target_size_range=(MIN_CRAB_SIZE, MAX_CRAB_SIZE)
            )

            # Try to find a valid position
            placed = False
            for attempt in range(max_attempts):
                # Random position
                max_x = self.image_size[1] - augmented_crab.width
                max_y = self.image_size[0] - augmented_crab.height

                if max_x <= 0 or max_y <= 0:
                    break

                x = random.randint(0, max_x)
                y = random.randint(0, max_y)

                # Calculate potential bbox
                potential_bbox = (x, y, x + augmented_crab.width,
                                  y + augmented_crab.height)

                # Check overlap
                if check_overlap(potential_bbox, bboxes, threshold=OVERLAP_THRESHOLD):
                    background, bbox = self.paste_crab_on_background(
                        background, augmented_crab, (x, y)
                    )
                    bboxes.append(bbox)

                    # Convert to YOLO format
                    yolo_bbox = bbox_to_yolo_format(
                        bbox, self.image_size[1], self.image_size[0]
                    )
                    annotations.append((class_id, *yolo_bbox))
                    placed = True
                    break

        return background, annotations

    def generate_dataset(
        self,
        reference_images: List[str],
        num_train: int = 1000,
        num_val: int = 200
    ) -> None:
        """
        Generate complete synthetic dataset.

        Args:
            reference_images: List of paths to reference crab images
            num_train: Number of training images to generate
            num_val: Number of validation images to generate
        """
        print("="*60)
        print("CRAB DETECTION DATASET GENERATION")
        print("="*60)

        # Step 1: Segment crabs from reference images
        print("\n[1/4] Segmenting crabs from reference images...")
        crab_templates = []
        for class_id, img_path in enumerate(reference_images):
            print(f"  - Processing {Path(img_path).name} (class {class_id})")
            segmented = self.segmentor.segment_crab(img_path)
            cropped = self.segmentor.crop_to_content(segmented)
            crab_templates.append((class_id, cropped))

        print(f"  ✓ Segmented {len(crab_templates)} crab templates")

        # Step 2: Generate training images
        print(f"\n[2/4] Generating {num_train} training images...")
        self._generate_split(crab_templates, num_train, self.train_img_dir,
                             self.train_label_dir, "train")

        # Step 3: Generate validation images
        print(f"\n[3/4] Generating {num_val} validation images...")
        self._generate_split(crab_templates, num_val, self.val_img_dir,
                             self.val_label_dir, "val")

        # Step 4: Create data.yaml
        print(f"\n[4/4] Creating dataset configuration...")
        class_names = [Path(img).stem.replace('_', ' ')
                       for img in reference_images]

        create_data_config(
            output_path=str(self.output_dir / 'data.yaml'),
            dataset_path=str(self.output_dir.absolute()),
            class_names=class_names
        )

        print(f"  ✓ Saved configuration to {self.output_dir / 'data.yaml'}")

        print("\n" + "="*60)
        print("DATASET GENERATION COMPLETE!")
        print("="*60)
        print(f"Training images:   {num_train}")
        print(f"Validation images: {num_val}")
        print(f"Output directory:  {self.output_dir}")
        print("="*60)

    def _generate_split(
        self,
        crab_templates: List[Tuple[int, Image.Image]],
        num_images: int,
        img_dir: Path,
        label_dir: Path,
        split_name: str
    ) -> None:
        """Generate images for a single split (train/val)."""
        for i in tqdm(range(num_images), desc=f"{split_name.capitalize()} set"):
            image, annotations = self.generate_single_image(crab_templates)

            # Save image
            img_path = img_dir / f"{split_name}_{i:05d}.jpg"
            image.save(img_path, quality=95)

            # Save annotations
            label_path = label_dir / f"{split_name}_{i:05d}.txt"
            with open(label_path, 'w') as f:
                for ann in annotations:
                    class_id, x_center, y_center, width, height = ann
                    f.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
