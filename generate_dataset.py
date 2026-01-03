"""
Synthetic Dataset Generator for Crab Detection
==============================================

This script generates a synthetic training dataset from reference images containing single crabs.

Strategy:
1. Segment crabs from reference images using background removal
2. Apply extensive augmentations to each crab
3. Compose multiple augmented crabs onto random backgrounds
4. Generate YOLO format annotations (normalized bbox coordinates)

Augmentation Pipeline:
- Rotation: 0-360 degrees (full rotation)
- Scale: 0.3-1.5x (handles size variation)
- Horizontal/Vertical flip: 50% probability each
- Color jitter: Hue (±30°), Saturation (±40%), Brightness (±30%), Contrast (±30%)
- Gaussian noise: σ=0-15
- Gaussian blur: kernel size 0-5
- Position: Random placement avoiding excessive overlap
"""

import cv2
import numpy as np
from pathlib import Path
import random
from rembg import remove
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from tqdm import tqdm
import yaml

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)


class CrabSegmentor:
    """Segments crabs from reference images using background removal."""

    def __init__(self):
        pass

    def segment_crab(self, image_path):
        """
        Remove background and return crab with alpha channel.

        Returns:
            PIL.Image: RGBA image with transparent background
        """
        input_image = Image.open(image_path)
        output_image = remove(input_image)
        return output_image

    def get_bounding_box(self, rgba_image):
        """
        Get tight bounding box around non-transparent pixels.

        Returns:
            tuple: (x_min, y_min, x_max, y_max)
        """
        alpha = np.array(rgba_image)[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)

        if not rows.any() or not cols.any():
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (x_min, y_min, x_max, y_max)

    def crop_to_content(self, rgba_image):
        """Crop image to content bounding box with small padding."""
        bbox = self.get_bounding_box(rgba_image)
        if bbox is None:
            return rgba_image

        x_min, y_min, x_max, y_max = bbox
        # Add 5% padding
        width, height = rgba_image.size
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        return rgba_image.crop((x_min, y_min, x_max, y_max))


class CrabAugmentor:
    """Applies augmentations to segmented crab images."""

    def augment_crab(self, crab_rgba, target_size_range=(50, 400)):
        """
        Apply random augmentations to a crab image.

        Args:
            crab_rgba: PIL RGBA image
            target_size_range: (min, max) size for the longest edge

        Returns:
            PIL.Image: Augmented RGBA image
        """
        # Convert to PIL for easier manipulation
        img = crab_rgba.copy()

        # 1. Random rotation (0-360 degrees)
        angle = random.uniform(0, 360)
        img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

        # 2. Random flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # 3. Random scale
        scale = random.uniform(0.3, 1.5)
        current_size = max(img.size)
        target_size = int(random.uniform(*target_size_range) * scale)
        scale_factor = target_size / current_size
        new_size = (int(img.width * scale_factor),
                    int(img.height * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)

        # 4. Color augmentations (only affect RGB, not alpha)
        if random.random() > 0.3:  # 70% chance
            # Convert to RGB for color operations, keep alpha separate
            rgb = img.convert('RGB')
            alpha = img.split()[3]

            # Brightness
            if random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(rgb)
                rgb = enhancer.enhance(random.uniform(0.7, 1.3))

            # Contrast
            if random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(rgb)
                rgb = enhancer.enhance(random.uniform(0.7, 1.3))

            # Saturation
            if random.random() > 0.5:
                enhancer = ImageEnhance.Color(rgb)
                rgb = enhancer.enhance(random.uniform(0.6, 1.4))

            # Hue shift (convert to HSV)
            if random.random() > 0.5:
                hsv = cv2.cvtColor(
                    np.array(rgb), cv2.COLOR_RGB2HSV).astype(np.float32)
                hue_shift = random.uniform(-30, 30)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                rgb = Image.fromarray(cv2.cvtColor(
                    hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))

            # Merge back with alpha
            img = Image.merge('RGBA', (*rgb.split(), alpha))

        # 5. Gaussian blur (on RGB only)
        if random.random() > 0.7:  # 30% chance
            rgb = img.convert('RGB')
            alpha = img.split()[3]
            blur_radius = random.uniform(0, 2)
            rgb = rgb.filter(ImageFilter.GaussianBlur(blur_radius))
            img = Image.merge('RGBA', (*rgb.split(), alpha))

        # 6. Gaussian noise
        if random.random() > 0.7:  # 30% chance
            img_array = np.array(img)
            noise = np.random.normal(0, random.uniform(
                5, 15), img_array[:, :, :3].shape)
            img_array[:, :, :3] = np.clip(
                img_array[:, :, :3] + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array, 'RGBA')

        return img


class SyntheticDatasetGenerator:
    """Generates synthetic training images with multiple crabs."""

    def __init__(self, output_dir, image_size=(640, 640)):
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

    def create_background(self):
        """
        Create a random background.
        Uses gradient patterns, noise, or solid colors.
        """
        bg = np.zeros((*self.image_size, 3), dtype=np.uint8)

        bg_type = random.choice(['gradient', 'noise', 'solid', 'texture'])

        if bg_type == 'gradient':
            # Create gradient background
            color1 = np.array([random.randint(100, 200) for _ in range(3)])
            color2 = np.array([random.randint(100, 200) for _ in range(3)])
            for i in range(self.image_size[0]):
                alpha = i / self.image_size[0]
                bg[i, :] = (1 - alpha) * color1 + alpha * color2

        elif bg_type == 'noise':
            # Noisy background
            base_color = random.randint(120, 180)
            bg = np.random.normal(base_color, 30, (*self.image_size, 3))
            bg = np.clip(bg, 0, 255).astype(np.uint8)

        elif bg_type == 'solid':
            # Solid color
            color = [random.randint(100, 200) for _ in range(3)]
            bg[:] = color

        else:  # texture
            # Create simple texture
            for i in range(0, self.image_size[0], 20):
                for j in range(0, self.image_size[1], 20):
                    color = [random.randint(100, 180) for _ in range(3)]
                    bg[i:i+20, j:j+20] = color
            # Blur to make it look more natural
            bg = cv2.GaussianBlur(bg, (15, 15), 0)

        return Image.fromarray(bg, 'RGB')

    def paste_crab_on_background(self, background, crab_rgba, position):
        """
        Paste augmented crab onto background at given position.

        Args:
            background: PIL RGB image
            crab_rgba: PIL RGBA image
            position: (x, y) top-left position

        Returns:
            PIL.Image: Background with crab pasted
            tuple: (x_min, y_min, x_max, y_max) in pixel coordinates
        """
        x, y = position
        background.paste(crab_rgba, (x, y), crab_rgba)

        # Calculate bounding box
        x_min = x
        y_min = y
        x_max = x + crab_rgba.width
        y_max = y + crab_rgba.height

        return background, (x_min, y_min, x_max, y_max)

    def check_overlap(self, new_bbox, existing_bboxes, threshold=0.3):
        """
        Check if new bbox overlaps too much with existing ones.

        Args:
            new_bbox: (x_min, y_min, x_max, y_max)
            existing_bboxes: List of existing bboxes
            threshold: Max allowed IoU

        Returns:
            bool: True if overlap is acceptable
        """
        if not existing_bboxes:
            return True

        x1_new, y1_new, x2_new, y2_new = new_bbox
        area_new = (x2_new - x1_new) * (y2_new - y1_new)

        for bbox in existing_bboxes:
            x1, y1, x2, y2 = bbox

            # Calculate intersection
            x1_i = max(x1_new, x1)
            y1_i = max(y1_new, y1)
            x2_i = min(x2_new, x2)
            y2_i = min(y2_new, y2)

            if x2_i > x1_i and y2_i > y1_i:
                intersection = (x2_i - x1_i) * (y2_i - y1_i)
                area_existing = (x2 - x1) * (y2 - y1)
                iou = intersection / (area_new + area_existing - intersection)

                if iou > threshold:
                    return False

        return True

    def bbox_to_yolo_format(self, bbox, img_width, img_height):
        """
        Convert pixel bbox to YOLO format (normalized center coordinates).

        Args:
            bbox: (x_min, y_min, x_max, y_max) in pixels
            img_width, img_height: Image dimensions

        Returns:
            tuple: (x_center, y_center, width, height) normalized to [0, 1]
        """
        x_min, y_min, x_max, y_max = bbox

        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # Clip to [0, 1]
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)

        return (x_center, y_center, width, height)

    def generate_single_image(self, crab_templates, max_crabs=8):
        """
        Generate one synthetic training image with multiple crabs.

        Args:
            crab_templates: List of tuples (class_id, segmented crab RGBA image)
            max_crabs: Maximum number of crabs per image

        Returns:
            PIL.Image: Generated image
            list: YOLO format annotations [(class_id, x_center, y_center, w, h), ...]
        """
        # Create background
        background = self.create_background()

        # Randomly decide number of crabs
        num_crabs = random.randint(1, max_crabs)

        bboxes = []
        annotations = []

        max_attempts = 50
        for _ in range(num_crabs):
            # Select random crab template with its class ID
            class_id, crab_template = random.choice(crab_templates)

            # Augment the crab
            augmented_crab = self.augmentor.augment_crab(crab_template)

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
                if self.check_overlap(potential_bbox, bboxes, threshold=0.3):
                    background, bbox = self.paste_crab_on_background(
                        background, augmented_crab, (x, y)
                    )
                    bboxes.append(bbox)

                    # Convert to YOLO format
                    yolo_bbox = self.bbox_to_yolo_format(
                        bbox, self.image_size[1], self.image_size[0]
                    )
                    # Use the actual class_id from the template
                    annotations.append((class_id, *yolo_bbox))
                    placed = True
                    break

            if not placed:
                # Could not place this crab, skip it
                continue

        return background, annotations

    def generate_dataset(self, reference_images, num_train=1000, num_val=200):
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
        for i in tqdm(range(num_train), desc="Training set"):
            image, annotations = self.generate_single_image(crab_templates)

            # Save image
            img_path = self.train_img_dir / f"train_{i:05d}.jpg"
            image.save(img_path, quality=95)

            # Save annotations
            label_path = self.train_label_dir / f"train_{i:05d}.txt"
            with open(label_path, 'w') as f:
                for ann in annotations:
                    class_id, x_center, y_center, width, height = ann
                    f.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Step 3: Generate validation images
        print(f"\n[3/4] Generating {num_val} validation images...")
        for i in tqdm(range(num_val), desc="Validation set"):
            image, annotations = self.generate_single_image(crab_templates)

            # Save image
            img_path = self.val_img_dir / f"val_{i:05d}.jpg"
            image.save(img_path, quality=95)

            # Save annotations
            label_path = self.val_label_dir / f"val_{i:05d}.txt"
            with open(label_path, 'w') as f:
                for ann in annotations:
                    class_id, x_center, y_center, width, height = ann
                    f.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Step 4: Create data.yaml
        print(f"\n[4/4] Creating dataset configuration...")
        # Extract class names from reference image filenames
        class_names = [Path(img).stem.replace('_', ' ')
                       for img in reference_images]
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(reference_images),
            'names': class_names
        }

        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"  ✓ Saved configuration to {yaml_path}")

        print("\n" + "="*60)
        print("DATASET GENERATION COMPLETE!")
        print("="*60)
        print(f"Training images:   {num_train}")
        print(f"Validation images: {num_val}")
        print(f"Output directory:  {self.output_dir}")
        print("="*60)


def main():
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
    # Using 1000 training + 200 validation images
    # This provides enough diversity while being computationally feasible
    generator.generate_dataset(
        reference_images=reference_images,
        num_train=1000,
        num_val=200
    )


if __name__ == '__main__':
    main()
