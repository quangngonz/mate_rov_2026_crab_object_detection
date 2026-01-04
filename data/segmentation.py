"""
Crab Segmentation Module
========================

Handles segmentation of crabs from reference images using background removal.
"""

from PIL import Image
import numpy as np
from rembg import remove
from pathlib import Path
from typing import Optional, Tuple


class CrabSegmentor:
    """Segments crabs from reference images using background removal."""

    def __init__(self):
        """Initialize the crab segmentor."""
        pass

    def segment_crab(self, image_path: str) -> Image.Image:
        """
        Remove background and return crab with alpha channel.

        Args:
            image_path: Path to input image

        Returns:
            PIL.Image: RGBA image with transparent background

        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        input_image = Image.open(image_path)
        output_image = remove(input_image)
        return output_image

    def get_bounding_box(self, rgba_image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Get tight bounding box around non-transparent pixels.

        Args:
            rgba_image: PIL Image in RGBA mode

        Returns:
            Bounding box as (x_min, y_min, x_max, y_max) or None if all transparent
        """
        alpha = np.array(rgba_image)[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)

        if not rows.any() or not cols.any():
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (x_min, y_min, x_max, y_max)

    def crop_to_content(self, rgba_image: Image.Image, padding: int = 10) -> Image.Image:
        """
        Crop image to content bounding box with padding.

        Args:
            rgba_image: PIL Image in RGBA mode
            padding: Padding to add around content in pixels

        Returns:
            Cropped PIL Image
        """
        bbox = self.get_bounding_box(rgba_image)
        if bbox is None:
            return rgba_image

        x_min, y_min, x_max, y_max = bbox
        width, height = rgba_image.size

        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        return rgba_image.crop((x_min, y_min, x_max, y_max))
