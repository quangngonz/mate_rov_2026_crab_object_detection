"""
Crab Augmentation Module
========================

Applies extensive augmentations to segmented crab images.
"""

from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import random
from typing import Tuple

from config.constants import (
    ROTATION_RANGE,
    SCALE_RANGE,
    FLIP_PROBABILITY,
    COLOR_AUG_PROBABILITY,
    BLUR_PROBABILITY,
    NOISE_PROBABILITY,
    BRIGHTNESS_RANGE,
    CONTRAST_RANGE,
    SATURATION_RANGE,
    HUE_SHIFT_RANGE,
    BLUR_RADIUS_RANGE,
    NOISE_SIGMA_RANGE,
)


class CrabAugmentor:
    """Applies augmentations to segmented crab images."""

    def augment_crab(
        self,
        crab_rgba: Image.Image,
        target_size_range: Tuple[int, int] = (50, 400)
    ) -> Image.Image:
        """
        Apply random augmentations to a crab image.

        Args:
            crab_rgba: PIL RGBA image
            target_size_range: (min, max) size for the longest edge

        Returns:
            Augmented RGBA PIL Image
        """
        img = crab_rgba.copy()

        # 1. Random rotation (0-360 degrees)
        angle = random.uniform(*ROTATION_RANGE)
        img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

        # 2. Random flip
        if random.random() > FLIP_PROBABILITY:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > FLIP_PROBABILITY:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # 3. Random scale
        scale = random.uniform(*SCALE_RANGE)
        current_size = max(img.size)
        target_size = int(random.uniform(*target_size_range) * scale)
        scale_factor = target_size / current_size
        new_size = (int(img.width * scale_factor),
                    int(img.height * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)

        # 4. Color augmentations (only affect RGB, not alpha)
        if random.random() < COLOR_AUG_PROBABILITY:
            img = self._apply_color_augmentations(img)

        # 5. Gaussian blur (on RGB only)
        if random.random() < BLUR_PROBABILITY:
            img = self._apply_blur(img)

        # 6. Gaussian noise
        if random.random() < NOISE_PROBABILITY:
            img = self._apply_noise(img)

        return img

    def _apply_color_augmentations(self, img: Image.Image) -> Image.Image:
        """Apply color-based augmentations to image."""
        # Convert to RGB for color operations, keep alpha separate
        rgb = img.convert('RGB')
        alpha = img.split()[3]

        # Brightness
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(rgb)
            rgb = enhancer.enhance(random.uniform(*BRIGHTNESS_RANGE))

        # Contrast
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(rgb)
            rgb = enhancer.enhance(random.uniform(*CONTRAST_RANGE))

        # Saturation
        if random.random() > 0.5:
            enhancer = ImageEnhance.Color(rgb)
            rgb = enhancer.enhance(random.uniform(*SATURATION_RANGE))

        # Hue shift (convert to HSV)
        if random.random() > 0.5:
            hsv = cv2.cvtColor(
                np.array(rgb), cv2.COLOR_RGB2HSV).astype(np.float32)
            hue_shift = random.uniform(*HUE_SHIFT_RANGE)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            rgb = Image.fromarray(cv2.cvtColor(
                hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))

        # Merge back with alpha
        return Image.merge('RGBA', (*rgb.split(), alpha))

    def _apply_blur(self, img: Image.Image) -> Image.Image:
        """Apply Gaussian blur to image."""
        rgb = img.convert('RGB')
        alpha = img.split()[3]
        blur_radius = random.uniform(*BLUR_RADIUS_RANGE)
        rgb = rgb.filter(ImageFilter.GaussianBlur(blur_radius))
        return Image.merge('RGBA', (*rgb.split(), alpha))

    def _apply_noise(self, img: Image.Image) -> Image.Image:
        """Apply Gaussian noise to image."""
        img_array = np.array(img)
        noise = np.random.normal(0, random.uniform(
            *NOISE_SIGMA_RANGE), img_array[:, :, :3].shape)
        img_array[:, :, :3] = np.clip(
            img_array[:, :, :3] + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array, 'RGBA')
