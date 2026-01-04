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
    MOTION_BLUR_PROBABILITY,
    PERSPECTIVE_PROBABILITY,
    NOISE_PROBABILITY,
    BRIGHTNESS_RANGE,
    CONTRAST_RANGE,
    SATURATION_RANGE,
    HUE_SHIFT_RANGE,
    BLUR_RADIUS_RANGE,
    MOTION_BLUR_KERNEL_RANGE,
    PERSPECTIVE_SCALE_RANGE,
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

        # 4. Perspective transform
        if random.random() < PERSPECTIVE_PROBABILITY:
            img = self._apply_perspective(img)

        # 5. Color augmentations (only affect RGB, not alpha)
        if random.random() < COLOR_AUG_PROBABILITY:
            img = self._apply_color_augmentations(img)

        # 6. Gaussian blur (on RGB only)
        if random.random() < BLUR_PROBABILITY:
            img = self._apply_blur(img)

        # 7. Motion blur
        if random.random() < MOTION_BLUR_PROBABILITY:
            img = self._apply_motion_blur(img)

        # 8. Gaussian noise
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

    def _apply_motion_blur(self, img: Image.Image) -> Image.Image:
        """Apply motion blur to simulate camera/object movement."""
        rgb = img.convert('RGB')
        alpha = img.split()[3]

        # Random kernel size (odd numbers only)
        kernel_size = random.choice(range(*MOTION_BLUR_KERNEL_RANGE, 2))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Random angle for motion direction
        angle = random.uniform(0, 360)

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Rotate kernel to random angle
        M = cv2.getRotationMatrix2D(
            (kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

        # Apply to RGB channels
        rgb_array = np.array(rgb)
        rgb_array = cv2.filter2D(rgb_array, -1, kernel)
        rgb = Image.fromarray(rgb_array)

        return Image.merge('RGBA', (*rgb.split(), alpha))

    def _apply_perspective(self, img: Image.Image) -> Image.Image:
        """Apply perspective transform to simulate ROV approach angles."""
        rgb = img.convert('RGB')
        alpha_array = np.array(img.split()[3])
        rgb_array = np.array(rgb)

        h, w = rgb_array.shape[:2]

        # Source points (original corners)
        src_pts = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ])

        # Choose random viewing angle type
        perspective_type = random.choice([
            'top_closer',     # ROV looking down, top edge closer
            'bottom_closer',  # Looking up at crab
            'left_closer',    # Approaching from left
            'right_closer',   # Approaching from right
            'tilt_left',      # Camera tilted left
            'tilt_right'      # Camera tilted right
        ])

        max_offset = int(min(w, h) * random.uniform(*PERSPECTIVE_SCALE_RANGE))

        # Create trapezoid distortions based on ROV viewing angle
        if perspective_type == 'top_closer':
            # Top edge is closer (wider), bottom edge farther (narrower)
            dst_pts = np.float32([
                [-max_offset, max_offset],
                [w - 1 + max_offset, max_offset],
                [w - 1 - max_offset//2, h - 1],
                [max_offset//2, h - 1]
            ])
        elif perspective_type == 'bottom_closer':
            # Bottom edge is closer (wider)
            dst_pts = np.float32([
                [max_offset//2, 0],
                [w - 1 - max_offset//2, 0],
                [w - 1 + max_offset, h - 1 - max_offset],
                [-max_offset, h - 1 - max_offset]
            ])
        elif perspective_type == 'left_closer':
            # Left edge is closer (taller)
            dst_pts = np.float32([
                [max_offset, -max_offset],
                [w - 1, max_offset//2],
                [w - 1, h - 1 - max_offset//2],
                [max_offset, h - 1 + max_offset]
            ])
        elif perspective_type == 'right_closer':
            # Right edge is closer (taller)
            dst_pts = np.float32([
                [0, max_offset//2],
                [w - 1 - max_offset, -max_offset],
                [w - 1 - max_offset, h - 1 + max_offset],
                [0, h - 1 - max_offset//2]
            ])
        elif perspective_type == 'tilt_left':
            # Camera tilted counterclockwise
            dst_pts = np.float32([
                [max_offset//2, -max_offset//2],
                [w - 1 + max_offset//2, max_offset//2],
                [w - 1 - max_offset//2, h - 1 + max_offset//2],
                [-max_offset//2, h - 1 - max_offset//2]
            ])
        else:  # tilt_right
            # Camera tilted clockwise
            dst_pts = np.float32([
                [-max_offset//2, max_offset//2],
                [w - 1 - max_offset//2, -max_offset//2],
                [w - 1 + max_offset//2, h - 1 - max_offset//2],
                [max_offset//2, h - 1 + max_offset//2]
            ])

        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply to both RGB and alpha
        rgb_array = cv2.warpPerspective(rgb_array, matrix, (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
        alpha_array = cv2.warpPerspective(alpha_array, matrix, (w, h),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=0)

        rgb = Image.fromarray(rgb_array)
        alpha = Image.fromarray(alpha_array)

        return Image.merge('RGBA', (*rgb.split(), alpha))
