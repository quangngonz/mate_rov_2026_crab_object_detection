"""
Image Processing Utilities
==========================

Common image manipulation functions.
"""

import cv2
import numpy as np
from PIL import Image
import random
from typing import Tuple


def create_background(
    image_size: Tuple[int, int] = (640, 640),
    bg_type: str = None
) -> Image.Image:
    """
    Create a random background image.

    Args:
        image_size: Size of background image (width, height)
        bg_type: Type of background ('gradient', 'noise', 'solid', 'texture')
                 If None, randomly selects a type

    Returns:
        PIL Image with generated background
    """
    if bg_type is None:
        bg_type = random.choice(['gradient', 'noise', 'solid', 'texture'])

    bg = np.zeros((*image_size, 3), dtype=np.uint8)

    if bg_type == 'gradient':
        # Create gradient background
        color1 = np.array([random.randint(100, 200) for _ in range(3)])
        color2 = np.array([random.randint(100, 200) for _ in range(3)])
        for i in range(image_size[0]):
            alpha = i / image_size[0]
            bg[i, :] = (1 - alpha) * color1 + alpha * color2

    elif bg_type == 'noise':
        # Noisy background
        base_color = random.randint(120, 180)
        bg = np.random.normal(base_color, 30, (*image_size, 3))
        bg = np.clip(bg, 0, 255).astype(np.uint8)

    elif bg_type == 'solid':
        # Solid color
        color = [random.randint(100, 200) for _ in range(3)]
        bg[:] = color

    else:  # texture
        # Create simple texture
        for i in range(0, image_size[0], 20):
            for j in range(0, image_size[1], 20):
                color = [random.randint(100, 180) for _ in range(3)]
                bg[i:i+20, j:j+20] = color
        # Blur to make it look more natural
        bg = cv2.GaussianBlur(bg, (15, 15), 0)

    return Image.fromarray(bg, 'RGB')


def paste_image_with_alpha(
    background: Image.Image,
    foreground: Image.Image,
    position: Tuple[int, int]
) -> Image.Image:
    """
    Paste an RGBA image onto a background using its alpha channel.

    Args:
        background: Background PIL Image (RGB or RGBA)
        foreground: Foreground PIL Image (must have alpha channel)
        position: Position to paste (x, y)

    Returns:
        Combined PIL Image
    """
    if foreground.mode != 'RGBA':
        raise ValueError("Foreground image must be in RGBA mode")

    background_copy = background.copy()
    background_copy.paste(foreground, position, foreground)
    return background_copy


def get_tight_bbox(rgba_image: Image.Image) -> Tuple[int, int, int, int]:
    """
    Get tight bounding box around non-transparent pixels in RGBA image.

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
