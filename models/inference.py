"""
Inference Engine
===============

High-level interface for running batch inference and saving results.
"""

import cv2
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
import numpy as np

from .detector import CrabDetector
from utils.drawing import draw_bounding_boxes
from utils.stats import calculate_detection_stats
from config.constants import CLASS_COLORS


class InferenceEngine:
    """High-level interface for running inference."""

    def __init__(self, detector: CrabDetector):
        """
        Initialize inference engine.

        Args:
            detector: Initialized CrabDetector instance
        """
        self.detector = detector

    def process_image(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        show_conf: bool = True,
        line_width: int = 2
    ) -> Dict[str, any]:
        """
        Process a single image.

        Args:
            image_path: Path to input image
            save_path: Path to save annotated image (optional)
            show_conf: Whether to show confidence scores
            line_width: Line width for bounding boxes

        Returns:
            Dictionary with detection results and annotated image
        """
        # Run detection
        result = self.detector.predict(image_path)

        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw boxes
        annotated = draw_bounding_boxes(
            image,
            result['boxes'],
            result['confidences'],
            result['class_ids'],
            self.detector.class_names,
            CLASS_COLORS,
            show_conf=show_conf,
            line_width=line_width
        )

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(annotated).save(save_path, quality=95)

        result['image'] = annotated
        result['filename'] = Path(image_path).name

        return result

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        image_extensions: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save annotated images
            image_extensions: Valid image extensions (default: common formats)

        Returns:
            Dictionary with batch statistics
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))

        image_files = sorted(set(image_files))

        if not image_files:
            return {'error': f'No images found in {input_dir}'}

        print(f"Processing {len(image_files)} images...")

        results = []
        for img_path in image_files:
            output_path = output_dir / f"detected_{img_path.name}"
            result = self.process_image(str(img_path), str(output_path))

            # Store only essential info for stats
            results.append({
                'filename': result['filename'],
                'num_detections': result['num_detections'],
                'confidences': result['confidences'].tolist(),
                'class_ids': result['class_ids'].tolist()
            })

            print(
                f"  ✓ {img_path.name}: {result['num_detections']} detection(s)")

        # Calculate statistics
        class_names = list(self.detector.class_names.values())
        stats = calculate_detection_stats(results, class_names)

        return stats
