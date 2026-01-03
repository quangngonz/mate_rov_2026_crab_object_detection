"""
Inference Script for Crab Detection
===================================

This script loads a trained YOLOv8 model and runs inference on test images,
drawing bounding boxes around detected crabs.

Features:
- Loads the best trained model
- Runs detection on all test images
- Draws bounding boxes with confidence scores
- Saves annotated images to output directory
- Displays detection statistics
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch


class CrabDetector:
    """Wrapper class for crab detection inference."""

    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the crab detector.

        Args:
            model_path: Path to trained model weights (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("✓ Model loaded successfully")

        # Check device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def detect(self, image_path, save_path=None, show_conf=True, line_width=2):
        """
        Run detection on a single image.

        Args:
            image_path: Path to input image
            save_path: Path to save annotated image (optional)
            show_conf: Whether to show confidence scores
            line_width: Thickness of bounding box lines

        Returns:
            dict: Detection results with 'num_detections', 'image', 'boxes', 'confidences'
        """
        image_path = Path(image_path)

        # Run inference
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]

        # Get detection info
        boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(
            int) if len(results.boxes) > 0 else np.array([])
        num_detections = len(boxes)

        # Load original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes
        annotated_image = self.draw_boxes(
            image.copy(),
            boxes,
            confidences,
            class_ids,
            show_conf=show_conf,
            line_width=line_width
        )

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_pil = Image.fromarray(annotated_image)
            annotated_pil.save(save_path, quality=95)

        return {
            'num_detections': num_detections,
            'image': annotated_image,
            'boxes': boxes,
            'confidences': confidences,
            'class_ids': class_ids
        }

    def draw_boxes(self, image, boxes, confidences, class_ids, show_conf=True, line_width=2):
        """
        Draw bounding boxes on image.

        Args:
            image: RGB numpy array
            boxes: Array of bounding boxes [x1, y1, x2, y2]
            confidences: Array of confidence scores
            class_ids: Array of class IDs
            show_conf: Whether to show confidence scores
            line_width: Thickness of box lines

        Returns:
            numpy.ndarray: Image with bounding boxes drawn
        """
        # Colors for each crab class (RGB)
        class_colors = [
            (0, 255, 0),    # Class 0: Green for European Green Crab
            (255, 165, 0),  # Class 1: Orange for Jonah Crab
            (255, 0, 255)   # Class 2: Magenta for Native Rock Crab
        ]

        # Get class names from model
        class_names = self.model.names

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box.astype(int)

            # Get color and name for this class
            color = class_colors[cls_id] if cls_id < len(
                class_colors) else (0, 255, 0)
            class_name = class_names[
                cls_id] if cls_id in class_names else f'class_{cls_id}'

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)

            # Draw label with confidence
            if show_conf:
                label = f'{class_name} {conf:.2f}'

                # Get text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw background rectangle for text
                cv2.rectangle(
                    image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1  # Filled
                )

                # Draw text
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    font,
                    font_scale,
                    (0, 0, 0),  # Black text
                    thickness,
                    cv2.LINE_AA
                )

        return image

    def detect_batch(self, image_dir, output_dir, image_extensions=None):
        """
        Run detection on all images in a directory.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save annotated images
            image_extensions: List of valid image extensions (default: common formats)

        Returns:
            dict: Statistics about detection results
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))

        image_files = sorted(set(image_files))

        if not image_files:
            print(f"No images found in {image_dir}")
            return {}

        print(f"\nProcessing {len(image_files)} images...")
        print("="*80)

        results_summary = {
            'total_images': len(image_files),
            'total_detections': 0,
            'images_with_detections': 0,
            'per_image_results': []
        }

        for img_path in image_files:
            print(f"\nProcessing: {img_path.name}")

            # Run detection
            output_path = output_dir / f"detected_{img_path.name}"
            result = self.detect(img_path, save_path=output_path)

            # Update statistics
            num_crabs = result['num_detections']
            results_summary['total_detections'] += num_crabs
            if num_crabs > 0:
                results_summary['images_with_detections'] += 1

            results_summary['per_image_results'].append({
                'filename': img_path.name,
                'num_detections': num_crabs,
                'confidences': result['confidences'].tolist()
            })

            print(f"  ✓ Detected {num_crabs} crab(s)")
            if num_crabs > 0:
                avg_conf = result['confidences'].mean()
                print(f"  ✓ Average confidence: {avg_conf:.3f}")
            print(f"  ✓ Saved to: {output_path}")

        return results_summary


def print_summary(stats):
    """Print detection statistics summary."""
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Total crabs detected:   {stats['total_detections']}")

    if stats['total_images'] > 0:
        avg_crabs = stats['total_detections'] / stats['total_images']
        print(f"Average crabs per image: {avg_crabs:.2f}")

    print("\nPer-image results:")
    print("-" * 80)
    for result in stats['per_image_results']:
        filename = result['filename']
        num_crabs = result['num_detections']
        confidences = result['confidences']

        print(f"  {filename:30s} → {num_crabs} crab(s)", end="")
        if confidences:
            avg_conf = np.mean(confidences)
            max_conf = np.max(confidences)
            min_conf = np.min(confidences)
            print(
                f" (conf: avg={avg_conf:.3f}, max={max_conf:.3f}, min={min_conf:.3f})")
        else:
            print()

    print("="*80)


def main():
    """
    Main inference function.

    Default configuration:
    - Model: runs/detect/crab_detector/weights/best.pt (best trained model)
    - Input: test_images/ directory
    - Output: detections/ directory
    - Confidence threshold: 0.25 (adjustable)
    - IoU threshold: 0.45 (for NMS)
    """

    # Configuration
    model_path = 'runs/detect/crab_detector/weights/best.pt'
    test_dir = 'test_images'
    output_dir = 'detections'
    conf_threshold = 0.8  # Lower for more detections, higher for more precision
    iou_threshold = 0.3   # NMS threshold

    print("="*80)
    print("CRAB DETECTION INFERENCE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Test images: {test_dir}/")
    print(f"  Output: {output_dir}/")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  IoU threshold: {iou_threshold}")

    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("\nPlease train the model first:")
        print("  1. Run: python generate_dataset.py")
        print("  2. Run: python train.py")
        print("  3. Then run this script again")
        return

    # Check if test directory exists
    if not Path(test_dir).exists():
        print(f"\n❌ Error: Test directory not found: {test_dir}")
        return

    # Initialize detector
    try:
        detector = CrabDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
    except Exception as e:
        print(f"\n❌ Error initializing detector: {e}")
        return

    # Run batch detection
    try:
        stats = detector.detect_batch(test_dir, output_dir)

        # Print summary
        print_summary(stats)

        print(f"\n✅ Inference complete!")
        print(f"📁 Annotated images saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
