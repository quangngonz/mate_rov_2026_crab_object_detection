"""
Inference Script
===============

CLI wrapper for running batch inference on test images.
"""

from utils.stats import print_detection_summary
from models import CrabDetector, InferenceEngine
from pathlib import Path


def main():
    """Main inference function."""

    # Configuration
    model_path = 'weights/best.pt'
    test_dir = 'test_images'
    output_dir = 'detections'
    conf_threshold = 0.8
    iou_threshold = 0.3

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
        print("  1. Run: python scripts/generate_dataset.py")
        print("  2. Run: python scripts/train.py")
        print("  3. Then run this script again")
        return

    # Check if test directory exists
    if not Path(test_dir).exists():
        print(f"\n❌ Error: Test directory not found: {test_dir}")
        return

    # Initialize detector and engine
    try:
        detector = CrabDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        print(f"\n✓ Model loaded successfully")
        print(f"✓ Using device: {detector.device}")

        engine = InferenceEngine(detector)

        # Run batch inference
        stats = engine.process_directory(test_dir, output_dir)

        if 'error' in stats:
            print(f"\n❌ {stats['error']}")
            return

        # Print summary
        print_detection_summary(stats)

        print(f"\n✅ Inference complete!")
        print(f"📁 Annotated images saved to: {output_dir}/")

    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
