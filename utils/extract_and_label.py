"""
Video Frame Extraction and Auto-Labeling
========================================

Extract frames from video and automatically generate labels using trained model.
"""

import cv2
import argparse
from pathlib import Path
from models import CrabDetector
import numpy as np
from typing import List, Tuple


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0
) -> List[Path]:
    """
    Extract frames from video at specified rate.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (e.g., 1.0 = 1 frame per second)

    Returns:
        List of paths to extracted frames
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(video_fps / fps)

    print(f"Video FPS: {video_fps:.2f}")
    print(f"Total frames in video: {total_frames}")
    print(f"Extracting 1 frame every {frame_interval} frames ({fps} fps)")

    saved_frames = []
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Save frame
            frame_filename = output_dir / f"frame_{extracted_count:05d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_frames.append(frame_filename)
            extracted_count += 1

            if extracted_count % 10 == 0:
                print(f"Extracted {extracted_count} frames...", end='\r')

        frame_count += 1

    cap.release()

    print(f"\n✓ Extracted {extracted_count} frames to {output_dir}")
    return saved_frames


def convert_yolo_bbox(detection, img_width: int, img_height: int) -> Tuple[int, float, float, float, float]:
    """
    Convert detection to YOLO format.

    Args:
        detection: Detection from model (x1, y1, x2, y2, conf, cls)
        img_width: Image width
        img_height: Image height

    Returns:
        Tuple of (class_id, x_center, y_center, width, height) in normalized coordinates
    """
    x1, y1, x2, y2 = detection[:4]
    class_id = int(detection[5])

    # Convert to center, width, height
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return class_id, x_center, y_center, width, height


def auto_label_frames(
    frame_paths: List[Path],
    model_path: str,
    labels_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    keep_only_detections: bool = True
) -> int:
    """
    Run model on frames and generate YOLO format labels.

    Args:
        frame_paths: List of paths to frame images
        model_path: Path to trained model
        labels_dir: Directory to save label files
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        keep_only_detections: If True, delete frames with no detections

    Returns:
        Number of labeled frames kept
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model from {model_path}...")
    detector = CrabDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

    print(f"Running inference on {len(frame_paths)} frames...")

    labeled_count = 0
    frames_with_detections = 0
    total_detections = 0
    frames_deleted = 0

    for i, frame_path in enumerate(frame_paths):
        # Read image to get dimensions
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"Warning: Could not read {frame_path}")
            continue

        img_height, img_width = img.shape[:2]

        # Run detection
        result = detector.predict(str(frame_path))

        # Check if we have detections
        has_detections = result['num_detections'] > 0

        if has_detections:
            # Create label file
            label_path = labels_dir / f"{frame_path.stem}.txt"

            with open(label_path, 'w') as f:
                frames_with_detections += 1
                total_detections += result['num_detections']

                # Combine detection data for convert_yolo_bbox
                for j in range(result['num_detections']):
                    box = result['boxes'][j]
                    conf = result['confidences'][j]
                    cls = result['class_ids'][j]

                    # Create detection array in expected format: [x1, y1, x2, y2, conf, cls]
                    detection = np.array(
                        [box[0], box[1], box[2], box[3], conf, cls])

                    class_id, x_center, y_center, width, height = convert_yolo_bbox(
                        detection, img_width, img_height
                    )
                    f.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            labeled_count += 1
        else:
            # No detections - delete frame if requested
            if keep_only_detections:
                frame_path.unlink()
                frames_deleted += 1

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frame_paths)} frames...", end='\r')

    print(f"\n✓ Processed {len(frame_paths)} frames")
    print(f"  - Frames with detections (kept): {frames_with_detections}")
    print(f"  - Frames without detections (deleted): {frames_deleted}")
    print(f"  - Total detections: {total_detections}")
    if labeled_count > 0:
        print(
            f"  - Avg detections per frame: {total_detections/labeled_count:.2f}")
    print(f"  - Labels saved to: {labels_dir}")

    return labeled_count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video and auto-label with trained model"
    )
    parser.add_argument(
        'video_path',
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1.0)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset/review',
        help="Output directory for frames and labels (default: dataset/review)"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='weights/best.pt',
        help="Path to trained model (default: weights/best.pt)"
    )
    parser.add_argument(
        '--conf_threshold',
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.25)"
    )
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    parser.add_argument(
        '--keep_all',
        action='store_true',
        help="Keep all frames even without detections (default: only keep frames with detections)"
    )

    args = parser.parse_args()

    print("="*80)
    print("VIDEO FRAME EXTRACTION & AUTO-LABELING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Video: {args.video_path}")
    print(f"  Extraction rate: {args.fps} fps")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model: {args.model_path}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print()

    # Create output directories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'

    # Extract frames
    print("Step 1: Extracting frames from video...")
    print("-" * 80)
    frame_paths = extract_frames(
        video_path=args.video_path,
        output_dir=images_dir,
        fps=args.fps
    )

    if len(frame_paths) == 0:
        print("❌ No frames extracted. Exiting.")
        return

    # Auto-label frames
    print("\nStep 2: Auto-labeling frames with model...")
    print("-" * 80)
    labeled_count = auto_label_frames(
        frame_paths=frame_paths,
        model_path=args.model_path,
        labels_dir=labels_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        keep_only_detections=not args.keep_all
    )

    print("\n" + "="*80)
    print("✓ PROCESSING COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Review and correct labels using: python review_and_train.py")
    print(f"  2. Images are in: {images_dir}")
    print(f"  3. Labels are in: {labels_dir}")


if __name__ == '__main__':
    main()
