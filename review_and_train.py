"""
Label Review and Training UI
============================

Interactive UI for reviewing and correcting auto-generated labels, then training.
"""

import cv2
import argparse
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional, Dict
import yaml
import copy

from config.constants import CRAB_CLASSES, CLASS_COLORS
from models import CrabTrainer


class BoundingBox:
    """Represents a bounding box with class label."""

    def __init__(self, class_id: int, x_center: float, y_center: float, width: float, height: float):
        """
        Initialize bounding box in YOLO format (normalized coordinates).

        Args:
            class_id: Class index
            x_center: X center (0-1)
            y_center: Y center (0-1)
            width: Width (0-1)
            height: Height (0-1)
        """
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def to_pixel_coords(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates (x1, y1, x2, y2)."""
        x_center_px = self.x_center * img_width
        y_center_px = self.y_center * img_height
        width_px = self.width * img_width
        height_px = self.height * img_height

        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)

        return x1, y1, x2, y2

    @staticmethod
    def from_pixel_coords(class_id: int, x1: int, y1: int, x2: int, y2: int,
                          img_width: int, img_height: int) -> 'BoundingBox':
        """Create from pixel coordinates."""
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        return BoundingBox(class_id, x_center, y_center, width, height)

    def to_yolo_string(self) -> str:
        """Convert to YOLO format string."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class LabelReviewUI:
    """Interactive UI for reviewing and correcting labels."""

    def __init__(self, images_dir: str, labels_dir: str):
        """
        Initialize the label review UI.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing label files
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.progress_file = self.labels_dir / '.review_progress.txt'

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}")

        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Load all images
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) +
                                  list(self.images_dir.glob('*.png')))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")

        self.current_index = 0
        self.current_image = None
        self.current_boxes: List[BoundingBox] = []
        self.display_image = None

        # Load last progress
        self.current_index = self.load_progress()

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_class = 0
        self.selected_box_index = None

        # UI state
        self.show_help = True
        self.modified = False

        # Undo functionality
        self.history: List[List[BoundingBox]] = []
        self.max_history = 20

        print(f"Loaded {len(self.image_files)} images from {images_dir}")

    def load_labels(self, image_path: Path) -> List[BoundingBox]:
        """Load labels for an image."""
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            boxes.append(BoundingBox(
                                class_id, x_center, y_center, width, height))

        return boxes

    def save_labels(self, image_path: Path, boxes: List[BoundingBox]):
        """Save labels for an image."""
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(box.to_yolo_string() + '\n')

    def autosave(self):
        """Automatically save current labels."""
        if self.modified:
            self.save_labels(
                self.image_files[self.current_index], self.current_boxes)
            self.modified = False

    def save_to_history(self):
        """Save current state to undo history."""
        self.history.append(copy.deepcopy(self.current_boxes))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo(self):
        """Undo last change."""
        if len(self.history) > 0:
            self.current_boxes = self.history.pop()
            self.modified = True
            self.autosave()
            self.selected_box_index = None
            return True
        return False

    def save_progress(self):
        """Save current image index for resuming later."""
        try:
            with open(self.progress_file, 'w') as f:
                f.write(str(self.current_index))
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")

    def load_progress(self) -> int:
        """Load last reviewed image index."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    saved_index = int(f.read().strip())
                    if 0 <= saved_index < len(self.image_files):
                        print(
                            f"Resuming from image {saved_index + 1}/{len(self.image_files)}")
                        return saved_index
            except Exception as e:
                print(f"Warning: Could not load progress: {e}")
        return 0

    def load_image(self, index: int):
        """Load image and its labels."""
        self.current_index = index % len(self.image_files)
        image_path = self.image_files[self.current_index]

        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.current_boxes = self.load_labels(image_path)
        self.modified = False
        self.selected_box_index = None

        # Clear history when loading new image
        self.history.clear()

        # Save progress
        self.save_progress()

    def draw_boxes(self):
        """Draw bounding boxes on the image."""
        img_height, img_width = self.current_image.shape[:2]

        # Fixed UI header height just for status bar
        ui_height = 40

        # Create canvas with extra space at top for status bar only
        canvas_height = img_height + ui_height
        self.display_image = np.zeros(
            (canvas_height, img_width, 3), dtype=np.uint8)
        self.display_image.fill(40)  # Dark gray background

        # Place original image below the status bar
        self.display_image[ui_height:, :] = self.current_image.copy()

        for i, box in enumerate(self.current_boxes):
            x1, y1, x2, y2 = box.to_pixel_coords(img_width, img_height)

            # Offset y coordinates for the status bar
            y1_offset = y1 + ui_height
            y2_offset = y2 + ui_height

            # Choose color
            color = CLASS_COLORS[box.class_id]
            thickness = 3 if i == self.selected_box_index else 2

            # Draw box (with offset)
            cv2.rectangle(self.display_image, (x1, y1_offset),
                          (x2, y2_offset), color, thickness)

            # Draw label (with offset)
            label = CRAB_CLASSES[box.class_id]
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(self.display_image, (x1, y1_offset - label_size[1] - 8),
                          (x1 + label_size[0] + 4, y1_offset), color, -1)
            cv2.putText(self.display_image, label, (x1 + 2, y1_offset - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_ui(self):
        """Draw UI elements - status bar at top, help text overlays image."""
        canvas_height, img_width = self.display_image.shape[:2]

        # Status bar at the very top (not overlaying)
        status = f"Image {self.current_index + 1}/{len(self.image_files)} | "
        status += f"Boxes: {len(self.current_boxes)}"
        if self.selected_box_index is not None:
            status += f" (#{self.selected_box_index + 1} selected)"
        status += f" | Class: {CRAB_CLASSES[self.current_class]} | "
        status += "Modified" if self.modified else "Saved"

        cv2.rectangle(self.display_image, (0, 0),
                      (img_width, 35), (50, 50, 50), -1)
        cv2.putText(self.display_image, status, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Help text (overlays the image - starts at y=40 where image begins)
        if self.show_help:
            help_lines = [
                "CONTROLS:",
                "  Left/Right Arrow: Navigate images",
                "  Up/Down Arrow: Cycle through boxes",
                "  Mouse: Click & drag to draw box",
                "  Click on box: Select it",
                "  1/2/3: Set/change class of selected box",
                "  D: Delete selected box",
                "  Backspace: Clear all boxes",
                "  Ctrl+Z: Undo last change",
                "  H: Toggle help | T: Train | Q/ESC: Quit"
            ]

            y_offset = 50  # Start below status bar but overlay the image
            for line in help_lines:
                cv2.rectangle(self.display_image, (0, y_offset),
                              (500, y_offset + 25), (50, 50, 50), -1)
                cv2.putText(self.display_image, line, (10, y_offset + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if self.current_image is None:
            return

        img_height, img_width = self.current_image.shape[:2]
        ui_height = 40  # Fixed status bar height

        # Adjust y coordinate to account for UI area
        y_adjusted = y - ui_height

        # Ignore clicks in the UI area
        if y < ui_height:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on existing box (using adjusted coordinates)
            for i, box in enumerate(self.current_boxes):
                x1, y1, x2, y2 = box.to_pixel_coords(img_width, img_height)
                if x1 <= x <= x2 and y1 <= y_adjusted <= y2:
                    self.selected_box_index = i
                    return

            # Start drawing new box (store adjusted coordinates)
            self.drawing = True
            self.start_point = (x, y_adjusted)
            self.selected_box_index = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                # Draw temporary box (with offset for display)
                temp_image = self.display_image.copy()
                start_display = (
                    self.start_point[0], self.start_point[1] + ui_height)
                end_display = (x, y)
                cv2.rectangle(temp_image, start_display, end_display,
                              CLASS_COLORS[self.current_class], 2)
                cv2.imshow('Label Review', temp_image)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.drawing = False

                # Create bounding box (using adjusted coordinates)
                x1 = min(self.start_point[0], x)
                y1 = min(self.start_point[1], y_adjusted)
                x2 = max(self.start_point[0], x)
                y2 = max(self.start_point[1], y_adjusted)

                # Only add if box has valid size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.save_to_history()
                    new_box = BoundingBox.from_pixel_coords(
                        self.current_class, x1, y1, x2, y2, img_width, img_height
                    )
                    self.current_boxes.append(new_box)
                    self.modified = True
                    self.autosave()

                self.start_point = None

    def run(self):
        """Run the UI main loop."""
        cv2.namedWindow('Label Review')
        cv2.setMouseCallback('Label Review', self.mouse_callback)

        self.load_image(self.current_index)

        while True:
            self.draw_boxes()
            self.draw_ui()
            cv2.imshow('Label Review', self.display_image)

            key = cv2.waitKey(1) & 0xFF

            # Image navigation
            if key == 81 or key == 2:  # Left arrow
                if self.modified:
                    self.save_labels(
                        self.image_files[self.current_index], self.current_boxes)
                self.load_image(self.current_index - 1)

            elif key == 83 or key == 3:  # Right arrow
                if self.modified:
                    self.save_labels(
                        self.image_files[self.current_index], self.current_boxes)
                self.load_image(self.current_index + 1)

            # Box selection cycling
            elif key == 82 or key == 0:  # Up arrow
                if len(self.current_boxes) > 0:
                    if self.selected_box_index is None:
                        self.selected_box_index = 0
                    else:
                        self.selected_box_index = (
                            self.selected_box_index - 1) % len(self.current_boxes)

            elif key == 84 or key == 1:  # Down arrow
                if len(self.current_boxes) > 0:
                    if self.selected_box_index is None:
                        self.selected_box_index = 0
                    else:
                        self.selected_box_index = (
                            self.selected_box_index + 1) % len(self.current_boxes)

            # Class selection (or change class of selected box)
            elif key == ord('1'):
                if self.selected_box_index is not None:
                    self.save_to_history()
                    self.current_boxes[self.selected_box_index].class_id = 0
                    self.modified = True
                    self.autosave()
                else:
                    self.current_class = 0
            elif key == ord('2'):
                if self.selected_box_index is not None:
                    self.save_to_history()
                    self.current_boxes[self.selected_box_index].class_id = 1
                    self.modified = True
                    self.autosave()
                else:
                    self.current_class = 1
            elif key == ord('3'):
                if self.selected_box_index is not None:
                    self.save_to_history()
                    self.current_boxes[self.selected_box_index].class_id = 2
                    self.modified = True
                    self.autosave()
                else:
                    self.current_class = 2

            # Delete box
            elif key == ord('d') or key == ord('D'):
                if self.selected_box_index is not None:
                    self.save_to_history()
                    del self.current_boxes[self.selected_box_index]
                    self.selected_box_index = None
                    self.modified = True
                    self.autosave()

            # Clear all boxes (Backspace)
            elif key == 8 or key == 127:  # Backspace key
                if len(self.current_boxes) > 0:
                    self.save_to_history()
                    self.current_boxes.clear()
                    self.selected_box_index = None
                    self.modified = True
                    self.autosave()
                    print(
                        f"Cleared all boxes for {self.image_files[self.current_index].name}")

            # Undo (Ctrl+Z)
            elif key == 26:  # Ctrl+Z
                if self.undo():
                    print(f"Undid last change")

            # Save
            elif key == ord('s') or key == ord('S'):
                self.save_labels(
                    self.image_files[self.current_index], self.current_boxes)
                self.modified = False
                print(
                    f"Saved labels for {self.image_files[self.current_index].name}")

            # Toggle help
            elif key == ord('h') or key == ord('H'):
                self.show_help = not self.show_help

            # Train
            elif key == ord('t') or key == ord('T'):
                if self.modified:
                    self.save_labels(
                        self.image_files[self.current_index], self.current_boxes)
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Allow window destruction to process
                return 'train'

            # Quit
            elif key == ord('q') or key == ord('Q') or key == 27:  # ESC
                if self.modified:
                    self.save_labels(
                        self.image_files[self.current_index], self.current_boxes)
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Allow window destruction to process
                return 'quit'


def prepare_training_data(review_dir: str, dataset_dir: str = 'dataset', val_split: float = 0.2):
    """
    Copy reviewed data to training dataset with 80/20 train/val split.
    Only copies images that haven't been copied before (prevents duplicates).

    Args:
        review_dir: Directory containing reviewed images and labels
        dataset_dir: Main dataset directory
        val_split: Fraction of new data to use for validation (default: 0.2)
    """
    import shutil
    import hashlib
    import random

    review_dir = Path(review_dir)
    dataset_dir = Path(dataset_dir)

    review_images = review_dir / 'images'
    review_labels = review_dir / 'labels'

    train_images_dir = dataset_dir / 'images' / 'train'
    train_labels_dir = dataset_dir / 'labels' / 'train'
    val_images_dir = dataset_dir / 'images' / 'val'
    val_labels_dir = dataset_dir / 'labels' / 'val'

    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # Track which images have been copied using a hash log
    hash_log_file = dataset_dir / '.copied_images_hashes.txt'
    copied_hashes = set()

    if hash_log_file.exists():
        with open(hash_log_file, 'r') as f:
            copied_hashes = set(line.strip() for line in f if line.strip())

    print(f"\nPreviously copied images: {len(copied_hashes)}")

    # Get existing file counts
    existing_train_count = len(list(train_images_dir.glob('*.jpg'))) + \
        len(list(train_images_dir.glob('*.png')))
    existing_val_count = len(list(val_images_dir.glob('*.jpg'))) + \
        len(list(val_images_dir.glob('*.png')))

    # Get review files
    image_files = list(review_images.glob('*.jpg')) + \
        list(review_images.glob('*.png'))

    print(f"Checking {len(image_files)} images from review directory...")

    # Filter out duplicates first
    new_images = []
    skipped_count = 0

    for img_file in image_files:
        # Calculate hash of image
        hash_md5 = hashlib.md5()
        with open(img_file, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        img_hash = hash_md5.hexdigest()

        # Skip if already copied
        if img_hash in copied_hashes:
            skipped_count += 1
            continue

        new_images.append((img_file, img_hash))

    # Shuffle and split new images
    random.shuffle(new_images)
    split_idx = int(len(new_images) * (1 - val_split))
    train_images = new_images[:split_idx]
    val_images = new_images[split_idx:]

    new_hashes = []
    train_copied = 0
    val_copied = 0

    # Copy training images
    for img_file, img_hash in train_images:
        label_file = review_labels / f"{img_file.stem}.txt"

        # Copy with new names to avoid conflicts
        new_img_name = f"train_{existing_train_count:05d}{img_file.suffix}"
        new_label_name = f"train_{existing_train_count:05d}.txt"

        # Copy image
        shutil.copy(str(img_file), str(train_images_dir / new_img_name))
        train_copied += 1

        # Copy label if exists
        if label_file.exists():
            shutil.copy(str(label_file), str(
                train_labels_dir / new_label_name))

        # Track this hash
        new_hashes.append(img_hash)
        existing_train_count += 1

    # Copy validation images
    for img_file, img_hash in val_images:
        label_file = review_labels / f"{img_file.stem}.txt"

        # Copy with new names to avoid conflicts
        new_img_name = f"val_{existing_val_count:05d}{img_file.suffix}"
        new_label_name = f"val_{existing_val_count:05d}.txt"

        # Copy image
        shutil.copy(str(img_file), str(val_images_dir / new_img_name))
        val_copied += 1

        # Copy label if exists
        if label_file.exists():
            shutil.copy(str(label_file), str(val_labels_dir / new_label_name))

        # Track this hash
        new_hashes.append(img_hash)
        existing_val_count += 1

    # Update hash log
    if new_hashes:
        with open(hash_log_file, 'a') as f:
            for h in new_hashes:
                f.write(f"{h}\n")

    print(f"\n✓ Processing complete:")
    print(f"  New images copied to train: {train_copied}")
    print(f"  New images copied to val: {val_copied}")
    print(f"  Duplicates skipped: {skipped_count}")
    print(f"  Total training images: {existing_train_count}")
    print(f"  Total validation images: {existing_val_count}")
    print(f"  Overall split: {existing_train_count/(existing_train_count + existing_val_count)*100:.1f}% train / {existing_val_count/(existing_train_count + existing_val_count)*100:.1f}% val")


def train_model(dataset_dir: str = 'dataset', epochs: int = 50, model_path: str = 'weights/best.pt'):
    """
    Fine-tune the model with additional data.

    Args:
        dataset_dir: Dataset directory
        epochs: Number of epochs to train
        model_path: Path to existing model weights
    """
    data_yaml = Path(dataset_dir) / 'data.yaml'

    if not data_yaml.exists():
        print(f"❌ Error: Dataset configuration not found at {data_yaml}")
        return

    print("\n" + "="*80)
    print("STARTING MODEL FINE-TUNING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {data_yaml}")
    print(f"  Base model: {model_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Project: runs/detect")
    print(f"  Name: crab_detector_finetuned")
    print()

    trainer = CrabTrainer(
        data_yaml=str(data_yaml),
        model_size='n'
    )

    # Train with existing checkpoint
    results = trainer.train(
        epochs=epochs,
        batch_size=16,
        img_size=640,
        project='runs/detect',
        name='crab_detector_finetuned',
        exist_ok=True,
        pretrained=True,
        resume_checkpoint=model_path
    )

    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest model saved to: runs/detect/crab_detector_finetuned/weights/best.pt")
    print(f"\nTo use the new model:")
    print(f"  1. Copy it to weights/: cp runs/detect/crab_detector_finetuned/weights/best.pt weights/")
    print(f"  2. Or specify path: --model_path runs/detect/crab_detector_finetuned/weights/best.pt")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Review auto-generated labels and train model"
    )
    parser.add_argument(
        '--review_dir',
        type=str,
        default='dataset/review',
        help="Directory containing images and labels to review (default: dataset/review)"
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='dataset',
        help="Main dataset directory (default: dataset)"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='weights/best.pt',
        help="Path to existing model weights (default: weights/best.pt)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help="Number of epochs for fine-tuning (default: 50)"
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help="Fraction of new data to use for validation (default: 0.2 for 80/20 split)"
    )

    args = parser.parse_args()

    print("="*80)
    print("LABEL REVIEW & TRAINING UI")
    print("="*80)
    print(f"\nReview directory: {args.review_dir}")
    print(f"Dataset directory: {args.dataset_dir}")
    print()

    images_dir = Path(args.review_dir) / 'images'
    labels_dir = Path(args.review_dir) / 'labels'

    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        print(f"\nPlease run extract_and_label.py first to generate frames and labels.")
        return

    # Start UI
    ui = LabelReviewUI(str(images_dir), str(labels_dir))
    result = ui.run()

    if result == 'train':
        # Ensure all CV2 windows are closed
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        print("\n" + "="*80)
        print("PREPARING TRAINING DATA")
        print("="*80)

        # Prepare data
        prepare_training_data(
            args.review_dir, args.dataset_dir, args.val_split)

        # Train model
        train_model(args.dataset_dir, args.epochs, args.model_path)
    else:
        print("\nQuitting without training.")


if __name__ == '__main__':
    main()
