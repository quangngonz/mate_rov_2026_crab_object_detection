"""
Label Review and Training UI
============================

Interactive UI for reviewing and correcting auto-generated labels, then training.
"""

import cv2
import argparse
import random
import shutil
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import yaml
import copy

from config.constants import CRAB_CLASSES, CLASS_COLORS
from models import CrabTrainer, CrabDetector


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
        self.reviewed_file = self.labels_dir / '.reviewed_images.txt'
        self.reviewed_stems: Set[str] = self.load_reviewed_stems()

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
        self.reset_button_rect = None

        # Undo functionality
        self.history: List[List[BoundingBox]] = []
        self.max_history = 20

        reviewed_count = len(self.reviewed_stems)
        print(f"Loaded {len(self.image_files)} images from {images_dir}")
        if reviewed_count:
            print(f"  {reviewed_count} images marked as reviewed")

    def load_reviewed_stems(self) -> Set[str]:
        """Load set of image stems the user has already reviewed."""
        if not self.reviewed_file.exists():
            return set()
        try:
            with open(self.reviewed_file, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            print(f"Warning: Could not load reviewed images list: {e}")
            return set()

    def save_reviewed_stems(self):
        """Persist reviewed image stems."""
        try:
            with open(self.reviewed_file, 'w') as f:
                for stem in sorted(self.reviewed_stems):
                    f.write(f"{stem}\n")
        except Exception as e:
            print(f"Warning: Could not save reviewed images list: {e}")

    def mark_current_reviewed(self):
        """Mark the current image as reviewed by the user."""
        stem = self.image_files[self.current_index].stem
        if stem not in self.reviewed_stems:
            self.reviewed_stems.add(stem)
            self.save_reviewed_stems()

    def reset_review_progress(self):
        """Clear reviewed tracking and return to the first image."""
        self.reviewed_stems.clear()
        self.save_reviewed_stems()

        if self.progress_file.exists():
            self.progress_file.unlink()

        self.load_image(0)
        print("Review progress reset. Starting from image 1.")

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
        reviewed_count = len(self.reviewed_stems)
        pending_count = len(self.image_files) - reviewed_count
        status = f"Image {self.current_index + 1}/{len(self.image_files)} | "
        status += f"Reviewed: {reviewed_count} | Pending: {pending_count} | "
        status += f"Boxes: {len(self.current_boxes)}"
        if self.selected_box_index is not None:
            status += f" (#{self.selected_box_index + 1} selected)"
        status += f" | Class: {CRAB_CLASSES[self.current_class]} | "
        status += "Modified" if self.modified else "Saved"

        cv2.rectangle(self.display_image, (0, 0),
                      (img_width, 35), (50, 50, 50), -1)
        cv2.putText(self.display_image, status, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        button_text = "Reset Progress [R]"
        button_size, _ = cv2.getTextSize(
            button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        button_width = button_size[0] + 20
        button_height = 28
        button_x1 = max(10, img_width - button_width - 10)
        button_y1 = 4
        button_x2 = button_x1 + button_width
        button_y2 = button_y1 + button_height
        self.reset_button_rect = (button_x1, button_y1, button_x2, button_y2)

        cv2.rectangle(self.display_image, (button_x1, button_y1),
                      (button_x2, button_y2), (90, 60, 60), -1)
        cv2.rectangle(self.display_image, (button_x1, button_y1),
                      (button_x2, button_y2), (180, 120, 120), 1)
        cv2.putText(
            self.display_image,
            button_text,
            (button_x1 + 10, button_y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

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
                "  R or Reset button: Clear review progress",
                "  H: Toggle help | P: Partial train & relabel | T: Full train",
                "  Q/ESC: Quit"
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

        # Ignore clicks in the UI area unless they hit the reset button
        if y < ui_height:
            if (
                event == cv2.EVENT_LBUTTONDOWN
                and self.reset_button_rect is not None
            ):
                x1, y1, x2, y2 = self.reset_button_rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.reset_review_progress()
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
                self.mark_current_reviewed()
                self.load_image(self.current_index - 1)

            elif key == 83 or key == 3:  # Right arrow
                if self.modified:
                    self.save_labels(
                        self.image_files[self.current_index], self.current_boxes)
                self.mark_current_reviewed()
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
                self.mark_current_reviewed()
                print(
                    f"Saved labels for {self.image_files[self.current_index].name}")

            # Toggle help
            elif key == ord('h') or key == ord('H'):
                self.show_help = not self.show_help

            # Reset review progress
            elif key == ord('r') or key == ord('R'):
                self.reset_review_progress()

            # Partial train on reviewed images, then relabel the rest
            elif key == ord('p') or key == ord('P'):
                if self.modified:
                    self.save_labels(
                        self.image_files[self.current_index], self.current_boxes)
                self.mark_current_reviewed()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                return 'partial_train'

            # Full train
            elif key == ord('t') or key == ord('T'):
                if self.modified:
                    self.save_labels(
                        self.image_files[self.current_index], self.current_boxes)
                self.mark_current_reviewed()
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Allow window destruction to process
                return 'train'

            # Quit
            elif key == ord('q') or key == ord('Q') or key == 27:  # ESC
                if self.modified:
                    self.save_labels(
                        self.image_files[self.current_index], self.current_boxes)
                self.mark_current_reviewed()
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


def resolve_trained_model_path(train_result: dict, run_name: str) -> Path:
    """Return the trained best.pt path, with fallback for legacy nested runs/."""
    best_model = Path(train_result['best_model']).resolve()
    if best_model.exists():
        return best_model

    nested_fallback = Path('runs/detect/runs/detect') / \
        run_name / 'weights' / 'best.pt'
    if nested_fallback.exists():
        print(
            f"Warning: Using model from legacy nested path: {nested_fallback}")
        return nested_fallback.resolve()

    raise FileNotFoundError(
        f"Trained model not found at {best_model} or {nested_fallback}"
    )


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
    train_result = trainer.train(
        epochs=epochs,
        batch_size=16,
        img_size=640,
        project='runs/detect',
        name='crab_detector_finetuned',
        exist_ok=True,
        pretrained=True,
        resume_checkpoint=model_path
    )

    best_model = resolve_trained_model_path(
        train_result, 'crab_detector_finetuned')

    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest model saved to: {best_model}")
    print(f"\nTo use the new model:")
    print(f"  1. Copy it to weights/: cp {best_model} weights/")
    print(f"  2. Or specify path: --model_path {best_model}")


def create_partial_training_dataset(
    review_dir: str,
    reviewed_stems: Set[str],
    val_split: float = 0.2
) -> Path:
    """
    Build a temporary YOLO dataset from reviewed images only.

    Args:
        review_dir: Review directory containing images/ and labels/
        reviewed_stems: Image stems the user has reviewed
        val_split: Fraction of reviewed data to use for validation

    Returns:
        Path to generated data.yaml
    """
    review_dir = Path(review_dir)
    review_images = review_dir / 'images'
    review_labels = review_dir / 'labels'
    partial_dir = review_dir / '_partial_train'

    if partial_dir.exists():
        shutil.rmtree(partial_dir)

    train_images_dir = partial_dir / 'images' / 'train'
    train_labels_dir = partial_dir / 'labels' / 'train'
    val_images_dir = partial_dir / 'images' / 'val'
    val_labels_dir = partial_dir / 'labels' / 'val'

    for directory in (train_images_dir, train_labels_dir, val_images_dir, val_labels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reviewed_files = sorted(
        img_file for img_file in list(review_images.glob('*.jpg')) + list(review_images.glob('*.png'))
        if img_file.stem in reviewed_stems
    )

    if not reviewed_files:
        raise ValueError(
            "No reviewed images found to build partial training dataset")

    shuffled_files = reviewed_files.copy()
    random.shuffle(shuffled_files)

    if len(shuffled_files) == 1:
        train_files = shuffled_files
        val_files = shuffled_files
    elif len(shuffled_files) < 5:
        val_files = shuffled_files[-1:]
        train_files = shuffled_files[:-1]
    else:
        split_idx = int(len(shuffled_files) * (1 - val_split))
        split_idx = max(1, min(split_idx, len(shuffled_files) - 1))
        train_files = shuffled_files[:split_idx]
        val_files = shuffled_files[split_idx:]

    def copy_split(image_files: List[Path], images_dir: Path, labels_dir: Path, prefix: str):
        for index, img_file in enumerate(image_files):
            label_file = review_labels / f"{img_file.stem}.txt"
            new_img_name = f"{prefix}_{index:05d}{img_file.suffix}"
            new_label_name = f"{prefix}_{index:05d}.txt"
            shutil.copy(str(img_file), str(images_dir / new_img_name))
            if label_file.exists():
                shutil.copy(str(label_file), str(labels_dir / new_label_name))
            else:
                (labels_dir / new_label_name).touch()

    copy_split(train_files, train_images_dir, train_labels_dir, 'train')
    copy_split(val_files, val_images_dir, val_labels_dir, 'val')

    data_yaml = partial_dir / 'data.yaml'
    with open(data_yaml, 'w') as f:
        yaml.dump({
            'path': str(partial_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(CRAB_CLASSES),
            'names': CRAB_CLASSES,
        }, f, default_flow_style=False)

    print(f"\n✓ Partial training dataset created at {partial_dir}")
    print(f"  Reviewed images: {len(reviewed_files)}")
    print(f"  Train split: {len(train_files)}")
    print(f"  Val split: {len(val_files)}")

    return data_yaml


def relabel_unreviewed_images(
    review_dir: str,
    reviewed_stems: Set[str],
    model_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> int:
    """
    Re-run model inference on images that have not been reviewed yet.

    Args:
        review_dir: Review directory containing images/ and labels/
        reviewed_stems: Image stems to skip
        model_path: Model checkpoint to use for inference
        conf_threshold: Detection confidence threshold
        iou_threshold: IoU threshold for NMS

    Returns:
        Number of images relabeled
    """
    review_dir = Path(review_dir)
    images_dir = review_dir / 'images'
    labels_dir = review_dir / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(images_dir.glob('*.jpg')) +
                         list(images_dir.glob('*.png')))
    unreviewed_files = [
        img for img in image_files if img.stem not in reviewed_stems]

    if not unreviewed_files:
        print("\nNo unreviewed images left to relabel.")
        return 0

    print(f"\nLoading model from {model_path}...")
    detector = CrabDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

    print(f"Relabeling {len(unreviewed_files)} unreviewed images...")
    relabeled_count = 0
    total_detections = 0

    for index, image_path in enumerate(unreviewed_files):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read {image_path}")
            continue

        img_height, img_width = img.shape[:2]
        result = detector.predict(str(image_path))
        label_path = labels_dir / f"{image_path.stem}.txt"

        with open(label_path, 'w') as f:
            for box_index in range(result['num_detections']):
                box = result['boxes'][box_index]
                class_id = int(result['class_ids'][box_index])
                x1, y1, x2, y2 = box[:4]
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_detections += 1

        relabeled_count += 1
        if (index + 1) % 10 == 0 or index + 1 == len(unreviewed_files):
            print(
                f"  Relabeled {index + 1}/{len(unreviewed_files)} images...", end='\r')

    print(
        f"\n✓ Relabeled {relabeled_count} images ({total_detections} detections)")
    return relabeled_count


def partial_train_and_relabel(
    review_dir: str,
    reviewed_stems: Set[str],
    model_path: str,
    partial_epochs: int = 10,
    val_split: float = 0.2,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    min_reviewed: int = 5
) -> Optional[str]:
    """
    Train briefly on reviewed labels, then refresh auto-labels for the rest.

    Returns:
        Path to the new model checkpoint, or None if training was skipped
    """
    if len(reviewed_stems) < min_reviewed:
        print(
            f"\n❌ Need at least {min_reviewed} reviewed images for partial training "
            f"(currently {len(reviewed_stems)})."
        )
        print("Review more frames first, then press P again.")
        return None

    unreviewed_count = 0
    review_images = Path(review_dir) / 'images'
    for img_file in list(review_images.glob('*.jpg')) + list(review_images.glob('*.png')):
        if img_file.stem not in reviewed_stems:
            unreviewed_count += 1

    print("\n" + "="*80)
    print("PARTIAL TRAIN & RELABEL")
    print("="*80)
    print(f"\nReviewed images: {len(reviewed_stems)}")
    print(f"Unreviewed images: {unreviewed_count}")
    print(f"Partial epochs: {partial_epochs}")
    print(f"Base model: {model_path}")

    data_yaml = create_partial_training_dataset(
        review_dir, reviewed_stems, val_split)
    reviewed_count = len(reviewed_stems)
    batch_size = min(16, max(2, reviewed_count // 4))

    print("\n" + "="*80)
    print("STARTING PARTIAL FINE-TUNING")
    print("="*80)

    trainer = CrabTrainer(data_yaml=str(data_yaml), model_size='n')
    train_result = trainer.train(
        epochs=partial_epochs,
        batch_size=batch_size,
        img_size=640,
        project='runs/detect',
        name='crab_detector_partial',
        exist_ok=True,
        pretrained=True,
        resume_checkpoint=model_path,
        patience=max(5, partial_epochs),
    )

    new_model_path = str(resolve_trained_model_path(
        train_result, 'crab_detector_partial'))
    print(f"\n✓ Partial training complete: {new_model_path}")

    if unreviewed_count > 0:
        relabel_unreviewed_images(
            review_dir=review_dir,
            reviewed_stems=reviewed_stems,
            model_path=new_model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
    else:
        print("\nAll images are already reviewed; skipping relabel step.")

    print("\n" + "="*80)
    print("✓ PARTIAL TRAIN & RELABEL COMPLETE")
    print("="*80)
    print("\nReturning to review UI. Continue correcting the refreshed labels,")
    print("then press P again or T for full training when done.")

    return new_model_path


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
    parser.add_argument(
        '--partial_epochs',
        type=int,
        default=10,
        help="Epochs for partial mid-review training when pressing P (default: 10)"
    )
    parser.add_argument(
        '--conf_threshold',
        type=float,
        default=0.25,
        help="Confidence threshold when relabeling unreviewed images (default: 0.25)"
    )
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.45,
        help="IoU threshold when relabeling unreviewed images (default: 0.45)"
    )
    parser.add_argument(
        '--min_reviewed',
        type=int,
        default=5,
        help="Minimum reviewed images required before partial training (default: 5)"
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

    current_model_path = args.model_path

    while True:
        ui = LabelReviewUI(str(images_dir), str(labels_dir))
        result = ui.run()

        if result == 'partial_train':
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            new_model_path = partial_train_and_relabel(
                review_dir=args.review_dir,
                reviewed_stems=ui.reviewed_stems,
                model_path=current_model_path,
                partial_epochs=args.partial_epochs,
                val_split=args.val_split,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                min_reviewed=args.min_reviewed,
            )

            if new_model_path:
                current_model_path = new_model_path

                image_files = sorted(list(images_dir.glob(
                    '*.jpg')) + list(images_dir.glob('*.png')))
                for index, image_path in enumerate(image_files):
                    if image_path.stem not in ui.reviewed_stems:
                        progress_file = labels_dir / '.review_progress.txt'
                        with open(progress_file, 'w') as f:
                            f.write(str(index))
                        print(
                            f"\nResuming review at first pending image: {index + 1}/{len(image_files)}")
                        break
            continue

        if result == 'train':
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            print("\n" + "="*80)
            print("PREPARING TRAINING DATA")
            print("="*80)

            prepare_training_data(
                args.review_dir, args.dataset_dir, args.val_split)

            train_model(args.dataset_dir, args.epochs, current_model_path)
            break

        print("\nQuitting without training.")
        break


if __name__ == '__main__':
    main()
