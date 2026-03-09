"""
Clean duplicate images from the dataset.
Uses file size comparison for fast duplicate detection.
"""

from pathlib import Path
import hashlib
from collections import defaultdict


def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_and_remove_duplicates(images_dir, labels_dir, dry_run=False):
    """Find and remove duplicate images based on file size and hash."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # Get all images
    image_files = sorted(list(images_dir.glob('*.jpg')) +
                         list(images_dir.glob('*.png')))

    print(f"Scanning {len(image_files)} images for duplicates...")

    # First pass: Group by file size (fast)
    print("  Step 1: Grouping by file size...")
    size_to_files = defaultdict(list)
    for img_file in image_files:
        size = img_file.stat().st_size
        size_to_files[size].append(img_file)

    potential_duplicates = sum(
        len(files) - 1 for files in size_to_files.values() if len(files) > 1)
    print(f"  Found {potential_duplicates} potential duplicates (same size)")

    if potential_duplicates == 0:
        print("  No duplicates found!")
        return 0

    # Second pass: Hash only files with same size
    print("  Step 2: Checking duplicates by content hash...")
    hash_to_files = {}
    files_to_check = [(size, files)
                      for size, files in size_to_files.items() if len(files) > 1]
    total_to_hash = sum(len(files) for _, files in files_to_check)

    hashed_count = 0
    for size, files in files_to_check:
        for img_file in files:
            hashed_count += 1
            if hashed_count % 500 == 0 or hashed_count == total_to_hash:
                print(f"    Progress: {hashed_count}/{total_to_hash} files...")

            file_hash = get_file_hash(img_file)
            if file_hash not in hash_to_files:
                hash_to_files[file_hash] = []
            hash_to_files[file_hash].append(img_file)

    # Find duplicates and remove them
    print("\n  Step 3: Removing duplicates...")
    duplicates_count = 0
    removed_count = 0

    for file_hash, files in hash_to_files.items():
        if len(files) > 1:
            duplicates_count += len(files) - 1
            # Keep the first one (lowest number), remove the rest
            files_sorted = sorted(files, key=lambda x: x.stem)

            if not dry_run:
                for i, f in enumerate(files_sorted):
                    if i == 0:
                        continue  # Keep first
                    else:
                        # Remove image
                        f.unlink()
                        removed_count += 1

                        # Remove corresponding label if exists
                        label_file = labels_dir / f"{f.stem}.txt"
                        if label_file.exists():
                            label_file.unlink()
            else:
                print(
                    f"\n  Would remove {len(files)-1} duplicates of: {files_sorted[0].name}")
                for f in files_sorted[1:]:
                    print(f"    - {f.name}")
                removed_count += len(files) - 1

    print(f"\n{'='*80}")
    print(f"CLEANUP SUMMARY")
    print(f"{'='*80}")
    print(f"Total images scanned: {len(image_files)}")
    print(
        f"Duplicate sets found: {len([h for h, f in hash_to_files.items() if len(f) > 1])}")
    if dry_run:
        print(f"Duplicates that would be removed: {removed_count}")
    else:
        print(f"Duplicate files removed: {removed_count}")
    print(f"Unique images: {len(hash_to_files)}")

    return removed_count


def main():
    """Main function."""
    import sys

    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv

    if dry_run:
        print("="*80)
        print("DRY RUN MODE - No files will be deleted")
        print("="*80)
        print()

    # Clean training data
    print("="*80)
    print("CLEANING TRAINING DATASET")
    print("="*80)

    train_images = 'dataset/images/train'
    train_labels = 'dataset/labels/train'

    if not Path(train_images).exists():
        print(f"Training dataset not found at {train_images}")
        return

    removed = find_and_remove_duplicates(
        train_images, train_labels, dry_run=dry_run)

    if dry_run:
        print(f"\n✓ Dry run complete - {removed} duplicates found")
        print(f"\nRun without --dry-run to actually remove duplicates")
    elif removed > 0:
        print(f"\n✓ Removed {removed} duplicate files from training dataset")
    else:
        print(f"\n✓ No duplicates found in training dataset")

    # Also check validation data
    val_images = 'dataset/images/val'
    val_labels = 'dataset/labels/val'

    if Path(val_images).exists():
        print(f"\n{'='*80}")
        print("CLEANING VALIDATION DATASET")
        print("="*80)

        removed_val = find_and_remove_duplicates(
            val_images, val_labels, dry_run=dry_run)
        if dry_run:
            print(f"\n✓ Dry run complete - {removed_val} duplicates found")
        elif removed_val > 0:
            print(
                f"\n✓ Removed {removed_val} duplicate files from validation dataset")
        else:
            print(f"\n✓ No duplicates found in validation dataset")


if __name__ == '__main__':
    main()
