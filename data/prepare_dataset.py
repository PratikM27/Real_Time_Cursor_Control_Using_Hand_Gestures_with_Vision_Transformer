"""
prepare_dataset.py — Dataset Preparation & Splitting
=====================================================
Takes raw collected images and creates a stratified
train/val/test split in ImageFolder format.

Usage:
    python data/prepare_dataset.py
"""

import os
import sys
import shutil
import random
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GESTURE_CLASSES, GESTURE_LABELS, PATHS, TRAINING, SEED


def get_image_files(directory):
    """Get all image files in a directory."""
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    files = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if os.path.splitext(f)[1].lower() in extensions:
                files.append(os.path.join(directory, f))
    return sorted(files)


def create_dataset_structure(dataset_dir):
    """Create train/val/test subdirectories for each class."""
    splits = ['train', 'val', 'test']
    for split in splits:
        for class_name in GESTURE_CLASSES.values():
            dir_path = os.path.join(dataset_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)


def split_dataset(raw_dir, dataset_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Perform stratified split of raw data into train/val/test.
    
    Args:
        raw_dir: Path to raw_data/ directory with per-class subdirectories
        dataset_dir: Path to gesture_dataset/ output directory
        train_ratio, val_ratio, test_ratio: Split ratios (must sum to 1.0)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    random.seed(SEED)
    
    print("=" * 60)
    print("  DATASET PREPARATION")
    print("=" * 60)
    print(f"\n  Source: {raw_dir}")
    print(f"  Output: {dataset_dir}")
    print(f"  Split:  {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")
    print()
    
    # Create directory structure
    create_dataset_structure(dataset_dir)
    
    stats = defaultdict(dict)
    total_train = total_val = total_test = 0
    
    for class_id, class_name in GESTURE_CLASSES.items():
        class_dir = os.path.join(raw_dir, class_name)
        images = get_image_files(class_dir)
        
        if len(images) == 0:
            print(f"  WARNING: No images found for class '{class_name}' in {class_dir}")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remaining goes to test
        n_test = n - n_train - n_val
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images to respective directories
        for split_name, split_images in [('train', train_images), 
                                          ('val', val_images), 
                                          ('test', test_images)]:
            dest_dir = os.path.join(dataset_dir, split_name, class_name)
            for img_path in split_images:
                filename = os.path.basename(img_path)
                dest_path = os.path.join(dest_dir, filename)
                shutil.copy2(img_path, dest_path)
        
        label = GESTURE_LABELS.get(class_name, class_name)
        stats[class_name] = {
            'total': n,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images),
        }
        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)
        
        print(f"  [{class_id}] {label:20s}: {n:4d} total → "
              f"Train: {len(train_images):4d} | "
              f"Val: {len(val_images):4d} | "
              f"Test: {len(test_images):4d}")
    
    # Print summary
    total = total_train + total_val + total_test
    print()
    print("-" * 60)
    print(f"  TOTAL: {total} images")
    print(f"    Train: {total_train:5d}  ({total_train/max(total,1):.1%})")
    print(f"    Val:   {total_val:5d}  ({total_val/max(total,1):.1%})")
    print(f"    Test:  {total_test:5d}  ({total_test/max(total,1):.1%})")
    print("-" * 60)
    
    # Warn about small classes
    min_count = min(s.get('total', 0) for s in stats.values()) if stats else 0
    if min_count < 100:
        print(f"\n  ⚠ WARNING: Smallest class has only {min_count} images.")
        print("    Consider collecting more data for better model performance.")
    
    if total == 0:
        print("\n  ❌ ERROR: No images found! Run collect_data.py first.")
    else:
        print(f"\n  ✓ Dataset prepared successfully at: {dataset_dir}")
    
    return stats


def validate_dataset(dataset_dir):
    """Validate the prepared dataset structure."""
    print("\n  Validating dataset structure...")
    
    splits = ['train', 'val', 'test']
    all_good = True
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"    ✗ Missing directory: {split_dir}")
            all_good = False
            continue
        
        classes = sorted(os.listdir(split_dir))
        if len(classes) == 0:
            print(f"    ✗ No classes in {split}/")
            all_good = False
        
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                count = len(get_image_files(class_dir))
                if count == 0:
                    print(f"    ✗ Empty class: {split}/{class_name}/")
                    all_good = False
    
    if all_good:
        print("    ✓ Dataset structure is valid!")
    
    return all_good


def main():
    raw_dir = PATHS["raw_data"]
    dataset_dir = PATHS["dataset"]
    
    # Check if raw data exists
    if not os.path.exists(raw_dir):
        print(f"ERROR: Raw data directory not found: {raw_dir}")
        print("Run 'python data/collect_data.py' first to collect gesture images.")
        return
    
    # Check if dataset already exists and ask for confirmation
    if os.path.exists(dataset_dir):
        # Count existing files
        existing = 0
        for root, dirs, files in os.walk(dataset_dir):
            existing += len(files)
        
        if existing > 0:
            print(f"WARNING: Dataset directory already contains {existing} files.")
            response = input("Overwrite? (y/n): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return
            
            # Clean existing dataset
            shutil.rmtree(dataset_dir)
    
    # Perform split
    stats = split_dataset(
        raw_dir, 
        dataset_dir,
        train_ratio=TRAINING["train_split"],
        val_ratio=TRAINING["val_split"],
        test_ratio=TRAINING["test_split"],
    )
    
    # Validate
    validate_dataset(dataset_dir)


if __name__ == "__main__":
    main()
