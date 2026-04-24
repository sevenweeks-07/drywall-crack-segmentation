import os
import shutil
import random
from pathlib import Path

def split_and_rename_dataset(src_img_dir, src_lbl_dir, base_dest_dir, prefix, split_ratio=0.8):
    """
    Splits a dataset into train/val, creates directories, and renames files sequentially.
    """
    src_img_path = Path(src_img_dir)
    src_lbl_path = Path(src_lbl_dir)
    
    # 1. Gather all valid images
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    all_images = [f for f in src_img_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    if not all_images:
        print(f"No images found in {src_img_path}")
        return

    # 2. Shuffle for randomized splitting
    random.seed(42) # Keeps the split reproducible if you run it again
    random.shuffle(all_images)
    
    # 3. Calculate split index
    split_index = int(len(all_images) * split_ratio)
    train_images = all_images[:split_index]
    valid_images = all_images[split_index:]
    
    # Define destination paths
    dest_paths = {
        "train": {
            "images": Path(base_dest_dir) / f"train_{prefix}" / "images",
            "labels": Path(base_dest_dir) / f"train_{prefix}" / "labels"
        },
        "valid": {
            "images": Path(base_dest_dir) / f"valid_{prefix}" / "images",
            "labels": Path(base_dest_dir) / f"valid_{prefix}" / "labels"
        }
    }
    
    # Create the necessary directories
    for split in dest_paths.values():
        split["images"].mkdir(parents=True, exist_ok=True)
        split["labels"].mkdir(parents=True, exist_ok=True)

    def process_files(image_list, split_type):
        print(f"Processing {len(image_list)} {split_type} files...")
        
        for index, img_file in enumerate(image_list, start=1):
            # Check for corresponding label
            lbl_file = src_lbl_path / (img_file.stem + '.txt')
            
            # Define new names based on your requested format (e.g., cracks_train_1.jpg)
            new_name_base = f"{prefix}_{split_type}_{index}"
            
            new_img_dest = dest_paths[split_type]["images"] / f"{new_name_base}{img_file.suffix}"
            new_lbl_dest = dest_paths[split_type]["labels"] / f"{new_name_base}.txt"
            
            # Copy image
            shutil.copy2(img_file, new_img_dest)
            
            # Copy label if it exists, otherwise create a blank one or skip (skipping is standard)
            if lbl_file.exists():
                shutil.copy2(lbl_file, new_lbl_dest)
            else:
                print(f"  -> Warning: No matching label for {img_file.name}. Skipped label copy.")

    # 4. Execute the processing
    process_files(train_images, "train")
    process_files(valid_images, "valid")
    print(f"\nSuccess! Dataset split into {len(train_images)} train and {len(valid_images)} valid samples.")

# ==========================================
# Configuration and Execution
# ==========================================
SOURCE_IMAGES = r"C:\Users\Dell\OneDrive\Desktop\origin\cracks\images"
SOURCE_LABELS = r"C:\Users\Dell\OneDrive\Desktop\origin\cracks\labels"
DESTINATION_BASE = r"C:\Users\Dell\OneDrive\Desktop\origin"

split_and_rename_dataset(
    src_img_dir=SOURCE_IMAGES, 
    src_lbl_dir=SOURCE_LABELS, 
    base_dest_dir=DESTINATION_BASE, 
    prefix="cracks", 
    split_ratio=0.8  # 80% Train, 20% Valid
)