import os
from pathlib import Path

def rename_yolo_dataset(image_dir, label_dir, prefix):
    """
    Renames images and corresponding YOLO label files sequentially.
    """
    image_path = Path(image_dir)
    label_path = Path(label_dir)

    if not image_path.exists():
        print(f"Directory not found: {image_path}")
        return

    # Look for common image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    images = [f for f in image_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    if not images:
        print(f"No images found in {image_path}")
        return

    print(f"Found {len(images)} images in {image_path.name}. Renaming with prefix '{prefix}'...")

    # Sort images to maintain consistent ordering
    images.sort()

    for index, img_file in enumerate(images, start=1):
        # The YOLO label file has the exact same name as the image, but a .txt extension
        lbl_file = label_path / (img_file.stem + '.txt')

        # Define the new names
        new_img_name = f"{prefix}_{index}{img_file.suffix}"
        new_lbl_name = f"{prefix}_{index}.txt"

        new_img_file = image_path / new_img_name
        new_lbl_file = label_path / new_lbl_name

        # Rename the image
        img_file.rename(new_img_file)

        # Rename the label if it exists
        if lbl_file.exists():
            lbl_file.rename(new_lbl_file)
        else:
            print(f"  -> Warning: No matching label file found for {img_file.name}")

    print(f"Finished renaming {prefix}!\n")


# ==========================================
# 1. Rename the Cracks Dataset
# ==========================================
cracks_img_dir = r"C:\Users\Dell\OneDrive\Desktop\origin\images_cracks"
cracks_lbl_dir = r"C:\Users\Dell\OneDrive\Desktop\origin\labels_cracks"

rename_yolo_dataset(cracks_img_dir, cracks_lbl_dir, "cracks")

drywall_train_dir = r"C:\Users\Dell\OneDrive\Desktop\origin\valid_drywall"

# Assuming images and labels are mixed in the same folder:
rename_yolo_dataset(drywall_train_dir+"\images", drywall_train_dir+"\labels", "drywall_train")

# ==========================================
# 3. Rename the Drywall Dataset (Valid)
# ==========================================
drywall_valid_dir = r"C:\Users\Dell\OneDrive\Desktop\origin\valid_drywall"

# Assuming images and labels are mixed in the same folder:
rename_yolo_dataset(drywall_valid_dir, drywall_valid_dir, "drywall_valid")