import os
import cv2
import numpy as np
from pathlib import Path

def draw_yolo_labels(image_dir, label_dir, output_dir, color=(0, 255, 0), thickness=2):
    """
    Reads both YOLO segmentation (polygons) and YOLO detection (bounding boxes)
    and draws them on the corresponding images.
    """
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print(f"Skipping {image_path} (Directory not found)")
        return

    valid_extensions = {'.jpg', '.jpeg', '.png'}
    images = [f for f in image_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    count = 0
    for img_file in images:
        lbl_file = label_path / (img_file.stem + '.txt')
        
        if not lbl_file.exists():
            continue 
            
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        img_height, img_width = img.shape[:2]
        
        with open(lbl_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Grab all coordinates after the class ID
            coords = np.array(parts[1:], dtype=float)
            
            # ====================================================
            # FORMAT 1: YOLO Object Detection (Bounding Box)
            # ====================================================
            if len(coords) == 4:
                x_center, y_center, box_width, box_height = coords
                
                # Denormalize to pixel coordinates
                x_center *= img_width
                y_center *= img_height
                box_width *= img_width
                box_height *= img_height
                
                # Calculate top-left and bottom-right corners
                x1 = int(x_center - (box_width / 2))
                y1 = int(y_center - (box_height / 2))
                x2 = int(x_center + (box_width / 2))
                y2 = int(y_center + (box_height / 2))
                
                # Draw the rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # ====================================================
            # FORMAT 2: YOLO Instance Segmentation (Polygon)
            # ====================================================
            elif len(coords) > 4:
                # Reshape into [x, y] pairs
                points = coords.reshape(-1, 2)
                
                # Denormalize
                points[:, 0] = points[:, 0] * img_width
                points[:, 1] = points[:, 1] * img_height
                
                # Draw the polygon
                polygon = np.int32(points)
                cv2.polylines(img, [polygon], isClosed=True, color=color, thickness=thickness)
                
        # Save the visualized image
        out_filename = out_path / f"viz_{img_file.name}"
        cv2.imwrite(str(out_filename), img)
        count += 1
        
    print(f"Processed {count} images from {image_path.parent.name}/{image_path.name}")

# ==========================================
# Configuration and Execution
# ==========================================
base_path = Path(r"C:\Users\Dell\OneDrive\Desktop\origin")
viz_output_dir = base_path / "visualization"

folders_to_process = [
    base_path / "cracks" / "train",
    base_path / "cracks" / "valid",
    base_path / "drywall" / "train",
    base_path / "drywall" / "valid"
]

print("Starting visualization process...\n")

for folder in folders_to_process:
    img_dir = folder / "images"
    lbl_dir = folder / "labels"
    draw_yolo_labels(img_dir, lbl_dir, viz_output_dir)

print(f"\nSuccess! Check the '{viz_output_dir}' folder to see your properly drawn labels.")