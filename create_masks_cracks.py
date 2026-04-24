import os
import cv2
import numpy as np
from pathlib import Path

# Set your paths here
IMAGE_DIR = Path("/root/origin/automated-drywall-inspection/cracks/train/images")
LABEL_DIR = Path("/root/origin/automated-drywall-inspection/cracks/train/labels")
MASK_OUT_DIR = Path("/root/origin/automated-drywall-inspection/cracks/train/masks")

# Create output directory if it doesn't exist
MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)

def create_masks():
    # Get all text files in the label directory
    label_files = [f for f in LABEL_DIR.iterdir() if f.suffix == '.txt']
    
    for lbl_path in label_files:
        # Find corresponding image to get the exact width and height
        # Try .jpg first, change to .png if your images are pngs
        img_path = IMAGE_DIR / f"{lbl_path.stem}.jpg" 
        
        if not img_path.exists():
            print(f"Skipping {lbl_path.name}: Matching image not found.")
            continue
            
        # Read image just to get its dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        
        # 1. Create the black background
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 2. Read the YOLO label file
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue # Skip empty or invalid lines
                
            # parts[0] is the class ID, parts[1:] are the normalized coordinates
            coords = [float(c) for c in parts[1:]]
            
            # YOLO coordinates are normalized (0 to 1). 
            # We must multiply them by the image width and height.
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * width)
                y = int(coords[i+1] * height)
                points.append([x, y])
                
            # Convert points to numpy array for OpenCV
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # 3. Make the label region white (255)
            cv2.fillPoly(mask, [pts], color=(255))
            
        # Save the resulting binary mask
        out_mask_path = MASK_OUT_DIR / f"{lbl_path.stem}_mask.png"
        cv2.imwrite(str(out_mask_path), mask)
        
    print(f"Done! Masks saved to {MASK_OUT_DIR}")

if __name__ == "__main__":
    create_masks()