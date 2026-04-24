import cv2
import numpy as np
from pathlib import Path

def generate_traditional_masks(image_dir, label_dir, output_dir):
    img_path = Path(image_dir)
    lbl_path = Path(label_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    images = [f for f in img_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    for img_file in images:
        lbl_file = lbl_path / (img_file.stem + ".txt")
        if not lbl_file.exists(): continue

        img = cv2.imread(str(img_file))
        if img is None: continue
        h, w = img.shape[:2]
        
        # Initialize a full-sized black mask [cite: 12]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                # Denormalize YOLO bbox to pixel coordinates 
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)

                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Crop to the bounding box region
                roi = img[y1:y2, x1:x2]
                if roi.size == 0: continue

                # 1. Edge Detection: Use Canny to find the seam boundaries
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)

                # 2. Morphological Closing: Bridge small gaps in the line 
                kernel = np.ones((5, 5), np.uint8)
                closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                # 3. Find and Filter Contours by geometry [cite: 442, 448]
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                best_cnt = None
                max_score = -1

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 50: continue # Area Filter [cite: 378, 445]

                    # Thinness check using Aspect Ratio [cite: 379, 443]
                    rect = cv2.minAreaRect(cnt)
                    _, (rw, rh), _ = rect
                    if rw == 0 or rh == 0: continue
                    aspect_ratio = max(rw, rh) / min(rw, rh)

                    # We want the largest and thinnest shape 
                    score = area * aspect_ratio 
                    if aspect_ratio > 3.0 and score > max_score:
                        max_score = score
                        best_cnt = cnt

                # 4. Draw the best candidate back onto the full mask [cite: 12]
                if best_cnt is not None:
                    roi_mask = np.zeros_like(closed)
                    cv2.drawContours(roi_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
                    full_mask[y1:y2, x1:x2] = cv2.bitwise_or(full_mask[y1:y2, x1:x2], roi_mask)

        # Save resulting binary mask as 0/255 PNG [cite: 12]
        cv2.imwrite(str(out_path / (img_file.stem + "_mask.png")), full_mask)

# Run for your directories
generate_traditional_masks(
    "/root/origin/automated-drywall-inspection/drywall/train/images",
    "/root/origin/automated-drywall-inspection/drywall/train/labels",
    "/root/origin/automated-drywall-inspection/drywall/train/masks"
)