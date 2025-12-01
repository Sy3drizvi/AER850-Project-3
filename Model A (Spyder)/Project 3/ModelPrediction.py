"""
AER850 – Project 3  
YOLOv11 Local Detection Script (White Box + Blue Text Style)

- White bounding boxes
- White label background
- Blue label text
- Black outline on text
- Clean layout identical to provided Mega 2560 image
"""

import os
import cv2
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.getcwd()
SAVE_DIR = os.path.join(BASE_DIR, "detection_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------
# Find data.yaml
# ---------------------------------------------------------

def find_data_yaml(base):
    for r, d, f in os.walk(base):
        if "data.yaml" in f:
            return os.path.join(r, "data.yaml")
    raise FileNotFoundError("data.yaml not found.")

DATA_YAML = find_data_yaml(BASE_DIR)


# ---------------------------------------------------------
# Load class names
# ---------------------------------------------------------

def load_classnames(path):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    return y["names"]

CLASS_NAMES = load_classnames(DATA_YAML)


# ---------------------------------------------------------
# Find best.pt
# ---------------------------------------------------------

def find_best_pt(base):
    for r, d, f in os.walk(base):
        if "best.pt" in f:
            return os.path.join(r, "best.pt")
    raise FileNotFoundError("best.pt missing.")

BEST_PT = find_best_pt(BASE_DIR)


# ---------------------------------------------------------
# Find evaluation images
# ---------------------------------------------------------

def find_eval_images(base):
    imgs = []
    exts = ("*.jpg", "*.jpeg", "*.png")
    for r, d, f in os.walk(base):
        folder = os.path.basename(r).lower()
        if "eval" in folder or "evaluation" in folder:
            for ext in exts:
                imgs.extend(glob.glob(os.path.join(r, ext)))
    return imgs

eval_images = sorted(find_eval_images(BASE_DIR))
if not eval_images:
    raise RuntimeError("No evaluation images found.")


# ---------------------------------------------------------
# Draw WHITE boxes + BLUE text (your requested style)
# ---------------------------------------------------------

def draw_boxes(img, results):
    out = img.copy()

    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
        cls = int(b.cls)
        conf = float(b.conf)

        label = f"{CLASS_NAMES[cls]} {conf:.2f}"

        # White bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # Label font setup
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thick = 2

        # Determine label box size
        (w, h), _ = cv2.getTextSize(label, font, scale, thick)

        # White background rectangle ABOVE the box
        cv2.rectangle(out,
            (x1, y1 - h - 12),
            (x1 + w + 10, y1),
            (255, 255, 255),
            -1
        )

        # BLUE text (R,G,B) = (255, 0, 0)
        text_x, text_y = x1 + 5, y1 - 5

        # Draw black outline first (thicker)
        cv2.putText(out, label, (text_x, text_y),
                    font, scale, (0, 0, 0), thick + 2)

        # Draw blue text inside
        cv2.putText(out, label, (text_x, text_y),
                    font, scale, (255, 0, 0), thick)

    return out


# ---------------------------------------------------------
# Resize for Spyder display (zoomed for clarity)
# ---------------------------------------------------------

def zoom_display(img, zoom=1.25, max_w=1600):
    img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    if w > max_w:
        s = max_w / w
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


# ---------------------------------------------------------
# Load YOLO model
# ---------------------------------------------------------

model = YOLO(BEST_PT)


# ---------------------------------------------------------
# Perform detection
# ---------------------------------------------------------

for img_path in eval_images:
    name = os.path.basename(img_path)
    print("Processing:", name)

    img = cv2.imread(img_path)
    res = model.predict(img, conf=0.20, verbose=False)[0]

    drawn = draw_boxes(img, res)
    shown = zoom_display(drawn)

    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
    plt.title(name)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Save original resolution output
    save_path = os.path.join(SAVE_DIR, "det_" + name)
    cv2.imwrite(save_path, drawn)
    print("Saved:", save_path)

print("Detection completed.")
