###############################################
# AER850 – PROJECT 3 – FULL PIPELINE (LOCAL)
# Highest Accuracy Edition – YOLOv11 + Masking
# Designed for Spyder / Desktop Python
###############################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import shutil
import glob

#--------------------------------------------------
# AUTO-INSTALL ULTRALYTICS (optional but handy)
#--------------------------------------------------
try:
    import ultralytics
except ImportError:
    print("Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

from ultralytics import YOLO

###############################################
# 0. USER SETTINGS
###############################################
# Choose mode:
#   "FULL_RUN"        -> Masking + Training + Evaluation + Consolidation
#   "EVALUATION_ONLY" -> Only Evaluation (uses saved weights)
MAIN_MODE = "FULL_RUN"   # "FULL_RUN" or "EVALUATION_ONLY"

# If using EVALUATION_ONLY, set your previous run path here
# (this folder must contain weights/best.pt)
LAST_RUN_PATH = os.path.join(os.getcwd(), "runs", "detect", "pcb_model_best")


###############################################
# 1. ENVIRONMENT SETUP
###############################################
def setup_environment():
    """
    Use the current working directory as the project folder.
    Make sure this folder contains:
      - data.zip  (or a 'data' folder with data.yaml inside)
      - motherboard_image.JPEG
      - evaluation/   (with evaluation images)
    """
    BASE_DIR = os.getcwd()
    print(f"\nWorking inside: {BASE_DIR}")

    # Auto-extract dataset if zip exists and 'data' folder doesn't
    zip_path = os.path.join(BASE_DIR, "data.zip")
    extract_dir = os.path.join(BASE_DIR, "data")

    if os.path.exists(zip_path) and not os.path.exists(extract_dir):
        print("\nExtracting data.zip...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(BASE_DIR)
        print("Dataset extracted successfully.")

    return BASE_DIR


###############################################
# 2. OBJECT MASKING (PART 1)
###############################################
def step1_masking(base_dir):
    print("\n===== STEP 1: PCB MASKING =====")

    img_path = os.path.join(base_dir, "motherboard_image.JPEG")
    if not os.path.exists(img_path):
        print(f"ERROR: motherboard_image.JPEG not found at {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: Could not read motherboard_image.JPEG")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu automatic threshold + inverse (PCB becomes white region)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Strong morphological closing → merge PCB region
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours and keep biggest one (assume PCB)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No PCB region detected. Try adjusting kernel size/threshold.")
        return

    biggest = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(gray)
    cv2.drawContours(final_mask, [biggest], -1, 255, thickness=cv2.FILLED)

    # Extract PCB
    extracted = cv2.bitwise_and(img, img, mask=final_mask)
    extr_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)

    # Create output folder for masking figures
    mask_out_dir = os.path.join(base_dir, "mask_outputs")
    os.makedirs(mask_out_dir, exist_ok=True)

    # Save raw intermediate images
    cv2.imwrite(os.path.join(mask_out_dir, "01_original.jpg"), img)
    cv2.imwrite(os.path.join(mask_out_dir, "02_gray.jpg"), gray)
    cv2.imwrite(os.path.join(mask_out_dir, "03_blur.jpg"), blur)
    cv2.imwrite(os.path.join(mask_out_dir, "04_thresh.jpg"), thresh)
    cv2.imwrite(os.path.join(mask_out_dir, "05_mask.jpg"), final_mask)
    cv2.imwrite(os.path.join(mask_out_dir, "06_extracted.jpg"), extracted)

    # Plot and save 3-panel figure (for report)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Motherboard")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("PCB Binary Mask")
    plt.imshow(final_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Extracted PCB")
    plt.imshow(extr_rgb)
    plt.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(mask_out_dir, "masking_overview.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"Masking images saved in: {mask_out_dir}")


###############################################
# 3. TRAIN YOLOv11 (PART 2)
###############################################
def step2_training(base_dir):
    print("\n===== STEP 2: TRAINING YOLOv11 =====")

    # data.yaml should be inside ./data/ or project root
    yaml_path = os.path.join(base_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        alt_yaml = os.path.join(base_dir, "data", "data.yaml")
        if os.path.exists(alt_yaml):
            yaml_path = alt_yaml
        else:
            raise FileNotFoundError(
                "ERROR: data.yaml not found.\n"
                "Place data.yaml in the project folder or in ./data/"
            )

    # Load YOLOv11-nano
    model = YOLO("yolo11n.pt")

    PROJECT = os.path.join(base_dir, "runs", "detect")
    NAME = "pcb_model_best"

    # NOTE: If GPU runs OOM, reduce imgsz to 800 or 640, or reduce batch size.
    results = model.train(
        data=yaml_path,
        imgsz=1024,         # Higher resolution → better accuracy (but heavier)
        epochs=150,         # Under project limit of 200
        batch=4,
        workers=2,
        patience=20,        # Early stopping
        augment=True,       # Enable general augmentation
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        plots=True          # saves loss + metrics plots (results.png, etc.)
    )

    run_path = os.path.join(PROJECT, NAME)
    print("\nTraining completed.")
    print("Run folder:", run_path)

    # Extra validation call to ensure all validation plots/metrics are saved
    print("\n===== STEP 2b: VALIDATION (plots + metrics) =====")
    metrics = model.val(
        data=yaml_path,
        split="val",
        imgsz=1024,
        save=True,
        plots=True,     # PR curves, F1, confusion matrix, etc.
        project=PROJECT,
        name=NAME,      # reuse same folder
        exist_ok=True
    )

    # Save a metrics summary as text for the report
    metrics_file = os.path.join(run_path, "metrics_summary.txt")
    with open(metrics_file, "w") as f:
        f.write(str(metrics))
    print("Validation metrics saved to:", metrics_file)

    return model, run_path


###############################################
# 4. EVALUATION (PART 3)
###############################################
def step3_eval(model, base_dir, run_path):
    print("\n===== STEP 3: EVALUATION =====")

    eval_folder = os.path.join(base_dir, "evaluation")
    if not os.path.exists(eval_folder):
        print(f"ERROR: evaluation folder missing at {eval_folder}")
        return

    out_dir = os.path.join(run_path, "evaluation_results")

    model.predict(
        source=eval_folder,
        imgsz=1024,
        conf=0.25,
        save=True,
        project=run_path,
        name="evaluation_results",
        exist_ok=True
    )

    print(f"Evaluation results saved to:\n{out_dir}")


###############################################
# 5. LOAD EXISTING MODEL
###############################################
def load_model(run_path):
    weight = os.path.join(run_path, "weights", "best.pt")
    if not os.path.exists(weight):
        print(f"ERROR: No best.pt found at {weight}")
        return None
    print(f"Loaded weights from {weight}")
    return YOLO(weight)


###############################################
# 6. CONSOLIDATE ALL OUTPUTS FOR REPORT
###############################################
def consolidate_outputs(base_dir, run_path):
    """
    Collect everything into project3_outputs/:
      - mask_outputs/
      - training_outputs/  (full YOLO run folder with all plots)
      - evaluation_outputs/ (annotated eval images)
      - report_notes.txt
    """
    print("\n===== STEP 4: CONSOLIDATING OUTPUTS FOR REPORT =====")

    final_dir = os.path.join(base_dir, "project3_outputs")
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    os.makedirs(final_dir, exist_ok=True)

    # 1) Masking outputs
    mask_dir = os.path.join(base_dir, "mask_outputs")
    if os.path.exists(mask_dir):
        shutil.copytree(mask_dir, os.path.join(final_dir, "masking"))
    else:
        print("Warning: mask_outputs not found, skipping masking/")

    # 2) Training outputs (YOLO run folder: results.png, curves, confusion matrix, metrics_summary.txt)
    if os.path.exists(run_path):
        shutil.copytree(run_path, os.path.join(final_dir, "training_outputs"))
    else:
        print("Warning: training run folder not found:", run_path)

    # 3) Evaluation outputs (evaluation_results inside run_path)
    eval_results_dir = os.path.join(run_path, "evaluation_results")
    if os.path.exists(eval_results_dir):
        shutil.copytree(eval_results_dir, os.path.join(final_dir, "evaluation_outputs"))
    else:
        # Fallback: look for predict* folders just in case
        predict_folders = sorted(glob.glob(os.path.join(run_path, "predict*")))
        if predict_folders:
            shutil.copytree(predict_folders[-1],
                            os.path.join(final_dir, "evaluation_outputs"))
        else:
            print("Warning: no evaluation_results or predict* outputs found.")

    # 4) Simple readme for your TA
    notes_path = os.path.join(final_dir, "report_notes.txt")
    with open(notes_path, "w") as f:
        f.write(
            "AER850 – Project 3 Output Structure\n"
            "===================================\n\n"
            "masking/              → PCB masking steps and overview figure\n"
            "training_outputs/     → YOLO training run (results.png, curves, confusion matrix, metrics_summary.txt)\n"
            "evaluation_outputs/   → Annotated predictions on evaluation images\n"
        )

    print("All outputs consolidated into:", final_dir)


###############################################
# 7. MAIN
###############################################
if __name__ == "__main__":
    base = setup_environment()

    if MAIN_MODE == "FULL_RUN":
        # 1) Masking
        step1_masking(base)
        # 2) Train + validate + plots
        model, run_path = step2_training(base)
        # 3) Evaluation on evaluation/
        step3_eval(model, base, run_path)
        # 4) Consolidate everything for the report
        consolidate_outputs(base, run_path)

    elif MAIN_MODE == "EVALUATION_ONLY":
        model = load_model(LAST_RUN_PATH)
        if model is not None:
            step3_eval(model, base, LAST_RUN_PATH)
            consolidate_outputs(base, LAST_RUN_PATH)

    else:
        print("Invalid MAIN_MODE. Use 'FULL_RUN' or 'EVALUATION_ONLY'.")
