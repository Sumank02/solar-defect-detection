import os
from ultralytics import YOLO
import torch

# Fix for PyTorch 2.8.0+ weights_only issue
original_load = torch.load
def safe_load(file, **kwargs):
    return original_load(file, weights_only=False, **kwargs)
torch.load = safe_load

# Path to trained YOLO model (updated to most recent training run)
MODEL_PATH = "runs/detect/solar_defect_train9/weights/best.pt"

# Input folder for testing images
SAMPLE_FOLDER = "sample"

# Output folder for detection results
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the YOLO model
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative model path...")
    # Try alternative path if the first one doesn't work
    MODEL_PATH = "runs/detect/solar_defect_train/weights/best.pt"
    model = YOLO(MODEL_PATH)
    print(f"Model loaded from alternative path: {MODEL_PATH}")

# Loop through sample images
for img_file in os.listdir(SAMPLE_FOLDER):
    if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(SAMPLE_FOLDER, img_file)

        # Run YOLO inference
        results = model(img_path, save=True, project=OUTPUT_FOLDER, name="", exist_ok=True)

        # Print detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                print(f"[{img_file}] Detected: {class_name} ({conf:.2f})")