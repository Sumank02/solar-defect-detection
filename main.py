import os
from ultralytics import YOLO
import torch
import glob

# Fix for PyTorch 2.8.0+ weights_only issue
original_load = torch.load
def safe_load(file, **kwargs):
    return original_load(file, weights_only=False, **kwargs)
torch.load = safe_load

# Resolve trained model path automatically (latest best.pt with fallbacks)
def resolve_model_path() -> str:
    candidates = glob.glob(os.path.join("runs", "detect", "*", "weights", "best.pt"))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    fallbacks = [
        os.path.join("runs", "detect", "solar_defect_train9", "weights", "best.pt"),
        os.path.join("runs", "detect", "solar_defect_train", "weights", "best.pt"),
    ]
    for p in fallbacks:
        if os.path.exists(p):
            return p
    # Final fallback: local yolov8n.pt or base config
    local_pt = os.path.join(os.getcwd(), "yolov8n.pt")
    return local_pt if os.path.exists(local_pt) else "yolov8n.yaml"

MODEL_PATH = resolve_model_path()

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
    raise

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