import os
import glob
import argparse
from typing import List

import torch
from ultralytics import YOLO


# Fix for PyTorch 2.8.0+ weights_only issue
_orig_load = torch.load
def _safe_load(file, **kwargs):
    return _orig_load(file, weights_only=False, **kwargs)
torch.load = _safe_load


def resolve_model_path() -> str:
    candidates = glob.glob(os.path.join("runs", "detect", "*", "weights", "best.pt"))
    if candidates:
        return max(candidates, key=os.path.getmtime)

    fallbacks = [
        os.path.join("runs", "detect", "solar_defect_train9", "weights", "best.pt"),
        os.path.join("runs", "detect", "solar_defect_train", "weights", "best.pt"),
    ]
    for path in fallbacks:
        if os.path.exists(path):
            return path

    local_pt = os.path.join(os.getcwd(), "yolov8n.pt")
    return local_pt if os.path.exists(local_pt) else "yolov8n.yaml"


def list_images(folder: str) -> List[str]:
    supported = (".jpg", ".jpeg", ".png")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(supported)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Software-only inference over a folder of images.")
    parser.add_argument("--input", default="sample", help="Input images folder (default: sample)")
    parser.add_argument("--output", default="outputs", help="Output folder for annotated results (default: outputs)")
    parser.add_argument("--model", default=None, help="Optional explicit path to a YOLO .pt file")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    model_path = args.model if args.model else resolve_model_path()
    print(f"Using model: {model_path}")
    model = YOLO(model_path)

    images = list_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}")
        return

    for img_path in images:
        results = model(img_path, save=True, project=output_dir, name="", exist_ok=True)
        for result in results:
            boxes = result.boxes
            for box in boxes or []:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                print(f"[{os.path.basename(img_path)}] {class_name} ({conf:.2f})")


if __name__ == "__main__":
    main()


