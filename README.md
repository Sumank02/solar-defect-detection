## Overview

Solar Defect Detection is a YOLO-based pipeline to detect hotspot-related defects on solar panels from thermal images. The repo contains three runnable parts:
- Software-only (CLI) for training and batch inference
- Web dashboard (FastAPI) for upload-and-analyze UI
- Hardware module for real-time camera inference (MLX90640)

Key tech: Ultralytics YOLOv8, PyTorch, OpenCV, FastAPI/Jinja2. Dataset follows standard YOLO layout under `dataset/`.

## Setup

1) Install Python 3.10+ and open PowerShell in the project folder, e.g. `D:\solar-defect-detection`.

2) Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4) Verify dataset structure (already included in this repo). It should look like:
```
dataset/
  train/
    images/   ← training pictures
    labels/   ← training labels (YOLO .txt files)
  valid/
    images/
    labels/
  test/
    images/
    labels/
```
Make sure `dataset/data.yaml` points to the correct absolute paths for your machine (edit if needed):
```yaml
train: D:/solar-defect-detection/dataset/train/images
val:   D:/solar-defect-detection/dataset/valid/images
test:  D:/solar-defect-detection/dataset/test/images
```

## Commands to Run (in order)

### 0) Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If your repo uses Git LFS for images/weights, fetch data:
```powershell
git lfs install
git lfs pull
```

### 1) Root scripts (training and quick prediction)

- Train (default 50 epochs; set `YOLO_EPOCHS` to shorten while testing):
```powershell
$env:YOLO_EPOCHS="10"; python train_yolo.py
```
Trained checkpoint is saved at `runs/detect/solar_defect_train/weights/best.pt`.

- Predict (auto-discovers latest `best.pt`; saves annotated images):
```powershell
python predict.py
```

- Optional clean start:
```powershell
Remove-Item -Recurse -Force runs, outputs -ErrorAction SilentlyContinue
```

### 2) Software-only module (organized CLI)

- Batch inference over a folder (defaults to `sample/` → `outputs/`):
```powershell
python -m software.run
```

- With options:
```powershell
python -m software.run --input D:\images --output D:\out --model D:\weights\best.pt
```
If `--model` is omitted, it auto-picks the latest `runs/detect/*/weights/best.pt`, else falls back to `yolov8n.pt` → `yolov8n.yaml`.

### 3) Web dashboard (upload and analyze)

Install web deps (separate file) and run:
```powershell
pip install -r requirements-web.txt
$env:WEB_MODEL_PATH="D:\solar-defect-detection\runs\detect\solar_defect_train\weights\best.pt"; `
python -m webapp.run_web
```
Open `http://127.0.0.1:8000`. Use `WEB_CONFIDENCE` (e.g., `0.15`) to adjust threshold. See `webapp/README.md` for details.

### 4) Hardware (MLX90640 real-time)

See `hardware/README_hardware.md` for platform-specific setup. The script auto-discovers the latest `best.pt` or falls back:
```powershell
python hardware/real_time_detection.py
```

## Results and Conclusion

### Where outputs appear
- Training: `runs/detect/<run_name>/` (weights under `weights/best.pt`, charts as `results.png`)
- Prediction (CLI): `runs/detect/predict*/` or `outputs/` depending on the script
- Web app: `webapp/static/results/`

### Conclusion
After training produces `best.pt`, all modes (software-only, web, hardware) auto-use it and generate detections on your thermal images. Before `best.pt` exists, scripts run with a fallback model (`yolov8n.pt`/`yolov8n.yaml`) to avoid errors, but detections will be limited. For stronger results, improve labels, increase epochs, and validate paths in `dataset/data.yaml`.
