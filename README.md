## Overview

Solar Defect Detection is a YOLO-based pipeline to detect hotspot-related defects on solar panels from thermal images. The repo contains three runnable parts:
- Software-only (CLI) for training and batch inference
- Web dashboard (FastAPI) for upload-and-analyze UI
- Hardware module for real-time camera inference (MLX90640)

Key tech: Ultralytics YOLOv8, PyTorch, OpenCV, FastAPI/Jinja2. Dataset follows standard YOLO layout under `dataset/`.

## Setup

**⚠️ IMPORTANT: This repository uses Git LFS for large files (dataset images).**

**Option 1: Using Git (Recommended)**
1. Install Git from https://git-scm.com/downloads (if not already installed)
2. Clone this repository:
   ```powershell
   git clone <repository-url>
   cd solar-defect-detection
   ```
3. Install Git LFS and pull the large files:
   ```powershell
   git lfs install
   git lfs pull
   ```

**Option 2: ZIP + Manual Dataset Folder (No Git Required!)**
If you receive this project as a ZIP file, the dataset images will be missing (Git LFS stores them separately). **You do NOT need Git for this option.**
1. Unzip the project
2. Get the `dataset/` folder separately (ask the sender to share it directly via file sharing service, USB, etc.)
3. Place the complete `dataset/` folder in the project root, maintaining the structure:
   ```
   dataset/
     train/images/  (with all .jpg files)
     train/labels/  (with all .txt files)
     valid/images/
     valid/labels/
     test/images/
     test/labels/
   ```
4. **Skip all Git commands** - you can proceed directly to Python setup below.

**After completing Option 1 or Option 2 above, continue with the setup steps below:**

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


### 1) Root scripts (training and quick prediction)

- Train (default 50 epochs; set `YOLO_EPOCHS` to shorten while testing):
```powershell
$env:YOLO_EPOCHS="50"; 
python train_yolo.py
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

### Troubleshooting

**Error: "git is not recognized" or "The term 'git' is not recognized"**
- If you received the project as a ZIP file, you **don't need Git!** Use **Option 2** above and skip all Git commands. Just unzip, add the `dataset/` folder manually, and proceed with Python setup.
- If you want to use Git (Option 1), install it from https://git-scm.com/downloads first, then restart PowerShell.

**Error: "Git LFS pointer detected" or missing dataset images**
- If using Option 1: Make sure you ran `git lfs install` and `git lfs pull` after cloning.
- If using Option 2: Make sure you manually added the complete `dataset/` folder with all actual image files (not pointer files).

**Dataset not found errors**
- Verify the `dataset/` folder exists in the project root with the correct structure (train/images, train/labels, etc.).
- Check that `dataset/data.yaml` has the correct absolute paths for your machine.

**Warning: "LF will be replaced by CRLF" for `.venv/` files**
- This happens because the virtual environment (`.venv/`) is being tracked by Git, which it shouldn't be.
- **Solution**: The project now includes a `.gitignore` file. If you see these warnings:
  1. Remove `.venv/` from Git tracking: `git rm -r --cached .venv`
  2. Commit the change: `git commit -m "Remove .venv from tracking"`
  3. The warnings will stop appearing. Virtual environments should never be committed to Git.

### Conclusion
After training produces `best.pt`, all modes (software-only, web, hardware) auto-use it and generate detections on your thermal images. Before `best.pt` exists, scripts run with a fallback model (`yolov8n.pt`/`yolov8n.yaml`) to avoid errors, but detections will be limited. For stronger results, improve labels, increase epochs, and validate paths in `dataset/data.yaml`.
