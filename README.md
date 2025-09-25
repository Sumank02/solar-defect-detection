## Overview

This project trains a YOLO model to detect defects in solar panel images. You provide a dataset of images and labels, the model learns the patterns, and then you can run it on new images to highlight possible defects.

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

## Commands to Run

- Train the model on your dataset:
```powershell
python train_yolo.py
```
This will download a pre-trained YOLO model (if needed) and train it on your data. It saves progress, charts, and the model weights.

- Run predictions on sample images using the trained model:
```powershell
python predict.py
```
This processes images in the `sample/` folder and saves results with detection boxes.

- Optional: clean all previous results to start fresh:
```powershell
Remove-Item -Recurse -Force runs, outputs -ErrorAction SilentlyContinue
```

## Results and Conclusion

- Training results (charts, metrics, and model files) are saved here:
  - `runs/detect/<your_run_name>/`
  - Best trained model: `runs/detect/<your_run_name>/weights/best.pt`
  - Training curves: `runs/detect/<your_run_name>/results.png`

- Prediction results (images with boxes) are saved here:
  - `runs/detect/predict*/`
  - If you used `main.py` previously, also check: `outputs/`

- Conclusion: After training, use `predict.py` to analyze new images. The model quality depends on your dataset and training time. If detections are weak, train longer, use larger image size (e.g., 640), or add more/better-labeled data. If you move the project folder, update paths in `dataset/data.yaml` before training again.
