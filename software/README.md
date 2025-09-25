## Software-Only Inference

Run YOLO detections on images from a folder without hardware or the web UI.

### Quick Start

1) Install deps (root requirements):
```powershell
pip install -r requirements.txt
```

2) Run over the default `sample/` folder:
```powershell
python -m software.run
```

3) Options:
```powershell
python -m software.run --input D:\path\to\images --output D:\path\to\out --model D:\path\to\best.pt
```

### Model selection

If `--model` is not provided, the script will:
1. Use the latest `runs/detect/*/weights/best.pt` if available
2. Try known paths like `runs/detect/solar_defect_train/weights/best.pt`
3. Fall back to `yolov8n.pt` in the repo, then `yolov8n.yaml`

### Outputs

Annotated images are saved to the `--output` directory (default: `outputs/`).


