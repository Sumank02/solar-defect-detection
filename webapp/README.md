## Web Dashboard (FastAPI)

This module provides a simple web UI to upload an image on the left and view detections on the right. It is isolated from the existing project and does not modify other files.

### Quick Start

1) Install deps (separate from main requirements):
```powershell
pip install -r requirements-web.txt
```

2) Launch the server (opens browser automatically):
```powershell
python -m webapp.run_web
```

3) Upload a JPG/PNG image. The app will run YOLO inference and display:
- Annotated image
- List of detections
- Brief summary
- Processing log (useful for debugging)

### Use Trained Weights

By default, the app tries in order:
1. `WEB_MODEL_PATH` (environment variable)
2. Latest `runs/detect/*/weights/best.pt`
3. Known project paths like `runs/detect/solar_defect_train/weights/best.pt`
4. Local `yolov8n.pt`
5. Fallback `yolov8n.yaml` (untrained)

Set your trained model explicitly:
```powershell
$env:WEB_MODEL_PATH="D:\solar-defect-detection\runs\detect\solar_defect_train\weights\best.pt"
$env:WEB_CONFIDENCE="0.15"
python -m webapp.run_web
```

### Logs & Troubleshooting

- The right panel shows a Processing Log for each request.
- The server console logs include model loading and detections.
- If you see "Git LFS pointer detected", run:
```powershell
git lfs install
git lfs pull
```
- If you see "No detections", confirm the log shows `Model: ...best.pt` and try lowering confidence via `WEB_CONFIDENCE`.

### File Locations

- App entry: `webapp/app.py`
- Model service: `webapp/model.py`
- Templates: `webapp/templates/index.html`
- Static assets and saved results: `webapp/static/`
- Launcher: `webapp/run_web.py`

### Notes

- The web app saves annotated images to `webapp/static/results/`.
- The app does not require a camera and only processes uploaded images.
- The module preserves the rest of the project; you can continue to use CLI scripts independently.


