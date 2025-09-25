import os
import uuid
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .model import ModelService


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
RESULTS_DIR = os.path.join(STATIC_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


app = FastAPI(title="Solar Hotspot Detection Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


model_service = ModelService()


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result_image_url": None,
            "detections": [],
            "summary": None,
        },
    )


def _read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode image. Supported formats: jpg, jpeg, png.")
    return image_bgr


def _save_result_image(image_bgr: np.ndarray) -> str:
    filename = f"{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(out_path, image_bgr)
    return f"/static/results/{filename}"


def _build_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for det in detections:
        cls = det.get("class_name", "Unknown")
        counts[cls] = counts.get(cls, 0) + 1
    total = sum(counts.values())
    return {"counts": counts, "total": total}


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        image_bgr = _read_image_from_upload(file_bytes)

        annotated_bgr, detections = model_service.run_inference(image_bgr)

        result_image_url = _save_result_image(annotated_bgr)
        summary = _build_summary(detections)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result_image_url": result_image_url,
                "detections": detections,
                "summary": summary,
            },
        )
    except Exception as exc:
        # On error, redirect back to home with a simple fallback. For production, show a friendly message.
        # Here we keep behavior simple and non-intrusive to existing project.
        return RedirectResponse(url="/", status_code=303)


