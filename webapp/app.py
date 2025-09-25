import os
import uuid
import time
import logging
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .model import ModelService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("webapp")


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
            "logs": [],
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
    logs: List[str] = []
    try:
        start_overall = time.time()

        file_bytes = await file.read()
        logs.append(f"Upload received: filename={file.filename} size={len(file_bytes)} bytes")
        logger.info(logs[-1])

        # Basic validation: require plausible image size and extension
        allowed_ext = {".jpg", ".jpeg", ".png"}
        _, ext = os.path.splitext(file.filename or "")
        ext = ext.lower()
        if ext and ext not in allowed_ext:
            logs.append(f"Rejected: unsupported extension '{ext}'. Use: jpg, jpeg, png.")
            logger.info(logs[-1])
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "result_image_url": None, "detections": [], "summary": None, "logs": logs},
            )

        # Detect common Git LFS pointer (typically ~130 bytes text file)
        lfs_signature = b"version https://git-lfs.github.com/spec/v1"
        if file_bytes.startswith(lfs_signature):
            logs.append("Rejected: Git LFS pointer detected. Run 'git lfs install' then 'git lfs pull' to fetch real images.")
            logger.info(logs[-1])
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "result_image_url": None, "detections": [], "summary": None, "logs": logs},
            )

        if len(file_bytes) < 1024:
            logs.append("Rejected: file too small (< 1KB). Ensure you selected the actual image, not a shortcut/label.")
            logger.info(logs[-1])
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "result_image_url": None, "detections": [], "summary": None, "logs": logs},
            )

        image_bgr = _read_image_from_upload(file_bytes)
        h, w = (image_bgr.shape[0], image_bgr.shape[1]) if image_bgr is not None else (0, 0)
        logs.append(f"Image decoded: shape={w}x{h}")
        logger.info(logs[-1])

        logs.append("Starting inference...")
        logger.info("Starting inference")
        t0 = time.time()
        annotated_bgr, detections = model_service.run_inference(image_bgr)
        infer_ms = int((time.time() - t0) * 1000)
        logs.append(f"Inference complete in {infer_ms} ms. Detections={len(detections)}")
        logger.info(logs[-1])

        result_image_url = _save_result_image(annotated_bgr)
        logs.append(f"Annotated image saved: {result_image_url}")
        logger.info(logs[-1])

        summary = _build_summary(detections)
        total_ms = int((time.time() - start_overall) * 1000)
        logs.append(f"Request finished in {total_ms} ms")
        logger.info(logs[-1])

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result_image_url": result_image_url,
                "detections": detections,
                "summary": summary,
                "logs": logs + [f"Model: {model_service.get_model_source()}"] ,
            },
        )
    except Exception as exc:
        err_msg = f"Error during analysis: {exc}"
        logs.append(err_msg)
        logger.exception(err_msg)
        # Return 200 so the page updates with logs instead of browser showing 500 error banner
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result_image_url": None, "detections": [], "summary": None, "logs": logs + [f"Model: {model_service.get_model_source()}"]},
        )


