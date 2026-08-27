"""FastAPI inference service.

Run:
    uvicorn deepfake_detection.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health
    POST /predict  (multipart file upload: video)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import PROJECT_ROOT
from .inference import DeepfakeVideoDetector

app = FastAPI(title="Deepfake Video Detection API", version="1.0.0")

SPATIAL = PROJECT_ROOT / "checkpoints" / "spatial_effb0.pt"
TEMPORAL = PROJECT_ROOT / "checkpoints" / "temporal_transformer.pt"

_detector: DeepfakeVideoDetector | None = None


def get_detector() -> DeepfakeVideoDetector:
    global _detector
    if _detector is None:
        if not SPATIAL.exists():
            raise HTTPException(503, f"Model checkpoint missing: {SPATIAL}")
        _detector = DeepfakeVideoDetector(SPATIAL, TEMPORAL if TEMPORAL.exists() else None)
    return _detector


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _detector is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(400, "Upload a video file (.mp4/.avi/.mov/.mkv/.webm)")
    data = await file.read()
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(413, "Video too large (max 200MB)")
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        det = get_detector()
        result = det.predict_video(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    if "error" in result:
        raise HTTPException(422, result["error"])
    return result
