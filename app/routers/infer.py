# app/routers/infer.py
import os
import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(tags=["inference"])

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")            # e.g. "workspace/model"
ROBOFLOW_MODEL_VERSION = os.getenv("ROBOFLOW_MODEL_VERSION", "1")

def _build_base_url() -> str:
    if not ROBOFLOW_API_KEY:
        raise HTTPException(500, "Missing ROBOFLOW_API_KEY")
    if not ROBOFLOW_MODEL_ID or "/" not in ROBOFLOW_MODEL_ID:
        raise HTTPException(500, "Missing or invalid ROBOFLOW_MODEL_ID (expected 'workspace/model')")
    return f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}/{ROBOFLOW_MODEL_VERSION}"

@router.get("/infer", include_in_schema=False)
def infer_get():
    # If you accidentally send GET, show a helpful message (instead of 405)
    return {"tip": "Use POST /api/infer with multipart/form-data 'file'", "base_url": _build_base_url()}

@router.post("/infer")
async def infer(file: UploadFile = File(...), confidence: float = 0.4, overlap: float = 0.5):
    base_url = _build_base_url()
    try:
        content = await file.read()
        files = {"file": (file.filename or "frame.jpg", content, file.content_type or "image/jpeg")}
        params = {
            "api_key": ROBOFLOW_API_KEY,
            "confidence": confidence,
            "overlap": overlap,
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(base_url, params=params, files=files)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference proxy error: {e}")

@router.get("/info", tags=["inference"])
def info():
    # Helpful for debugging (does NOT reveal your key)
    return {
        "model_id": ROBOFLOW_MODEL_ID,
        "version": ROBOFLOW_MODEL_VERSION,
        "base_url": _build_base_url().replace(ROBOFLOW_MODEL_ID, "<model_id>").replace(ROBOFLOW_MODEL_VERSION, "<version>")
    }
