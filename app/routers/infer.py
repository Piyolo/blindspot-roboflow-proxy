import os
import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(tags=["inference"])

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")            # e.g. "your-workspace/your-model"
ROBOFLOW_MODEL_VERSION = os.getenv("ROBOFLOW_MODEL_VERSION", "1")
BASE_URL = None

def _ensure_config():
    global BASE_URL
    if not ROBOFLOW_API_KEY:
        raise HTTPException(500, "Missing ROBOFLOW_API_KEY")
    if not ROBOFLOW_MODEL_ID:
        raise HTTPException(500, "Missing ROBOFLOW_MODEL_ID (e.g. 'workspace/model')")
    if BASE_URL is None:
        BASE_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}/{ROBOFLOW_MODEL_VERSION}"

@router.post("/infer")
async def infer(file: UploadFile = File(...), confidence: float = 0.4, overlap: float = 0.5):
    _ensure_config()
    # Forward the uploaded image to Roboflow Hosted API
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
            resp = await client.post(BASE_URL, params=params, files=files)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference proxy error: {e}")
