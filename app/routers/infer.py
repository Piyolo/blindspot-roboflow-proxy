# app/routers/infer.py
import os
import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(tags=["inference"])

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")  # Private key (non-rf_)
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")  # e.g. "workspace/project" OR just "project"
ROBOFLOW_MODEL_VERSION = os.getenv("ROBOFLOW_MODEL_VERSION", "1")

def _candidate_urls():
    """
    Build a list of possible detect endpoints to try, to avoid 405s from
    model-id format differences across Roboflow setups.
    """
    if not ROBOFLOW_API_KEY:
        raise HTTPException(500, "Missing ROBOFLOW_API_KEY")
    if not ROBOFLOW_MODEL_ID:
        raise HTTPException(500, "Missing ROBOFLOW_MODEL_ID")

    candidates = []

    # If the user supplied workspace/project, try that first
    if "/" in ROBOFLOW_MODEL_ID:
        workspace, project = ROBOFLOW_MODEL_ID.split("/", 1)
        candidates.append(f"https://detect.roboflow.com/{workspace}/{project}/{ROBOFLOW_MODEL_VERSION}")
        candidates.append(f"https://detect.roboflow.com/{project}/{ROBOFLOW_MODEL_VERSION}")
    else:
        # Only project provided
        project = ROBOFLOW_MODEL_ID
        candidates.append(f"https://detect.roboflow.com/{project}/{ROBOFLOW_MODEL_VERSION}")

    return candidates

@router.get("/info", tags=["inference"])
def info():
    # Show the model info and the candidate URL shapes (without API key)
    return {
        "model_id": ROBOFLOW_MODEL_ID,
        "version": ROBOFLOW_MODEL_VERSION,
        "candidates": [u.replace("https://detect.roboflow.com/", "detect://") for u in _candidate_urls()],
        "tip": "POST /api/infer with multipart form 'file'"
    }

@router.get("/infer", include_in_schema=False)
def infer_get_tip():
    return {"tip": "Use POST /api/infer with multipart/form-data 'file'"}

@router.post("/infer")
async def infer(file: UploadFile = File(...), confidence: float = 0.4, overlap: float = 0.5):
    # Read once so we can retry to alternate URL if needed
    content = await file.read()
    files = {"file": (file.filename or "frame.jpg", content, file.content_type or "image/jpeg")}
    params = {"api_key": ROBOFLOW_API_KEY, "confidence": confidence, "overlap": overlap, "format": "json"}

    last_text = None
    async with httpx.AsyncClient(timeout=30) as client:
        for url in _candidate_urls():
            try:
                resp = await client.post(url, params=params, files=files)
                if resp.status_code == 200:
                    return resp.json()
                # collect text and try next candidate on 404/405
                last_text = f"{resp.status_code} {resp.text}"
                if resp.status_code in (404, 405):
                    continue
                # For other error codes, surface immediately
                raise HTTPException(resp.status_code, resp.text)
            except httpx.HTTPError as e:
                last_text = str(e)
                continue

    raise HTTPException(502, f"Roboflow request failed. Tried {len(_candidate_urls())} URL forms. Last error: {last_text}")
