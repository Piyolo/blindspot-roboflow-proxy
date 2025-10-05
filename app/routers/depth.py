# app/routers/depth.py
import io, os
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import httpx

from .depth_local import _run_depth  # reuse the local MiDaS runner

router = APIRouter(tags=["DepthMerge"])

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")
ROBOFLOW_MODEL_VERSION = os.getenv("ROBOFLOW_MODEL_VERSION", "1")

def _rf_url() -> str:
    if not (ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID):
        raise HTTPException(500, "Missing Roboflow env")
    return f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}/{ROBOFLOW_MODEL_VERSION}"

def _center_to_tlwh(cx, cy, w, h):
    l = int(round(cx - w/2.0))
    t = int(round(cy - h/2.0))
    return l, t, int(round(w)), int(round(h))

def _median_in_box(depth_map: np.ndarray, tlwh):
    H, W = depth_map.shape
    x, y, bw, bh = tlwh
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + bw); y1 = min(H, y + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    region = depth_map[y0:y1, x0:x1]
    if region.size == 0:
        return None
    return float(np.median(region))

def _norm_to_meters(norm_val: float, min_m: float, max_m: float, scale: float):
    if norm_val is None:
        return None
    # 1.0 (near) -> min_m ; 0.0 (far) -> max_m
    meters = min_m + (1.0 - norm_val) * (max_m - min_m)
    return max(0.05, meters * scale)

@router.post("/api/infer_depth")
async def infer_depth(
    file: UploadFile = File(...),
    confidence: float = 0.4,
    overlap: float = 0.5,
    min_m: float = 0.3,
    max_m: float = 5.0,
    scale: float = 1.0
):
    # read image once
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Invalid image")

    # 1) Roboflow detections
    params = {
        "api_key": ROBOFLOW_API_KEY,
        "confidence": confidence,
        "overlap": overlap,
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=45) as client:
        rf_resp = await client.post(_rf_url(), params=params, files={
            "file": (file.filename or "frame.jpg", content, file.content_type or "image/jpeg")
        })
    if rf_resp.status_code != 200:
        raise HTTPException(rf_resp.status_code, rf_resp.text)
    rf_json = rf_resp.json()
    preds = rf_json.get("predictions", [])

    # 2) Local MiDaS depth
    depth_map = _run_depth(img)

    # 3) Attach distances to detections
    out = []
    for p in preds:
        cx = float(p["x"]); cy = float(p["y"])
        w  = float(p["width"]); h = float(p["height"])
        tlwh = _center_to_tlwh(cx, cy, w, h)
        dnorm = _median_in_box(depth_map, tlwh)
        meters = _norm_to_meters(dnorm, min_m=min_m, max_m=max_m, scale=scale)

        out.append({
            "class_id": p.get("class_id", -1),
            "class_name": p.get("class", "object"),
            "conf": float(p.get("confidence", 0.0)),
            "box": {"x": tlwh[0], "y": tlwh[1], "w": tlwh[2], "h": tlwh[3]},
            "distance_m": meters
        })

    return {"detections": out, "image_b64": None}
