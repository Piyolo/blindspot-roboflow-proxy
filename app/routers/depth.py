# app/routers/depth.py
import io, os
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import httpx

router = APIRouter(tags=["inference"])

# ---- Roboflow env ----
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")
ROBOFLOW_MODEL_VERSION = os.getenv("ROBOFLOW_MODEL_VERSION", "1")

def rf_url():
    if not (ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID):
        raise HTTPException(500, "Missing Roboflow env")
    return f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}/{ROBOFLOW_MODEL_VERSION}"

# ---- Load MiDaS (small) once
_midas = None
_transform = None
_device = torch.device("cpu")

def _load_midas():
    global _midas, _transform
    if _midas is not None: 
        return
    # Small, faster backbone
    model_type = "MiDaS_small"  # alternatives: DPT_Large / DPT_Hybrid (slower, better)
    _midas = torch.hub.load("intel-isl/MiDaS", model_type)
    _midas.eval().to(_device)
    _transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def _infer_depth_pil(img_pil: Image.Image) -> np.ndarray:
    _load_midas()
    # to RGB and tensor
    img_rgb = img_pil.convert("RGB")
    input_batch = _transform(img_rgb).to(_device)
    with torch.no_grad():
        prediction = _midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.size[::-1],  # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    # MiDaS: higher = *closer* (relative). Normalize for stability.
    d = prediction.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)  # 0..1 (1 ~ closest)
    return d  # HxW in [0,1]

def _bbox_to_tlwh(center_x, center_y, width, height):
    # Roboflow returns center x/y + w/h
    left = center_x - width/2.0
    top  = center_y - height/2.0
    return int(left), int(top), int(width), int(height)

def _median_depth_in_box(depth_map: np.ndarray, box):
    h, w = depth_map.shape
    x, y, bw, bh = box
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(w, x + bw); y1 = min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    region = depth_map[y0:y1, x0:x1]
    if region.size == 0: 
        return None
    return float(np.median(region))

def _depth_to_meters(depth_norm: float, scale: float, min_m: float, max_m: float):
    """
    Convert normalized depth (1=near, 0=far) to meters.
    Simple linear mapping: meters = min_m + (1 - depth_norm) * (max_m - min_m)
    Then apply scale factor for quick tuning.
    """
    meters = min_m + (1.0 - depth_norm) * (max_m - min_m)
    return max(0.05, meters * scale)

@router.post("/infer_depth")
async def infer_depth(
    file: UploadFile = File(...),
    confidence: float = 0.4,
    overlap: float = 0.5,
    # calibration knobs (tweak in requests without redeploy)
    min_m: float = 0.3,
    max_m: float = 5.0,
    scale: float = 1.0,
):
    # Read image
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Unsupported image")

    # 1) Roboflow detections
    rf_params = {
        "api_key": ROBOFLOW_API_KEY,
        "confidence": confidence,
        "overlap": overlap,
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=45) as client:
        rf_resp = await client.post(rf_url(), params=rf_params, files={
            "file": (file.filename or "frame.jpg", content, file.content_type or "image/jpeg")
        })
    if rf_resp.status_code != 200:
        raise HTTPException(rf_resp.status_code, rf_resp.text)
    rf_json = rf_resp.json()
    preds = rf_json.get("predictions", [])

    # 2) MiDaS depth
    depth_map = _infer_depth_pil(img)  # HxW in [0,1]

    # 3) For each detection, sample depth and convert to meters
    out_dets = []
    for p in preds:
        cx = float(p["x"]); cy = float(p["y"])
        w  = float(p["width"]); h = float(p["height"])
        tlwh = _bbox_to_tlwh(cx, cy, w, h)
        dnorm = _median_depth_in_box(depth_map, tlwh)
        if dnorm is None:
            meters = None
        else:
            meters = _depth_to_meters(dnorm, scale=scale, min_m=min_m, max_m=max_m)

        out_dets.append({
            "class_id": p.get("class_id", -1),
            "class_name": p.get("class", "object"),
            "conf": float(p.get("confidence", 0.0)),
            "box": { "x": tlwh[0], "y": tlwh[1], "w": tlwh[2], "h": tlwh[3] },
            "distance_m": meters
        })

    return {
        "time_ms": 0.0,
        "detections": out_dets,
        "image_b64": None
    }
