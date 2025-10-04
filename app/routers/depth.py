# app/routers/depth_local.py
import io, os, sys, time
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import httpx

# Torch imports (CPU)
import torch
import torch.nn.functional as F

router = APIRouter(tags=["depth_local"])

# Local model path (create app/models and ensure write permission)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MIDAS_PATH = MODEL_DIR / "midas_small.pt"

# Utility: load MiDaS (try local state_dict, otherwise torch.hub and save)
_midas_model = None
_midas_transform = None
_device = torch.device("cpu")

def load_midas(model_path: Path = LOCAL_MIDAS_PATH):
    global _midas_model, _midas_transform
    if _midas_model is not None:
        return _midas_model, _midas_transform

    # Use torch.hub to fetch model architecture and transforms
    try:
        # This will ensure code for MiDaS and transforms is available
        midas_net = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = transforms.small_transform
    except Exception as e:
        raise RuntimeError(f"Failed to load midas code from torch.hub: {e}")

    # If local weights exist, load them as state_dict (preferred)
    if model_path.exists():
        try:
            state = torch.load(str(model_path), map_location=_device)
            # For some hub models, returned object is already the model
            if isinstance(state, dict):
                midas_net.load_state_dict(state)
            else:
                # state may be a full model object
                midas_net = state
        except Exception as e:
            # if local load fails, continue to let hub download weights automatically
            print("Warning: failed to load local midas weights:", e, file=sys.stderr)

    # If weights not loaded (fresh), let hub handle downloading weights (first-run)
    # torch.hub.load above already downloaded weights into cache when building model
    midas_net.eval().to(_device)
    _midas_model = midas_net
    _midas_transform = transform

    # Try to save state_dict to model_path for future offline use
    try:
        state_dict = _midas_model.state_dict()
        torch.save(state_dict, str(model_path))
        print(f"Saved MiDaS weights to {model_path}", file=sys.stderr)
    except Exception as e:
        print("Warning: could not save midas weights locally:", e, file=sys.stderr)

    return _midas_model, _midas_transform

def infer_depth_map(img_pil: Image.Image) -> np.ndarray:
    """
    Returns a normalized depth map HxW with values in [0,1] where 1 ~ near, 0 ~ far.
    """
    model, transform = load_midas()
    img_rgb = img_pil.convert("RGB")
    inp = transform(img_rgb).unsqueeze(0).to(_device)  # shape 1x3xHxW
    with torch.no_grad():
        pred = model(inp)
        pred = F.interpolate(pred.unsqueeze(1),
                             size=img_rgb.size[::-1],  # PIL size = (W,H) -> we pass (H,W)
                             mode="bicubic",
                             align_corners=False).squeeze().cpu().numpy()
    # normalize (0..1)
    d = pred.astype("float32")
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d  # H x W, values 0..1 (1 = near)

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
    # Simple linear mapping: norm 1 => min_m (near), norm 0 => max_m (far)
    if norm_val is None:
        return None
    meters = min_m + (1.0 - norm_val) * (max_m - min_m)
    return max(0.01, meters * scale)

@router.post("/infer_depth_local")
async def infer_depth_local(
    file: UploadFile = File(...),
    min_m: float = 0.3,
    max_m: float = 6.0,
    scale: float = 1.0,
    # you can still include detection params if you proxy to other detector here
):
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Cannot open image")

    # Run local MiDaS depth
    try:
        depth_map = infer_depth_map(img)  # HxW normalized
    except Exception as e:
        raise HTTPException(500, f"Depth inference error: {e}")

    # If you also want detections: you can run your detector here (Roboflow or local detector).
    # For demonstration we'll *not* run object detector — we will return full-depth map + example stats.
    H, W = depth_map.shape

    # Example: compute coarse distance grid for debugging (downsample)
    small = depth_map[:: max(1, H//64), :: max(1, W//64)]
    summary = {
        "H": H, "W": W,
        "min": float(depth_map.min()), "max": float(depth_map.max()),
        "median": float(np.median(depth_map)),
        "grid_shape": small.shape
    }

    return {
        "time": time.time(),
        "depth_summary": summary,
        # do NOT return the full float array in prod — for debugging only
    }
