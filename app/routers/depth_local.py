# app/routers/depth_local.py
"""
Depth inference router using Depth Anything V2 (replacing MiDaS small).

Endpoints (backward-compatible):
- GET  /api/depth_info
- POST /api/infer_depth_local   (multipart form 'file')
"""
import io, sys, time, traceback
from pathlib import Path

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import torch

router = APIRouter(tags=["Depth"])

# Where the Dockerfile places code/weights
VENDOR_DIR = Path("/app/vendor/Depth-Anything-V2")
CHECKPOINT = Path("/app/checkpoints/depth_anything_v2_vits.pth")  # Small (ViT-S)
ENCODER = "vits"  # 'vits' | 'vitb' | 'vitl'

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_da2 = None

def _load_da2():
    """Load Depth Anything V2 from vendored repo + local checkpoint."""
    global _da2
    if _da2 is not None:
        return
    if not VENDOR_DIR.exists():
        raise RuntimeError(f"Depth-Anything-V2 code not found at {VENDOR_DIR}")
    if not CHECKPOINT.exists():
        raise RuntimeError(f"DAv2 checkpoint not found at {CHECKPOINT}")

    sys.path.insert(0, str(VENDOR_DIR))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as e:
        raise RuntimeError(f"Failed to import Depth Anything V2 modules: {e}")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    print("🔹 Loading Depth Anything V2 (small)...", flush=True)
    model = DepthAnythingV2(**model_configs[ENCODER])
    state = torch.load(str(CHECKPOINT), map_location="cpu")
    model.load_state_dict(state)
    globals()["_da2"] = model.to(_device).eval()

def _run_depth(pil_image: Image.Image) -> np.ndarray:
    """Run DAv2 and return a [0,1] normalized depth map (1=near)."""
    _load_da2()
    # DAv2 expects numpy BGR (OpenCV style). Convert PIL RGB -> BGR.
    arr = np.array(pil_image.convert("RGB"))[:, :, ::-1].copy()
    with torch.no_grad():
        depth = _da2.infer_image(arr)  # HxW float numpy
    depth = depth.astype(np.float32)
    # normalize like your MiDaS path: near=1, far=0
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

@router.get("/api/depth_info")
def depth_info():
    try:
        _load_da2()
        return {"ok": True, "device": str(_device)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.post("/api/infer_depth_local")
async def infer_depth_local(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    try:
        t0 = time.time()
        depth_map = _run_depth(image)
        dt_ms = round((time.time() - t0) * 1000.0, 2)
        H, W = depth_map.shape[:2]
        small = depth_map[:: max(1, H // 64), :: max(1, W // 64)]
        return {
            "status": "ok",
            "inference_time_ms": dt_ms,
            "depth_summary": {
                "H": H, "W": W,
                "min": float(depth_map.min()),
                "max": float(depth_map.max()),
                "median": float(np.median(depth_map)),
                "grid_shape": small.shape,
            },
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")
