# app/routers/depth_local.py
import io, os, time, traceback
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException

import torch
import torch.nn.functional as F

router = APIRouter(tags=["Depth"])

_midas = None
_transform = None
_device = torch.device("cpu")

def _load_midas():
    global _midas, _transform
    if _midas is not None:
        return
    try:
        print("🔹 Loading MiDaS_small via torch.hub ...", flush=True)
        _midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        _midas.to(_device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _transform = transforms.small_transform
        print("✅ MiDaS loaded.", flush=True)
    except Exception as e:
        print("❌ MiDaS load failed:", e, flush=True)
        traceback.print_exc()
        raise

def _run_depth(pil_image: Image.Image):
    _load_midas()
    img = pil_image.convert("RGB")
    inp = _transform(img).to(_device)
    with torch.no_grad():
        pred = _midas(inp)
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    d = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return d

@router.get("/api/depth_info")
def depth_info():
    try:
        _load_midas()
        return {"ok": True, "device": str(_device)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.post("/api/infer_depth_local")
async def infer_depth_local(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {e}")

    try:
        t0 = time.time()
        depth_map = _run_depth(image)
        dt = round((time.time() - t0) * 1000, 2)
        return {
            "status": "ok",
            "inference_time_ms": dt,
            "median_norm_depth": float(np.median(depth_map)),
            "near_norm_depth": float(np.percentile(depth_map, 90)),
            "far_norm_depth": float(np.percentile(depth_map, 10)),
        }
    except Exception as e:
        # surface the real reason to Swagger and Render logs
        err = f"{type(e).__name__}: {e}"
        print("❌ Depth error:", err, flush=True)
        traceback.print_exc()
        raise HTTPException(500, err)
