import io, os, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(tags=["Depth"])

# -------- MiDaS setup --------
_midas = None
_transform = None
_device = torch.device("cpu")

def _load_midas():
    """Load MiDaS_small model (CPU friendly)."""
    global _midas, _transform
    if _midas is not None:
        return
    print("🔹 Loading MiDaS model...")
    _midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    _midas.to(_device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    _transform = transforms.small_transform

def _run_depth(pil_image: Image.Image):
    """Run MiDaS and return normalized depth map."""
    _load_midas()
    img = pil_image.convert("RGB")
    inp = _transform(img).to(_device)
    with torch.no_grad():
        pred = _midas(inp)
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=img.size[::-1],  # (H,W)
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    d = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return d

def _depth_to_meters(norm_val: float, min_m=0.3, max_m=5.0):
    """Map normalized depth (1=near) → meters."""
    return min_m + (1.0 - norm_val) * (max_m - min_m)

# -------- Endpoint --------
@router.post("/api/infer_depth_local")
async def infer_depth_local(file: UploadFile = File(...)):
    """Runs MiDaS locally on uploaded image and returns estimated distances."""
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Invalid image file")

    start = time.time()
    depth_map = _run_depth(image)
    duration = round((time.time() - start) * 1000, 2)

    # Compute simple stats for sanity
    median_depth = float(np.median(depth_map))
    near_depth = float(np.percentile(depth_map, 90))
    far_depth = float(np.percentile(depth_map, 10))

    return {
        "status": "ok",
        "inference_time_ms": duration,
        "median_norm_depth": median_depth,
        "near_norm_depth": near_depth,
        "far_norm_depth": far_depth,
        "message": "Depth map computed successfully!"
    }
