# app/routers/depth_local.py
import io, os, sys, time, traceback
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

import torch
import torch.nn.functional as F

router = APIRouter(tags=["Depth"])

VENDOR_DIR = Path("/app/vendor/MiDaS")
WEIGHTS = Path("/app/models/midas_v21_small-70d6b9c8.pt")

_midas = None
_transform = None
_device = torch.device("cpu")

def _load_midas():
    """
    Load MiDaS from the local vendor checkout and local weights file,
    so no GitHub/torch.hub calls are made at runtime.
    """
    global _midas, _transform
    if _midas is not None:
        return
    if not VENDOR_DIR.exists():
        raise RuntimeError(f"MiDaS code not found at {VENDOR_DIR}")
    if not WEIGHTS.exists():
        raise RuntimeError(f"MiDaS weights not found at {WEIGHTS}")

    sys.path.insert(0, str(VENDOR_DIR))  # import local MiDaS package
    try:
        from midas.models.midas_net import MidasNet_small
        from midas.transforms import Resize, NormalizeImage, PrepareForNet
        import torchvision.transforms as T
    except Exception as e:
        raise RuntimeError(f"Failed to import local MiDaS: {e}")

    print("🔹 Loading local MiDaS_small...", flush=True)
    model = MidasNet_small(
        str(WEIGHTS),
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True
    )
    model.to(_device).eval()
    _midas = model

    # same transform used by MiDaS small
    _transform = T.Compose([
        Resize(
            256, 256,  # keep small for CPU; adjust to 384 for a bit more detail
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method="bicubic",
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    print("✅ MiDaS_small ready.", flush=True)

def _run_depth(pil_image: Image.Image) -> np.ndarray:
    _load_midas()
    img = pil_image.convert("RGB")
    inp = _transform({"image": np.array(img)})["image"]  # HxWxC -> CxHxW float32
    inp = torch.from_numpy(inp).unsqueeze(0).to(_device)

    with torch.no_grad():
        pred = _midas.forward(inp)  # shape: 1x1xH'xW'
        pred = F.interpolate(
            pred,
            size=(img.height, img.width),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Normalize to [0,1] (1 = near)
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
        raise HTTPException(400, f"Invalid image: {e}")

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
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")
