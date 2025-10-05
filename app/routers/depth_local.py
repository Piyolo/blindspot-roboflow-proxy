# app/routers/depth_local.py
import io, sys, time, traceback
from pathlib import Path
import numpy as np
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
    """Load MiDaS_small from vendored repo + local weights (no network)."""
    global _midas, _transform
    if _midas is not None:
        return

    if not VENDOR_DIR.exists():
        raise RuntimeError(f"MiDaS code not found: {VENDOR_DIR}")
    if not WEIGHTS.exists():
        raise RuntimeError(f"MiDaS weights not found: {WEIGHTS}")

    sys.path.insert(0, str(VENDOR_DIR))
    try:
        # Repo layout differs by commit; try both import paths
        try:
            from midas.models.midas_net import MidasNet_small
        except ImportError:
            from midas.midas_net import MidasNet_small

        from midas.transforms import Resize, NormalizeImage, PrepareForNet
        import torchvision.transforms as T
    except Exception as e:
        raise RuntimeError(f"Failed to import MiDaS local modules: {e}")

    print("🔹 Loading local MiDaS_small...", flush=True)
    model = MidasNet_small(
        str(WEIGHTS),
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True,
    )
    model.to(_device).eval()
    _midas = model

    _transform = T.Compose([
        Resize(
            256, 256,
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
    data = _transform({"image": np.array(img)})
    inp = torch.from_numpy(data["image"]).unsqueeze(0).to(_device)  # 1xCxHxW
    with torch.no_grad():
        pred = _midas(inp)  # 1x1xH'xW'
        pred = F.interpolate(
            pred,
            size=(img.height, img.width),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    d = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # [0,1], 1=near
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
