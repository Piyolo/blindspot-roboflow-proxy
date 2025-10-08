# app/routers/depth_local.py
"""
Depth inference router using Depth Anything V2 (offline, vendored).
- Loads code from vendor/Depth-Anything-V2/depth_anything_v2
- Loads weights from checkpoints/
- Limits CPU threads and downscales input to avoid OOM on Render
Endpoints:
  GET  /api/depth_info
  POST /api/infer_depth_local
"""
import io, os, sys, time, traceback
from pathlib import Path

# Keep small instances stable
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import torch

router = APIRouter(tags=["Depth"])

# Resolve project paths: <project>/app/routers/depth_local.py -> up two levels -> <project>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR   = PROJECT_ROOT / "vendor" / "Depth-Anything-V2" / "depth_anything_v2"
CHECKPOINT   = PROJECT_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"
ENCODER      = os.getenv("DA2_ENCODER", "vits").lower()   # keep 'vits' on CPU
MAX_SIDE     = int(os.getenv("DA2_MAX_SIDE", "640"))      # downscale longest side

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.set_num_threads(1)
except Exception:
    pass

_da2 = None

def _assert_layout():
    if not VENDOR_DIR.exists():
        raise RuntimeError(
            f"Depth-Anything-V2 package not found at {VENDOR_DIR}.\n"
            f"Upload the *depth_anything_v2* folder here."
        )
    if not CHECKPOINT.exists():
        raise RuntimeError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            f"It will be downloaded during the Render *build* step."
        )

def _load_da2():
    global _da2
    if _da2 is not None:
        return

    _assert_layout()

    # import vendored package
    if str(VENDOR_DIR.parent) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR.parent))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as e:
        raise RuntimeError(f"Failed to import Depth Anything V2: {e}")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    cfg = model_configs.get(ENCODER)
    if cfg is None:
        raise RuntimeError(f"Unsupported ENCODER '{ENCODER}'")

    print(f"🔹 Loading DAv2 ({ENCODER}) from {VENDOR_DIR.parent}", flush=True)
    state = torch.load(str(CHECKPOINT), map_location="cpu")
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(state)
    _da2 = model.to(_device).eval()
    print("✅ DAv2 loaded (offline)", flush=True)

def _resize_keep_ar(pil_img: Image.Image, max_side: int) -> Image.Image:
    w, h = pil_img.size
    s = max(w, h)
    if s <= max_side:
        return pil_img
    scale = max_side / float(s)
    new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
    return pil_img.resize((new_w, new_h), Image.BILINEAR)

def _run_depth(pil_image: Image.Image) -> np.ndarray:
    """Run DAv2 and return a [0,1] normalized depth map (1=near)."""
    _load_da2()

    img_small = _resize_keep_ar(pil_image.convert("RGB"), MAX_SIDE)
    arr = np.array(img_small)[:, :, ::-1].copy()  # RGB -> BGR

    # Preferred helper
    try:
        depth = _da2.infer_image(arr)  # HxW float numpy
    except AttributeError:
        # Fallback minimal forward if infer_image isn't available
        x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float() / 255.0
        x = x.to(_device)
        with torch.no_grad():
            pred = _da2(x)
        depth = pred.squeeze().detach().cpu().numpy()

    depth = depth.astype(np.float32)
    mn, mx = float(depth.min()), float(depth.max())
    if mx - mn < 1e-9:
        return np.zeros_like(depth, dtype=np.float32)
    return (depth - mn) / (mx - mn)

@router.get("/api/depth_info")
def depth_info():
    try:
        _load_da2()
        return {
            "ok": True,
            "device": str(_device),
            "vendor_dir": str(VENDOR_DIR.parent),
            "checkpoint": str(CHECKPOINT),
            "encoder": ENCODER,
            "max_side": MAX_SIDE,
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
        }
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
        stride_h = max(1, H // 64)
        stride_w = max(1, W // 64)
        small = depth_map[::stride_h, ::stride_w]
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
