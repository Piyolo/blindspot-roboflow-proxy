# app/routers/depth_local.py
"""
Depth inference router using Depth Anything V2 (replacing MiDaS small).

Endpoints (backward-compatible):
- GET  /api/depth_info
- POST /api/infer_depth_local   (multipart form 'file')
"""
import io, os, sys, time, traceback, zipfile, urllib.request, shutil
from pathlib import Path

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import torch

router = APIRouter(tags=["Depth"])

# ── Writable base on Render Native runtime ─────────────────────────────────────
BASE_DIR   = Path(os.getenv("DA2_BASE_DIR", "/var/tmp/da2"))
VENDOR_ROOT = BASE_DIR / "vendor"           # will contain an extracted repo
CKPT_ROOT   = BASE_DIR / "checkpoints"

# pick encoder via env if needed: vits | vitb | vitl
ENCODER = os.getenv("DA2_ENCODER", "vits").lower()

# Checkpoint names/URLs
HF_URLS = {
    "vits": ("depth_anything_v2_vits.pth",
             "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"),
    "vitb": ("depth_anything_v2_vitb.pth",
             "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"),
    "vitl": ("depth_anything_v2_vitl.pth",
             "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"),
}
CKPT_NAME, CKPT_URL = HF_URLS.get(ENCODER, HF_URLS["vits"])
CHECKPOINT = CKPT_ROOT / CKPT_NAME

# Github ZIP (no git needed)
GITHUB_ZIP = "https://codeload.github.com/DepthAnything/Depth-Anything-V2/zip/refs/heads/main"

# Torch device
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loaded model singleton
_da2 = None
_vendor_module_parent = None  # path to put on sys.path

def _download(url: str, dst: Path, desc: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"⬇️  Downloading {desc} … {url}", flush=True)
        urllib.request.urlretrieve(url, dst)
    except Exception as e:
        raise RuntimeError(f"Failed to download {desc} from {url}: {e}")

def _ensure_vendor_repo() -> Path:
    """
    Ensures the Depth-Anything-V2 repo is available under VENDOR_ROOT.
    Returns the directory that contains the 'depth_anything_v2' package.
    """
    # If already extracted, locate the package folder
    if VENDOR_ROOT.exists():
        for p in VENDOR_ROOT.glob("Depth-Anything-V2-*"):
            if (p / "depth_anything_v2").exists():
                return p

    # Otherwise download+extract
    tmp_zip = BASE_DIR / "tmp" / "da2.zip"
    _download(GITHUB_ZIP, tmp_zip, "Depth-Anything-V2 repo ZIP")
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        extract_dir = VENDOR_ROOT
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_dir)

    for p in VENDOR_ROOT.glob("Depth-Anything-V2-*"):
        if (p / "depth_anything_v2").exists():
            return p

    raise RuntimeError("Depth-Anything-V2 zip extracted, but module folder not found.")

def _ensure_checkpoint() -> Path:
    if not CHECKPOINT.exists():
        _download(CKPT_URL, CHECKPOINT, f"DAv2 {ENCODER.upper()} checkpoint")
        print(f"✅ Checkpoint ready at {CHECKPOINT}", flush=True)
    return CHECKPOINT

def _load_da2():
    """Load Depth Anything V2; downloads vendor + checkpoint into /var/tmp if missing."""
    global _da2, _vendor_module_parent
    if _da2 is not None:
        return

    vendor_repo_root = _ensure_vendor_repo()
    _vendor_module_parent = vendor_repo_root  # parent that contains 'depth_anything_v2'
    if str(_vendor_module_parent) not in sys.path:
        sys.path.insert(0, str(_vendor_module_parent))

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as e:
        raise RuntimeError(f"Failed to import Depth Anything V2 modules: {e}")

    ckpt = _ensure_checkpoint()

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    cfg = model_configs.get(ENCODER)
    if cfg is None:
        raise RuntimeError(f"Unsupported ENCODER '{ENCODER}'")

    print(f"🔹 Loading Depth Anything V2 ({ENCODER}) …", flush=True)
    state = torch.load(str(ckpt), map_location="cpu")
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(state)
    _da2 = model.to(_device).eval()
    print("✅ DAv2 loaded", flush=True)

def _run_depth(pil_image: Image.Image) -> np.ndarray:
    """Run DAv2 and return a [0,1] normalized depth map (1=near)."""
    _load_da2()

    # DAv2 expects ndarray BGR
    arr = np.array(pil_image.convert("RGB"))[:, :, ::-1].copy()

    # Preferred path: use model's helper if available
    try:
        depth = _da2.infer_image(arr)  # HxW float numpy
    except AttributeError:
        # Fallback: try direct forward (repo usually ships an infer helper; this is a best-effort)
        # If your repo revision lacks infer_image, we can wire transforms explicitly if needed.
        import torch.nn.functional as F
        x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float() / 255.0
        x = x.to(_device)
        with torch.no_grad():
            pred = _da2(x)            # may require exact preprocessing in some revisions
        depth = pred.squeeze().detach().cpu().numpy()

    depth = depth.astype(np.float32)
    # normalize to 0..1 (near=1)
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
            "base_dir": str(BASE_DIR),
            "vendor_root": str(VENDOR_ROOT),
            "checkpoint": str(CHECKPOINT),
            "encoder": ENCODER,
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
        # lightweight summary to keep response small like your old MiDaS path
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
