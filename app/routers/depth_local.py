import io, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import torch

router = APIRouter(tags=["Depth"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR   = PROJECT_ROOT / "vendor" / "Depth-Anything-V2" / "depth_anything_v2"
CHECKPOINT   = PROJECT_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"
ENCODER      = os.getenv("DA2_ENCODER","vits").lower()
MAX_SIDE     = int(os.getenv("DA2_MAX_SIDE","320"))  # start small on free tier

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try: torch.set_num_threads(1)
except: pass
_da2 = None

def _assert_layout():
    if not VENDOR_DIR.exists():
        raise RuntimeError(f"Missing package at {VENDOR_DIR} (upload depth_anything_v2/ here)")
    if not CHECKPOINT.exists():
        raise RuntimeError(f"Missing checkpoint {CHECKPOINT} (fetched at *build* time)")

def _load_da2():
    global _da2
    if _da2 is not None: return
    _assert_layout()
    # import vendored package
    pkg_parent = VENDOR_DIR.parent
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))
    from depth_anything_v2.dpt import DepthAnythingV2
    cfg = {
        "vits": {"encoder":"vits","features":64,"out_channels":[48,96,192,384]},
        "vitb": {"encoder":"vitb","features":128,"out_channels":[96,192,384,768]},
        "vitl": {"encoder":"vitl","features":256,"out_channels":[256,512,1024,1024]},
    }[ENCODER]
    state = torch.load(str(CHECKPOINT), map_location="cpu")
    model = DepthAnythingV2(**cfg); model.load_state_dict(state)
    _da2 = model.to(_device).eval()

def _resize(pil, max_side):
    w,h = pil.size; s = max(w,h)
    if s <= max_side: return pil
    r = max_side/float(s); return pil.resize((max(1,int(w*r)), max(1,int(h*r))), Image.BILINEAR)

def _run_depth(pil_img):
    _load_da2()
    img = _resize(pil_img.convert("RGB"), MAX_SIDE)
    arr = np.array(img)[:, :, ::-1].copy()    # RGB->BGR
    try:
        depth = _da2.infer_image(arr)         # HxW float
    except AttributeError:
        x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()/255.0
        with torch.no_grad(): depth = _da2(x.to(_device)).squeeze().cpu().numpy()
    d = depth.astype(np.float32)
    mn, mx = float(d.min()), float(d.max())
    return np.zeros_like(d) if mx-mn<1e-9 else (d-mn)/(mx-mn)

@router.get("/api/depth_info")
def depth_info():
    try:
        _load_da2()
        return {"ok": True, "device": str(_device),
                "vendor_dir": str(VENDOR_DIR.parent),
                "checkpoint": str(CHECKPOINT),
                "encoder": ENCODER, "max_side": MAX_SIDE,
                "omp": os.environ.get("OMP_NUM_THREADS"),
                "mkl": os.environ.get("MKL_NUM_THREADS")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@router.post("/api/infer_depth_local")
async def infer_depth_local(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    try:
        t0 = time.time(); d = _run_depth(img); ms = round((time.time()-t0)*1000,2)
        H,W = d.shape; sh=max(1,H//64); sw=max(1,W//64)
        return {"status":"ok","inference_time_ms":ms,
                "depth_summary":{"H":H,"W":W,"min":float(d.min()),
                                 "max":float(d.max()),"median":float(np.median(d)),
                                 "grid_shape": d[::sh,::sw].shape}}
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500, f"{type(e).__name__}: {e}")
