# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.routers import infer, depth_local, depth  # always enabled

app = FastAPI(
    title="BlindSpot Inference API",
    description="Roboflow proxy + local depth endpoints",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def warmup():
    """
    Prefer Depth Anything v2 warmup if available; otherwise fall back to MiDaS.
    If warmup fails, the depth router will lazy-load on first request.
    """
    # Pick the available warmup function without breaking either setup
    _warm = None
    try:
        from app.routers.depth_local import _load_da2 as _warm  # DAv2 path
    except Exception:
        try:
            from app.routers.depth_local import _load_midas as _warm  # legacy MiDaS
        except Exception as e:
            print(f"⚠️ No depth warmup function found: {e}", flush=True)
            return

    try:
        _warm()
        print("✅ Depth warmup OK", flush=True)
    except Exception as e:
        print(f"⚠️ Depth warmup failed (will lazy-load on first request): {e}", flush=True)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["meta"])
def health():
    return {"ok": True}

# Always mount routers
app.include_router(infer.router, prefix="/api")   # Roboflow proxy
app.include_router(depth_local.router)            # /api/depth_info, /api/infer_depth_local
app.include_router(depth.router)                  # /api/infer_depth
