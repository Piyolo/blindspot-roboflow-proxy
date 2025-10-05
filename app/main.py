# app/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.routers import infer  # keep
# from app.routers import depth_local, depth  # <-- keep code but don't import when disabled

ENABLE_DEPTH = os.getenv("ENABLE_DEPTH", "false").lower() in ("1", "true", "yes")

app = FastAPI(
    title="BlindSpot Inference API",
    description="Roboflow Hosted API proxy (depth temporarily disabled).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["meta"])
def health():
    return {"ok": True, "depth_enabled": ENABLE_DEPTH}

# Always mount detections
app.include_router(infer.router, prefix="/api")

# Mount depth only if explicitly enabled
if ENABLE_DEPTH:
    from app.routers import depth_local, depth  # import here so it doesn't load otherwise
    @app.on_event("startup")
    async def warmup():
        try:
            from app.routers.depth_local import _load_midas
            _load_midas()
        except Exception as e:
            print("⚠️ MiDaS warmup failed (will retry on first request):", e, flush=True)
    app.include_router(depth_local.router)  # /api/depth_info, /api/infer_depth_local
    app.include_router(depth.router)        # /api/infer_depth
