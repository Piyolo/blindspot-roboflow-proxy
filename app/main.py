from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.routers import infer, depth_local

app = FastAPI(
    title="BlindSpot Roboflow Proxy",
    description="Detections + Local MiDaS depth",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def warmup():
    # try to preload MiDaS when the container boots
    try:
        from app.routers.depth_local import _load_midas
        _load_midas()
    except Exception as e:
        print("⚠️ MiDaS warmup failed (will retry on first request):", e, flush=True)

@app.get("/", include_in_schema=False)
def root(): return RedirectResponse(url="/docs")

@app.get("/health", tags=["meta"])
def health(): return {"ok": True}

app.include_router(infer.router, prefix="/api")
app.include_router(depth_local.router)   # note: depth routes already include /api/...
