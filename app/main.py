# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.routers import infer

app = FastAPI(
    title="BlindSpot Roboflow Proxy (Slim)",
    description="Forwards frames to Roboflow Detect; no local depth.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["meta"])
def health():
    return {"ok": True}

app.include_router(infer.router, prefix="/api")
