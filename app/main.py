# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.routers import infer

app = FastAPI(
    title="BlindSpot Roboflow Proxy",
    description="Forwards images to Roboflow Hosted API and returns JSON detections.",
    version="1.0.0",
    docs_url="/docs",         # Swagger UI
    redoc_url="/redoc"        # ReDoc (optional)
)

# CORS — open for now; tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    # send people straight to Swagger
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["meta"])
def health():
    return {"ok": True}

# /api/infer is defined in the router
app.include_router(infer.router, prefix="/api")
