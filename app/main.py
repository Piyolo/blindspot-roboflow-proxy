from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import infer

app = FastAPI(title="BlindSpot Roboflow Proxy")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

app.include_router(infer.router, prefix="/api")
