# BlindSpot Roboflow Proxy (Slim)
Fast minimal proxy that forwards images to Roboflow Detect and returns JSON (no local depth).

## Environment
- `ROBOFLOW_API_KEY` (required)
- `ROBOFLOW_MODEL_ID` (required)  e.g. `workspace/project` or just `project`
- `ROBOFLOW_MODEL_VERSION` (optional, default `1`)

## Run locally
```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=rf_xxx
export ROBOFLOW_MODEL_ID=workspace/project
uvicorn app.main:app --reload --port 8000
```

## API
- `GET /health` → `{ "ok": true }`
- `GET /api/info` → model/version and candidate endpoint shapes
- `POST /api/infer` (multipart `file`) → Roboflow JSON

## Deploy on Render
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Env vars: set ROBOFLOW_* above
