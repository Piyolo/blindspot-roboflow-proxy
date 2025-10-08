# Dockerfile
FROM python:3.11-slim

# minimal tools to fetch weights
RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- install deps (CPU torch only) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- app code (includes vendor/ and scripts/) ---
COPY . .

# --- fetch DAv2 weight once at build time ---
RUN bash scripts/fetch_da2_weights.sh

# --- runtime env (low memory) ---
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    WEB_CONCURRENCY=1 \
    UVICORN_WORKERS=1 \
    PORT=10000

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","10000"]
