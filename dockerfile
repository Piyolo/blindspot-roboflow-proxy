FROM python:3.11-slim

# --- tools MiDaS needs (git+wget) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- vendor MiDaS code locally & fetch weights (NO runtime GitHub calls) ---
RUN mkdir -p /app/vendor && \
    git clone --depth 1 https://github.com/isl-org/MiDaS.git /app/vendor/MiDaS && \
    mkdir -p /app/models && \
    wget -O /app/models/midas_v21_small-70d6b9c8.pt \
      https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small-70d6b9c8.pt


# app code
COPY . .
ENV PORT=8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
