FROM python:3.11-slim

# --- NEW: tools MiDaS needs ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional: a directory to cache/save weights
ENV TORCH_HOME=/app/.cache/torch
RUN mkdir -p /app/.cache/torch /app/models

COPY . .
ENV PORT=8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
