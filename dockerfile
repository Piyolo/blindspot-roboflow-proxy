# Slim Dockerfile for Roboflow proxy only (no Torch)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000     WEB_CONCURRENCY=1     UVICORN_WORKERS=1

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","10000"]
