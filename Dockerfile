# Multi-stage build for lightweight, non-root ONNX inference API
FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements-serving.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_TYPE=distilbert_onnx \
    MODEL_DIR=/app/artifacts/distilbert_sst2_int8 \
    MODEL_VERSION=1.0.0

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --shell /bin/false --create-home appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ /app/app/
COPY artifacts/distilbert_sst2_int8/ /app/artifacts/distilbert_sst2_int8/

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

