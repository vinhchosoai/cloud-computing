"""FastAPI entrypoint for the AI text classification inference service."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.drift import DriftMonitor
from app.inference import ONNXInferenceEngine
from app.metrics import (
    ACTIVE_REQUESTS,
    BATCH_SIZE_HISTOGRAM,
    DRIFT_DETECTED_GAUGE,
    DRIFT_PSI_GAUGE,
    INPUT_LENGTH_HISTOGRAM,
    MODEL_INFERENCE_LATENCY,
    MODEL_LOAD_SECONDS,
    PREDICTION_CONFIDENCE,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
)
from app.schemas import (
    DriftStatusResponse,
    HealthResponse,
    PredictionItem,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    VersionResponse,
)
from app.security import (
    get_dynamic_rate_limit,
    hash_text_for_log,
    limiter,
    verify_api_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("inference-api")

engine: ONNXInferenceEngine | None = None
drift_monitor: DriftMonitor | None = None


def init_runtime() -> None:
    global engine, drift_monitor
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    engine = ONNXInferenceEngine(
        model_dir=model_dir,
        model_type=settings.model_type,
        expected_sha256=settings.expected_sha256,
        max_token_length=settings.max_token_length,
    )
    engine.load()
    MODEL_LOAD_SECONDS.set(engine.load_seconds)
    logger.info(
        "Model loaded: type=%s version=%s load_s=%.3f sha256=%s",
        settings.model_type,
        settings.model_version,
        engine.load_seconds,
        engine.artifact_sha256[:12],
    )

    ref_path = model_dir / "reference_stats.json"
    drift_monitor = DriftMonitor(
        reference_path=ref_path if ref_path.exists() else None,
        window_size=settings.drift_window_size,
        threshold=settings.drift_psi_threshold,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_runtime()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Failed to load model on startup: %s", exc)
    yield


app = FastAPI(
    title="AI Sentiment Inference Service",
    version=get_settings().model_version,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    REQUESTS_TOTAL.labels(endpoint="/health", status="200").inc()
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(response: Response) -> ReadyResponse:
    loaded = engine is not None and engine.is_loaded
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        REQUESTS_TOTAL.labels(endpoint="/ready", status="503").inc()
        return ReadyResponse(status="not_ready", model_loaded=False, model_load_seconds=None)
    REQUESTS_TOTAL.labels(endpoint="/ready", status="200").inc()
    return ReadyResponse(
        status="ready",
        model_loaded=True,
        model_load_seconds=round(engine.load_seconds, 4),
    )


@app.get("/version", response_model=VersionResponse, tags=["ops"])
def version() -> VersionResponse:
    settings = get_settings()
    sha = engine.artifact_sha256 if engine and engine.is_loaded else "unloaded"
    REQUESTS_TOTAL.labels(endpoint="/version", status="200").inc()
    return VersionResponse(
        model_name=settings.model_name,
        model_type=settings.model_type,
        model_version=settings.model_version,
        artifact_sha256=sha,
        git_commit=settings.git_commit,
    )


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    "/v1/drift",
    response_model=DriftStatusResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["monitoring"],
)
def drift_status() -> DriftStatusResponse:
    if drift_monitor is None:
        raise HTTPException(status_code=503, detail="Drift monitor not initialized")
    return DriftStatusResponse(**drift_monitor.status())


@app.post(
    "/v1/predict",
    response_model=PredictResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["inference"],
)
@limiter.limit(get_dynamic_rate_limit)
def predict(request: Request, payload: PredictRequest) -> PredictResponse:
    t_start = time.perf_counter()
    if engine is None or not engine.is_loaded:
        REQUESTS_TOTAL.labels(endpoint="/v1/predict", status="503").inc()
        raise HTTPException(status_code=503, detail="Model not ready")

    settings = get_settings()
    request_id = str(uuid.uuid4())
    batch_size = len(payload.texts)

    ACTIVE_REQUESTS.inc()
    try:
        t_inf = time.perf_counter()
        raw_preds = engine.predict(payload.texts)
        inf_dur = time.perf_counter() - t_inf
        MODEL_INFERENCE_LATENCY.observe(inf_dur)
    finally:
        ACTIVE_REQUESTS.dec()

    items: list[PredictionItem] = []
    lengths: list[int] = []
    confs: list[float] = []
    input_hashes: list[str] = []

    for text, (label, score) in zip(payload.texts, raw_preds):
        items.append(PredictionItem(label=label, score=round(score, 6)))
        tlen = len(text)
        lengths.append(tlen)
        confs.append(score)
        input_hashes.append(hash_text_for_log(text))
        INPUT_LENGTH_HISTOGRAM.observe(tlen)
        PREDICTION_CONFIDENCE.observe(score)

    BATCH_SIZE_HISTOGRAM.observe(batch_size)

    if drift_monitor is not None:
        _, _, max_psi = drift_monitor.record(lengths, confs)
        DRIFT_PSI_GAUGE.set(max_psi)
        DRIFT_DETECTED_GAUGE.set(1.0 if max_psi > settings.drift_psi_threshold else 0.0)

    total_dur = time.perf_counter() - t_start
    REQUEST_LATENCY.labels(endpoint="/v1/predict", status="200").observe(total_dur)
    REQUESTS_TOTAL.labels(endpoint="/v1/predict", status="200").inc()

    # Privacy-safe structured log: only hashes and lengths, NEVER raw text
    logger.info(
        "predict request_id=%s batch=%d hashes=%s lengths=%s latency_ms=%.2f",
        request_id,
        batch_size,
        ",".join(input_hashes[:3]),
        lengths[:3],
        total_dur * 1000,
    )

    return PredictResponse(
        request_id=request_id,
        model_name=settings.model_name,
        model_version=settings.model_version,
        batch_size=batch_size,
        latency_ms=round(total_dur * 1000, 3),
        predictions=items,
    )
