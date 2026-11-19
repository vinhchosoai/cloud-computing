"""Prometheus metrics definitions for the inference service."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "End-to-end inference request latency in seconds",
    labelnames=("endpoint", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

MODEL_INFERENCE_LATENCY = Histogram(
    "inference_model_duration_seconds",
    "Pure model forward-pass latency in seconds",
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of HTTP requests",
    labelnames=("endpoint", "status"),
)

BATCH_SIZE_HISTOGRAM = Histogram(
    "inference_batch_size",
    "Distribution of request batch sizes",
    buckets=(1, 2, 4, 8, 16, 32),
)

INPUT_LENGTH_HISTOGRAM = Histogram(
    "inference_input_length_chars",
    "Distribution of input text character lengths",
    buckets=(10, 25, 50, 100, 200, 350, 512),
)

PREDICTION_CONFIDENCE = Histogram(
    "inference_prediction_confidence",
    "Distribution of top-class prediction confidence scores",
    buckets=(0.5, 0.65, 0.8, 0.9, 0.95, 0.99, 1.0),
)

MODEL_LOAD_SECONDS = Gauge(
    "inference_model_load_seconds",
    "Time taken to load and verify the ONNX model at startup",
)

DRIFT_PSI_GAUGE = Gauge(
    "inference_drift_psi",
    "Current Population Stability Index (max of length PSI and confidence PSI)",
)

DRIFT_DETECTED_GAUGE = Gauge(
    "inference_drift_detected",
    "1 if current drift PSI exceeds threshold, else 0",
)

ACTIVE_REQUESTS = Gauge(
    "inference_active_requests",
    "Number of inference requests currently in flight",
)

