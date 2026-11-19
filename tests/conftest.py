"""Shared test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.config import get_settings
from app.drift import DriftMonitor


class DummyEngine:
    def __init__(self) -> None:
        self.is_loaded = True
        self.load_seconds = 0.012
        self.artifact_sha256 = "deadbeef" * 8

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for t in texts:
            if "bad" in t.lower() or "terrible" in t.lower():
                out.append(("NEGATIVE", 0.92))
            else:
                out.append(("POSITIVE", 0.96))
        return out


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("API_KEYS", "test-key-1")
    monkeypatch.setenv("BENCHMARK_API_KEY", "bench-key-1")
    monkeypatch.setenv("RATE_LIMIT", "5/minute")
    monkeypatch.setenv("BENCHMARK_RATE_LIMIT", "1000/minute")
    get_settings.cache_clear()

    # Reset rate limiter storage between tests
    app_main.limiter.reset()
    app_main.engine = DummyEngine()
    app_main.drift_monitor = DriftMonitor(reference_path=None, window_size=50, threshold=0.20)

    with TestClient(app_main.app, raise_server_exceptions=False) as c:
        # Override lifespan-loaded runtime with dummy engine for fast isolated unit tests
        app_main.engine = DummyEngine()
        app_main.drift_monitor = DriftMonitor(reference_path=None, window_size=50, threshold=0.20)
        yield c
    get_settings.cache_clear()

