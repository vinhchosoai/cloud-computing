"""Unit tests for API endpoints, schema validation, auth, and rate limiting."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_and_ready_and_version(client: TestClient) -> None:
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok"}

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    assert r_ready.json()["model_loaded"] is True

    r_ver = client.get("/version")
    assert r_ver.status_code == 200
    assert "artifact_sha256" in r_ver.json()

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "inference_requests_total" in r_metrics.text


def test_auth_required_and_invalid_key(client: TestClient) -> None:
    r_missing = client.post("/v1/predict", json={"texts": ["great movie"]})
    assert r_missing.status_code == 401

    r_bad = client.post(
        "/v1/predict",
        headers={"X-API-Key": "wrong-key"},
        json={"texts": ["great movie"]},
    )
    assert r_bad.status_code == 403


def test_predict_single_and_batch(client: TestClient) -> None:
    r = client.post(
        "/v1/predict",
        headers={"X-API-Key": "test-key-1"},
        json={"texts": ["This movie is great!", "Terrible waste of time."]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["batch_size"] == 2
    assert data["predictions"][0]["label"] == "POSITIVE"
    assert data["predictions"][1]["label"] == "NEGATIVE"


def test_schema_validation_errors(client: TestClient) -> None:
    headers = {"X-API-Key": "test-key-1"}

    # Empty list
    assert client.post("/v1/predict", headers=headers, json={"texts": []}).status_code == 422

    # Whitespace-only text
    assert client.post("/v1/predict", headers=headers, json={"texts": ["   "]}).status_code == 422

    # Too long text (> 512 chars)
    assert (
        client.post("/v1/predict", headers=headers, json={"texts": ["a" * 513]}).status_code == 422
    )

    # Batch size > 32
    assert (
        client.post("/v1/predict", headers=headers, json={"texts": ["ok"] * 33}).status_code == 422
    )


def test_rate_limiting(client: TestClient) -> None:
    headers = {"X-API-Key": "test-key-1"}
    statuses = [
        client.post("/v1/predict", headers=headers, json={"texts": ["good"]}).status_code
        for _ in range(7)
    ]
    assert statuses[:5] == [200, 200, 200, 200, 200]
    assert 429 in statuses[5:]

