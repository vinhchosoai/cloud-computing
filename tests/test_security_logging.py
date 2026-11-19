"""Verify that raw user input text (potential PII) is never written to logs."""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient


def test_no_raw_input_in_logs(client: TestClient, caplog) -> None:
    secret_pii = "SECRET_PII_JOHN_DOE_SSN_999_88_7777"
    with caplog.at_level(logging.INFO):
        resp = client.post(
            "/v1/predict",
            headers={"X-API-Key": "test-key-1"},
            json={"texts": [f"Customer review from {secret_pii}"]},
        )
    assert resp.status_code == 200
    full_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_pii not in full_logs
    assert "hashes=" in full_logs
