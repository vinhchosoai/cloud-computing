"""Pydantic request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config import get_settings


class PredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of input texts to classify (1 to 32 items).",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        settings = get_settings()
        if len(values) > settings.max_batch_size:
            raise ValueError(f"Batch size exceeds maximum of {settings.max_batch_size}")
        cleaned: list[str] = []
        for idx, text in enumerate(values):
            stripped = text.strip()
            if not stripped:
                raise ValueError(f"Text at index {idx} must not be empty or whitespace-only")
            if len(stripped) > settings.max_text_length:
                raise ValueError(
                    f"Text at index {idx} exceeds max length of {settings.max_text_length} chars"
                )
            cleaned.append(stripped)
        return cleaned


class PredictionItem(BaseModel):
    label: Literal["POSITIVE", "NEGATIVE"]
    score: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    batch_size: int
    latency_ms: float
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    model_load_seconds: float | None = None


class VersionResponse(BaseModel):
    model_name: str
    model_type: str
    model_version: str
    artifact_sha256: str
    git_commit: str


class DriftStatusResponse(BaseModel):
    window_size: int
    current_samples: int
    length_psi: float
    confidence_psi: float
    max_psi: float
    threshold: float
    drift_detected: bool
