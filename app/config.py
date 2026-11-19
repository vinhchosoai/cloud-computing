"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_keys: str = "dev-secret-key-change-me"
    benchmark_api_key: str = ""
    rate_limit: str = "60/minute"
    benchmark_rate_limit: str = "10000/minute"

    model_type: str = "distilbert_onnx"  # distilbert_onnx | baseline_onnx
    model_dir: str = "artifacts/distilbert_sst2_int8"
    model_name: str = "sst2-distilbert-int8"
    model_version: str = "1.0.0"
    expected_sha256: str = ""
    git_commit: str = "local"

    max_batch_size: int = 32
    max_text_length: int = 512
    max_token_length: int = 128

    drift_window_size: int = 200
    drift_psi_threshold: float = 0.20

    def get_valid_api_keys(self) -> list[str]:
        keys = [k.strip() for k in self.api_keys.split(",") if k.strip()]
        if self.benchmark_api_key.strip() and self.benchmark_api_key.strip() not in keys:
            keys.append(self.benchmark_api_key.strip())
        return keys


@lru_cache
def get_settings() -> Settings:
    return Settings()
