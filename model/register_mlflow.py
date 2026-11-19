"""Register DistilBERT INT8 ONNX and Baseline Sklearn ONNX models in MLflow."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.onnx
import numpy as np
import onnx
from mlflow.models import infer_signature

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "sst2-text-classification"


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def register_distilbert(git_commit: str) -> None:
    model_dir = Path("artifacts/distilbert_sst2_int8")
    onnx_path = model_dir / "model_quantized.onnx"
    stats = json.loads((model_dir / "reference_stats.json").read_text(encoding="utf-8"))

    onnx_model = onnx.load(str(onnx_path))
    sample_in = {
        "input_ids": np.array([[101, 2023, 2003, 2204, 102]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64),
    }
    sample_out = np.array([[-2.1, 2.3]], dtype=np.float32)
    signature = infer_signature(sample_in, sample_out)

    with mlflow.start_run(run_name="distilbert-sst2-int8-onnx"):
        mlflow.log_params(
            {
                "base_model": "distilbert-base-uncased-finetuned-sst-2-english",
                "quantization": "INT8_dynamic_avx2",
                "max_length": 128,
                "git_commit": git_commit,
            }
        )
        mlflow.log_metrics(
            {
                "val_accuracy": stats["val_accuracy"],
                "model_size_mb": stats["model_size_mb"],
            }
        )
        mlflow.set_tag("sha256", stats["sha256"])
        mlflow.log_artifact(str(model_dir / "reference_stats.json"), artifact_path="metadata")
        mlflow.log_artifact(str(model_dir / "checksum.sha256"), artifact_path="metadata")
        mlflow.log_artifact(str(model_dir / "tokenizer.json"), artifact_path="metadata")

        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            name="model",
            signature=signature,
            pip_requirements=[
                "onnxruntime>=1.20.0",
                "tokenizers>=0.20.0",
                "numpy>=1.26.0",
            ],
            registered_model_name="sst2-distilbert-int8",
        )
    print("Registered sst2-distilbert-int8 in MLflow.")


def register_baseline(git_commit: str) -> None:
    model_dir = Path("artifacts/baseline_sklearn")
    onnx_path = model_dir / "model.onnx"
    meta = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))

    onnx_model = onnx.load(str(onnx_path))
    sample_in = np.array([["this movie is great"]], dtype=object)
    sample_out = np.array([1], dtype=np.int64)
    signature = infer_signature(sample_in, sample_out)

    with mlflow.start_run(run_name="baseline-tfidf-logreg-onnx"):
        mlflow.log_params(
            {
                "model_type": "tfidf_logreg",
                "max_features": 20000,
                "ngram_range": "(1, 2)",
                "git_commit": git_commit,
            }
        )
        mlflow.log_metrics(
            {
                "val_accuracy": meta["val_accuracy"],
                "model_size_mb": meta["model_size_mb"],
            }
        )
        mlflow.set_tag("sha256", meta["sha256"])
        mlflow.log_artifact(str(model_dir / "checksum.sha256"), artifact_path="metadata")

        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            name="model",
            signature=signature,
            pip_requirements=["onnxruntime>=1.20.0", "numpy>=1.26.0"],
            registered_model_name="sst2-tfidf-logreg",
        )
    print("Registered sst2-tfidf-logreg in MLflow.")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    git_commit = get_git_commit()
    register_distilbert(git_commit)
    register_baseline(git_commit)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
