"""ONNX Runtime model loader and batch inference engine."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class ONNXInferenceEngine:
    def __init__(
        self,
        model_dir: Path,
        model_type: str = "distilbert_onnx",
        expected_sha256: str = "",
        max_token_length: int = 128,
    ):
        self.model_dir = model_dir
        self.model_type = model_type
        self.max_token_length = max_token_length
        self.artifact_sha256 = ""
        self.load_seconds = 0.0
        self.session: ort.InferenceSession | None = None
        self.tokenizer: Tokenizer | None = None
        self.expected_sha256 = expected_sha256.strip()

    def load(self) -> None:
        t0 = time.perf_counter()
        if self.model_type == "distilbert_onnx":
            onnx_path = self.model_dir / "model_quantized.onnx"
            if not onnx_path.exists():
                onnx_path = self.model_dir / "model.onnx"
        else:
            onnx_path = self.model_dir / "model.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        actual_sha = sha256_file(onnx_path)
        if self.expected_sha256 and actual_sha != self.expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {onnx_path}: expected {self.expected_sha256}, got {actual_sha}"
            )
        self.artifact_sha256 = actual_sha

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        if self.model_type == "distilbert_onnx":
            tok_path = self.model_dir / "tokenizer.json"
            if not tok_path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
            tok = Tokenizer.from_file(str(tok_path))
            tok.enable_truncation(max_length=self.max_token_length)
            tok.enable_padding(pad_id=0, pad_token="[PAD]")
            self.tokenizer = tok

        self.load_seconds = time.perf_counter() - t0

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        if self.session is None:
            raise RuntimeError("Model is not loaded")

        if self.model_type == "distilbert_onnx":
            assert self.tokenizer is not None
            encodings = self.tokenizer.encode_batch(texts)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            logits = self.session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )[0]
            probs = softmax(logits)
            results: list[tuple[str, float]] = []
            for row in probs:
                idx = int(np.argmax(row))
                label = "POSITIVE" if idx == 1 else "NEGATIVE"
                results.append((label, float(row[idx])))
            return results

        # Baseline sklearn ONNX (string_input -> [labels, probabilities])
        arr = np.array([[t] for t in texts], dtype=object)
        labels, probs = self.session.run(None, {"string_input": arr})
        results = []
        for lbl, prob_row in zip(labels, probs):
            idx = int(lbl)
            label = "POSITIVE" if idx == 1 else "NEGATIVE"
            score = float(prob_row[idx])
            results.append((label, score))
        return results

