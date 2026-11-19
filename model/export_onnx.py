"""Export DistilBERT SST-2 to ONNX, quantize to INT8, and compute reference stats."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from tokenizers import Tokenizer
from transformers import AutoTokenizer

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
OUT_DIR = Path("artifacts/distilbert_sst2_int8")
FP32_TMP_DIR = Path("artifacts/_tmp_distilbert_fp32")

TEXT_LENGTH_EDGES = [0, 20, 40, 60, 100, 150, 250, 513]
CONFIDENCE_EDGES = [0.5, 0.65, 0.8, 0.9, 0.95, 1.01]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def compute_hist_probs(values: list[float], edges: list[float]) -> list[float]:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return [1.0 / (len(edges) - 1)] * (len(edges) - 1)
    # Add small epsilon smoothing to avoid zero-division in PSI
    probs = (counts + 1e-6) / (total + 1e-6 * len(counts))
    return [float(p) for p in probs]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FP32_TMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {MODEL_ID} to FP32 ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ort_model.save_pretrained(FP32_TMP_DIR)
    tokenizer.save_pretrained(FP32_TMP_DIR)

    print("Quantizing ONNX model to INT8 (dynamic AVX2)...")
    quantizer = ORTQuantizer.from_pretrained(FP32_TMP_DIR)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=OUT_DIR, quantization_config=qconfig)
    tokenizer.save_pretrained(OUT_DIR)

    onnx_path = OUT_DIR / "model_quantized.onnx"
    if not onnx_path.exists():
        # Fallback if optimum names it model.onnx
        alt = OUT_DIR / "model.onnx"
        if alt.exists():
            alt.rename(onnx_path)
        else:
            raise FileNotFoundError(f"Expected quantized ONNX model in {OUT_DIR}")

    digest = sha256_file(onnx_path)
    (OUT_DIR / "checksum.sha256").write_text(f"{digest}  {onnx_path.name}\n", encoding="utf-8")
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Saved {onnx_path} ({size_mb:.2f} MB), SHA256={digest}")

    # Evaluate INT8 model and compute reference stats on SST-2 validation split
    print("Evaluating INT8 model on SST-2 validation split for reference stats...")
    ds = load_dataset("glue", "sst2", split="validation")
    tok = Tokenizer.from_file(str(OUT_DIR / "tokenizer.json"))
    tok.enable_truncation(max_length=128)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    texts = ds["sentence"]
    labels = ds["label"]
    lengths: list[float] = [float(len(t)) for t in texts]
    confidences: list[float] = []
    correct = 0

    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        encodings = tok.encode_batch(batch_texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        logits = session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        probs = softmax(logits)
        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1)

        confidences.extend([float(c) for c in confs])
        correct += int(np.sum(preds == np.array(batch_labels)))

    val_acc = correct / len(texts)
    print(f"SST-2 Validation Accuracy (INT8 ONNX): {val_acc:.4f} ({correct}/{len(texts)})")

    ref_stats = {
        "dataset": "glue/sst2/validation",
        "num_samples": len(texts),
        "val_accuracy": round(val_acc, 4),
        "model_size_mb": round(size_mb, 2),
        "sha256": digest,
        "text_length_edges": TEXT_LENGTH_EDGES,
        "text_length_probs": compute_hist_probs(lengths, TEXT_LENGTH_EDGES),
        "confidence_edges": CONFIDENCE_EDGES,
        "confidence_probs": compute_hist_probs(confidences, CONFIDENCE_EDGES),
    }
    (OUT_DIR / "reference_stats.json").write_text(
        json.dumps(ref_stats, indent=2), encoding="utf-8"
    )
    print(f"Saved reference stats to {OUT_DIR / 'reference_stats.json'}")


if __name__ == "__main__":
    main()

