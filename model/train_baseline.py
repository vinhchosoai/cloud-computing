"""Train TF-IDF + LogisticRegression baseline on SST-2 and export to ONNX."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

OUT_DIR = Path("artifacts/baseline_sklearn")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading glue/sst2 dataset...")
    train_ds = load_dataset("glue", "sst2", split="train")
    val_ds = load_dataset("glue", "sst2", split="validation")

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]
    )
    print(f"Training TF-IDF + LogisticRegression on {len(train_ds)} samples...")
    pipe.fit(train_ds["sentence"], train_ds["label"])

    preds = pipe.predict(val_ds["sentence"])
    val_acc = accuracy_score(val_ds["label"], preds)
    print(f"SST-2 Validation Accuracy (sklearn): {val_acc:.4f}")

    # Export to ONNX
    initial_type = [("string_input", StringTensorType([None, 1]))]
    onnx_model = convert_sklearn(
        pipe,
        initial_types=initial_type,
        target_opset=15,
        options={id(pipe): {"zipmap": False}},
    )
    onnx_path = OUT_DIR / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())

    digest = sha256_file(onnx_path)
    (OUT_DIR / "checksum.sha256").write_text(f"{digest}  {onnx_path.name}\n", encoding="utf-8")
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Saved {onnx_path} ({size_mb:.2f} MB), SHA256={digest}")

    # Verify ONNX inference matches sklearn
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample_inputs = np.array([[s] for s in val_ds["sentence"][:10]])
    ort_labels, _ = sess.run(None, {"string_input": sample_inputs})
    assert list(ort_labels) == list(preds[:10]), "ONNX output mismatch vs sklearn!"

    meta = {
        "dataset": "glue/sst2",
        "val_accuracy": round(float(val_acc), 4),
        "model_size_mb": round(size_mb, 2),
        "sha256": digest,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Baseline ONNX export verified successfully.")


if __name__ == "__main__":
    main()

