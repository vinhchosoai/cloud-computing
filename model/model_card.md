# Model Card: SST-2 Sentiment Classifier (DistilBERT INT8 ONNX)

## 1. Model Details
- **Base Model:** `distilbert-base-uncased-finetuned-sst-2-english` (66M parameters, 6-layer DistilBERT).
- **Format & Optimization:** Exported to ONNX and dynamically quantized to INT8 (AVX2) via Hugging Face Optimum.
- **Baseline Comparison Model:** TF-IDF (20,000 features, 1–2 n-grams) + Logistic Regression exported to ONNX via `skl2onnx`.
- **Registry:** Logged and versioned in MLflow (`sst2-distilbert-int8`, `sst2-tfidf-logreg`) with explicit `ModelSignature`, `pip_requirements`, and SHA-256 artifact checksum.

## 2. Intended Use
- **Task:** Binary English sentiment classification (`POSITIVE` / `NEGATIVE`) on short texts (1–512 characters, truncated at 128 tokens).
- **Out-of-Scope Uses:** Non-English text (e.g. Vietnamese), multi-class emotion detection, medical/legal/safety-critical decisions, or inputs exceeding 512 characters.

## 3. Evaluation Data & Measured Results
- **Dataset:** GLUE SST-2 (`glue/sst2`) validation split (872 movie review sentences).
- **DistilBERT INT8 ONNX (`sst2-distilbert-int8` v1):**
  - Validation Accuracy: **90.37%** (`788/872`)
  - Artifact Size: **64.25 MB** (`model_quantized.onnx`)
  - SHA-256: `93bbcb19c062823da6c4da7aa607663e966467e411dae80bfe3d5bd85da08658`
- **Baseline TF-IDF + LogisticRegression ONNX (`sst2-tfidf-logreg` v1):**
  - Validation Accuracy: **81.54%** (trained on 67,349 SST-2 train samples)
  - Artifact Size: **0.66 MB** (`model.onnx`)
  - SHA-256: `b2e437e7037767ff09142d49ff7d30269cd30f8d4a5c5680eb905539843fb127`

## 4. Risks, Bias, and Limitations
- **Domain Bias:** Trained on Rotten Tomatoes movie reviews (SST-2); performance degrades on social media slang, domain-specific jargon, or non-English text.
- **Quantization Trade-off:** Dynamic INT8 quantization significantly reduces model size and CPU inference latency, with minor potential accuracy variance compared to FP32.
- **Scope of Benchmark Claims:** Latency, throughput, and cost measurements apply strictly to the tested Azure `eastasia` configurations (`1 vCPU / 2 GiB` Container Apps vs `Standard_B2s` VM) and client-to-region network conditions; results must not be extrapolated to GPU workloads or different regions without re-measuring.

