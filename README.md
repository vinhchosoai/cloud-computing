# AI Text Classification Inference Service (Topic 30)

Cloud-native text classification inference service comparing **Azure Container Apps (serverless, scale-to-zero)** and **Azure VM (`Standard_B2s` + Docker Compose)** with automated load testing, Prometheus/Grafana monitoring, security controls, PSI drift detection, and cost/break-even analysis.

## Project Structure

- `model/` — DistilBERT SST-2 ONNX INT8 export (`export_onnx.py`), TF-IDF + LogisticRegression baseline (`train_baseline.py`), MLflow registration (`register_mlflow.py`), and `model_card.md`.
- `app/` — FastAPI inference service (`/v1/predict`, `/health`, `/ready`, `/version`, `/metrics`, `/v1/drift`) with API key auth, `slowapi` rate limiting, SHA-256 privacy-safe logging, and sliding-window PSI drift detection.
- `tests/` — Unit tests for schema validation, auth, rate limiting, PII-free logging, and drift detection.
- `monitoring/` — Prometheus config, alert rules (`HighInferenceLatencyP95`, `HighErrorRate`, `DataDriftPSIHigh`), and auto-provisioned Grafana dashboard.
- `deploy/` — Azure CLI scripts for Container Apps (`deploy_aca.ps1`), VM (`deploy_vm.ps1`), and resource cleanup (`cleanup.ps1`).
- `loadtest/` — k6 scripts (`predict.js`, `spike.js`, `error_scenarios.js`, `drift_sim.js`, `run_matrix.ps1`).
- `analysis/` — Result aggregation, statistical summary (mean ± std), and Azure cost/break-even modeling (`analyze_results.py`).

## Quickstart (Local)

```powershell
# 1. Install dependencies
uv sync --all-groups

# 2. Export DistilBERT INT8 ONNX + Train Baseline + Register in MLflow
uv run python model/export_onnx.py
uv run python model/train_baseline.py
uv run python model/register_mlflow.py

# 3. Run unit tests
uv run pytest -v

# 4. Start API locally
Copy-Item .env.example .env
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Full Local Stack (API + Prometheus + Grafana)

```powershell
docker compose up -d --build
# API: http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

## Azure Deployment & Cleanup (`eastasia` / `rg-ai-inference`)

```powershell
# Deploy Serverless (Azure Container Apps)
.\deploy\deploy_aca.ps1

# Deploy Fixed VM (Standard_B2s)
.\deploy\deploy_vm.ps1

# Cleanup ALL cloud resources after benchmarking
.\deploy\cleanup.ps1
```

