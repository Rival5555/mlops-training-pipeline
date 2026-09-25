# Automated Production MLOps Training Pipeline

A production-grade, automated, reproducible, and versioned MLOps training pipeline for Computer Vision (Image Classification) built with **PyTorch**, **MLflow**, **Prefect 3.x**, and **DVC**.

---

## 1. Architecture & Pipeline Stages

The pipeline is decoupled into 6 independently runnable, testable modules orchestrated via Prefect:

```
[Raw Data] 
    │
    ▼
[Stage 1: Ingest & Validate] ──(Fails loudly on corrupt images/schema mismatch)
    │
    ▼
[Stage 2: Preprocess & Split] ──(Deterministic stratified split + DVC config hash tagging)
    │
    ▼
[Stage 3: PyTorch AMP Training] ──(Cosine LR, early stopping, best/last checkpoints, MLflow logs)
    │
    ▼
[Stage 4: Evaluation & Gate] ──(Held-out test set vs MLflow production model baseline + delta)
    │   ├── Rejection: Abort registration, log failure report
    │   └── Approval: Pass to Stage 5
    ▼
[Stage 5: Model Registration] ──(MLflow Model Registry: Git/Data/Config metadata tags -> 'staging')
    │
    ▼
[Stage 6: Packaging & Export] ──(ONNX export + numerical parity verification tolerance check)
```

---

## 2. Quickstart & Local Setup

### Prerequisites
- Python 3.10+ (or Python 3.11+)
- [`uv`](https://github.com/astral-sh/uv) (fast Python package manager)
- Docker (optional, for local MLflow server)

### Step 1: Clone and Set Up Virtual Environment
```bash
# Create and activate virtual environment using uv
uv venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies in editable mode with dev tools
uv pip install -e ".[dev]"
```

### Step 2: Configure Environment
Copy the example environment configuration:
```bash
cp .env.example .env
```
*(If running MLflow locally, default tracking URI is `http://localhost:5000` or local filesystem fallback `file:./mlruns`)*

### Step 3: Start Local MLflow Server (Optional)
```bash
# Spin up local containerized MLflow server with SQLite backend
docker compose -f docker/docker-compose.yml up -d mlflow
```
Access the MLflow UI at: `http://localhost:5000`

### Step 4: Generate Demo Dataset
To run the pipeline immediately without external dataset downloads:
```bash
python scripts/generate_synthetic_data.py --samples-per-class 25
```
This populates `data/raw/healthy` and `data/raw/referable_dr` with realistic test samples.

### Step 5: Execute the Pipeline
Run all stages end-to-end via the Prefect orchestrator:
```bash
python -m src.pipeline.flow --config configs/config.yaml
# Or via Makefile:
make train
```

---

## 3. Running Stages Independently

Each stage is an isolated CLI module for debugging and targeted development:

```bash
# Stage 1: Data ingestion and schema validation
python -m src.pipeline.stages.s01_ingest_validate --config configs/config.yaml

# Stage 2: Preprocessing, stratified split, and data versioning
python -m src.pipeline.stages.s02_preprocess_split --config configs/config.yaml

# Stage 3: PyTorch model training
python -m src.pipeline.stages.s03_train --config configs/config.yaml

# Stage 4: Held-out test evaluation & production validation gate
python -m src.pipeline.stages.s04_eval_gate --config configs/config.yaml

# Stage 5: Model registration to MLflow
python -m src.pipeline.stages.s05_register --config configs/config.yaml

# Stage 6: ONNX export & parity smoke test
python -m src.pipeline.stages.s06_export_package --config configs/config.yaml
```

---

## 4. How the Automated Promotion Gate Works

The validation gate (`s04_eval_gate.py`) guarantees that substandard or regressed models never touch production:

1. **Held-Out Evaluation**:
   The candidate model is evaluated on the held-out `test` split (never touched during training or hyperparameter tuning).
2. **Registry Lookup**:
   The pipeline queries MLflow Model Registry for the currently active `production` tagged model and extracts its baseline metric (e.g. `test_f1_macro`).
3. **Threshold Check**:
   $$\text{Candidate Metric} \ge \max(\text{Production Baseline} + \Delta_{\text{min}}, \text{Absolute Floor})$$
   - Default $\Delta_{\text{min}} = +0.01$ (must beat baseline by at least $1\%$)
   - Default Absolute Floor $= 0.70$
4. **Enforcement**:
   - If the candidate passes: `models/checkpoints/gate_decision.json` records `passed: true` and Stage 5 proceeds.
   - If the candidate fails: An `EvaluationGateError` is raised loudly, execution halts, and the model is blocked from registration.
5. **Cold-Start Handling**:
   If no existing model is tagged `production`, the gate allows promotion as long as the candidate satisfies the absolute minimum quality floor (`allow_first_model: true`).

---

## 5. Rollback Procedure for Production Models

If an anomalous issue is identified in production after promotion:

### Automated Rollback via Python API / CLI
You can rollback to any previous stable model version using the MLflow Client:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")
model_name = "retinopathy_classifier"

# 1. Archive the problematic current production version
bad_version = "4"
client.transition_model_version_stage(name=model_name, version=bad_version, stage="Archived")

# 2. Re-promote previous stable version back to Production
stable_version = "3"
client.transition_model_version_stage(name=model_name, version=stable_version, stage="Production")
```

### UI Rollback via MLflow Dashboard
1. Open MLflow UI at `http://localhost:5000`.
2. Navigate to **Models** > `retinopathy_classifier`.
3. Open the previously verified stable version (e.g., Version 3).
4. Click **Stage** > **Transition to Production**. This automatically demotes/archives the active faulty version.

---

## 6. Testing & Quality Assurance

Run the test suite:
```bash
# Run unit tests with code coverage
pytest tests/unit/ -v --cov=src/pipeline

# Run the promotion gate test suite specifically
pytest tests/unit/test_eval_gate.py -v

# Run the end-to-end integration smoke test
pytest tests/integration/test_pipeline_smoke.py -v
```

Code formatting and static analysis:
```bash
ruff check .
ruff format --check .
mypy src/
```

---

## 7. CI/CD Integration

The repository includes GitHub Actions workflows under `.github/workflows/`:
- **`ci.yml`**: Triggers on pull requests. Runs `ruff` linting, `mypy` type checking, and unit tests (with explicit gate testing).
- **`retrain_pipeline.yml`**: Scheduled (weekly) or manually triggered (`workflow_dispatch`) retrain execution that logs directly to MLflow and builds release artifacts.
