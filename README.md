# MLOps Pipeline — Predictive Maintenance

> End-to-end MLOps project: turbofan engine RUL prediction using NASA CMAPSS data.  
> Demonstrates data versioning (DVC), experiment tracking (MLflow), containerisation (Docker),  
> CI/CD (GitHub Actions), live monitoring (Prometheus + Grafana), and drift detection (KS test).

[![CI](https://github.com/YOUR_USERNAME/MLOP_Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/MLOP_Pipeline/actions/workflows/ci.yml)
[![CD](https://github.com/YOUR_USERNAME/MLOP_Pipeline/actions/workflows/cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/MLOP_Pipeline/actions/workflows/cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](https://www.docker.com/)

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 — Data & Model | ✅ Complete | DVC pipeline · XGBoost · MLflow tracking |
| Phase 2 — Containerisation | ✅ Complete | FastAPI · Docker Compose · Prometheus · Grafana |
| Phase 3 — CI/CD | ✅ Complete | GitHub Actions lint → test → build/push → promote |
| Phase 4 — Drift Detection | ✅ Complete | KS-test detector · `/drift-report` endpoint · Grafana panels |
| Phase 5 — Documentation | ✅ Complete | Architecture diagram · Full README · badges |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Local / CI Training                        │
│                                                                  │
│  NASA CMAPSS Data                                                │
│       │                                                          │
│       ▼                                                          │
│  [ingest.py] ──DVC──► train/val/test.parquet                     │
│       │                                                          │
│       ▼                                                          │
│  [features.py] ──DVC──► *_features.parquet ◄── DriftDetector    │
│       │                       (reference dist)                   │
│       ▼                                                          │
│  [train.py] ──────────► MLflow Experiment + Model Registry       │
│                                  │                               │
│               GitHub Actions CD: evaluate → promote to Prod      │
└───────────────────────┬──────────────────────────────────────────┘
                        │ mlruns/ volume
┌───────────────────────▼──────────────────────────────────────────┐
│                     Docker Compose Stack                          │
│                                                                  │
│  ┌─────────────────┐   ┌──────────────────┐                      │
│  │  FastAPI :8000  │   │  MLflow  :5000   │                      │
│  │  /predict       │──►│  Model Registry  │                      │
│  │  /health        │   └──────────────────┘                      │
│  │  /model-info    │                                              │
│  │  /drift-report  │──► KS-test per feature                      │
│  │  /metrics       │──► Prometheus metrics                        │
│  └────────┬────────┘                                              │
│           │ scrape /metrics every 15s                             │
│  ┌────────▼────────┐   ┌──────────────────┐                      │
│  │ Prometheus :9090│   │  Grafana  :3000  │                      │
│  │  TSDB storage   │──►│  Auto-provisioned │                      │
│  └─────────────────┘   │  Dashboard       │                      │
│                         └──────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘

GitHub Actions
  ├── ci.yml   push/PR  → lint (ruff+black) → test (pytest+cov) → docker build
  └── cd.yml   weekly/dispatch → dvc repro → evaluate → promote to Production
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop (or Docker Engine + Compose v2)

---

### Phase 1 — Train the Model Locally

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install all dependencies (including dev tools)
pip install -e ".[dev]"

# 3. Run the full DVC pipeline: ingest → featurize → train
dvc repro

# 4. View results in MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000
```

The pipeline will:
1. Download NASA CMAPSS data (FD001) — or generate synthetic data if unavailable
2. Engineer rolling mean/std, lag, trend, and cycle-progress features per engine
3. Train an XGBoost regressor and log all metrics + artifacts to MLflow
4. Register the model as `turbofan_rul` in the MLflow Model Registry

---

### Phase 2 — Run the Full Stack (Docker Compose)

```bash
# Start all four services
docker compose up --build -d

# Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @monitoring/example_request.json

# Check drift detection status
curl "http://localhost:8000/drift-report?force=true"
```

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI | http://localhost:8000 | — |
| Swagger UI | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

---

### Phase 3 — CI/CD (GitHub Actions)

Two workflows ship in `.github/workflows/`:

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `ci.yml` | Push / PR to `main` or `develop` | lint → test → docker build (+push on main) |
| `cd.yml` | Weekly Sun 02:00 UTC · manual dispatch · post-CI | `dvc repro` → evaluate → promote to Production |

**Required GitHub secrets** (for CD push to GHCR and MLflow server):

| Secret | Description |
|--------|-------------|
| `GITHUB_TOKEN` | Auto-provided — used for GHCR image push |
| `MLFLOW_TRACKING_URI` | Optional — remote MLflow server URL |

---

### Phase 4 — Drift Detection

The API automatically detects **feature distribution drift** at inference time using the two-sample Kolmogorov-Smirnov test.

**How it works:**
1. At startup, the detector loads the training-set feature statistics from `data/processed/train_features.parquet`
2. Every `/predict` call records the raw feature values into a rolling buffer (default: 500 samples)
3. When the buffer fills, KS tests run automatically and results are pushed to Prometheus
4. Grafana plots KS statistics and p-values per feature; alerts fire when p < 0.05

**API endpoints:**

```bash
# Get the latest drift report (based on last full 500-sample batch)
GET /drift-report

# Force an immediate check on whatever is currently buffered
GET /drift-report?force=true
```

**Prometheus metrics added:**

| Metric | Labels | Description |
|--------|--------|-------------|
| `drift_ks_statistic` | `feature` | KS statistic (0=no drift → 1=full drift) |
| `drift_ks_pvalue` | `feature` | KS p-value (< 0.05 = alert) |
| `drift_detection_runs_total` | — | Total completed detection runs |
| `drift_alerts_total` | `feature` | Cumulative drift alerts per feature |

---

## Project Structure

```
MLOP_Pipeline/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint → Test → Docker build/push
│       └── cd.yml              # DVC retrain → evaluate → promote
├── src/
│   ├── data/
│   │   ├── ingest.py           # Download, parse, RUL-label, split
│   │   └── features.py         # Rolling, lag, slope features
│   ├── train/
│   │   └── train.py            # Train, evaluate, MLflow log + register
│   ├── api/
│   │   ├── app.py              # FastAPI: /predict /health /drift-report /metrics
│   │   ├── model_loader.py     # MLflow registry loader (Production → Staging)
│   │   └── schemas.py          # Pydantic request/response schemas
│   ├── monitoring/
│   │   └── drift.py            # KS-test drift detector (thread-safe)
│   └── utils/
│       └── config.py           # Pydantic settings (params.yaml + env vars)
├── tests/
│   ├── test_ingest.py          # Schema, RUL, anti-leakage checks
│   ├── test_features.py        # Rolling, lag, trend feature tests
│   ├── test_api.py             # FastAPI endpoint tests
│   └── test_drift.py           # KS drift detector tests
├── monitoring/
│   ├── prometheus.yml          # Scrape config (API every 15s)
│   ├── example_request.json    # Sample /predict payload
│   └── grafana/
│       ├── provisioning/       # Auto-provisioned datasource + dashboards
│       └── dashboards/
│           └── mlops_dashboard.json   # Pre-built Grafana dashboard
├── data/
│   ├── raw/                    # Raw CMAPSS files (DVC-tracked)
│   └── processed/              # Feature parquets (DVC-tracked)
├── models/                     # MLflow run ID + metrics (DVC-tracked)
├── Dockerfile                  # Multi-stage: builder + runtime
├── docker-compose.yml          # 4-service stack
├── dvc.yaml                    # Pipeline DAG: ingest → featurize → train
└── params.yaml                 # All hyperparameters (DVC-tracked)
```

---

## Configuration

All hyperparameters live in [`params.yaml`](params.yaml). DVC detects changes and reruns only affected stages.

```yaml
model:
  type: xgboost      # switch to "random_forest" to compare
  n_estimators: 300
  learning_rate: 0.05
  ...

features:
  rolling_windows: [5, 10, 20]
  lag_steps: [1, 3, 5]
  trend_window: 20
```

---

## Running Tests

```bash
# Full test suite with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Individual test files
pytest tests/test_drift.py -v     # drift detection tests
pytest tests/test_api.py -v       # API endpoint tests
pytest tests/test_features.py -v  # feature engineering tests
pytest tests/test_ingest.py -v    # data ingestion tests
```

---

## Dataset

**NASA CMAPSS Turbofan Engine Degradation Simulation Dataset**
- 100 training engine units, 21 sensor readings + 3 operational settings
- Target: Remaining Useful Life (RUL) in engine cycles
- Piece-wise linear RUL labelling with clip at 125 cycles

---

*Built as a portfolio project demonstrating end-to-end MLOps engineering: data versioning, experiment tracking, containerisation, CI/CD, monitoring, and drift detection.*
