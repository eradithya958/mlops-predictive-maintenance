#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot setup script for the MLOps Pipeline project
# Run this from your terminal (not the IDE) as it requires sudo/password.
# =============================================================================
set -euo pipefail

echo "============================================================"
echo "  MLOps Pipeline — Phase 1 Setup"
echo "============================================================"

# ---- Step 1: Install Xcode Command Line Tools (needed for git, Python venv) ----
if ! xcode-select -p &>/dev/null; then
    echo "→ Installing Xcode Command Line Tools…"
    xcode-select --install
    echo "  Please click 'Install' in the popup, then re-run this script."
    exit 0
fi
echo "✓ Xcode CLT already installed"

# ---- Step 2: Install Homebrew ----
if ! command -v brew &>/dev/null; then
    echo "→ Installing Homebrew…"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    fi
fi
echo "✓ Homebrew ready: $(brew --version | head -1)"

# ---- Step 3: Install Python 3.11 ----
if ! brew list python@3.11 &>/dev/null; then
    echo "→ Installing Python 3.11 via Homebrew…"
    brew install python@3.11
fi
PYTHON=$(brew --prefix python@3.11)/bin/python3.11
echo "✓ Python: $($PYTHON --version)"

# ---- Step 4: Create virtualenv ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d .venv ]]; then
    echo "→ Creating virtual environment (.venv)…"
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate
echo "✓ Virtualenv activated: $(which python)"

# ---- Step 5: Install project ----
echo "→ Installing project + dev dependencies…"
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
echo "✓ Installation complete"

# ---- Step 6: Init git ----
if [[ ! -d .git ]]; then
    echo "→ Initialising git repository…"
    git init
    git add .
    git commit -m "feat: Phase 1 — data pipeline, feature engineering, XGBoost baseline"
fi
echo "✓ Git ready"

# ---- Step 7: Init DVC ----
if [[ ! -d .dvc ]]; then
    echo "→ Initialising DVC…"
    dvc init
    # Configure a local DVC remote for versioning data
    mkdir -p /tmp/dvc_remote
    dvc remote add -d local_remote /tmp/dvc_remote
    git add .dvc .dvcignore
    git commit -m "chore: init DVC with local remote"
fi
echo "✓ DVC ready"

# ---- Step 8: Run tests ----
echo ""
echo "→ Running test suite…"
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
echo ""

# ---- Step 9: Run the pipeline ----
echo "→ Running DVC pipeline (ingest → featurize → train)…"
echo "  (This will download CMAPSS data or generate synthetic data)"
dvc repro

echo ""
echo "============================================================"
echo "  ✓ Phase 1 complete!"
echo "============================================================"
echo ""
echo "  To view experiment results:"
echo "    source .venv/bin/activate"
echo "    mlflow ui --port 5000"
echo "    → Open http://localhost:5000"
echo ""
echo "  To re-run with different params (e.g. change model type):"
echo "    Edit params.yaml → model.type = 'random_forest'"
echo "    dvc repro"
echo ""
echo "------------------------------------------------------------"
echo "  Phase 2 — Start the full Docker Compose stack:"
echo ""
echo "    docker compose up --build -d"
echo ""
echo "    API      → http://localhost:8000"
echo "    API docs → http://localhost:8000/docs"
echo "    MLflow   → http://localhost:5000"
echo "    Prometheus → http://localhost:9090"
echo "    Grafana  → http://localhost:3000  (admin / admin)"
echo ""
echo "  Test predict endpoint:"
echo "    curl -X POST http://localhost:8000/predict \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d @monitoring/example_request.json"
echo "------------------------------------------------------------"

