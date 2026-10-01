# =============================================================================
# Multi-stage Dockerfile for the Predictive Maintenance FastAPI service
# =============================================================================

# ---- Stage 1: Builder — install all Python dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (needed for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency declarations first (layer-cache friendly)
COPY pyproject.toml ./
COPY README.md ./

# Install into an isolated venv at /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime deps only (no dev extras like pytest/ruff)
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
        "pandas>=2.0" \
        "numpy>=1.24" \
        "scikit-learn>=1.3" \
        "xgboost>=2.0" \
        "mlflow>=2.13" \
        "pydantic>=2.5" \
        "pydantic-settings>=2.2" \
        "pyyaml>=6.0" \
        "requests>=2.31" \
        "pyarrow>=14.0" \
        "scipy>=1.11" \
        "matplotlib>=3.7" \
        "fastapi>=0.111" \
        "uvicorn[standard]>=0.29" \
        "prometheus-client>=0.20"

# ---- Stage 2: Runtime — lean image with non-root user ----
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code
COPY src/ ./src/
COPY params.yaml ./

# Ensure Python can find the src package (equivalent to pip install -e .)
ENV PYTHONPATH="/app"

# The mlruns/ and models/ dirs are mounted as volumes at runtime
# (see docker-compose.yml) — don't COPY them into the image.

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
