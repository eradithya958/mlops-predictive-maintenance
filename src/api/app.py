"""
FastAPI prediction service for the MLOps predictive maintenance pipeline.

Endpoints:
  GET  /health        — liveness + model status
  GET  /model-info    — registry metadata for loaded model
  POST /predict       — RUL prediction from raw sensor readings
  GET  /metrics       — Prometheus metrics exposition

Startup:
  Reads MLFLOW_TRACKING_URI and MLOPS_MODEL_NAME from environment.
  Loads the model from the registry (Production → Staging fallback).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.api.model_loader import get_loader, init_loader
from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.data.features import (
    _sensor_cols,
    add_cycle_progress,
    add_lag_features,
    add_rolling_features,
    add_trend_features,
)
from src.monitoring.drift import (
    get_detector,
    init_detector,
)
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PREDICT_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["status"],  # labels: success / error
)

PREDICT_LATENCY = Histogram(
    "prediction_latency_seconds",
    "End-to-end prediction request latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTED_RUL = Histogram(
    "predicted_rul_value",
    "Distribution of predicted RUL values",
    buckets=[0, 10, 20, 30, 50, 75, 100, 125],
)

MODEL_INFO_GAUGE = Gauge(
    "model_info",
    "Currently loaded model version (value=1 means active)",
    ["model_name", "version", "stage"],
)

# ---------------------------------------------------------------------------
# App startup / shutdown
# ---------------------------------------------------------------------------

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    cfg = get_settings()

    # --- Model loader ---
    loader = init_loader(
        tracking_uri=cfg.mlflow_tracking_uri,
        model_name=cfg.mlflow.model_registry_name,
    )
    success = loader.load()
    if success:
        lm = loader.get()
        MODEL_INFO_GAUGE.labels(
            model_name=lm.model_name,
            version=lm.model_version,
            stage=lm.model_stage,
        ).set(1)
        logger.info("API ready — model %s v%s loaded.", lm.model_name, lm.model_version)
    else:
        logger.warning(
            "API started but NO model loaded. /predict will return 503 until a model is registered."
        )

    # --- Drift detector ---
    ref_parquet = cfg.processed_dir / "train_features.parquet"
    if ref_parquet.exists():
        try:
            init_detector(
                reference_parquet=ref_parquet,
                window_size=500,
                alert_threshold=0.05,
            )
            logger.info("Drift detector initialised from %s", ref_parquet)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Drift detector could not be initialised: %s", exc)
    else:
        logger.warning(
            "Reference parquet not found at %s — drift detection disabled. "
            "Run 'dvc repro' to generate training features.",
            ref_parquet,
        )

    yield
    # Shutdown — nothing to clean up
    logger.info("API shutting down.")


# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Predict the Remaining Useful Life (RUL) of turbofan engines "
        "using an XGBoost model trained on NASA CMAPSS data."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Feature engineering helper  (mirrors the training pipeline)
# ---------------------------------------------------------------------------


def _engineer_features_for_request(req: PredictRequest, cfg: Any) -> pd.DataFrame:
    """
    Build a feature-engineered DataFrame from a PredictRequest.

    If the client supplies `history`, we reconstruct a per-unit DataFrame
    with multiple cycles and compute rolling/lag features properly.
    Otherwise we use the single current cycle (rolling features degenerate
    to the raw sensor value — still valid, just less accurate).
    """
    # Build the cycle rows: history (if any) + current cycle
    rows = []
    history = req.history or []

    for i, h in enumerate(history, start=1):
        row = {"unit": 1, "cycle": i}
        row.update({k: v for k, v in h.items() if k.startswith("sensor_")})
        row.update(req.settings)
        rows.append(row)

    # Current cycle
    current_row = {"unit": 1, "cycle": len(history) + 1 if history else req.cycle}
    current_row.update({k: v for k, v in req.sensors.items() if k.startswith("sensor_")})
    current_row.update(req.settings)
    rows.append(current_row)

    df = pd.DataFrame(rows)

    # Add a dummy RUL column (required by feature functions, not used in prediction)
    df["rul"] = 0

    sensor_cols = _sensor_cols(df)

    # Apply same feature engineering as training
    df = add_cycle_progress(df)
    df = add_rolling_features(df, cfg.features.rolling_windows, sensor_cols)
    df = add_lag_features(df, cfg.features.lag_steps, sensor_cols)
    df = add_trend_features(df, cfg.features.trend_window, sensor_cols)

    # Fill any residual NaN (short history → head rows)
    df = df.fillna(0)

    # Return only the last row (current cycle)
    return df.iloc[[-1]]


def _align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Align DataFrame columns to the exact feature list the model was trained on.
    Missing columns are filled with 0; extra columns are dropped.
    """
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    available = [c for c in feature_names if c in df.columns]
    return df[available]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and model status check",
    tags=["operations"],
)
async def health() -> HealthResponse:
    loader = get_loader()
    uptime = time.time() - _start_time

    if loader is None or not loader.is_loaded():
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            uptime_seconds=round(uptime, 2),
        )

    lm = loader.get()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=lm.model_name,
        model_version=lm.model_version,
        model_stage=lm.model_stage,
        uptime_seconds=round(uptime, 2),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Loaded model metadata",
    tags=["operations"],
)
async def model_info() -> ModelInfoResponse:
    loader = get_loader()
    if loader is None or not loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model loaded. Run 'dvc repro' to train and register a model.",
        )
    lm = loader.get()
    return ModelInfoResponse(
        model_name=lm.model_name,
        model_version=lm.model_version,
        model_stage=lm.model_stage,
        run_id=lm.run_id,
        feature_names=lm.feature_names,
        feature_count=len(lm.feature_names),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict remaining useful life from sensor readings",
    tags=["prediction"],
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        422: {"description": "Validation error (missing sensors etc.)"},
    },
)
async def predict(req: PredictRequest) -> PredictResponse:
    loader = get_loader()

    if loader is None or not loader.is_loaded():
        PREDICT_REQUESTS.labels(status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model loaded. Run 'dvc repro' to train and register a model first.",
        )

    lm = loader.get()
    cfg = get_settings()

    t_start = time.perf_counter()

    try:
        # Feature engineering
        df_feat = _engineer_features_for_request(req, cfg)

        # Align to training feature set
        if lm.feature_names:
            df_feat = _align_features(df_feat, lm.feature_names)
        else:
            # Fallback: drop non-feature columns
            non_feat = {"unit", "cycle", "rul", "setting_1", "setting_2", "setting_3"}
            df_feat = df_feat.drop(columns=[c for c in non_feat if c in df_feat.columns])

        pred = float(lm.predict(df_feat.values)[0])
        pred = max(0.0, pred)  # RUL is non-negative

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Prometheus
        PREDICT_REQUESTS.labels(status="success").inc()
        PREDICT_LATENCY.observe(latency_ms / 1000)
        PREDICTED_RUL.observe(pred)

        # --- Drift detection: record this inference sample ---
        detector = get_detector()
        if detector is not None:
            try:
                feature_record = df_feat.iloc[0].to_dict()
                detector.record(feature_record)
                # Trigger a KS check if the buffer is full (non-blocking)
                detector.check()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Drift record failed (non-fatal): %s", exc)

        warning = None
        if pred < 30:
            warning = f"⚠️  Low RUL ({pred:.1f} cycles): schedule maintenance soon."

        return PredictResponse(
            unit_id=req.unit_id,
            predicted_rul=round(pred, 2),
            model_name=lm.model_name,
            model_version=lm.model_version,
            model_stage=lm.model_stage,
            latency_ms=round(latency_ms, 2),
            warning=warning,
        )

    except Exception as exc:
        PREDICT_REQUESTS.labels(status="error").inc()
        logger.exception("Prediction failed for unit_id=%s", req.unit_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    tags=["operations"],
    response_class=PlainTextResponse,
)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    "/drift-report",
    summary="Latest feature drift report (KS test)",
    tags=["monitoring"],
    responses={
        200: {"description": "Drift report from the last completed detection run"},
        202: {"description": "Buffer not yet full — partial report returned"},
        503: {"description": "Drift detector not initialised"},
    },
)
async def drift_report(force: bool = False):
    """
    Returns the latest feature-distribution drift report.

    - **force=false** (default): returns the last *completed* report
      (based on a full `window_size` batch of inferences).
    - **force=true**: immediately runs KS tests on whatever is currently
      buffered, regardless of window size — useful for smoke-testing.
    """
    detector = get_detector()
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Drift detector not initialised. "
                "Ensure 'data/processed/train_features.parquet' exists "
                "(run 'dvc repro' first)."
            ),
        )

    if force:
        report = detector.force_check()
    else:
        report = detector.last_report

    if report is None:
        return {
            "status": "pending",
            "message": f"Buffer has {detector.buffer_size}/{detector.window_size} samples. "
            "Send more /predict requests or use ?force=true.",
            "buffer_size": detector.buffer_size,
            "window_size": detector.window_size,
        }

    return {
        "status": "ok",
        "buffer_size": detector.buffer_size,
        "window_size": detector.window_size,
        **report.to_dict(),
    }


# ---------------------------------------------------------------------------
# Middleware — log every request
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
