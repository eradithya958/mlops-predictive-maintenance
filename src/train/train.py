"""
Model training with MLflow experiment tracking.

Reads feature-engineered parquets, trains an XGBoost (or Random Forest)
regressor on the RUL target, evaluates on val + test sets, logs everything
to MLflow, and registers the model in the MLflow Model Registry.

Run directly:
    python -m src.train.train

Or via DVC:
    dvc repro train
"""

from __future__ import annotations

import json
import logging
import pickle
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Columns that are NOT model features
_NON_FEATURE_COLS = {"unit", "cycle", "rul", "setting_1", "setting_2", "setting_3"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split(processed_dir: Path, split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a feature parquet and return (X, y)."""
    path = processed_dir / f"{split}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run features.py first.")
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["rul"]
    return X, y


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_xgboost(cfg: Settings) -> XGBRegressor:
    mc = cfg.model
    return XGBRegressor(
        n_estimators=mc.n_estimators,
        max_depth=mc.max_depth,
        learning_rate=mc.learning_rate,
        subsample=mc.subsample,
        colsample_bytree=mc.colsample_bytree,
        min_child_weight=mc.min_child_weight,
        reg_alpha=mc.reg_alpha,
        reg_lambda=mc.reg_lambda,
        early_stopping_rounds=mc.early_stopping_rounds,
        tree_method="hist",  # fast CPU training
        random_state=cfg.data.random_seed,
        n_jobs=-1,
        verbosity=0,
    )


def build_random_forest(cfg: Settings) -> RandomForestRegressor:
    mc = cfg.model
    return RandomForestRegressor(
        n_estimators=mc.n_estimators,
        max_depth=mc.max_depth,
        min_samples_leaf=mc.min_child_weight,
        random_state=cfg.data.random_seed,
        n_jobs=-1,
    )


def build_model(cfg: Settings):
    model_type = cfg.model.type.lower()
    if model_type == "xgboost":
        return build_xgboost(cfg)
    elif model_type in ("random_forest", "rf"):
        return build_random_forest(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    tag = f"{prefix}_" if prefix else ""
    return {
        f"{tag}rmse": rmse,
        f"{tag}mae": mae,
        f"{tag}r2": r2,
    }


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------


def plot_feature_importance(model, feature_names: list[str], top_n: int = 20) -> plt.Figure:
    """Return a matplotlib figure of the top-N feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return plt.figure()  # fallback for models without importances

    idx = np.argsort(importances)[-top_n:][::-1]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_vals)), top_vals[::-1], color="#4C72B0")
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title(f"Top {top_n} features by importance")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(cfg: Settings | None = None) -> str:
    """
    Full training pipeline. Returns the MLflow run ID.
    """
    if cfg is None:
        cfg = get_settings()

    # ----- MLflow setup -----
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    logger.info("MLflow tracking URI: %s", cfg.mlflow_tracking_uri)
    logger.info("Experiment: %s", cfg.mlflow.experiment_name)

    # ----- Load data -----
    X_train, y_train = load_split(cfg.processed_dir, "train")
    X_val, y_val = load_split(cfg.processed_dir, "val")
    X_test, y_test = load_split(cfg.processed_dir, "test")
    feature_names = list(X_train.columns)

    logger.info(
        "Data loaded: train=%s  val=%s  test=%s  features=%d",
        X_train.shape,
        X_val.shape,
        X_test.shape,
        len(feature_names),
    )

    # ----- Build model -----
    model = build_model(cfg)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run ID: %s", run_id)

        # Log all hyperparameters
        mlflow.log_params(
            {
                "model_type": cfg.model.type,
                "n_estimators": cfg.model.n_estimators,
                "max_depth": cfg.model.max_depth,
                "learning_rate": cfg.model.learning_rate,
                "subsample": cfg.model.subsample,
                "colsample_bytree": cfg.model.colsample_bytree,
                "min_child_weight": cfg.model.min_child_weight,
                "reg_alpha": cfg.model.reg_alpha,
                "reg_lambda": cfg.model.reg_lambda,
                "rul_clip": cfg.data.rul_clip,
                "dataset": cfg.data.dataset,
                "n_features": len(feature_names),
                "rolling_windows": str(cfg.features.rolling_windows),
                "lag_steps": str(cfg.features.lag_steps),
                "trend_window": cfg.features.trend_window,
            }
        )

        mlflow.set_tags(
            {
                "phase": "1",
                "dataset": cfg.data.dataset,
                "model_type": cfg.model.type,
                "task": "rul_regression",
            }
        )

        # ----- Train -----
        logger.info("Training %s …", cfg.model.type)
        if cfg.model.type.lower() == "xgboost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        # ----- Evaluate -----
        val_metrics = compute_metrics(y_val, model.predict(X_val), prefix="val")
        test_metrics = compute_metrics(y_test, model.predict(X_test), prefix="test")
        all_metrics = {**val_metrics, **test_metrics}

        mlflow.log_metrics(all_metrics)

        # ----- Artifacts -----
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 1. Serialised model
            model_pkl = tmp_path / "model.pkl"
            with open(model_pkl, "wb") as fh:
                pickle.dump(model, fh)
            mlflow.log_artifact(str(model_pkl), artifact_path="model")

            # 2. Feature importance plot
            fig = plot_feature_importance(model, feature_names, top_n=25)
            plot_path = tmp_path / "feature_importance.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

            # 3. Feature names list (for inference-time column alignment)
            features_json = tmp_path / "feature_names.json"
            features_json.write_text(json.dumps(feature_names))
            mlflow.log_artifact(str(features_json), artifact_path="model")

            # 4. params.yaml snapshot
            params_src = cfg.project_root / "params.yaml"
            if params_src.exists():
                mlflow.log_artifact(str(params_src), artifact_path="config")

        # 5. Log model to MLflow model registry via sklearn flavour
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_model",
            registered_model_name=cfg.mlflow.model_registry_name,
            input_example=X_train.head(5),
        )

        # ----- Print summary -----
        logger.info("\n" + "=" * 52)
        logger.info("  Training complete — MLflow run %s", run_id[:8])
        logger.info("=" * 52)
        for k, v in sorted(all_metrics.items()):
            logger.info("  %-18s %.4f", k, v)
        logger.info("=" * 52)

        # Save run ID to file so DVC / downstream scripts can read it
        run_id_path = cfg.models_dir / "latest_run_id.txt"
        run_id_path.write_text(run_id)

        # Write metrics.json for DVC metric tracking
        metrics_path = cfg.models_dir / "metrics.json"
        metrics_path.write_text(json.dumps(all_metrics, indent=2))

    return run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train()
