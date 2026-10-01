"""
MLflow model loader — handles registry queries, model caching, and version management.

The loader is designed as a singleton: loaded once at startup and cached.
It exposes a reload() method used by Phase 3 auto-promotion logic.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

# Stage precedence: prefer Production, fall back to Staging
_STAGE_PREFERENCE = ["Production", "Staging"]


@dataclass
class LoadedModel:
    """Container for a loaded MLflow model and its metadata."""

    model: Any  # sklearn-compatible estimator
    model_name: str
    model_version: str
    model_stage: str
    run_id: str
    feature_names: list[str]  # ordered list of expected feature column names

    def predict(self, X) -> Any:  # noqa: N803
        return self.model.predict(X)


class ModelLoader:
    """
    Manages loading and caching of the MLflow-registered prediction model.

    Usage:
        loader = ModelLoader(tracking_uri="...", model_name="turbofan_rul")
        loader.load()                # call once at startup
        loaded = loader.get()        # returns LoadedModel or None
    """

    def __init__(
        self,
        tracking_uri: str,
        model_name: str = "turbofan_rul",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self._loaded: LoadedModel | None = None

        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load the best available model version from the registry.
        Tries Production first, then Staging.
        Returns True if a model was loaded successfully.
        """
        for stage in _STAGE_PREFERENCE:
            try:
                loaded = self._load_from_stage(stage)
                if loaded:
                    self._loaded = loaded
                    logger.info(
                        "Model loaded: %s v%s (%s) — %d features",
                        loaded.model_name,
                        loaded.model_version,
                        loaded.model_stage,
                        len(loaded.feature_names),
                    )
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s model: %s", stage, exc)

        logger.error(
            "No model found in registry '%s' at stages %s. Run Phase 1 training first (dvc repro).",
            self.model_name,
            _STAGE_PREFERENCE,
        )
        return False

    def reload(self) -> bool:
        """Force re-load from registry (called after auto-promotion in CI/CD)."""
        logger.info("Reloading model from registry…")
        return self.load()

    def get(self) -> LoadedModel | None:
        """Return the currently cached model, or None if not loaded."""
        return self._loaded

    def is_loaded(self) -> bool:
        return self._loaded is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_stage(self, stage: str) -> LoadedModel | None:
        """Fetch and deserialise the latest model at the given stage."""
        versions = self._client.get_latest_versions(self.model_name, stages=[stage])
        if not versions:
            return None

        # Pick the highest version number
        mv = max(versions, key=lambda v: int(v.version))
        run_id = mv.run_id

        # Download feature names artifact (written by train.py)
        feature_names = self._load_feature_names(run_id)

        # Load the raw pickle model (lighter than pyfunc for prediction)
        model = self._load_model_artifact(run_id)

        return LoadedModel(
            model=model,
            model_name=self.model_name,
            model_version=mv.version,
            model_stage=stage,
            run_id=run_id,
            feature_names=feature_names,
        )

    def _load_feature_names(self, run_id: str) -> list[str]:
        """Download feature_names.json artifact from the MLflow run."""
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="model/feature_names.json",
                tracking_uri=self.tracking_uri,
            )
            with open(local_path) as fh:
                return json.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load feature_names.json: %s — using empty list", exc)
            return []

    def _load_model_artifact(self, run_id: str) -> Any:
        """Download model.pkl artifact and unpickle it."""
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="model/model.pkl",
                tracking_uri=self.tracking_uri,
            )
            with open(local_path, "rb") as fh:
                return pickle.load(fh)  # noqa: S301
        except Exception:  # noqa: BLE001
            # Fall back to the sklearn MLflow flavour
            logger.info("model.pkl not found — loading via mlflow.sklearn")
            model_uri = f"runs:/{run_id}/sklearn_model"
            return mlflow.sklearn.load_model(model_uri)


# ---------------------------------------------------------------------------
# Module-level singleton — imported by app.py
# ---------------------------------------------------------------------------

_loader_instance: ModelLoader | None = None


def init_loader(tracking_uri: str, model_name: str = "turbofan_rul") -> ModelLoader:
    """Initialise (or replace) the global loader instance."""
    global _loader_instance  # noqa: PLW0603
    _loader_instance = ModelLoader(tracking_uri=tracking_uri, model_name=model_name)
    return _loader_instance


def get_loader() -> ModelLoader | None:
    """Return the global loader (may be None if not yet initialised)."""
    return _loader_instance
