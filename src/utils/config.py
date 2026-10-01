"""
Central configuration — reads params.yaml and optional environment variables.
All other modules import from here rather than hardcoding paths or values.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Root directory of the project (the folder containing params.yaml)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Pydantic sub-models for each YAML section
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    dataset: str = "FD001"
    rul_clip: int = 125
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42

    @field_validator("test_size", "val_size")
    @classmethod
    def _valid_fraction(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("Split fractions must be between 0 and 1 exclusive")
        return v


class FeaturesConfig(BaseModel):
    rolling_windows: list[int] = [5, 10, 20]
    lag_steps: list[int] = [1, 3, 5]
    trend_window: int = 20
    drop_constant_sensors: bool = True


class ModelConfig(BaseModel):
    type: str = "xgboost"
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30


class MLflowConfig(BaseModel):
    experiment_name: str = "predictive_maintenance"
    run_name: str = "baseline"
    tracking_uri: str = "mlruns"
    model_registry_name: str = "turbofan_rul"


class ParamsFile(BaseModel):
    """Mirrors the top-level structure of params.yaml."""

    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    mlflow: MLflowConfig = MLflowConfig()


# ---------------------------------------------------------------------------
# Main settings class — combines params.yaml with env-var overrides
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Usage:
        from src.utils.config import get_settings
        cfg = get_settings()
        cfg.data.rul_clip       # 125
        cfg.paths.raw_dir       # Path to data/raw/
    """

    model_config = SettingsConfigDict(env_prefix="MLOPS_", env_nested_delimiter="__")

    # Resolved at runtime from params.yaml — not env-var backed individually
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    mlflow: MLflowConfig = MLflowConfig()

    # Paths (can be overridden via MLOPS_DATA_DIR, MLOPS_MODELS_DIR, etc.)
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def mlflow_tracking_uri(self) -> str:
        """Resolve tracking URI: env var > params.yaml > default."""
        env_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if env_uri:
            return env_uri
        uri = self.mlflow.tracking_uri
        # If relative, resolve against project root
        if not uri.startswith(("http://", "https://", "file://", "sqlite:///")):
            return str(PROJECT_ROOT / uri)
        return uri


def _load_params_yaml(path: Path | None = None) -> dict:
    """Load params.yaml from project root (or a custom path)."""
    yaml_path = path or (PROJECT_ROOT / "params.yaml")
    if not yaml_path.exists():
        return {}
    with open(yaml_path) as fh:
        return yaml.safe_load(fh) or {}


def get_settings(params_path: Path | None = None) -> Settings:
    """
    Factory: parse params.yaml then overlay with environment variables.
    Call once at module level and cache the result, or call per-script.
    """
    raw = _load_params_yaml(params_path)
    parsed = ParamsFile.model_validate(raw)

    settings = Settings(
        data=parsed.data,
        features=parsed.features,
        model=parsed.model,
        mlflow=parsed.mlflow,
    )

    # Ensure directories exist
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    return settings
