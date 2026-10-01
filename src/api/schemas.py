"""
Pydantic schemas for the FastAPI prediction service.
Defines request/response models and sensor validation logic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Sensor & settings definitions (FD001 after constant-sensor removal)
# ---------------------------------------------------------------------------

# Sensors that survive the drop_constant_sensors step on FD001
ACTIVE_SENSORS = {
    "sensor_02",
    "sensor_03",
    "sensor_04",
    "sensor_07",
    "sensor_08",
    "sensor_09",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
}

SETTING_COLS = {"setting_1", "setting_2", "setting_3"}


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """
    Single-cycle prediction request.

    The client sends raw sensor readings for one engine at one cycle.
    The API handles feature engineering internally.

    If you have a rolling window of historical cycles available, put the
    current cycle last in `history` (optional — used for richer rolling features).
    """

    unit_id: str = Field(
        ...,
        description="Unique identifier for the engine unit (arbitrary string).",
        examples=["engine_001"],
    )
    cycle: int = Field(
        ...,
        ge=1,
        description="Current cycle number (1-indexed).",
        examples=[50],
    )
    sensors: dict[str, float] = Field(
        ...,
        description=(
            "Sensor readings keyed by sensor name (e.g. 'sensor_02'). "
            "Must include at least the active sensors for FD001."
        ),
        examples=[{"sensor_02": 641.82, "sensor_03": 1589.70, "sensor_04": 1400.60}],
    )
    settings: dict[str, float] = Field(
        default_factory=lambda: {"setting_1": 0.0, "setting_2": 0.0002, "setting_3": 100.0},
        description="Three operational settings.",
    )
    # Optional history of previous cycles for this unit — improves rolling features
    history: list[dict[str, float]] | None = Field(
        default=None,
        description=(
            "Optional list of prior-cycle sensor dicts (oldest first, current cycle excluded). "
            "Providing at least 20 entries gives the best rolling/lag accuracy."
        ),
    )

    @model_validator(mode="after")
    def _check_required_sensors(self) -> PredictRequest:
        provided = set(self.sensors.keys())
        missing = ACTIVE_SENSORS - provided
        if missing:
            raise ValueError(
                f"Missing required sensor readings: {sorted(missing)}. "
                f"Provide at least the {len(ACTIVE_SENSORS)} active sensors."
            )
        return self


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PredictResponse(BaseModel):
    """Prediction response — predicted RUL in engine cycles."""

    unit_id: str
    predicted_rul: float = Field(
        ..., description="Predicted remaining useful life in engine cycles."
    )
    model_name: str = Field(..., description="MLflow registered model name.")
    model_version: str = Field(..., description="MLflow model version number.")
    model_stage: str = Field(..., description="MLflow model stage (Production/Staging).")
    latency_ms: float = Field(..., description="End-to-end prediction latency in milliseconds.")
    warning: str | None = Field(
        default=None,
        description="Warning message if RUL is critically low (< 30 cycles).",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "ok" | "degraded" | "error"
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    model_stage: str | None = None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Detailed model registry info."""

    model_name: str
    model_version: str
    model_stage: str
    run_id: str
    feature_names: list[str]
    feature_count: int


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: str | None = None
