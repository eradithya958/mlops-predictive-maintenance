"""
Tests for the FastAPI prediction service (src/api/app.py).
Uses FastAPI's TestClient — no real MLflow server or Docker required.
The model loader is mocked so tests are fast and hermetic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Sensor values covering all ACTIVE_SENSORS in schemas.py
_FULL_SENSORS = {
    "sensor_02": 641.82,
    "sensor_03": 1589.70,
    "sensor_04": 1400.60,
    "sensor_07": 554.36,
    "sensor_08": 2388.02,
    "sensor_09": 9063.29,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.02,
    "sensor_14": 8127.55,
    "sensor_15": 8.4195,
    "sensor_17": 392.0,
    "sensor_20": 38.86,
    "sensor_21": 23.3735,
}

_SETTINGS = {"setting_1": 0.0, "setting_2": 0.0002, "setting_3": 100.0}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_loader():
    """Returns a mock LoadedModel that always predicts RUL=75."""
    from src.api.model_loader import LoadedModel

    feature_names = list(_FULL_SENSORS.keys()) + ["cycle_progress"]
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([75.0])

    loaded = LoadedModel(
        model=mock_model,
        model_name="turbofan_rul",
        model_version="1",
        model_stage="Staging",
        run_id="abc123",
        feature_names=feature_names,
    )
    return loaded


@pytest.fixture()
def client(mock_loader):
    """
    TestClient with the model loader mocked to return mock_loader.
    Patches get_loader() and init_loader() so no MLflow is needed.
    """
    with (
        patch("src.api.app.get_loader") as mock_get,
        patch("src.api.app.init_loader") as mock_init,
        patch("src.api.app.get_settings") as mock_cfg,
    ):
        mock_get.return_value = MagicMock(
            is_loaded=lambda: True,
            get=lambda: mock_loader,
        )
        mock_init.return_value = mock_get.return_value

        # Minimal config stub
        mock_cfg.return_value = MagicMock(
            mlflow_tracking_uri="mlruns",
            mlflow=MagicMock(model_registry_name="turbofan_rul"),
            features=MagicMock(
                rolling_windows=[5, 10, 20],
                lag_steps=[1, 3, 5],
                trend_window=20,
            ),
        )

        from src.api.app import app

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def client_no_model():
    """TestClient where no model is loaded (simulates cold start without training)."""
    with (
        patch("src.api.app.get_loader") as mock_get,
        patch("src.api.app.init_loader"),
        patch("src.api.app.get_settings") as mock_cfg,
    ):
        mock_get.return_value = MagicMock(
            is_loaded=lambda: False,
            get=lambda: None,
        )
        mock_cfg.return_value = MagicMock(
            mlflow_tracking_uri="mlruns",
            mlflow=MagicMock(model_registry_name="turbofan_rul"),
        )

        from src.api.app import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "ok"

    def test_health_no_model_degraded(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_health_includes_uptime(self, client):
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# /predict endpoint — happy path
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def _payload(self, **overrides):
        p = {
            "unit_id": "engine_001",
            "cycle": 50,
            "sensors": _FULL_SENSORS,
            "settings": _SETTINGS,
        }
        p.update(overrides)
        return p

    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=self._payload())
        assert resp.status_code == 200, resp.text

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json=self._payload()).json()
        assert "predicted_rul" in data
        assert "model_version" in data
        assert "latency_ms" in data
        assert "unit_id" in data
        assert data["unit_id"] == "engine_001"

    def test_predict_rul_non_negative(self, client):
        data = client.post("/predict", json=self._payload()).json()
        assert data["predicted_rul"] >= 0

    def test_predict_warning_when_low_rul(self, client, mock_loader):
        """When predicted RUL < 30, a warning string must be present."""
        mock_loader.model.predict.return_value = np.array([15.0])
        data = client.post("/predict", json=self._payload()).json()
        assert data["warning"] is not None
        assert "Low RUL" in data["warning"]

    def test_predict_no_warning_high_rul(self, client, mock_loader):
        mock_loader.model.predict.return_value = np.array([80.0])
        data = client.post("/predict", json=self._payload()).json()
        assert data["warning"] is None

    def test_predict_with_history(self, client):
        history = [{f"sensor_{i:02d}": float(i) for i in range(2, 22)} for _ in range(10)]
        payload = self._payload(history=history)
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /predict — validation / error cases
# ---------------------------------------------------------------------------


class TestPredictValidation:
    def test_missing_sensors_returns_422(self, client):
        """Request without required sensors should fail schema validation."""
        payload = {
            "unit_id": "bad_engine",
            "cycle": 10,
            "sensors": {"sensor_02": 641.82},  # missing most sensors
            "settings": _SETTINGS,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_cycle_returns_422(self, client):
        payload = {
            "unit_id": "e1",
            "cycle": 0,  # must be >= 1
            "sensors": _FULL_SENSORS,
            "settings": _SETTINGS,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_unit_id_returns_422(self, client):
        payload = {"cycle": 10, "sensors": _FULL_SENSORS, "settings": _SETTINGS}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_no_model_returns_503(self, client_no_model):
        payload = {
            "unit_id": "e1",
            "cycle": 10,
            "sensors": _FULL_SENSORS,
            "settings": _SETTINGS,
        }
        resp = client_no_model.post("/predict", json=payload)
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_contains_expected_names(self, client):
        # Trigger a prediction first to populate counters
        client.post(
            "/predict",
            json={
                "unit_id": "x",
                "cycle": 1,
                "sensors": _FULL_SENSORS,
                "settings": _SETTINGS,
            },
        )
        body = client.get("/metrics").text
        assert "prediction_requests_total" in body
        assert "prediction_latency_seconds" in body
        assert "predicted_rul_value" in body


# ---------------------------------------------------------------------------
# /model-info endpoint
# ---------------------------------------------------------------------------


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        with patch("src.api.app.get_loader") as mock_get:
            from src.api.model_loader import LoadedModel

            lm = LoadedModel(
                model=MagicMock(),
                model_name="turbofan_rul",
                model_version="2",
                model_stage="Production",
                run_id="xyz789",
                feature_names=["f1", "f2"],
            )
            mock_get.return_value = MagicMock(is_loaded=lambda: True, get=lambda: lm)
            resp = client.get("/model-info")
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_version"] == "2"
            assert data["feature_count"] == 2

    def test_model_info_503_when_no_model(self, client_no_model):
        resp = client_no_model.get("/model-info")
        assert resp.status_code == 503
