"""
src/monitoring/drift.py
=======================
Feature-distribution drift detector using the Kolmogorov-Smirnov (KS) test.

Design
------
• At API startup the detector loads the **training-set feature statistics**
  (mean, std, and raw sample arrays for each feature) from a snapshot parquet
  saved during training.
• Incoming prediction requests feed a **rolling window buffer** (configurable
  size, default 500 rows).  Once the buffer is full it is flushed and compared
  against the training distribution via the two-sample KS test.
• Results are exposed as Prometheus Gauges so Grafana can alert when
  p-values drop below a threshold.

Public API
----------
    detector = DriftDetector.from_parquet(path, window_size=500)
    detector.record(feature_dict)          # called after each /predict
    report   = detector.check()            # returns DriftReport (or None if buffer not full)

Prometheus metrics exposed
--------------------------
    drift_ks_statistic{feature="..."}      KS statistic (0–1; higher = more drift)
    drift_ks_pvalue{feature="..."}         KS p-value   (lower = more significant drift)
    drift_detection_runs_total             Total number of drift checks performed
    drift_alerts_total{feature="..."}      Number of times a feature crossed the alert threshold
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_KS_STATISTIC = Gauge(
    "drift_ks_statistic",
    "Kolmogorov-Smirnov statistic per feature (0=no drift, 1=full drift)",
    ["feature"],
)

_KS_PVALUE = Gauge(
    "drift_ks_pvalue",
    "Kolmogorov-Smirnov p-value per feature (low p = significant drift)",
    ["feature"],
)

_DRIFT_RUNS = Counter(
    "drift_detection_runs_total",
    "Total number of completed drift detection runs",
)

_DRIFT_ALERTS = Counter(
    "drift_alerts_total",
    "Number of drift alerts fired per feature",
    ["feature"],
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftResult:
    """Result of a KS test for a single feature."""

    feature: str
    ks_statistic: float
    p_value: float
    drifted: bool  # True if p_value < alert_threshold
    reference_mean: float
    reference_std: float
    window_mean: float
    window_std: float
    n_reference: int
    n_window: int


@dataclass
class DriftReport:
    """Aggregated drift report for one detection run."""

    run_id: int
    n_features_checked: int
    n_features_drifted: int
    alert_threshold: float  # p-value below which we flag drift
    results: list[FeatureDriftResult] = field(default_factory=list)

    @property
    def drifted_features(self) -> list[str]:
        return [r.feature for r in self.results if r.drifted]

    @property
    def overall_drift(self) -> bool:
        return self.n_features_drifted > 0

    def summary(self) -> str:
        if not self.overall_drift:
            return (
                f"[Run {self.run_id}] ✅ No drift detected "
                f"({self.n_features_checked} features checked)"
            )
        return (
            f"[Run {self.run_id}] ⚠️  Drift detected in "
            f"{self.n_features_drifted}/{self.n_features_checked} features: "
            f"{self.drifted_features}"
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "n_features_checked": self.n_features_checked,
            "n_features_drifted": self.n_features_drifted,
            "alert_threshold": self.alert_threshold,
            "overall_drift": self.overall_drift,
            "drifted_features": self.drifted_features,
            "results": [
                {
                    "feature": r.feature,
                    "ks_statistic": round(r.ks_statistic, 6),
                    "p_value": round(r.p_value, 6),
                    "drifted": r.drifted,
                    "reference_mean": round(r.reference_mean, 4),
                    "reference_std": round(r.reference_std, 4),
                    "window_mean": round(r.window_mean, 4),
                    "window_std": round(r.window_std, 4),
                    "n_reference": r.n_reference,
                    "n_window": r.n_window,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Thread-safe drift detector backed by a sliding window buffer.

    Parameters
    ----------
    reference_data : Dict[str, np.ndarray]
        Mapping of feature_name → reference sample array (from training set).
    window_size : int
        Number of inference samples to accumulate before running a KS test.
    alert_threshold : float
        KS p-value below which drift is flagged (default 0.05).
    max_reference_samples : int
        Cap on how many reference samples are kept per feature (for memory).
    """

    def __init__(
        self,
        reference_data: dict[str, np.ndarray],
        window_size: int = 500,
        alert_threshold: float = 0.05,
        max_reference_samples: int = 5_000,
    ) -> None:
        self.reference_data: dict[str, np.ndarray] = {
            feat: arr[:max_reference_samples] for feat, arr in reference_data.items()
        }
        self.window_size = window_size
        self.alert_threshold = alert_threshold

        self._lock = threading.Lock()
        self._buffer: list[dict[str, float]] = []
        self._run_count = 0
        self._last_report: DriftReport | None = None

        logger.info(
            "DriftDetector initialised: %d reference features, window=%d, threshold=%.3f",
            len(self.reference_data),
            self.window_size,
            self.alert_threshold,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(
        cls,
        parquet_path: Path | str,
        feature_cols: list[str] | None = None,
        window_size: int = 500,
        alert_threshold: float = 0.05,
        max_reference_samples: int = 5_000,
    ) -> DriftDetector:
        """
        Build a DriftDetector from a training-set parquet file.

        Parameters
        ----------
        parquet_path : path to the training features parquet (e.g. data/processed/train_features.parquet)
        feature_cols : list of columns to monitor; defaults to all numeric sensor/feature columns
        """
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reference parquet not found: {path}. "
                "Run 'dvc repro' to generate processed data first."
            )

        df = pd.read_parquet(path)

        # Default: monitor all numeric columns except metadata cols
        _meta = {"unit", "cycle", "rul", "setting_1", "setting_2", "setting_3"}
        if feature_cols is None:
            feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns if c not in _meta
            ]

        reference_data = {
            col: df[col].dropna().to_numpy(dtype=float)
            for col in feature_cols
            if col in df.columns and len(df[col].dropna()) > 0
        }

        logger.info(
            "Loaded reference data from %s — %d features, %d rows",
            path.name,
            len(reference_data),
            len(df),
        )

        return cls(
            reference_data=reference_data,
            window_size=window_size,
            alert_threshold=alert_threshold,
            max_reference_samples=max_reference_samples,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, features: dict[str, float]) -> None:
        """
        Record one inference sample.  Thread-safe.

        Parameters
        ----------
        features : flat dict of feature_name → value (as passed to the model).
        """
        with self._lock:
            # Keep only features we track
            row = {k: v for k, v in features.items() if k in self.reference_data}
            if row:
                self._buffer.append(row)

    def check(self) -> DriftReport | None:
        """
        If the buffer is full, run KS tests and return a DriftReport.
        Returns None if not enough samples have accumulated yet.
        Thread-safe — the buffer is atomically swapped out before testing.
        """
        with self._lock:
            if len(self._buffer) < self.window_size:
                return None
            # Swap out the buffer atomically
            batch = self._buffer[: self.window_size]
            self._buffer = self._buffer[self.window_size :]
            self._run_count += 1
            run_id = self._run_count

        # Run KS tests outside the lock (can be slow for many features)
        report = self._run_ks_tests(batch, run_id)
        self._last_report = report

        _DRIFT_RUNS.inc()
        logger.info(report.summary())
        return report

    def force_check(self) -> DriftReport | None:
        """
        Run a drift check on whatever is currently in the buffer,
        even if it is not yet full.  Useful for the /drift-report endpoint.
        Returns None if the buffer is empty.
        """
        with self._lock:
            if not self._buffer:
                return None
            batch = list(self._buffer)
            self._run_count += 1
            run_id = self._run_count

        report = self._run_ks_tests(batch, run_id)
        self._last_report = report
        _DRIFT_RUNS.inc()
        logger.info(report.summary())
        return report

    @property
    def buffer_size(self) -> int:
        """Current number of unprocessed inference samples in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def last_report(self) -> DriftReport | None:
        return self._last_report

    @property
    def reference_feature_count(self) -> int:
        return len(self.reference_data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_ks_tests(
        self,
        batch: list[dict[str, float]],
        run_id: int,
    ) -> DriftReport:
        """Run KS test for each tracked feature against the reference distribution."""
        window_df = pd.DataFrame(batch)
        results: list[FeatureDriftResult] = []

        for feat, ref_arr in self.reference_data.items():
            if feat not in window_df.columns:
                continue

            win_arr = window_df[feat].dropna().to_numpy(dtype=float)
            if len(win_arr) < 2:
                continue

            ks_stat, p_val = stats.ks_2samp(ref_arr, win_arr)
            drifted = bool(p_val < self.alert_threshold)

            result = FeatureDriftResult(
                feature=feat,
                ks_statistic=float(ks_stat),
                p_value=float(p_val),
                drifted=drifted,
                reference_mean=float(ref_arr.mean()),
                reference_std=float(ref_arr.std()),
                window_mean=float(win_arr.mean()),
                window_std=float(win_arr.std()),
                n_reference=len(ref_arr),
                n_window=len(win_arr),
            )
            results.append(result)

            # Push to Prometheus
            _KS_STATISTIC.labels(feature=feat).set(ks_stat)
            _KS_PVALUE.labels(feature=feat).set(p_val)
            if drifted:
                _DRIFT_ALERTS.labels(feature=feat).inc()

        n_drifted = sum(1 for r in results if r.drifted)
        return DriftReport(
            run_id=run_id,
            n_features_checked=len(results),
            n_features_drifted=n_drifted,
            alert_threshold=self.alert_threshold,
            results=results,
        )


# ---------------------------------------------------------------------------
# Module-level singleton — imported by app.py
# ---------------------------------------------------------------------------

_detector_instance: DriftDetector | None = None


def init_detector(
    reference_parquet: Path | str,
    feature_cols: list[str] | None = None,
    window_size: int = 500,
    alert_threshold: float = 0.05,
) -> DriftDetector:
    """Initialise the global detector singleton."""
    global _detector_instance  # noqa: PLW0603
    _detector_instance = DriftDetector.from_parquet(
        parquet_path=reference_parquet,
        feature_cols=feature_cols,
        window_size=window_size,
        alert_threshold=alert_threshold,
    )
    return _detector_instance


def get_detector() -> DriftDetector | None:
    """Return the global detector (may be None if not yet initialised)."""
    return _detector_instance
