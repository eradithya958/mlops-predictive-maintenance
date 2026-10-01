"""
tests/test_drift.py
===================
Tests for the KS-test based drift detector (src/monitoring/drift.py).

Covers:
  • DriftDetector initialisation from reference arrays
  • Buffer accumulation and flush-on-full behaviour
  • KS test detects obvious distributional shift
  • KS test correctly passes for samples drawn from the same distribution
  • force_check() works on partial buffers
  • Thread-safety (concurrent record() calls)
  • DriftReport serialisation
"""

from __future__ import annotations

import threading

import numpy as np

from src.monitoring.drift import DriftDetector, DriftReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector(
    n_ref: int = 1000,
    window_size: int = 50,
    alert_threshold: float = 0.05,
    n_features: int = 3,
    seed: int = 0,
) -> DriftDetector:
    """Build a DriftDetector with synthetic Gaussian reference data."""
    rng = np.random.default_rng(seed)
    reference_data = {
        f"sensor_{i:02d}": rng.normal(loc=float(i), scale=1.0, size=n_ref)
        for i in range(n_features)
    }
    return DriftDetector(
        reference_data=reference_data,
        window_size=window_size,
        alert_threshold=alert_threshold,
    )


def _sample_row(
    means: dict[str, float],
    std: float = 1.0,
    seed: int | None = None,
) -> dict[str, float]:
    """Draw one sample row from Gaussian distributions."""
    rng = np.random.default_rng(seed)
    return {feat: float(rng.normal(loc=mu, scale=std)) for feat, mu in means.items()}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestDriftDetectorInit:
    def test_reference_feature_count(self):
        det = _make_detector(n_features=5)
        assert det.reference_feature_count == 5

    def test_buffer_starts_empty(self):
        det = _make_detector()
        assert det.buffer_size == 0

    def test_last_report_is_none_initially(self):
        det = _make_detector()
        assert det.last_report is None

    def test_max_reference_samples_capped(self):
        det = DriftDetector(
            reference_data={"f": np.ones(10_000)},
            window_size=10,
            max_reference_samples=500,
        )
        assert len(det.reference_data["f"]) == 500


# ---------------------------------------------------------------------------
# Buffer accumulation
# ---------------------------------------------------------------------------


class TestBufferAccumulation:
    def test_record_increases_buffer(self):
        det = _make_detector(window_size=20)
        for _ in range(10):
            det.record({"sensor_00": 1.0, "sensor_01": 2.0, "sensor_02": 3.0})
        assert det.buffer_size == 10

    def test_check_returns_none_before_full(self):
        det = _make_detector(window_size=50)
        for _ in range(30):
            det.record({"sensor_00": 1.0, "sensor_01": 2.0, "sensor_02": 3.0})
        assert det.check() is None

    def test_check_returns_report_when_full(self):
        det = _make_detector(window_size=20)
        for _ in range(20):
            det.record({"sensor_00": 1.0, "sensor_01": 2.0, "sensor_02": 3.0})
        report = det.check()
        assert report is not None
        assert isinstance(report, DriftReport)

    def test_buffer_flushed_after_check(self):
        det = _make_detector(window_size=10)
        for _ in range(10):
            det.record({"sensor_00": 1.0, "sensor_01": 2.0, "sensor_02": 3.0})
        det.check()
        assert det.buffer_size == 0

    def test_overflow_stays_in_buffer(self):
        """Records beyond window_size remain buffered for next check."""
        det = _make_detector(window_size=10)
        for _ in range(15):
            det.record({"sensor_00": 1.0, "sensor_01": 2.0, "sensor_02": 3.0})
        det.check()
        assert det.buffer_size == 5

    def test_unknown_feature_is_ignored(self):
        """Recording a feature not in reference_data does not crash."""
        det = _make_detector(n_features=2)
        det.record({"sensor_99": 999.0})  # not in reference
        assert det.buffer_size == 0  # nothing tracked

    def test_partial_feature_dict_is_recorded(self):
        """Recording a dict with a mix of known and unknown features is fine."""
        det = _make_detector(n_features=3)  # sensor_00, sensor_01, sensor_02
        det.record({"sensor_00": 1.0, "sensor_UNKNOWN": 9.9})
        assert det.buffer_size == 1


# ---------------------------------------------------------------------------
# Statistical correctness
# ---------------------------------------------------------------------------


class TestKSTestResults:
    def test_no_drift_same_distribution(self):
        """Samples from the same distribution should not trigger drift alert."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, size=2000)
        det = DriftDetector(
            reference_data={"feat": ref},
            window_size=300,
            alert_threshold=0.05,
        )
        for _ in range(300):
            det.record({"feat": float(rng.normal(0, 1))})

        report = det.check()
        assert report is not None
        # With a clean same-distribution draw, p_value should be >> 0.05
        feat_result = report.results[0]
        assert not feat_result.drifted, (
            f"False positive: p={feat_result.p_value:.4f}, ks={feat_result.ks_statistic:.4f}"
        )

    def test_drift_detected_shifted_mean(self):
        """Samples with a 5-sigma mean shift must be flagged."""
        rng = np.random.default_rng(7)
        ref = rng.normal(0, 1, size=2000)
        det = DriftDetector(
            reference_data={"feat": ref},
            window_size=200,
            alert_threshold=0.05,
        )
        # Inject data from a heavily shifted distribution
        for _ in range(200):
            det.record({"feat": float(rng.normal(5, 1))})  # 5-sigma shift

        report = det.check()
        assert report is not None
        feat_result = report.results[0]
        assert feat_result.drifted, (
            f"Drift not detected: p={feat_result.p_value:.6f}, ks={feat_result.ks_statistic:.4f}"
        )
        assert feat_result.ks_statistic > 0.5

    def test_drift_report_counts(self):
        """n_features_drifted is correctly counted."""
        rng = np.random.default_rng(0)
        ref_data = {
            "stable": rng.normal(0, 1, size=1000),
            "drifted": rng.normal(0, 1, size=1000),
        }
        det = DriftDetector(
            reference_data=ref_data,
            window_size=100,
            alert_threshold=0.05,
        )
        for _ in range(100):
            det.record(
                {
                    "stable": float(rng.normal(0, 1)),  # same distribution
                    "drifted": float(rng.normal(10, 1)),  # obvious shift
                }
            )

        report = det.check()
        assert report is not None
        assert report.n_features_drifted == 1
        assert "drifted" in report.drifted_features
        assert "stable" not in report.drifted_features


# ---------------------------------------------------------------------------
# force_check
# ---------------------------------------------------------------------------


class TestForceCheck:
    def test_force_check_on_partial_buffer(self):
        """force_check() works even if buffer < window_size."""
        det = _make_detector(window_size=100)
        for _ in range(30):
            det.record({"sensor_00": 0.0, "sensor_01": 0.0, "sensor_02": 0.0})
        report = det.force_check()
        assert report is not None
        assert report.n_features_checked > 0

    def test_force_check_on_empty_buffer_returns_none(self):
        det = _make_detector()
        assert det.force_check() is None

    def test_force_check_updates_last_report(self):
        det = _make_detector(window_size=100)
        for _ in range(10):
            det.record({"sensor_00": 1.0, "sensor_01": 1.0, "sensor_02": 1.0})
        det.force_check()
        assert det.last_report is not None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_calls(self):
        """Multiple threads recording simultaneously should not corrupt state."""
        det = _make_detector(window_size=1000)
        n_threads = 10
        records_per_thread = 50

        def worker():
            for _ in range(records_per_thread):
                det.record({"sensor_00": 1.0, "sensor_01": 2.0, "sensor_02": 3.0})

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert det.buffer_size == n_threads * records_per_thread


# ---------------------------------------------------------------------------
# DriftReport serialisation
# ---------------------------------------------------------------------------


class TestDriftReportSerialization:
    def _get_report(self) -> DriftReport:
        rng = np.random.default_rng(1)
        det = DriftDetector(
            reference_data={"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 1, 500)},
            window_size=50,
        )
        for _ in range(50):
            det.record({"f1": float(rng.normal(0, 1)), "f2": float(rng.normal(5, 1))})
        return det.check()

    def test_to_dict_keys(self):
        report = self._get_report()
        d = report.to_dict()
        assert "run_id" in d
        assert "n_features_checked" in d
        assert "n_features_drifted" in d
        assert "overall_drift" in d
        assert "drifted_features" in d
        assert "results" in d

    def test_to_dict_results_structure(self):
        report = self._get_report()
        d = report.to_dict()
        for r in d["results"]:
            assert "feature" in r
            assert "ks_statistic" in r
            assert "p_value" in r
            assert "drifted" in r

    def test_summary_string(self):
        report = self._get_report()
        s = report.summary()
        assert isinstance(s, str)
        assert len(s) > 0
