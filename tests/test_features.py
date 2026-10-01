"""
Unit tests for src/data/features.py.
All tests operate on small in-memory DataFrames — fast, deterministic, no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    _sensor_cols,
    add_cycle_progress,
    add_lag_features,
    add_rolling_features,
    add_trend_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_df():
    """
    A minimal DataFrame with 2 units, 10 cycles each.
    Unit 1: sensor values 1,2,...,10
    Unit 2: sensor values 10,9,...,1 (decreasing)
    """
    rows = []
    for uid in (1, 2):
        for cycle in range(1, 11):
            val = cycle if uid == 1 else 11 - cycle
            rows.append(
                {
                    "unit": uid,
                    "cycle": cycle,
                    "sensor_01": float(val),
                    "sensor_02": float(val * 2),
                    "rul": 10 - cycle,
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cycle progress
# ---------------------------------------------------------------------------


class TestCycleProgress:
    def test_column_exists(self, simple_df):
        df = add_cycle_progress(simple_df)
        assert "cycle_progress" in df.columns

    def test_last_cycle_is_one(self, simple_df):
        df = add_cycle_progress(simple_df)
        last = df.groupby("unit")["cycle_progress"].max()
        np.testing.assert_allclose(last.values, 1.0)

    def test_first_cycle_is_min(self, simple_df):
        df = add_cycle_progress(simple_df)
        first = df[df["cycle"] == 1]["cycle_progress"]
        assert (first > 0).all()
        assert (first < 1).all()

    def test_monotonically_increasing_within_unit(self, simple_df):
        df = add_cycle_progress(simple_df)
        for uid, grp in df.groupby("unit"):
            progress = grp.sort_values("cycle")["cycle_progress"].values
            assert np.all(np.diff(progress) > 0), f"Unit {uid}: cycle_progress not increasing"


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------


class TestRollingFeatures:
    def test_expected_number_of_columns(self, simple_df):
        sensors = _sensor_cols(simple_df)
        windows = [3, 5]
        df = add_rolling_features(simple_df, windows, sensors)
        # Each sensor × each window × (mean + std)
        expected_new = len(sensors) * len(windows) * 2
        new_cols = [c for c in df.columns if "roll_" in c]
        assert len(new_cols) == expected_new

    def test_roll_mean_first_value_equals_sensor(self, simple_df):
        """With min_periods=1, the first row of each unit should equal the sensor value itself."""
        sensors = _sensor_cols(simple_df)
        df = add_rolling_features(simple_df, [3], sensors)
        first_rows = df[df["cycle"] == 1]
        for sensor in sensors:
            np.testing.assert_allclose(
                first_rows[f"{sensor}_roll_mean_3"].values,
                first_rows[sensor].values,
                err_msg=f"{sensor}: first-row rolling mean ≠ sensor value",
            )

    def test_roll_std_at_first_row_is_zero(self, simple_df):
        """Std of a single observation is 0."""
        sensors = _sensor_cols(simple_df)
        df = add_rolling_features(simple_df, [3], sensors)
        first_rows = df[df["cycle"] == 1]
        for sensor in sensors:
            np.testing.assert_allclose(first_rows[f"{sensor}_roll_std_3"].values, 0.0)

    def test_no_cross_unit_contamination(self, simple_df):
        """Rolling computation must not bleed across unit boundaries."""
        sensors = _sensor_cols(simple_df)
        df = add_rolling_features(simple_df, [5], sensors)
        # Unit 1 row at cycle=1 should NOT be influenced by unit 2
        u1_c1 = df[(df["unit"] == 1) & (df["cycle"] == 1)]
        assert u1_c1[f"{sensors[0]}_roll_mean_5"].values[0] == pytest.approx(
            simple_df[(simple_df["unit"] == 1) & (simple_df["cycle"] == 1)][sensors[0]].values[0]
        )

    def test_no_nan_in_rolling_output(self, simple_df):
        sensors = _sensor_cols(simple_df)
        df = add_rolling_features(simple_df, [3, 5], sensors)
        roll_cols = [c for c in df.columns if "roll_" in c]
        assert df[roll_cols].isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


class TestLagFeatures:
    def test_expected_number_of_columns(self, simple_df):
        sensors = _sensor_cols(simple_df)
        lags = [1, 2]
        df = add_lag_features(simple_df, lags, sensors)
        expected_new = len(sensors) * len(lags)
        new_cols = [c for c in df.columns if "_lag_" in c]
        assert len(new_cols) == expected_new

    def test_lag_correctness(self, simple_df):
        """Lag-1 of cycle N should equal the sensor value at cycle N-1 (within unit)."""
        sensors = _sensor_cols(simple_df)
        df = add_lag_features(simple_df, [1], sensors)

        for uid, grp in df.groupby("unit"):
            grp = grp.sort_values("cycle").reset_index(drop=True)
            for sensor in sensors:
                lag_col = f"{sensor}_lag_1"
                for i in range(1, len(grp)):
                    expected = grp.loc[i - 1, sensor]
                    actual = grp.loc[i, lag_col]
                    assert actual == pytest.approx(expected), (
                        f"Unit {uid} cycle {grp.loc[i, 'cycle']}: {lag_col}={actual} ≠ {expected}"
                    )

    def test_no_nan_in_lag_output(self, simple_df):
        """After bfill, there should be no NaN remaining."""
        sensors = _sensor_cols(simple_df)
        df = add_lag_features(simple_df, [1, 3], sensors)
        lag_cols = [c for c in df.columns if "_lag_" in c]
        assert df[lag_cols].isnull().sum().sum() == 0

    def test_no_cross_unit_contamination_in_lags(self, simple_df):
        """Lag of cycle=1 in unit 2 must NOT pick up values from unit 1."""
        sensors = _sensor_cols(simple_df)
        # Unit 1 has sensor_01 = [1,2,...,10], unit 2 = [10,9,...,1]
        df = add_lag_features(simple_df, [1], sensors)
        u2_c2 = df[(df["unit"] == 2) & (df["cycle"] == 2)]
        # lag_1 of unit2, cycle2 = unit2, cycle1 = 10.0
        assert u2_c2["sensor_01_lag_1"].values[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Trend features
# ---------------------------------------------------------------------------


class TestTrendFeatures:
    def test_expected_columns(self, simple_df):
        sensors = _sensor_cols(simple_df)
        df = add_trend_features(simple_df, window=5, sensor_cols=sensors)
        trend_cols = [c for c in df.columns if "_slope_" in c]
        assert len(trend_cols) == len(sensors)

    def test_increasing_sensor_positive_slope(self, simple_df):
        """Unit 1 has sensor_01 increasing → slope must be positive."""
        sensors = ["sensor_01"]
        df = add_trend_features(simple_df, window=5, sensor_cols=sensors)
        u1 = df[(df["unit"] == 1) & (df["cycle"] >= 5)]  # after window fills
        assert (u1["sensor_01_slope_5"] > 0).all()

    def test_decreasing_sensor_negative_slope(self, simple_df):
        """Unit 2 has sensor_01 decreasing → slope must be negative."""
        sensors = ["sensor_01"]
        df = add_trend_features(simple_df, window=5, sensor_cols=sensors)
        u2 = df[(df["unit"] == 2) & (df["cycle"] >= 5)]
        assert (u2["sensor_01_slope_5"] < 0).all()

    def test_no_nan_in_trend_output(self, simple_df):
        sensors = _sensor_cols(simple_df)
        df = add_trend_features(simple_df, window=5, sensor_cols=sensors)
        slope_cols = [c for c in df.columns if "_slope_" in c]
        assert df[slope_cols].isnull().sum().sum() == 0
