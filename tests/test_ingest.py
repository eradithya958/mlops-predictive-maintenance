"""
Unit tests for src/data/ingest.py.
Tests run entirely in-memory / on synthetic data — no network calls, no disk I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.ingest import (
    _generate_synthetic_cmapss,
    attach_rul_labels,
    attach_test_rul_labels,
    drop_constant_sensors,
    split_by_unit,
    validate_schema,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


@pytest.fixture()
def synthetic_data(rng):
    train_df, test_df, test_rul = _generate_synthetic_cmapss(
        "FD001", rng, n_train_units=20, n_test_units=10
    )
    return train_df, test_df, test_rul


@pytest.fixture()
def labelled_train(synthetic_data):
    train_df, _, _ = synthetic_data
    return attach_rul_labels(train_df, rul_clip=125)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestRawDataSchema:
    def test_expected_columns(self, synthetic_data):
        train_df, test_df, _ = synthetic_data
        for df in (train_df, test_df):
            for col in ("unit", "cycle"):
                assert col in df.columns, f"Missing required column: {col}"

    def test_sensor_count(self, synthetic_data):
        train_df, _, _ = synthetic_data
        sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
        assert len(sensor_cols) == 21, "Expected 21 sensor columns"

    def test_no_nan_in_core_columns(self, synthetic_data):
        train_df, test_df, _ = synthetic_data
        for df in (train_df, test_df):
            assert not df["unit"].isnull().any()
            assert not df["cycle"].isnull().any()

    def test_cycles_start_at_one(self, synthetic_data):
        train_df, _, _ = synthetic_data
        first_cycles = train_df.groupby("unit")["cycle"].min()
        assert (first_cycles == 1).all(), "All units should start at cycle 1"

    def test_cycles_are_sequential(self, synthetic_data):
        train_df, _, _ = synthetic_data
        for uid, grp in train_df.groupby("unit"):
            cycles = grp["cycle"].values
            assert np.all(np.diff(cycles) == 1), f"Unit {uid}: cycles not sequential"


# ---------------------------------------------------------------------------
# RUL labelling tests
# ---------------------------------------------------------------------------


class TestRULLabelling:
    def test_rul_column_exists(self, labelled_train):
        assert "rul" in labelled_train.columns

    def test_rul_non_negative(self, labelled_train):
        assert (labelled_train["rul"] >= 0).all()

    def test_rul_clipped_at_max(self, labelled_train):
        assert (labelled_train["rul"] <= 125).all()

    def test_last_cycle_rul_is_zero(self, labelled_train):
        """The final cycle of each unit must have RUL = 0."""
        last_rul = labelled_train.groupby("unit").apply(lambda g: g.loc[g["cycle"].idxmax(), "rul"])
        assert (last_rul == 0).all(), "Last cycle of each unit must have RUL=0"

    def test_rul_decreases_within_unit(self, labelled_train):
        """Within each unit, RUL should be non-increasing (once un-clipped region is reached)."""
        for _, grp in labelled_train.groupby("unit"):
            rul_vals = grp.sort_values("cycle")["rul"].values
            # After the clip plateau, values must be non-increasing
            unclipped_idx = np.where(rul_vals < 125)[0]
            if len(unclipped_idx) >= 2:
                assert np.all(np.diff(rul_vals[unclipped_idx[0] :]) <= 0), (
                    "RUL should be non-increasing in the un-clipped region"
                )

    def test_test_rul_labelling(self, synthetic_data):
        _, test_df, test_rul = synthetic_data
        labelled = attach_test_rul_labels(test_df, test_rul, rul_clip=125)
        assert "rul" in labelled.columns
        assert (labelled["rul"] >= 0).all()
        assert (labelled["rul"] <= 125).all()


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_drop_constant_sensors(self, labelled_train):
        # Inject a truly constant column
        df = labelled_train.copy()
        df["sensor_99"] = 42.0
        cleaned, dropped = drop_constant_sensors(df, threshold=1e-6)
        assert "sensor_99" in dropped
        assert "sensor_99" not in cleaned.columns

    def test_variable_sensors_not_dropped(self, labelled_train):
        """Sensors with actual variance should survive."""
        df = labelled_train.copy()
        _, dropped = drop_constant_sensors(df, threshold=1e-6)
        variable_cols = [
            c for c in labelled_train.columns if c.startswith("sensor_") and c not in dropped
        ]
        assert len(variable_cols) > 0


# ---------------------------------------------------------------------------
# Split tests — the most critical for data leakage
# ---------------------------------------------------------------------------


class TestSplitByUnit:
    @pytest.fixture()
    def splits(self, labelled_train):
        return split_by_unit(labelled_train, val_size=0.15, test_size=0.15, random_seed=42)

    def test_no_unit_in_multiple_splits(self, splits):
        """Core anti-leakage test: no engine unit may appear in more than one split."""
        train, val, test = splits
        train_units = set(train["unit"].unique())
        val_units = set(val["unit"].unique())
        test_units = set(test["unit"].unique())

        assert train_units.isdisjoint(val_units), "Unit overlap between train and val!"
        assert train_units.isdisjoint(test_units), "Unit overlap between train and test!"
        assert val_units.isdisjoint(test_units), "Unit overlap between val and test!"

    def test_all_units_accounted_for(self, labelled_train, splits):
        train, val, test = splits
        original_units = set(labelled_train["unit"].unique())
        recovered_units = (
            set(train["unit"].unique()) | set(val["unit"].unique()) | set(test["unit"].unique())
        )
        assert original_units == recovered_units

    def test_split_sizes_are_reasonable(self, labelled_train, splits):
        train, val, test = splits
        total = len(labelled_train)
        # At least 50% should be in training
        assert len(train) / total >= 0.50


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_frame_passes(self, labelled_train):
        validate_schema(labelled_train, "train")  # Should not raise

    def test_missing_rul_raises(self, labelled_train):
        bad = labelled_train.drop(columns=["rul"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_schema(bad)

    def test_negative_rul_raises(self, labelled_train):
        bad = labelled_train.copy()
        bad.loc[bad.index[0], "rul"] = -1
        with pytest.raises(ValueError, match="negative RUL"):
            validate_schema(bad)

    def test_nan_rul_raises(self, labelled_train):
        bad = labelled_train.copy()
        bad.loc[bad.index[0], "rul"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_schema(bad)
