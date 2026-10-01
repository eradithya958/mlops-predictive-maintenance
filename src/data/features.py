"""
Feature engineering for the CMAPSS turbofan degradation dataset.

Applied after ingestion (reads train/val/test.parquet, writes
train_features/val_features/test_features.parquet).

Features added per engine unit (grouped before computation to avoid
cross-unit contamination):
  • Rolling mean & std for each sensor  (windows: 5, 10, 20 cycles)
  • Lag features for each sensor        (lags: 1, 3, 5 cycles)
  • Linear degradation slope            (last `trend_window` cycles)
  • Cycle progress                      (cycle / max_cycle_per_unit)

NaN values at window/lag boundaries are forward-filled within the unit,
then backward-filled for any remaining head NaNs.

Run directly:
    python -m src.data.features
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: identify sensor columns in a DataFrame
# ---------------------------------------------------------------------------


def _sensor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sensor_")]


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    sensor_cols: list[str],
) -> pd.DataFrame:
    """
    Add rolling mean and std per sensor per engine unit.
    min_periods=1 ensures the very first cycle always gets a value.
    """
    new_cols: dict[str, pd.Series] = {}
    for col in sensor_cols:
        for w in windows:
            rolled = df.groupby("unit")[col].transform(
                lambda s, _w=w: s.rolling(_w, min_periods=1).mean()
            )
            new_cols[f"{col}_roll_mean_{w}"] = rolled

            rolled_std = df.groupby("unit")[col].transform(
                lambda s, _w=w: s.rolling(_w, min_periods=1).std().fillna(0)
            )
            new_cols[f"{col}_roll_std_{w}"] = rolled_std

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int],
    sensor_cols: list[str],
) -> pd.DataFrame:
    """
    Add per-unit lag features. NaN at the head is forward-then-backward filled.
    """
    new_cols: dict[str, pd.Series] = {}
    for col in sensor_cols:
        for lag in lags:
            lagged = df.groupby("unit")[col].transform(lambda s, _l=lag: s.shift(_l))
            # Fill head NaN: ffill within group not applicable for head,
            # so bfill covers the first <lag> rows of each unit.
            lagged = lagged.groupby(df["unit"]).transform(lambda s: s.bfill())
            new_cols[f"{col}_lag_{lag}"] = lagged

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Degradation trend (linear slope)
# ---------------------------------------------------------------------------


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute the OLS slope over a rolling window within a single unit's series."""
    slopes = []
    vals = series.values
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        chunk = vals[start : i + 1]
        if len(chunk) < 2:
            slopes.append(0.0)
        else:
            x = np.arange(len(chunk), dtype=float)
            # Fast OLS slope via covariance formula
            xm = x - x.mean()
            ym = chunk - chunk.mean()
            denom = (xm**2).sum()
            slope = (xm * ym).sum() / denom if denom > 0 else 0.0
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)


def add_trend_features(
    df: pd.DataFrame,
    window: int,
    sensor_cols: list[str],
) -> pd.DataFrame:
    """
    Add per-unit linear degradation slope for each sensor over the last
    `window` cycles. This captures rate-of-change which is highly
    predictive of remaining useful life.
    """
    new_cols: dict[str, pd.Series] = {}
    for col in sensor_cols:
        slope_series = df.groupby("unit")[col].transform(lambda s, _w=window: _rolling_slope(s, _w))
        new_cols[f"{col}_slope_{window}"] = slope_series

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Cycle-progress feature
# ---------------------------------------------------------------------------


def add_cycle_progress(df: pd.DataFrame) -> pd.DataFrame:
    """
    cycle_progress = cycle / max_cycle_of_unit  ∈ (0, 1].
    Gives the model a sense of how far along in the unit's observed life
    it currently is (relative degradation phase).
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["cycle_progress"] = df["cycle"] / max_cycle
    return df


# ---------------------------------------------------------------------------
# Main featurize function
# ---------------------------------------------------------------------------


def featurize(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a single split DataFrame.
    The `rul` and metadata columns are preserved unchanged.
    """
    sensor_cols = _sensor_cols(df)

    logger.info("Featurizing %d rows, %d sensors …", len(df), len(sensor_cols))

    # Sort by unit then cycle to guarantee correct rolling/lag direction
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)

    # 1. Cycle progress
    df = add_cycle_progress(df)

    # 2. Rolling mean + std
    df = add_rolling_features(df, cfg.features.rolling_windows, sensor_cols)

    # 3. Lag features
    df = add_lag_features(df, cfg.features.lag_steps, sensor_cols)

    # 4. Degradation trends (slope)
    df = add_trend_features(df, cfg.features.trend_window, sensor_cols)

    # 5. Final NaN sweep — should be minimal after min_periods/bfill above
    n_nan = df.isnull().sum().sum()
    if n_nan > 0:
        logger.warning("%d NaN values remain after feature engineering — filling with 0", n_nan)
        df = df.fillna(0)

    logger.info(
        "Feature engineering complete: %d columns (was %d)",
        len(df.columns),
        len(sensor_cols) + 2,
    )
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_features(cfg: Settings | None = None) -> None:
    """
    Read raw split parquets, apply feature engineering, write feature parquets.
    """
    if cfg is None:
        cfg = get_settings()

    for split in ("train", "val", "test"):
        in_path = cfg.processed_dir / f"{split}.parquet"
        out_path = cfg.processed_dir / f"{split}_features.parquet"

        if not in_path.exists():
            raise FileNotFoundError(f"{in_path} not found — run ingest.py first.")

        df = pd.read_parquet(in_path)
        df_feat = featurize(df, cfg)
        df_feat.to_parquet(out_path, index=False)
        logger.info(
            "Saved %s → %s (%d rows, %d cols)", split, out_path, len(df_feat), len(df_feat.columns)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_features()
