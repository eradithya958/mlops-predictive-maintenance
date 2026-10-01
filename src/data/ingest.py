"""
Data ingestion for the NASA CMAPSS Turbofan Engine Degradation Dataset.

Steps:
1. Download raw data from NASA (with synthetic fallback if unavailable).
2. Parse fixed-width text files into DataFrames.
3. Drop zero-variance sensors.
4. Attach piece-wise linear RUL labels.
5. Split by engine unit (no leakage) → train / val / test parquets.

Run directly:
    python -m src.data.ingest
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CMAPSS column schema
# ---------------------------------------------------------------------------
_SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
_SENSOR_COLS = [f"sensor_{i:02d}" for i in range(1, 22)]  # sensor_01 … sensor_21
CMAPSS_COLUMNS = ["unit", "cycle"] + _SETTING_COLS + _SENSOR_COLS

# Sensors known to have near-zero variance on FD001 (will be dropped automatically
# if drop_constant_sensors=True; listed here as documentation).
_KNOWN_CONSTANT_FD001 = {
    "sensor_01",
    "sensor_05",
    "sensor_06",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
}

# NASA data URLs — direct download of the zip from the PHM data challenge repo
_NASA_ZIP_URL = "https://ti.arc.nasa.gov/c/6/"  # official NASA redirect
_FALLBACK_URL = "https://raw.githubusercontent.com/microsoft/MLOpsPython/master/data/CMAPSSData.zip"

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _attempt_download(url: str, dest: Path, timeout: int = 60) -> bool:
    """Try to download `url` to `dest`. Returns True on success."""
    try:
        logger.info("Attempting download from %s", url)
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(dest, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Download from %s failed: %s", url, exc)
        if dest.exists():
            dest.unlink()
        return False


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    logger.info("Extracting %s → %s", zip_path.name, dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def download_cmapss(raw_dir: Path, dataset: str = "FD001") -> bool:
    """
    Download the CMAPSS zip if not already present.
    Tries the official NASA URL first, then a GitHub mirror.
    Returns True if successful.
    """
    zip_path = raw_dir / "CMAPSSData.zip"
    train_file = raw_dir / f"train_{dataset}.txt"

    if train_file.exists():
        logger.info("CMAPSS %s already present, skipping download.", dataset)
        return True

    raw_dir.mkdir(parents=True, exist_ok=True)
    success = _attempt_download(_NASA_ZIP_URL, zip_path) or _attempt_download(
        _FALLBACK_URL, zip_path
    )

    if success:
        _extract_zip(zip_path, raw_dir)
        zip_path.unlink(missing_ok=True)
        return True

    logger.warning("All download attempts failed — will use synthetic data instead.")
    return False


# ---------------------------------------------------------------------------
# Synthetic data generator (CMAPSS-like)
# ---------------------------------------------------------------------------


def _generate_synthetic_cmapss(
    dataset: str,
    rng: np.random.Generator,
    n_train_units: int = 100,
    n_test_units: int = 100,
    max_life_range: tuple[int, int] = (150, 350),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Generate CMAPSS-like degradation data when real data is unavailable.
    Sensors degrade linearly with added Gaussian noise.
    """
    logger.info("Generating synthetic CMAPSS-like data for %s …", dataset)

    def _make_units(n_units: int, provide_rul: bool = False):
        frames = []
        true_rul_vals = []
        for uid in range(1, n_units + 1):
            life = int(rng.integers(*max_life_range))
            cycles = np.arange(1, life + 1)
            progress = cycles / life  # 0→1 over the unit's lifetime

            unit_df = pd.DataFrame({"unit": uid, "cycle": cycles})
            # Three operational settings (discrete clusters)
            for s in range(1, 4):
                unit_df[f"setting_{s}"] = rng.choice([1.0, 2.0, 3.0])

            # 21 sensors — some degrade, some remain stable
            for i in range(1, 22):
                if i in {1, 5, 6, 10, 16, 18, 19}:
                    # Constant sensors
                    unit_df[f"sensor_{i:02d}"] = rng.normal(100, 0.01, size=life)
                elif i % 3 == 0:
                    # Increasing degradation
                    unit_df[f"sensor_{i:02d}"] = (
                        rng.normal(0, 1)
                        + progress * rng.uniform(5, 15)
                        + rng.normal(0, 0.5, size=life)
                    )
                else:
                    # Decreasing / stable
                    unit_df[f"sensor_{i:02d}"] = (
                        rng.normal(100, 0.5)
                        - progress * rng.uniform(2, 8)
                        + rng.normal(0, 0.3, size=life)
                    )

            frames.append(unit_df)
            if provide_rul:
                true_rul_vals.append(0)  # last known RUL for test data
        return pd.concat(frames, ignore_index=True), true_rul_vals

    train_df, _ = _make_units(n_train_units)
    test_df, rul_list = _make_units(n_test_units, provide_rul=True)
    rul_series = pd.Series(rul_list, name="rul")
    return train_df, test_df, rul_series


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_cmapss_file(path: Path) -> pd.DataFrame:
    """Parse a CMAPSS fixed-width text file into a DataFrame."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
        index_col=False,
    )
    # Drop trailing NaN columns that sometimes appear due to trailing whitespace
    df = df.dropna(axis=1, how="all")
    return df


def _parse_rul_file(path: Path) -> pd.Series:
    """Parse the RUL ground-truth file (one value per engine unit in test set)."""
    return pd.read_csv(path, header=None, names=["rul"]).squeeze("columns")


def load_raw_data(
    raw_dir: Path, dataset: str = "FD001", rng: np.random.Generator | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load raw CMAPSS data from disk (or generate synthetic if unavailable).

    Returns:
        train_df, test_df, test_rul_series
    """
    if rng is None:
        rng = np.random.default_rng(42)

    train_path = raw_dir / f"train_{dataset}.txt"
    test_path = raw_dir / f"test_{dataset}.txt"
    rul_path = raw_dir / f"RUL_{dataset}.txt"

    if train_path.exists() and test_path.exists() and rul_path.exists():
        logger.info("Loading CMAPSS %s from %s", dataset, raw_dir)
        train_df = _parse_cmapss_file(train_path)
        test_df = _parse_cmapss_file(test_path)
        test_rul = _parse_rul_file(rul_path)
    else:
        train_df, test_df, test_rul = _generate_synthetic_cmapss(dataset, rng)

    return train_df, test_df, test_rul


# ---------------------------------------------------------------------------
# RUL labelling
# ---------------------------------------------------------------------------


def attach_rul_labels(df: pd.DataFrame, rul_clip: int = 125) -> pd.DataFrame:
    """
    Add a `rul` column using piece-wise linear labelling:
    RUL = min(max_cycle − current_cycle, rul_clip)

    This is the de-facto standard for CMAPSS and avoids the model being
    penalised for predicting high degradation early in life.
    """
    max_cycles = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = (max_cycles - df["cycle"]).clip(upper=rul_clip)
    return df


def attach_test_rul_labels(
    test_df: pd.DataFrame, test_rul: pd.Series, rul_clip: int = 125
) -> pd.DataFrame:
    """
    For the test set, the true RUL at the last observed cycle is given by
    `test_rul`. We back-fill to label every cycle.
    """
    test_df = test_df.copy()
    # Map unit index (1-based) to ground truth RUL at the last cycle
    rul_map = {uid: int(test_rul.iloc[uid - 1]) for uid in test_df["unit"].unique()}
    max_cycles = test_df.groupby("unit")["cycle"].transform("max")
    test_df["rul"] = (test_df["unit"].map(rul_map) + (max_cycles - test_df["cycle"])).clip(
        upper=rul_clip
    )
    return test_df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def drop_constant_sensors(
    df: pd.DataFrame, threshold: float = 1e-6
) -> tuple[pd.DataFrame, list[str]]:
    """Remove sensor columns with near-zero variance across the dataset."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    variances = df[sensor_cols].var()
    to_drop = variances[variances < threshold].index.tolist()
    if to_drop:
        logger.info("Dropping %d constant sensor(s): %s", len(to_drop), to_drop)
    return df.drop(columns=to_drop), to_drop


def validate_schema(df: pd.DataFrame, name: str = "dataframe") -> None:
    """Assert basic schema integrity."""
    required = {"unit", "cycle", "rul"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    if df["rul"].isnull().any():
        raise ValueError(f"{name} contains NaN in 'rul' column")
    if (df["rul"] < 0).any():
        raise ValueError(f"{name} contains negative RUL values")


# ---------------------------------------------------------------------------
# Train / val / test split  (by engine unit — avoids any data leakage)
# ---------------------------------------------------------------------------


def split_by_unit(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split engine units into train/val/test. All cycles of a given unit land
    in exactly one split — preventing temporal leakage across splits.
    """
    rng = np.random.default_rng(random_seed)
    units = df["unit"].unique()
    rng.shuffle(units)

    n = len(units)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size))

    test_units = set(units[:n_test])
    val_units = set(units[n_test : n_test + n_val])
    train_units = set(units[n_test + n_val :])

    train = df[df["unit"].isin(train_units)].copy()
    val = df[df["unit"].isin(val_units)].copy()
    test = df[df["unit"].isin(test_units)].copy()

    logger.info(
        "Split: %d train units (%d rows) | %d val units (%d rows) | %d test units (%d rows)",
        len(train_units),
        len(train),
        len(val_units),
        len(val),
        len(test_units),
        len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def ingest(cfg: Settings | None = None) -> dict[str, pd.DataFrame]:
    """
    Full ingestion pipeline.

    Returns a dict with keys 'train', 'val', 'test'.
    Also saves parquets to cfg.processed_dir.
    """
    if cfg is None:
        cfg = get_settings()

    dataset = cfg.data.dataset
    rng = np.random.default_rng(cfg.data.random_seed)

    # 1. Download (best-effort) and load raw data
    download_cmapss(cfg.raw_dir, dataset)
    train_raw, test_raw, test_rul = load_raw_data(cfg.raw_dir, dataset, rng)

    # 2. Attach RUL labels
    train_labelled = attach_rul_labels(train_raw, cfg.data.rul_clip)
    test_labelled = attach_test_rul_labels(test_raw, test_rul, cfg.data.rul_clip)

    # 3. Combine into one frame then split by engine unit
    #    (test set from CMAPSS is kept separate; we carve val from train)
    if cfg.features.drop_constant_sensors:
        train_labelled, dropped = drop_constant_sensors(train_labelled)
        # Apply same drops to test set
        existing_drops = [c for c in dropped if c in test_labelled.columns]
        test_labelled = test_labelled.drop(columns=existing_drops)

    train_split, val_split, internal_test = split_by_unit(
        train_labelled,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
    )

    # Validate
    for name, df in [("train", train_split), ("val", val_split), ("test", internal_test)]:
        validate_schema(df, name)

    # 4. Save parquets
    splits: dict[str, pd.DataFrame] = {
        "train": train_split,
        "val": val_split,
        "test": internal_test,
    }
    for split_name, df in splits.items():
        out_path = cfg.processed_dir / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved %s → %s (%d rows)", split_name, out_path, len(df))

    return splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest()
