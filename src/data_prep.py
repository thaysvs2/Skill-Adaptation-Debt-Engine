from __future__ import annotations

import pandas as pd
import numpy as np

from . import config


def load_raw() -> pd.DataFrame:
    if not config.DATA_RAW.exists():
        raise FileNotFoundError(f"Dataset not found: {config.DATA_RAW}")
    return pd.read_csv(config.DATA_RAW)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop known constant / useless columns when present
    for c in config.DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Strip whitespace in object columns
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # Normalize target to {0,1}
    if config.TARGET_COL in df.columns:
        df[config.TARGET_COL] = (
            df[config.TARGET_COL].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        )
    # Ensure ID is int-ish but keep as string-safe
    if config.ID_COL in df.columns:
        df[config.ID_COL] = pd.to_numeric(df[config.ID_COL], errors="coerce").astype("Int64")

    # Coerce numeric-looking columns
    for c in df.columns:
        if c in [config.TARGET_COL]:
            continue
        if df[c].dtype == "object":
            # try numeric conversion; keep original if many failures
            conv = pd.to_numeric(df[c], errors="coerce")
            # if at least half non-null after conversion, use numeric
            if conv.notna().mean() >= 0.5:
                df[c] = conv

    # Basic missing handling
    # Numeric: keep NaN (handled by imputer in pipeline)
    # Categorical: fill with "Unknown"
    for c in df.columns:
        if c == config.TARGET_COL:
            continue
        if df[c].dtype == "object":
            df[c] = df[c].fillna("Unknown")

    return df


def prepare_processed() -> pd.DataFrame:
    df = load_raw()
    df = _clean_columns(df)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.DATA_PROCESSED, index=False)
    return df


def load_processed() -> pd.DataFrame:
    if config.DATA_PROCESSED.exists():
        return pd.read_parquet(config.DATA_PROCESSED)
    return prepare_processed()
