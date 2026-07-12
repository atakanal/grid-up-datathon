from __future__ import annotations

import numpy as np
import pandas as pd


def add_datetime_features(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """Add leakage-safe calendar features derived only from the timestamp itself."""
    out = df.copy()
    dt = pd.to_datetime(out[datetime_column], errors="coerce")
    out[f"{datetime_column}_year"] = dt.dt.year
    out[f"{datetime_column}_month"] = dt.dt.month
    out[f"{datetime_column}_day"] = dt.dt.day
    out[f"{datetime_column}_dayofweek"] = dt.dt.dayofweek
    out[f"{datetime_column}_hour"] = dt.dt.hour
    out[f"{datetime_column}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    out[f"{datetime_column}_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    out[f"{datetime_column}_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    return out.drop(columns=[datetime_column])


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in df.columns[df.isna().any()]:
        out[f"{column}__missing"] = df[column].isna().astype("int8")
    return out


def align_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align one-hot encoded train/test frames without silently losing columns."""
    train_aligned, test_aligned = X_train.align(X_test, join="left", axis=1, fill_value=0)
    return train_aligned, test_aligned
