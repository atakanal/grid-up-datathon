from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .submit import validate_submission
from .utils import utc_timestamp


TARGET = "consumption_kwh"
ID_COLUMN = "row_id"
TIME_COLUMN = "timestamp"
GROUP_COLUMN = "meter_id"


@dataclass(frozen=True)
class RehearsalArtifacts:
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    submission_path: Path
    oof_path: Path
    results_path: Path
    feature_importance_path: Path
    model_path: Path
    cv_score: float
    fold_scores: list[float]


def generate_synthetic_competition_data(
    seed: int = 42,
    n_meters: int = 18,
    train_days: int = 60,
    test_days: int = 14,
    frequency_hours: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a compact electricity-consumption competition rehearsal dataset."""
    if 24 % frequency_hours != 0:
        raise ValueError("frequency_hours must divide 24 exactly.")
    if n_meters < 3:
        raise ValueError("At least three meters are required.")

    rng = np.random.default_rng(seed)
    periods_per_day = 24 // frequency_hours
    total_periods = (train_days + test_days) * periods_per_day
    timestamps = pd.date_range("2026-01-01", periods=total_periods, freq=f"{frequency_hours}h")

    regions = np.array(["north", "south", "east"])
    customer_types = np.array(["residential", "commercial", "industrial"])
    meter_ids = np.array([f"M{i:03d}" for i in range(n_meters)])

    meter_meta = pd.DataFrame(
        {
            GROUP_COLUMN: meter_ids,
            "region": regions[np.arange(n_meters) % len(regions)],
            "customer_type": customer_types[np.arange(n_meters) % len(customer_types)],
            "base_load": rng.uniform(30, 90, size=n_meters),
            "temperature_sensitivity": rng.uniform(0.25, 1.10, size=n_meters),
            "meter_bias": rng.normal(0, 3.0, size=n_meters),
        }
    )

    frame = pd.MultiIndex.from_product(
        [timestamps, meter_ids], names=[TIME_COLUMN, GROUP_COLUMN]
    ).to_frame(index=False)
    frame = frame.merge(meter_meta, on=GROUP_COLUMN, how="left", validate="many_to_one")

    dt = pd.to_datetime(frame[TIME_COLUMN])
    day_index = (dt - dt.min()).dt.total_seconds() / 86_400
    hour = dt.dt.hour
    weekend = (dt.dt.dayofweek >= 5).astype(float)

    weather_daily = 17 + 7 * np.sin(2 * np.pi * day_index / 45)
    weather_intraday = 4 * np.sin(2 * np.pi * (hour - 8) / 24)
    late_period_shift = np.where(day_index >= train_days, 2.5, 0.0)
    frame["temperature_c"] = weather_daily + weather_intraday + late_period_shift + rng.normal(0, 1.2, len(frame))
    frame["humidity_pct"] = 63 - 0.75 * frame["temperature_c"] + rng.normal(0, 4.0, len(frame))
    frame["planned_maintenance"] = rng.binomial(1, 0.018, len(frame)).astype("int8")

    residential_pattern = np.where((hour >= 18) | (hour <= 6), 10.0, 2.0)
    commercial_pattern = np.where((hour >= 8) & (hour <= 18), 14.0, -3.0)
    industrial_pattern = 5.0 + 2.0 * np.sin(2 * np.pi * hour / 24)
    customer_pattern = np.select(
        [
            frame["customer_type"].eq("residential"),
            frame["customer_type"].eq("commercial"),
        ],
        [residential_pattern, commercial_pattern],
        default=industrial_pattern,
    )

    region_effect = frame["region"].map({"north": 3.5, "south": -1.0, "east": 1.5}).astype(float)
    cooling_heating_need = np.abs(frame["temperature_c"] - 20)
    trend = 0.055 * day_index
    maintenance_effect = -12.0 * frame["planned_maintenance"]
    weekend_effect = np.where(frame["customer_type"].eq("commercial"), -7.0 * weekend, 2.5 * weekend)

    load = (
        frame["base_load"]
        + frame["meter_bias"]
        + customer_pattern
        + region_effect
        + frame["temperature_sensitivity"] * cooling_heating_need
        + trend
        + maintenance_effect
        + weekend_effect
        + rng.normal(0, 2.8, len(frame))
    )
    frame[TARGET] = np.clip(load, 3.0, None)
    frame["voltage_v"] = 232.0 - 0.035 * frame[TARGET] + rng.normal(0, 1.1, len(frame))

    # Simulate imperfect sensor feeds. Missingness indicators and imputation must handle these.
    for column, probability in (("humidity_pct", 0.035), ("voltage_v", 0.025)):
        mask = rng.random(len(frame)) < probability
        frame.loc[mask, column] = np.nan

    frame[ID_COLUMN] = [f"R{i:07d}" for i in range(len(frame))]
    public_columns = [
        ID_COLUMN,
        TIME_COLUMN,
        GROUP_COLUMN,
        "region",
        "customer_type",
        "temperature_c",
        "humidity_pct",
        "voltage_v",
        "planned_maintenance",
        TARGET,
    ]
    frame = frame[public_columns].sort_values([TIME_COLUMN, GROUP_COLUMN]).reset_index(drop=True)

    split_time = timestamps[train_days * periods_per_day]
    train = frame[frame[TIME_COLUMN] < split_time].reset_index(drop=True)
    test = frame[frame[TIME_COLUMN] >= split_time].drop(columns=[TARGET]).reset_index(drop=True)
    sample_submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET: 0.0})
    return train, test, sample_submission


def build_safe_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create features without reading the target column.

    Target-derived lags are deliberately excluded because future labels are unavailable.
    Lags below use only past *observed input variables* within each meter.
    """
    required = {
        ID_COLUMN,
        TIME_COLUMN,
        GROUP_COLUMN,
        "region",
        "customer_type",
        "temperature_c",
        "humidity_pct",
        "voltage_v",
        "planned_maintenance",
    }
    missing_train = required.difference(train.columns)
    missing_test = required.difference(test.columns)
    if missing_train or missing_test:
        raise ValueError(f"Missing columns - train: {sorted(missing_train)}, test: {sorted(missing_test)}")

    train_inputs = train.drop(columns=[TARGET], errors="ignore").copy()
    test_inputs = test.drop(columns=[TARGET], errors="ignore").copy()
    train_inputs["__source"] = "train"
    test_inputs["__source"] = "test"
    combined = pd.concat([train_inputs, test_inputs], ignore_index=True)
    combined[TIME_COLUMN] = pd.to_datetime(combined[TIME_COLUMN], errors="raise")
    combined = combined.sort_values([GROUP_COLUMN, TIME_COLUMN]).reset_index(drop=True)

    dt = combined[TIME_COLUMN]
    combined["month"] = dt.dt.month.astype("int8")
    combined["dayofweek"] = dt.dt.dayofweek.astype("int8")
    combined["hour"] = dt.dt.hour.astype("int8")
    combined["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    combined["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    combined["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    combined["elapsed_days"] = (dt - dt.min()).dt.total_seconds() / 86_400

    grouped = combined.groupby(GROUP_COLUMN, sort=False)
    for column in ["temperature_c", "humidity_pct", "voltage_v"]:
        combined[f"{column}_lag_1"] = grouped[column].shift(1)
        combined[f"{column}_past_mean_4"] = grouped[column].transform(
            lambda series: series.shift(1).rolling(4, min_periods=1).mean()
        )

    feature_columns = [
        column
        for column in combined.columns
        if column not in {ID_COLUMN, TIME_COLUMN, "__source", TARGET}
    ]
    train_features = (
        combined.loc[combined["__source"].eq("train"), feature_columns]
        .reset_index(drop=True)
    )
    test_features = (
        combined.loc[combined["__source"].eq("test"), feature_columns]
        .reset_index(drop=True)
    )

    # Restore the exact original row order after groupwise feature calculation.
    train_order = train[[ID_COLUMN]].merge(
        pd.concat(
            [
                combined.loc[combined["__source"].eq("train"), [ID_COLUMN]].reset_index(drop=True),
                train_features,
            ],
            axis=1,
        ),
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    test_order = test[[ID_COLUMN]].merge(
        pd.concat(
            [
                combined.loc[combined["__source"].eq("test"), [ID_COLUMN]].reset_index(drop=True),
                test_features,
            ],
            axis=1,
        ),
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    return train_order.drop(columns=[ID_COLUMN]), test_order.drop(columns=[ID_COLUMN])


def make_timestamp_block_splits(
    timestamps: pd.Series,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split unique timestamps so no timestamp is shared across train and validation."""
    parsed = pd.to_datetime(timestamps, errors="raise")
    unique_times = pd.Series(parsed.unique()).sort_values().reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for train_time_idx, valid_time_idx in splitter.split(unique_times):
        train_times = set(unique_times.iloc[train_time_idx])
        valid_times = set(unique_times.iloc[valid_time_idx])
        train_idx = np.flatnonzero(parsed.isin(train_times).to_numpy())
        valid_idx = np.flatnonzero(parsed.isin(valid_times).to_numpy())
        splits.append((train_idx, valid_idx))
    return splits


def build_pipeline(X: pd.DataFrame, seed: int, n_estimators: int = 120) -> Pipeline:
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=14,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _save_plots(
    y: pd.Series,
    oof: np.ndarray,
    valid_mask: np.ndarray,
    fold_scores: list[float],
    experiment_dir: Path,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y.to_numpy()[valid_mask], oof[valid_mask], alpha=0.35, s=12)
    low = min(float(y.min()), float(np.nanmin(oof)))
    high = max(float(y.max()), float(np.nanmax(oof)))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual consumption (kWh)")
    ax.set_ylabel("OOF prediction (kWh)")
    ax.set_title("Competition rehearsal: OOF predictions")
    fig.tight_layout()
    fig.savefig(experiment_dir / "rehearsal_oof_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    fold_numbers = np.arange(1, len(fold_scores) + 1)
    ax.bar(fold_numbers, fold_scores)
    ax.set_xticks(fold_numbers)
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.set_title("Competition rehearsal: fold scores")
    fig.tight_layout()
    fig.savefig(experiment_dir / "rehearsal_fold_scores.png", dpi=150)
    plt.close(fig)


def run_rehearsal(
    output_root: str | Path = ".",
    seed: int = 42,
    n_splits: int = 3,
    n_estimators: int = 120,
    n_meters: int = 18,
    train_days: int = 60,
    test_days: int = 14,
) -> RehearsalArtifacts:
    root = Path(output_root)
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    submissions_dir = root / "submissions"
    experiments_dir = root / "experiments"
    models_dir = root / "models"
    for directory in [raw_dir, processed_dir, submissions_dir, experiments_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train, test, sample = generate_synthetic_competition_data(
        seed=seed,
        n_meters=n_meters,
        train_days=train_days,
        test_days=test_days,
    )
    train_path = raw_dir / "rehearsal_train.csv"
    test_path = raw_dir / "rehearsal_test.csv"
    sample_path = raw_dir / "rehearsal_sample_submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)

    X, X_test = build_safe_features(train, test)
    y = train[TARGET].reset_index(drop=True)
    timestamps = train[TIME_COLUMN].reset_index(drop=True)

    oof = np.full(len(train), np.nan, dtype=float)
    test_predictions = np.zeros(len(test), dtype=float)
    fold_scores: list[float] = []
    splits = make_timestamp_block_splits(timestamps, n_splits=n_splits)

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        pipeline = build_pipeline(X, seed=seed + fold, n_estimators=n_estimators)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        valid_prediction = pipeline.predict(X.iloc[valid_idx])
        test_predictions += pipeline.predict(X_test) / len(splits)
        oof[valid_idx] = valid_prediction
        fold_rmse = float(np.sqrt(mean_squared_error(y.iloc[valid_idx], valid_prediction)))
        fold_scores.append(fold_rmse)
        print(
            f"Fold {fold}: RMSE={fold_rmse:.4f} | "
            f"train through {timestamps.iloc[train_idx].max()} | "
            f"validate {timestamps.iloc[valid_idx].min()} -> {timestamps.iloc[valid_idx].max()}"
        )

    valid_mask = np.isfinite(oof)
    cv_score = float(np.sqrt(mean_squared_error(y[valid_mask], oof[valid_mask])))

    final_pipeline = build_pipeline(X, seed=seed, n_estimators=n_estimators)
    final_pipeline.fit(X, y)
    model_path = models_dir / "rehearsal_randomforest.joblib"
    joblib.dump(final_pipeline, model_path)

    submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET: test_predictions})
    errors = validate_submission(submission, sample)
    if errors:
        raise RuntimeError("Submission validation failed: " + " | ".join(errors))
    submission_path = submissions_dir / "rehearsal_baseline.csv"
    submission.to_csv(submission_path, index=False)

    oof_frame = train[[ID_COLUMN, TIME_COLUMN, GROUP_COLUMN, TARGET]].copy()
    oof_frame["prediction"] = oof
    oof_frame["is_validated"] = valid_mask
    oof_path = processed_dir / "rehearsal_oof.csv"
    oof_frame.to_csv(oof_path, index=False)

    preprocessor = final_pipeline.named_steps["preprocessor"]
    model = final_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance_path = experiments_dir / "rehearsal_feature_importance.csv"
    importance.to_csv(feature_importance_path, index=False)

    top = importance.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Random Forest importance")
    ax.set_title("Competition rehearsal: top features")
    fig.tight_layout()
    fig.savefig(experiments_dir / "rehearsal_feature_importance.png", dpi=150)
    plt.close(fig)

    _save_plots(y, oof, valid_mask, fold_scores, experiments_dir)

    result_record = {
        "timestamp_utc": utc_timestamp(),
        "experiment": "rehearsal_randomforest_timeseries",
        "model": "randomforest",
        "task": "regression",
        "metric": "rmse",
        "cv": "timestamp_block_timeseries",
        "n_splits": n_splits,
        "seed": seed,
        "fold_scores": json.dumps(fold_scores),
        "cv_score": cv_score,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "submission": str(submission_path),
        "public_lb": "",
        "notes": "Synthetic end-to-end competition rehearsal; no target-derived features.",
    }
    results_path = experiments_dir / "rehearsal_results.csv"
    pd.DataFrame([result_record]).to_csv(results_path, index=False)

    experiment_log_path = experiments_dir / "experiments.csv"
    log_columns = [
        "timestamp_utc",
        "experiment",
        "model",
        "task",
        "metric",
        "cv",
        "n_splits",
        "seed",
        "fold_scores",
        "cv_score",
        "submission",
        "public_lb",
        "notes",
    ]
    pd.DataFrame([{key: result_record[key] for key in log_columns}]).to_csv(
        experiment_log_path,
        mode="a",
        header=not experiment_log_path.exists() or experiment_log_path.stat().st_size == 0,
        index=False,
    )

    return RehearsalArtifacts(
        train_path=train_path,
        test_path=test_path,
        sample_submission_path=sample_path,
        submission_path=submission_path,
        oof_path=oof_path,
        results_path=results_path,
        feature_importance_path=feature_importance_path,
        model_path=model_path,
        cv_score=cv_score,
        fold_scores=fold_scores,
    )
