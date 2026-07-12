from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

SEED = 42
N_SPLITS = 5


@dataclass
class EvaluationResult:
    scenario: str
    method: str
    rmse: float
    interpretation: str


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def build_model(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=180,
        max_depth=14,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=seed,
        n_jobs=-1,
    )


def generate_panel_data(seed: int = SEED) -> pd.DataFrame:
    """Create a realistic-enough panel dataset for validation experiments.

    The data contains:
    - repeated meter/transformer entities,
    - a time trend that tree models cannot extrapolate cleanly,
    - seasonality and weather effects,
    - a late operational regime change,
    - strong entity-specific effects.
    """
    rng = np.random.default_rng(seed)
    n_meters = 45
    n_days = 140

    meter_ids = [f"M{idx:03d}" for idx in range(n_meters)]
    regions = np.array(["North", "South", "East", "West"])
    customer_types = np.array(["residential", "commercial", "industrial"])

    meter_meta = pd.DataFrame(
        {
            "meter_id": meter_ids,
            "region": rng.choice(regions, size=n_meters, p=[0.25, 0.25, 0.25, 0.25]),
            "customer_type": rng.choice(
                customer_types, size=n_meters, p=[0.55, 0.30, 0.15]
            ),
            "capacity": rng.uniform(50, 250, size=n_meters).round(1),
            "meter_effect": rng.normal(0, 13, size=n_meters),
        }
    )

    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    rows: list[dict[str, object]] = []

    region_effect = {"North": 2.5, "South": -1.0, "East": 1.0, "West": -2.0}
    type_effect = {"residential": 0.0, "commercial": 8.0, "industrial": 18.0}

    for _, meta in meter_meta.iterrows():
        for day_index, date in enumerate(dates):
            annual_wave = 8 * np.sin(2 * np.pi * day_index / 60)
            weekly_wave = 3 * np.sin(2 * np.pi * day_index / 7)
            temperature = 16 + annual_wave + rng.normal(0, 1.8)
            is_weekend = int(date.dayofweek >= 5)

            # A hidden process change starts late in the sample.
            regime_shift = 0.0 if day_index < 90 else 10.0
            time_trend = 0.28 * day_index

            target = (
                25
                + float(meta["meter_effect"])
                + 0.20 * float(meta["capacity"])
                + 1.15 * temperature
                + region_effect[str(meta["region"])]
                + type_effect[str(meta["customer_type"])]
                - 4.0 * is_weekend
                + weekly_wave
                + time_trend
                + regime_shift
                + rng.normal(0, 3.8)
            )

            rows.append(
                {
                    "date": date,
                    "day_index": day_index,
                    "meter_id": meta["meter_id"],
                    "region": meta["region"],
                    "customer_type": meta["customer_type"],
                    "capacity": meta["capacity"],
                    "temperature": temperature,
                    "dayofweek": date.dayofweek,
                    "is_weekend": is_weekend,
                    "target": target,
                }
            )

    df = pd.DataFrame(rows).sort_values(["date", "meter_id"]).reset_index(drop=True)

    # Safe historical features: every value uses only earlier target observations.
    by_meter = df.groupby("meter_id", sort=False)["target"]
    df["lag_1"] = by_meter.shift(1)
    df["lag_7"] = by_meter.shift(7)
    df["safe_roll_3"] = by_meter.transform(lambda s: s.shift(1).rolling(3).mean())

    # Deliberately leaky: the current target participates in the rolling mean.
    df["leaky_roll_3"] = by_meter.transform(lambda s: s.rolling(3).mean())

    return df


def encode_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train[feature_columns], valid[feature_columns]], axis=0, ignore_index=True
    )
    categorical = combined.select_dtypes(include=["object", "category", "bool"]).columns
    combined = pd.get_dummies(combined, columns=list(categorical), dummy_na=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.fillna(combined.median(numeric_only=True)).fillna(0)
    return (
        combined.iloc[: len(train)].reset_index(drop=True),
        combined.iloc[len(train) :].reset_index(drop=True),
    )


def cross_validate(
    df: pd.DataFrame,
    feature_columns: list[str],
    splitter,
    groups: pd.Series | None = None,
    seed: int = SEED,
) -> tuple[float, list[float]]:
    X, _ = encode_features(df, df.iloc[:0], feature_columns)
    y = df["target"].reset_index(drop=True)
    fold_scores: list[float] = []

    split_iter = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for fold, (train_idx, valid_idx) in enumerate(split_iter, start=1):
        model = build_model(seed + fold)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        fold_scores.append(rmse(y.iloc[valid_idx], pred))

    return float(np.mean(fold_scores)), fold_scores


def holdout_score(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    seed: int = SEED,
) -> float:
    X_train, X_valid = encode_features(train, valid, feature_columns)
    model = build_model(seed)
    model.fit(X_train, train["target"])
    pred = model.predict(X_valid)
    return rmse(valid["target"], pred)


def future_forecasting_experiment(df: pd.DataFrame) -> list[EvaluationResult]:
    features = [
        "meter_id",
        "region",
        "customer_type",
        "capacity",
        "temperature",
        "day_index",
        "dayofweek",
        "is_weekend",
    ]
    train = df[df["day_index"] < 112].copy().sort_values(["date", "meter_id"])
    future = df[df["day_index"] >= 112].copy().sort_values(["date", "meter_id"])

    random_cv, _ = cross_validate(
        train,
        features,
        KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
    )
    time_cv, _ = cross_validate(
        train,
        features,
        TimeSeriesSplit(n_splits=N_SPLITS),
    )
    actual_future = holdout_score(train, future, features)

    return [
        EvaluationResult(
            "Future forecasting",
            "Random KFold CV",
            random_cv,
            "Future rows are mixed into training folds; usually optimistic.",
        ),
        EvaluationResult(
            "Future forecasting",
            "TimeSeriesSplit CV",
            time_cv,
            "Training always precedes validation; closer to deployment.",
        ),
        EvaluationResult(
            "Future forecasting",
            "Actual future holdout",
            actual_future,
            "The final 28 days; our deployment-like reference.",
        ),
    ]


def unseen_asset_experiment(df: pd.DataFrame) -> list[EvaluationResult]:
    all_meters = sorted(df["meter_id"].unique())
    train_meters = set(all_meters[:36])
    unseen_meters = set(all_meters[36:])
    train = df[df["meter_id"].isin(train_meters) & (df["day_index"] < 112)].copy()
    unseen = df[df["meter_id"].isin(unseen_meters) & (df["day_index"] < 112)].copy()

    features = [
        "meter_id",
        "region",
        "customer_type",
        "capacity",
        "temperature",
        "day_index",
        "dayofweek",
        "is_weekend",
    ]

    random_cv, _ = cross_validate(
        train,
        features,
        KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
    )
    group_cv, _ = cross_validate(
        train,
        features,
        GroupKFold(n_splits=N_SPLITS),
        groups=train["meter_id"].reset_index(drop=True),
    )
    actual_unseen = holdout_score(train, unseen, features)

    return [
        EvaluationResult(
            "Unseen assets",
            "Random KFold CV",
            random_cv,
            "The same meters appear in train and validation; identity is memorized.",
        ),
        EvaluationResult(
            "Unseen assets",
            "GroupKFold CV",
            group_cv,
            "Each validation fold contains meters absent from that fold's training data.",
        ),
        EvaluationResult(
            "Unseen assets",
            "Actual unseen-meter holdout",
            actual_unseen,
            "Nine completely unseen meters; our deployment-like reference.",
        ),
    ]


def leakage_experiment(df: pd.DataFrame) -> list[EvaluationResult]:
    lab = df[df["day_index"] < 112].dropna(subset=["lag_7", "safe_roll_3"]).copy()
    base_features = [
        "meter_id",
        "region",
        "customer_type",
        "capacity",
        "temperature",
        "day_index",
        "dayofweek",
        "is_weekend",
        "lag_1",
        "lag_7",
    ]
    safe_features = base_features + ["safe_roll_3"]
    leaky_features = base_features + ["leaky_roll_3"]

    splitter_safe = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splitter_leaky = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    safe_cv, _ = cross_validate(lab, safe_features, splitter_safe)
    leaky_cv, _ = cross_validate(lab, leaky_features, splitter_leaky)

    corr_safe = float(lab[["safe_roll_3", "target"]].corr().iloc[0, 1])
    corr_leaky = float(lab[["leaky_roll_3", "target"]].corr().iloc[0, 1])

    return [
        EvaluationResult(
            "Feature leakage",
            "Safe shifted 3-step rolling CV",
            safe_cv,
            f"Uses only past labels; feature-target correlation={corr_safe:.3f}.",
        ),
        EvaluationResult(
            "Feature leakage",
            "Leaky current-inclusive 3-step rolling CV",
            leaky_cv,
            f"Current target contaminates the feature; correlation={corr_leaky:.3f}.",
        ),
    ]


def save_plot(results: pd.DataFrame, output_path: Path) -> None:
    plot_df = results.copy()
    labels = plot_df["scenario"] + "\n" + plot_df["method"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(plot_df)), plot_df["rmse"])
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Validation method changes the story")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "validation_lab.csv"
    results_path = project_root / "experiments" / "validation_lab_results.csv"
    plot_path = project_root / "experiments" / "validation_lab_rmse.png"

    data_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_panel_data()
    df.to_csv(data_path, index=False)

    records = (
        future_forecasting_experiment(df)
        + unseen_asset_experiment(df)
        + leakage_experiment(df)
    )
    results = pd.DataFrame([record.__dict__ for record in records])
    results.to_csv(results_path, index=False)
    save_plot(results, plot_path)

    print("\nVALIDATION LAB RESULTS")
    print("=" * 80)
    for scenario, group in results.groupby("scenario", sort=False):
        print(f"\n{scenario}")
        for row in group.itertuples(index=False):
            print(f"  {row.method:<38} RMSE={row.rmse:8.3f}")
            print(f"    {row.interpretation}")

    print("\nGenerated files:")
    print(f"  Data:    {data_path.relative_to(project_root)}")
    print(f"  Results: {results_path.relative_to(project_root)}")
    print(f"  Plot:    {plot_path.relative_to(project_root)}")
    print("\nCore lesson: validation must reproduce the information boundary at inference time.")


if __name__ == "__main__":
    main()
