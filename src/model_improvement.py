from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .rehearsal import (
    GROUP_COLUMN,
    ID_COLUMN,
    TARGET,
    TIME_COLUMN,
    build_safe_features,
    make_timestamp_block_splits,
)
from .submit import validate_submission
from .utils import utc_timestamp


@dataclass(frozen=True)
class ModelImprovementArtifacts:
    results_path: Path
    oof_path: Path
    feature_importance_path: Path
    raw_submission_path: Path
    calibrated_submission_path: Path
    report_path: Path
    model_dir: Path
    baseline_rmse: float
    catboost_rmse: float
    causal_calibrated_rmse: float
    raw_bias: float
    causal_calibrated_bias: float
    final_test_shift: float
    fold_scores: list[float]


def build_diagnosis_driven_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extend target-independent rehearsal features using diagnosed failure modes.

    All new columns are derived only from fields available in train and test inputs.
    No target values are read, including during rolling or interaction creation.
    """
    train_features, test_features = build_safe_features(train, test)
    train_features = train_features.copy()
    test_features = test_features.copy()

    combined = pd.concat(
        [
            train_features.assign(__source="train"),
            test_features.assign(__source="test"),
        ],
        ignore_index=True,
    )

    combined["temperature_distance_20"] = (combined["temperature_c"] - 20.0).abs()
    combined["temperature_distance_20_sq"] = combined["temperature_distance_20"] ** 2
    combined["voltage_drop_from_232"] = 232.0 - combined["voltage_v"]

    combined["temperature_delta_1"] = (
        combined["temperature_c"] - combined["temperature_c_lag_1"]
    )
    combined["humidity_delta_1"] = combined["humidity_pct"] - combined["humidity_pct_lag_1"]
    combined["voltage_delta_1"] = combined["voltage_v"] - combined["voltage_v_lag_1"]

    combined["dayofweek_sin"] = np.sin(2 * np.pi * combined["dayofweek"] / 7)
    combined["dayofweek_cos"] = np.cos(2 * np.pi * combined["dayofweek"] / 7)

    for column in [
        "humidity_pct",
        "voltage_v",
        "temperature_c_lag_1",
        "humidity_pct_lag_1",
        "voltage_v_lag_1",
    ]:
        combined[f"{column}_missing"] = combined[column].isna().astype("int8")

    combined["customer_hour"] = (
        combined["customer_type"].astype(str) + "__" + combined["hour"].astype(str)
    )
    combined["meter_hour"] = (
        combined[GROUP_COLUMN].astype(str) + "__" + combined["hour"].astype(str)
    )
    combined["region_customer"] = (
        combined["region"].astype(str) + "__" + combined["customer_type"].astype(str)
    )

    train_out = combined.loc[combined["__source"].eq("train")].drop(columns="__source")
    test_out = combined.loc[combined["__source"].eq("test")].drop(columns="__source")
    return train_out.reset_index(drop=True), test_out.reset_index(drop=True)


def causal_fold_bias_correction(
    y_true: np.ndarray,
    predictions: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, dict[int, float]]:
    """Correct each fold using residuals from earlier folds only.

    Fold 1 receives no correction. Fold 2 uses fold 1 residuals, and so on.
    This produces an honest diagnostic score without using the current or future fold.
    """
    y_true = np.asarray(y_true, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    fold_ids = np.asarray(fold_ids, dtype=int)
    if not (len(y_true) == len(predictions) == len(fold_ids)):
        raise ValueError("y_true, predictions and fold_ids must have equal lengths.")

    corrected = predictions.copy()
    previous_residuals: list[np.ndarray] = []
    corrections: dict[int, float] = {}

    for fold in sorted(int(value) for value in np.unique(fold_ids) if value > 0):
        mask = fold_ids == fold
        if previous_residuals:
            historical_residuals = np.concatenate(previous_residuals)
            shift = -float(np.mean(historical_residuals))
        else:
            shift = 0.0
        corrected[mask] = predictions[mask] + shift
        corrections[fold] = shift
        previous_residuals.append(predictions[mask] - y_true[mask])

    return corrected, corrections


def _metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "bias": float(np.mean(predictions - y_true)),
    }




def _markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a small dataframe as Markdown without optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _save_comparison_plots(results: pd.DataFrame, importance: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(results["experiment"], results["rmse"])
    ax.set_ylabel("OOF RMSE")
    ax.set_title("Model improvement: same validation folds")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "model_rmse_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(results["experiment"], results["bias"])
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel("Bias: prediction - actual")
    ax.set_title("Model improvement: systematic prediction bias")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "model_bias_comparison.png", dpi=150)
    plt.close(fig)

    top = importance.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Mean CatBoost feature importance")
    ax.set_title("Diagnosis-driven CatBoost: top features")
    fig.tight_layout()
    fig.savefig(output_dir / "catboost_feature_importance.png", dpi=150)
    plt.close(fig)


def run_model_improvement(
    output_root: str | Path = ".",
    seed: int = 42,
    n_splits: int = 3,
    iterations: int = 350,
    thread_count: int = 4,
) -> ModelImprovementArtifacts:
    root = Path(output_root)
    train_path = root / "data" / "raw" / "rehearsal_train.csv"
    test_path = root / "data" / "raw" / "rehearsal_test.csv"
    sample_path = root / "data" / "raw" / "rehearsal_sample_submission.csv"
    baseline_oof_path = root / "data" / "processed" / "rehearsal_oof.csv"

    required_paths = [train_path, test_path, sample_path, baseline_oof_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run scripts/run_competition_rehearsal.py first. Missing: " + ", ".join(missing)
        )

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("CatBoost is not installed. Run pip install -r requirements.txt") from exc

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    baseline_oof = pd.read_csv(baseline_oof_path)

    X, X_test = build_diagnosis_driven_features(train, test)
    y = train[TARGET].to_numpy(dtype=float)
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for column in categorical_columns:
        X[column] = X[column].astype(str)
        X_test[column] = X_test[column].astype(str)

    splits = make_timestamp_block_splits(train[TIME_COLUMN], n_splits=n_splits)
    oof = np.full(len(train), np.nan, dtype=float)
    fold_ids = np.zeros(len(train), dtype=int)
    test_predictions = np.zeros(len(test), dtype=float)
    fold_scores: list[float] = []
    feature_importances: list[np.ndarray] = []

    model_dir = root / "models" / "rehearsal_catboost_diagnosis"
    model_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.05,
            depth=7,
            loss_function="RMSE",
            l2_leaf_reg=5.0,
            random_strength=0.3,
            max_ctr_complexity=2,
            verbose=False,
            allow_writing_files=False,
            random_seed=seed + fold,
            thread_count=thread_count,
        )
        model.fit(X.iloc[train_idx], y[train_idx], cat_features=categorical_columns)
        valid_prediction = model.predict(X.iloc[valid_idx])
        test_prediction = model.predict(X_test)

        oof[valid_idx] = valid_prediction
        fold_ids[valid_idx] = fold
        test_predictions += test_prediction / len(splits)
        fold_rmse = float(np.sqrt(mean_squared_error(y[valid_idx], valid_prediction)))
        fold_scores.append(fold_rmse)
        feature_importances.append(np.asarray(model.get_feature_importance(), dtype=float))
        model.save_model(model_dir / f"fold_{fold}.cbm")

        print(f"CatBoost fold {fold}: RMSE={fold_rmse:.4f}")

    valid_mask = np.isfinite(oof)
    if not valid_mask.any():
        raise RuntimeError("No CatBoost OOF predictions were generated.")

    baseline_by_id = baseline_oof.set_index(ID_COLUMN)["prediction"]
    baseline_predictions = train[ID_COLUMN].map(baseline_by_id).to_numpy(dtype=float)
    comparison_mask = valid_mask & np.isfinite(baseline_predictions)
    if not comparison_mask.any():
        raise RuntimeError("Baseline and CatBoost OOF rows do not overlap.")

    corrected_oof, fold_corrections = causal_fold_bias_correction(y, oof, fold_ids)
    raw_metrics = _metrics(y[comparison_mask], oof[comparison_mask])
    corrected_metrics = _metrics(y[comparison_mask], corrected_oof[comparison_mask])
    baseline_metrics = _metrics(y[comparison_mask], baseline_predictions[comparison_mask])

    # OOF residuals estimate a one-parameter calibration for the future test prediction.
    final_test_shift = -raw_metrics["bias"]
    calibrated_test_predictions = test_predictions + final_test_shift

    raw_submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET: test_predictions})
    calibrated_submission = pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], TARGET: calibrated_test_predictions}
    )
    for name, submission in [("raw", raw_submission), ("calibrated", calibrated_submission)]:
        errors = validate_submission(submission, sample)
        if errors:
            raise RuntimeError(f"{name} submission validation failed: " + " | ".join(errors))

    submissions_dir = root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    raw_submission_path = submissions_dir / "rehearsal_catboost_raw.csv"
    calibrated_submission_path = submissions_dir / "rehearsal_catboost_calibrated.csv"
    raw_submission.to_csv(raw_submission_path, index=False)
    calibrated_submission.to_csv(calibrated_submission_path, index=False)

    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    oof_frame = train[[ID_COLUMN, TIME_COLUMN, GROUP_COLUMN, TARGET]].copy()
    oof_frame["fold"] = fold_ids
    oof_frame["baseline_prediction"] = baseline_predictions
    oof_frame["catboost_prediction"] = oof
    oof_frame["catboost_causal_calibrated"] = corrected_oof
    oof_frame["catboost_error"] = oof - y
    oof_frame["catboost_causal_calibrated_error"] = corrected_oof - y
    oof_path = processed_dir / "rehearsal_model_improvement_oof.csv"
    oof_frame.to_csv(oof_path, index=False)

    experiments_dir = root / "experiments" / "model_improvement"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(
        [
            {"experiment": "RandomForest baseline", **baseline_metrics},
            {"experiment": "CatBoost diagnosis features", **raw_metrics},
            {"experiment": "CatBoost causal calibration", **corrected_metrics},
        ]
    )
    results["timestamp_utc"] = utc_timestamp()
    results["fold_scores"] = ""
    results.loc[results["experiment"].eq("CatBoost diagnosis features"), "fold_scores"] = json.dumps(
        fold_scores
    )
    results_path = experiments_dir / "model_improvement_results.csv"
    results.to_csv(results_path, index=False)

    mean_importance = np.mean(np.vstack(feature_importances), axis=0)
    importance = (
        pd.DataFrame({"feature": X.columns, "importance": mean_importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance_path = experiments_dir / "catboost_feature_importance.csv"
    importance.to_csv(feature_importance_path, index=False)
    _save_comparison_plots(results, importance, experiments_dir)

    report_path = experiments_dir / "model_improvement_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Grid Up Model Improvement Report",
                "",
                "## Same-fold comparison",
                "",
                _markdown_table(results[["experiment", "rmse", "mae", "bias"]]),
                "",
                "## CatBoost fold RMSE",
                "",
                *[f"- Fold {idx}: {score:.4f}" for idx, score in enumerate(fold_scores, start=1)],
                "",
                "## Causal fold corrections",
                "",
                *[f"- Fold {fold}: {shift:+.4f} kWh" for fold, shift in fold_corrections.items()],
                "",
                "## Final test calibration",
                "",
                f"- OOF-estimated shift applied to the calibrated test submission: {final_test_shift:+.4f} kWh",
                "- Keep both raw and calibrated submissions. Real competitions may react differently under distribution shift.",
                "",
                "## Top features",
                "",
                _markdown_table(importance.head(15)),
                "",
                "## Interpretation",
                "",
                "CatBoost reduces regression-to-the-mean by modelling categorical identity and nonlinear interactions more effectively. The causal calibration diagnostic uses only earlier-fold residuals, while the final test shift is estimated from all available OOF residuals.",
            ]
        ),
        encoding="utf-8",
    )

    metadata = {
        "timestamp_utc": utc_timestamp(),
        "experiment": "rehearsal_catboost_diagnosis",
        "iterations": iterations,
        "seed": seed,
        "n_splits": n_splits,
        "fold_scores": fold_scores,
        "baseline_rmse": baseline_metrics["rmse"],
        "catboost_rmse": raw_metrics["rmse"],
        "causal_calibrated_rmse": corrected_metrics["rmse"],
        "raw_bias": raw_metrics["bias"],
        "causal_calibrated_bias": corrected_metrics["bias"],
        "final_test_shift": final_test_shift,
        "raw_submission": str(raw_submission_path),
        "calibrated_submission": str(calibrated_submission_path),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ModelImprovementArtifacts(
        results_path=results_path,
        oof_path=oof_path,
        feature_importance_path=feature_importance_path,
        raw_submission_path=raw_submission_path,
        calibrated_submission_path=calibrated_submission_path,
        report_path=report_path,
        model_dir=model_dir,
        baseline_rmse=baseline_metrics["rmse"],
        catboost_rmse=raw_metrics["rmse"],
        causal_calibrated_rmse=corrected_metrics["rmse"],
        raw_bias=raw_metrics["bias"],
        causal_calibrated_bias=corrected_metrics["bias"],
        final_test_shift=final_test_shift,
        fold_scores=fold_scores,
    )
