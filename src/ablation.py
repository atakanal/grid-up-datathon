from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .model_improvement import build_diagnosis_driven_features
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
class AblationArtifacts:
    results_path: Path
    oof_path: Path
    report_path: Path
    safe_submission_path: Path
    no_identity_submission_path: Path
    model_dirs: list[Path]
    baseline_rmse: float
    safe_catboost_rmse: float
    diagnosis_catboost_rmse: float
    no_identity_rmse: float
    calibrated_rmse: float


def drop_direct_meter_identity(features: pd.DataFrame) -> pd.DataFrame:
    """Remove direct meter identity columns while retaining observed sensor history.

    This is a stress test, not a claim that meter identity is always invalid. It asks
    how much the temporal validation score depends on direct identity memorisation.
    """
    direct_identity_columns = [GROUP_COLUMN, "meter_hour"]
    return features.drop(
        columns=[column for column in direct_identity_columns if column in features.columns]
    )


def compute_ablation_attribution(results: pd.DataFrame) -> dict[str, float]:
    """Attribute RMSE reductions to model, features and calibration.

    Expected experiment names are intentionally explicit so report errors fail loudly.
    """
    required = {
        "RandomForest baseline",
        "CatBoost safe features",
        "CatBoost diagnosis features",
        "CatBoost diagnosis + causal calibration",
        "CatBoost diagnosis without direct meter identity",
    }
    indexed = results.set_index("experiment")
    missing = required.difference(indexed.index)
    if missing:
        raise ValueError(f"Missing ablation experiments: {sorted(missing)}")

    baseline = float(indexed.loc["RandomForest baseline", "rmse"])
    safe = float(indexed.loc["CatBoost safe features", "rmse"])
    diagnosis = float(indexed.loc["CatBoost diagnosis features", "rmse"])
    calibrated = float(indexed.loc["CatBoost diagnosis + causal calibration", "rmse"])
    no_identity = float(
        indexed.loc["CatBoost diagnosis without direct meter identity", "rmse"]
    )
    return {
        "model_gain_rmse": baseline - safe,
        "feature_gain_rmse": safe - diagnosis,
        "calibration_gain_rmse": diagnosis - calibrated,
        "direct_identity_value_rmse": no_identity - diagnosis,
        "total_gain_rmse": baseline - calibrated,
        "total_gain_pct": 100.0 * (baseline - calibrated) / baseline,
    }


def _metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "bias": float(np.mean(predictions - y_true)),
    }


def _markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
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


def _train_catboost_variant(
    *,
    name: str,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_dir: Path,
    seed: int,
    iterations: int,
    thread_count: int,
) -> tuple[np.ndarray, np.ndarray, list[float], pd.DataFrame]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CatBoost is not installed. Run pip install -r requirements.txt") from exc

    X = X.copy()
    X_test = X_test.copy()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    for column in categorical_columns:
        X[column] = X[column].astype(str)
        X_test[column] = X_test[column].astype(str)

    oof = np.full(len(X), np.nan, dtype=float)
    test_predictions = np.zeros(len(X_test), dtype=float)
    fold_scores: list[float] = []
    feature_importances: list[np.ndarray] = []
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
        test_predictions += test_prediction / len(splits)
        fold_rmse = float(np.sqrt(mean_squared_error(y[valid_idx], valid_prediction)))
        fold_scores.append(fold_rmse)
        feature_importances.append(np.asarray(model.get_feature_importance(), dtype=float))
        model.save_model(model_dir / f"fold_{fold}.cbm")
        print(f"{name} fold {fold}: RMSE={fold_rmse:.4f}")

    mean_importance = np.mean(np.vstack(feature_importances), axis=0)
    importance = (
        pd.DataFrame({"feature": X.columns, "importance": mean_importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "timestamp_utc": utc_timestamp(),
                "experiment": name,
                "iterations": iterations,
                "seed": seed,
                "fold_scores": fold_scores,
                "feature_count": int(X.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return oof, test_predictions, fold_scores, importance


def _save_plots(results: pd.DataFrame, attribution: dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_frame = results.sort_values("rmse", ascending=False)
    ax.barh(plot_frame["experiment"], plot_frame["rmse"])
    ax.set_xlabel("OOF RMSE (lower is better)")
    ax.set_title("Ablation: model, features, identity and calibration")
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_rmse.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(results["experiment"], results["bias"])
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Bias: prediction - actual")
    ax.set_title("Ablation: systematic prediction bias")
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_bias.png", dpi=150)
    plt.close(fig)

    contribution_frame = pd.DataFrame(
        {
            "component": ["Model", "Diagnosis features", "Calibration"],
            "rmse_reduction": [
                attribution["model_gain_rmse"],
                attribution["feature_gain_rmse"],
                attribution["calibration_gain_rmse"],
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(contribution_frame["component"], contribution_frame["rmse_reduction"])
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel("RMSE reduction")
    ax.set_title("Where did the improvement come from?")
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_gain_attribution.png", dpi=150)
    plt.close(fig)


def run_ablation(
    output_root: str | Path = ".",
    seed: int = 42,
    n_splits: int = 3,
    iterations: int = 350,
    thread_count: int = 4,
) -> AblationArtifacts:
    root = Path(output_root)
    train_path = root / "data" / "raw" / "rehearsal_train.csv"
    test_path = root / "data" / "raw" / "rehearsal_test.csv"
    sample_path = root / "data" / "raw" / "rehearsal_sample_submission.csv"
    baseline_oof_path = root / "data" / "processed" / "rehearsal_oof.csv"
    improvement_oof_path = root / "data" / "processed" / "rehearsal_model_improvement_oof.csv"
    diagnosis_submission_path = root / "submissions" / "rehearsal_catboost_raw.csv"

    required_paths = [
        train_path,
        test_path,
        sample_path,
        baseline_oof_path,
        improvement_oof_path,
        diagnosis_submission_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run competition rehearsal and model improvement first. Missing: "
            + ", ".join(missing)
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    baseline_oof = pd.read_csv(baseline_oof_path)
    improvement_oof = pd.read_csv(improvement_oof_path)
    diagnosis_submission = pd.read_csv(diagnosis_submission_path)

    y = train[TARGET].to_numpy(dtype=float)
    splits = make_timestamp_block_splits(train[TIME_COLUMN], n_splits=n_splits)

    safe_X, safe_X_test = build_safe_features(train, test)
    diagnosis_X, diagnosis_X_test = build_diagnosis_driven_features(train, test)
    no_identity_X = drop_direct_meter_identity(diagnosis_X)
    no_identity_X_test = drop_direct_meter_identity(diagnosis_X_test)

    safe_model_dir = root / "models" / "rehearsal_catboost_safe"
    no_identity_model_dir = root / "models" / "rehearsal_catboost_no_meter_identity"

    safe_oof, safe_test, safe_fold_scores, safe_importance = _train_catboost_variant(
        name="CatBoost safe features",
        X=safe_X,
        X_test=safe_X_test,
        y=y,
        splits=splits,
        model_dir=safe_model_dir,
        seed=seed,
        iterations=iterations,
        thread_count=thread_count,
    )
    no_identity_oof, no_identity_test, no_identity_fold_scores, no_identity_importance = (
        _train_catboost_variant(
            name="CatBoost diagnosis without direct meter identity",
            X=no_identity_X,
            X_test=no_identity_X_test,
            y=y,
            splits=splits,
            model_dir=no_identity_model_dir,
            seed=seed,
            iterations=iterations,
            thread_count=thread_count,
        )
    )

    baseline_predictions = train[ID_COLUMN].map(
        baseline_oof.set_index(ID_COLUMN)["prediction"]
    ).to_numpy(dtype=float)
    improvement_by_id = improvement_oof.set_index(ID_COLUMN)
    diagnosis_predictions = train[ID_COLUMN].map(
        improvement_by_id["catboost_prediction"]
    ).to_numpy(dtype=float)
    calibrated_predictions = train[ID_COLUMN].map(
        improvement_by_id["catboost_causal_calibrated"]
    ).to_numpy(dtype=float)

    common_mask = (
        np.isfinite(baseline_predictions)
        & np.isfinite(safe_oof)
        & np.isfinite(diagnosis_predictions)
        & np.isfinite(calibrated_predictions)
        & np.isfinite(no_identity_oof)
    )
    if not common_mask.any():
        raise RuntimeError("No common OOF rows were available for ablation comparison.")

    experiments = [
        ("RandomForest baseline", baseline_predictions, safe_X.shape[1], "reference"),
        ("CatBoost safe features", safe_oof, safe_X.shape[1], "model effect"),
        (
            "CatBoost diagnosis features",
            diagnosis_predictions,
            diagnosis_X.shape[1],
            "feature effect",
        ),
        (
            "CatBoost diagnosis + causal calibration",
            calibrated_predictions,
            diagnosis_X.shape[1],
            "calibration effect",
        ),
        (
            "CatBoost diagnosis without direct meter identity",
            no_identity_oof,
            no_identity_X.shape[1],
            "identity stress test",
        ),
    ]
    results = pd.DataFrame(
        [
            {
                "experiment": name,
                **_metrics(y[common_mask], predictions[common_mask]),
                "feature_count": feature_count,
                "purpose": purpose,
            }
            for name, predictions, feature_count, purpose in experiments
        ]
    )
    results["timestamp_utc"] = utc_timestamp()
    attribution = compute_ablation_attribution(results)

    submissions_dir = root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    safe_submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET: safe_test})
    no_identity_submission = pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], TARGET: no_identity_test}
    )
    safe_submission_path = submissions_dir / "rehearsal_catboost_safe.csv"
    no_identity_submission_path = (
        submissions_dir / "rehearsal_catboost_no_meter_identity.csv"
    )
    for name, frame, path in [
        ("safe", safe_submission, safe_submission_path),
        ("no_identity", no_identity_submission, no_identity_submission_path),
        ("diagnosis", diagnosis_submission, diagnosis_submission_path),
    ]:
        errors = validate_submission(frame, sample)
        if errors:
            raise RuntimeError(f"{name} submission validation failed: " + " | ".join(errors))
        if name != "diagnosis":
            frame.to_csv(path, index=False)

    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    oof_frame = train[[ID_COLUMN, TIME_COLUMN, GROUP_COLUMN, TARGET]].copy()
    oof_frame["randomforest_prediction"] = baseline_predictions
    oof_frame["catboost_safe_prediction"] = safe_oof
    oof_frame["catboost_diagnosis_prediction"] = diagnosis_predictions
    oof_frame["catboost_calibrated_prediction"] = calibrated_predictions
    oof_frame["catboost_no_identity_prediction"] = no_identity_oof
    oof_path = processed_dir / "rehearsal_ablation_oof.csv"
    oof_frame.to_csv(oof_path, index=False)

    experiments_dir = root / "experiments" / "ablation"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    results_path = experiments_dir / "ablation_results.csv"
    results.to_csv(results_path, index=False)
    safe_importance.to_csv(experiments_dir / "catboost_safe_feature_importance.csv", index=False)
    no_identity_importance.to_csv(
        experiments_dir / "catboost_no_identity_feature_importance.csv", index=False
    )
    pd.DataFrame([attribution]).to_csv(
        experiments_dir / "ablation_attribution.csv", index=False
    )
    _save_plots(results, attribution, experiments_dir)

    report_path = experiments_dir / "ablation_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Grid Up Ablation Report",
                "",
                "## Same-fold results",
                "",
                _markdown_table(
                    results[["experiment", "rmse", "mae", "bias", "feature_count", "purpose"]]
                ),
                "",
                "## Improvement attribution",
                "",
                f"- Model change (Random Forest -> CatBoost, same safe features): {attribution['model_gain_rmse']:+.4f} RMSE",
                f"- Diagnosis-driven feature engineering: {attribution['feature_gain_rmse']:+.4f} RMSE",
                f"- Causal calibration: {attribution['calibration_gain_rmse']:+.4f} RMSE",
                f"- Total baseline-to-calibrated gain: {attribution['total_gain_rmse']:+.4f} RMSE ({attribution['total_gain_pct']:.2f}%)",
                "",
                "## Direct meter identity stress test",
                "",
                f"- Removing `meter_id` and `meter_hour` changes RMSE by {attribution['direct_identity_value_rmse']:+.4f}.",
                "- A large deterioration means the temporal split rewards known-meter identity. It does not prove leakage because the same meters exist at inference time in this rehearsal.",
                "- If the real test contains unseen meters, repeat this with GroupKFold before trusting identity features.",
                "",
                "## Fold scores",
                "",
                f"- CatBoost safe features: {json.dumps([round(value, 4) for value in safe_fold_scores])}",
                f"- CatBoost without direct identity: {json.dumps([round(value, 4) for value in no_identity_fold_scores])}",
                "",
                "## Interpretation rule",
                "",
                "Ablation changes one component at a time. Prefer changes that improve the same validation rows and remain defensible under the real test-generation process.",
            ]
        ),
        encoding="utf-8",
    )

    return AblationArtifacts(
        results_path=results_path,
        oof_path=oof_path,
        report_path=report_path,
        safe_submission_path=safe_submission_path,
        no_identity_submission_path=no_identity_submission_path,
        model_dirs=[safe_model_dir, no_identity_model_dir],
        baseline_rmse=float(
            results.loc[results["experiment"].eq("RandomForest baseline"), "rmse"].iloc[0]
        ),
        safe_catboost_rmse=float(
            results.loc[results["experiment"].eq("CatBoost safe features"), "rmse"].iloc[0]
        ),
        diagnosis_catboost_rmse=float(
            results.loc[results["experiment"].eq("CatBoost diagnosis features"), "rmse"].iloc[0]
        ),
        no_identity_rmse=float(
            results.loc[
                results["experiment"].eq(
                    "CatBoost diagnosis without direct meter identity"
                ),
                "rmse",
            ].iloc[0]
        ),
        calibrated_rmse=float(
            results.loc[
                results["experiment"].eq("CatBoost diagnosis + causal calibration"),
                "rmse",
            ].iloc[0]
        ),
    )
