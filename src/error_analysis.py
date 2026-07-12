from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .rehearsal import ID_COLUMN, GROUP_COLUMN, TARGET, TIME_COLUMN, make_timestamp_block_splits


PREDICTION_COLUMN = "prediction"


@dataclass(frozen=True)
class ErrorAnalysisArtifacts:
    analysis_path: Path
    segment_summary_path: Path
    worst_errors_path: Path
    report_path: Path
    plot_paths: list[Path]
    overall_rmse: float
    overall_mae: float
    overall_bias: float


def regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Calculate regression diagnostics using prediction - actual as signed error."""
    if frame.empty:
        raise ValueError("Cannot calculate metrics for an empty frame.")
    error = frame[PREDICTION_COLUMN].to_numpy(dtype=float) - frame[TARGET].to_numpy(dtype=float)
    return {
        "count": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "mean_actual": float(frame[TARGET].mean()),
        "mean_prediction": float(frame[PREDICTION_COLUMN].mean()),
    }


def build_error_frame(
    train: pd.DataFrame,
    oof: pd.DataFrame,
    n_splits: int = 3,
) -> pd.DataFrame:
    """Join OOF predictions to training metadata and add deployable error diagnostics."""
    required_train = {
        ID_COLUMN,
        TIME_COLUMN,
        GROUP_COLUMN,
        TARGET,
        "region",
        "customer_type",
        "planned_maintenance",
    }
    required_oof = {ID_COLUMN, PREDICTION_COLUMN}
    missing_train = required_train.difference(train.columns)
    missing_oof = required_oof.difference(oof.columns)
    if missing_train or missing_oof:
        raise ValueError(
            f"Missing columns - train: {sorted(missing_train)}, oof: {sorted(missing_oof)}"
        )
    if train[ID_COLUMN].duplicated().any() or oof[ID_COLUMN].duplicated().any():
        raise ValueError("row_id must be unique in both train and OOF files.")

    metadata_columns = [
        ID_COLUMN,
        TIME_COLUMN,
        GROUP_COLUMN,
        "region",
        "customer_type",
        "planned_maintenance",
        TARGET,
    ]
    prediction_columns = [ID_COLUMN, PREDICTION_COLUMN]
    if "is_validated" in oof.columns:
        prediction_columns.append("is_validated")

    frame = train[metadata_columns].merge(
        oof[prediction_columns], on=ID_COLUMN, how="left", validate="one_to_one"
    )
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="raise")

    fold = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for fold_number, (_, valid_idx) in enumerate(
        make_timestamp_block_splits(frame[TIME_COLUMN], n_splits=n_splits), start=1
    ):
        fold.iloc[valid_idx] = fold_number
    frame["fold"] = fold

    validated = np.isfinite(pd.to_numeric(frame[PREDICTION_COLUMN], errors="coerce"))
    if "is_validated" in frame.columns:
        validated &= frame["is_validated"].fillna(False).astype(bool)
    frame = frame.loc[validated].copy()
    if frame.empty:
        raise ValueError("No validated OOF predictions were found.")
    if frame["fold"].isna().any():
        raise ValueError("Some validated rows could not be assigned to a fold.")

    frame["error"] = frame[PREDICTION_COLUMN] - frame[TARGET]
    frame["residual"] = -frame["error"]  # positive means the model underpredicted
    frame["abs_error"] = frame["error"].abs()
    frame["squared_error"] = frame["error"].pow(2)
    frame["hour"] = frame[TIME_COLUMN].dt.hour.astype("int8")
    frame["dayofweek"] = frame[TIME_COLUMN].dt.dayofweek.astype("int8")
    frame["is_weekend"] = (frame["dayofweek"] >= 5).astype("int8")

    quantile_count = min(5, int(frame[TARGET].nunique()))
    if quantile_count >= 2:
        labels = [f"Q{i}" for i in range(1, quantile_count + 1)]
        frame["consumption_band"] = pd.qcut(
            frame[TARGET], q=quantile_count, labels=labels, duplicates="drop"
        ).astype(str)
    else:
        frame["consumption_band"] = "Q1"

    return frame.reset_index(drop=True)


def summarize_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one tidy table for the most useful error-analysis segments."""
    segment_columns = [
        "fold",
        "hour",
        GROUP_COLUMN,
        "region",
        "customer_type",
        "consumption_band",
        "is_weekend",
        "planned_maintenance",
    ]
    rows: list[dict[str, object]] = []

    overall = regression_metrics(frame)
    rows.append({"segment_type": "overall", "segment_value": "all", **overall})

    for column in segment_columns:
        for value, group in frame.groupby(column, observed=True, dropna=False):
            rows.append(
                {
                    "segment_type": column,
                    "segment_value": str(value),
                    **regression_metrics(group),
                }
            )
    return pd.DataFrame(rows)


def _plot_hourly_rmse(summary: pd.DataFrame, path: Path) -> None:
    hourly = summary.loc[summary["segment_type"].eq("hour")].copy()
    hourly["segment_numeric"] = pd.to_numeric(hourly["segment_value"])
    hourly = hourly.sort_values("segment_numeric")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hourly["segment_numeric"], hourly["rmse"], marker="o")
    ax.set_xticks(hourly["segment_numeric"])
    ax.set_xlabel("Hour")
    ax.set_ylabel("RMSE")
    ax.set_title("Rehearsal error analysis: RMSE by hour")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_meter_rmse(summary: pd.DataFrame, path: Path) -> None:
    meter = summary.loc[summary["segment_type"].eq(GROUP_COLUMN)].nlargest(12, "rmse")
    meter = meter.sort_values("rmse")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(meter["segment_value"], meter["rmse"])
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Meter")
    ax.set_title("Rehearsal error analysis: hardest meters")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_consumption_band_bias(summary: pd.DataFrame, path: Path) -> None:
    band = summary.loc[summary["segment_type"].eq("consumption_band")].copy()
    band["band_order"] = band["segment_value"].str.extract(r"(\d+)").astype(int)
    band = band.sort_values("band_order")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(band["segment_value"], band["bias"])
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Actual consumption quantile")
    ax.set_ylabel("Mean error: prediction - actual")
    ax.set_title("Rehearsal error analysis: bias by consumption level")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_error_distribution(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(frame["error"], bins=40)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Prediction - actual (kWh)")
    ax.set_ylabel("Validated rows")
    ax.set_title("Rehearsal error analysis: signed error distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def write_markdown_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    report_path: Path,
    feature_importance: pd.DataFrame | None = None,
) -> None:
    overall = regression_metrics(frame)

    def worst_segment(segment_type: str) -> pd.Series:
        subset = summary.loc[summary["segment_type"].eq(segment_type)]
        return subset.loc[subset["rmse"].idxmax()]

    worst_fold = worst_segment("fold")
    worst_hour = worst_segment("hour")
    worst_meter = worst_segment(GROUP_COLUMN)
    worst_band = worst_segment("consumption_band")
    bias_word = "overpredicts" if overall["bias"] > 0 else "underpredicts"

    lines = [
        "# Rehearsal Error Analysis Report",
        "",
        "This report uses only out-of-fold rows that were never used to fit the model producing their prediction.",
        "Signed error is defined as `prediction - actual`; positive bias means overprediction.",
        "",
        "## Overall diagnostics",
        "",
        f"- Validated rows: **{int(overall['count'])}**",
        f"- RMSE: **{_format_metric(overall['rmse'])} kWh**",
        f"- MAE: **{_format_metric(overall['mae'])} kWh**",
        f"- Mean bias: **{_format_metric(overall['bias'])} kWh**; the model {bias_word} on average.",
        "",
        "## Hardest segments",
        "",
        f"- Fold **{worst_fold['segment_value']}** has the highest RMSE: **{_format_metric(worst_fold['rmse'])}**.",
        f"- Hour **{worst_hour['segment_value']}:00** has the highest RMSE: **{_format_metric(worst_hour['rmse'])}**.",
        f"- Meter **{worst_meter['segment_value']}** has the highest RMSE: **{_format_metric(worst_meter['rmse'])}**.",
        f"- Consumption band **{worst_band['segment_value']}** has the highest RMSE: **{_format_metric(worst_band['rmse'])}**.",
        "",
        "## Interpretation checklist",
        "",
        "1. A bad fold can indicate limited history or distribution shift, not merely weak hyperparameters.",
        "2. Meter-level errors can reveal identity effects or underrepresented operating profiles.",
        "3. Bias by consumption band shows whether peaks are systematically flattened or exaggerated.",
        "4. The largest individual errors should be inspected before adding model complexity.",
    ]

    if feature_importance is not None and {"feature", "importance"}.issubset(feature_importance.columns):
        lines.extend(["", "## Top model features", ""])
        for row in feature_importance.nlargest(10, "importance").itertuples(index=False):
            lines.append(f"- `{row.feature}`: {_format_metric(float(row.importance))}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_error_analysis(
    train_path: str | Path = "data/raw/rehearsal_train.csv",
    oof_path: str | Path = "data/processed/rehearsal_oof.csv",
    feature_importance_path: str | Path = "experiments/rehearsal_feature_importance.csv",
    output_dir: str | Path = "experiments/error_analysis",
    n_splits: int = 3,
    top_n_errors: int = 50,
) -> ErrorAnalysisArtifacts:
    train_path = Path(train_path)
    oof_path = Path(oof_path)
    feature_importance_path = Path(feature_importance_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}")

    train = pd.read_csv(train_path)
    oof = pd.read_csv(oof_path)
    analysis = build_error_frame(train, oof, n_splits=n_splits)
    summary = summarize_segments(analysis)

    analysis_path = output_dir / "rehearsal_error_rows.csv"
    segment_summary_path = output_dir / "rehearsal_error_segments.csv"
    worst_errors_path = output_dir / "rehearsal_worst_errors.csv"
    report_path = output_dir / "rehearsal_error_report.md"
    analysis.to_csv(analysis_path, index=False)
    summary.to_csv(segment_summary_path, index=False)
    analysis.nlargest(top_n_errors, "abs_error").to_csv(worst_errors_path, index=False)

    feature_importance = (
        pd.read_csv(feature_importance_path) if feature_importance_path.exists() else None
    )
    write_markdown_report(analysis, summary, report_path, feature_importance)

    plot_paths = [
        output_dir / "rehearsal_rmse_by_hour.png",
        output_dir / "rehearsal_hardest_meters.png",
        output_dir / "rehearsal_bias_by_consumption_band.png",
        output_dir / "rehearsal_error_distribution.png",
    ]
    _plot_hourly_rmse(summary, plot_paths[0])
    _plot_meter_rmse(summary, plot_paths[1])
    _plot_consumption_band_bias(summary, plot_paths[2])
    _plot_error_distribution(analysis, plot_paths[3])

    metrics = regression_metrics(analysis)
    return ErrorAnalysisArtifacts(
        analysis_path=analysis_path,
        segment_summary_path=segment_summary_path,
        worst_errors_path=worst_errors_path,
        report_path=report_path,
        plot_paths=plot_paths,
        overall_rmse=metrics["rmse"],
        overall_mae=metrics["mae"],
        overall_bias=metrics["bias"],
    )
