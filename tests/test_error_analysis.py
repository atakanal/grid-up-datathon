from __future__ import annotations

import numpy as np
import pandas as pd

from src.error_analysis import build_error_frame, regression_metrics, run_error_analysis
from src.rehearsal import (
    ID_COLUMN,
    TARGET,
    TIME_COLUMN,
    generate_synthetic_competition_data,
    make_timestamp_block_splits,
    run_rehearsal,
)


def test_regression_metrics_preserve_bias_direction() -> None:
    frame = pd.DataFrame({TARGET: [10.0, 20.0], "prediction": [12.0, 19.0]})
    metrics = regression_metrics(frame)
    assert metrics["count"] == 2
    assert np.isclose(metrics["bias"], 0.5)
    assert np.isclose(metrics["mae"], 1.5)
    assert np.isclose(metrics["rmse"], np.sqrt(2.5))


def test_error_frame_keeps_only_validated_oof_rows() -> None:
    train, _, _ = generate_synthetic_competition_data(
        seed=17, n_meters=6, train_days=12, test_days=3
    )
    oof = pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN],
            "prediction": np.nan,
            "is_validated": False,
        }
    )
    validated_indices = np.concatenate(
        [valid_idx for _, valid_idx in make_timestamp_block_splits(train[TIME_COLUMN], n_splits=3)]
    )
    oof.loc[validated_indices, "prediction"] = train.loc[validated_indices, TARGET] + 1.0
    oof.loc[validated_indices, "is_validated"] = True

    frame = build_error_frame(train, oof, n_splits=3)
    assert len(frame) == len(validated_indices)
    assert frame["prediction"].notna().all()
    assert frame["fold"].notna().all()
    assert np.allclose(frame["error"], 1.0)


def test_error_analysis_creates_report_tables_and_plots(tmp_path) -> None:
    rehearsal = run_rehearsal(
        output_root=tmp_path,
        seed=23,
        n_splits=2,
        n_estimators=8,
        n_meters=6,
        train_days=12,
        test_days=3,
    )
    artifacts = run_error_analysis(
        train_path=rehearsal.train_path,
        oof_path=rehearsal.oof_path,
        feature_importance_path=rehearsal.feature_importance_path,
        output_dir=tmp_path / "experiments" / "error_analysis",
        n_splits=2,
        top_n_errors=10,
    )

    assert artifacts.analysis_path.exists()
    assert artifacts.segment_summary_path.exists()
    assert artifacts.worst_errors_path.exists()
    assert artifacts.report_path.exists()
    assert all(path.exists() for path in artifacts.plot_paths)
    assert np.isfinite(artifacts.overall_rmse)
    assert "Overall diagnostics" in artifacts.report_path.read_text(encoding="utf-8")
    assert len(pd.read_csv(artifacts.worst_errors_path)) == 10
