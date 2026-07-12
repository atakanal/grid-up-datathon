from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_improvement import (
    build_diagnosis_driven_features,
    causal_fold_bias_correction,
    run_model_improvement,
)
from src.rehearsal import TARGET, generate_synthetic_competition_data, run_rehearsal
from src.submit import validate_submission


def test_diagnosis_features_do_not_depend_on_target() -> None:
    train, test, _ = generate_synthetic_competition_data(
        seed=21, n_meters=6, train_days=12, test_days=3
    )
    original_train, original_test = build_diagnosis_driven_features(train, test)
    mutated = train.copy()
    mutated[TARGET] = mutated[TARGET] * 10_000 + np.arange(len(mutated))
    mutated_train, mutated_test = build_diagnosis_driven_features(mutated, test)
    pd.testing.assert_frame_equal(original_train, mutated_train)
    pd.testing.assert_frame_equal(original_test, mutated_test)


def test_causal_correction_uses_only_previous_folds() -> None:
    y = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    predictions = np.array([8.0, 8.0, 9.0, 9.0, 100.0, 100.0])
    folds = np.array([1, 1, 2, 2, 3, 3])
    corrected, shifts = causal_fold_bias_correction(y, predictions, folds)

    assert shifts[1] == 0.0
    assert shifts[2] == 2.0  # learned from fold 1 residuals only
    assert shifts[3] == 1.5  # learned from folds 1 and 2, not fold 3
    np.testing.assert_allclose(corrected[:2], predictions[:2])
    np.testing.assert_allclose(corrected[2:4], predictions[2:4] + 2.0)


def test_small_model_improvement_creates_valid_submissions(tmp_path) -> None:
    run_rehearsal(
        output_root=tmp_path,
        seed=23,
        n_splits=2,
        n_estimators=10,
        n_meters=6,
        train_days=12,
        test_days=3,
    )
    artifacts = run_model_improvement(
        output_root=tmp_path,
        seed=23,
        n_splits=2,
        iterations=20,
        thread_count=1,
    )

    sample = pd.read_csv(tmp_path / "data" / "raw" / "rehearsal_sample_submission.csv")
    raw = pd.read_csv(artifacts.raw_submission_path)
    calibrated = pd.read_csv(artifacts.calibrated_submission_path)
    assert validate_submission(raw, sample) == []
    assert validate_submission(calibrated, sample) == []
    assert np.isfinite(artifacts.catboost_rmse)
    assert artifacts.report_path.exists()
