from __future__ import annotations

import numpy as np
import pandas as pd

from src.rehearsal import (
    ID_COLUMN,
    TARGET,
    TIME_COLUMN,
    build_safe_features,
    generate_synthetic_competition_data,
    make_timestamp_block_splits,
    run_rehearsal,
)
from src.submit import validate_submission


def test_generated_data_has_a_strict_future_test_period() -> None:
    train, test, sample = generate_synthetic_competition_data(
        seed=7, n_meters=6, train_days=12, test_days=3
    )
    assert TARGET in train.columns
    assert TARGET not in test.columns
    assert pd.to_datetime(train[TIME_COLUMN]).max() < pd.to_datetime(test[TIME_COLUMN]).min()
    assert sample[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()


def test_features_are_independent_of_target_values() -> None:
    train, test, _ = generate_synthetic_competition_data(
        seed=9, n_meters=6, train_days=12, test_days=3
    )
    X_original, X_test_original = build_safe_features(train, test)
    mutated = train.copy()
    mutated[TARGET] = mutated[TARGET] * 1000 + np.arange(len(mutated))
    X_mutated, X_test_mutated = build_safe_features(mutated, test)

    pd.testing.assert_frame_equal(X_original, X_mutated)
    pd.testing.assert_frame_equal(X_test_original, X_test_mutated)


def test_timestamp_block_splits_do_not_mix_time_boundaries() -> None:
    train, _, _ = generate_synthetic_competition_data(
        seed=11, n_meters=6, train_days=12, test_days=3
    )
    timestamps = pd.to_datetime(train[TIME_COLUMN])
    for train_idx, valid_idx in make_timestamp_block_splits(timestamps, n_splits=3):
        assert timestamps.iloc[train_idx].max() < timestamps.iloc[valid_idx].min()
        assert set(timestamps.iloc[train_idx]).isdisjoint(set(timestamps.iloc[valid_idx]))


def test_small_rehearsal_creates_a_valid_submission(tmp_path) -> None:
    artifacts = run_rehearsal(
        output_root=tmp_path,
        seed=13,
        n_splits=2,
        n_estimators=12,
        n_meters=6,
        train_days=12,
        test_days=3,
    )
    submission = pd.read_csv(artifacts.submission_path)
    sample = pd.read_csv(artifacts.sample_submission_path)
    assert validate_submission(submission, sample) == []
    assert artifacts.model_path.exists()
    assert np.isfinite(artifacts.cv_score)
