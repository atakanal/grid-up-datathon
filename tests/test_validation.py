import pandas as pd
import pytest

from src.validation import make_splits


def test_group_split_has_no_overlap():
    X = pd.DataFrame({"x": range(12)})
    y = pd.Series([0, 1] * 6)
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    for train_idx, valid_idx in make_splits(X, y, "group", n_splits=3, groups=groups):
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[valid_idx]))


def test_group_split_requires_groups():
    X = pd.DataFrame({"x": range(8)})
    y = pd.Series([0, 1] * 4)
    with pytest.raises(ValueError):
        list(make_splits(X, y, "group", n_splits=2))


def test_timeseries_split_respects_order():
    X = pd.DataFrame({"x": range(20)})
    y = pd.Series(range(20))
    for train_idx, valid_idx in make_splits(X, y, "timeseries", n_splits=4):
        assert max(train_idx) < min(valid_idx)
