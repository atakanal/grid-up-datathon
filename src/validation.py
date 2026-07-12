from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    n_splits: int = 5,
    seed: int = 42,
    groups: pd.Series | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Return CV indices while enforcing the information required by each strategy."""
    strategy = strategy.lower()

    if strategy == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return splitter.split(X, y)

    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return splitter.split(X, y)

    if strategy == "group":
        if groups is None:
            raise ValueError("GroupKFold için groups verilmelidir.")
        splitter = GroupKFold(n_splits=n_splits)
        return splitter.split(X, y, groups=groups)

    if strategy == "timeseries":
        splitter = TimeSeriesSplit(n_splits=n_splits)
        return splitter.split(X, y)

    raise ValueError(
        f"Bilinmeyen CV stratejisi: {strategy}. "
        "Geçerli seçenekler: kfold, stratified, group, timeseries"
    )
