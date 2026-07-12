from __future__ import annotations

from typing import Any


def build_model(name: str, task: str, seed: int, params: dict[str, Any] | None = None):
    """Build a model with lazy imports so CLI/help/tests remain fast and robust."""
    name = name.lower()
    task = task.lower()
    params = dict(params or {})

    if task not in {"regression", "classification"}:
        raise ValueError("Task regression veya classification olmalıdır.")

    if name == "randomforest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
        defaults = {
            "n_estimators": 150,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_jobs": 1,
            "random_state": seed,
        }
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMRegressor if task == "regression" else LGBMClassifier
        defaults = {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "verbosity": -1,
            "random_state": seed,
        }
    elif name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostRegressor if task == "regression" else CatBoostClassifier
        defaults = {
            "iterations": 800,
            "learning_rate": 0.03,
            "depth": 7,
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": seed,
        }
    elif name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBRegressor if task == "regression" else XGBClassifier
        defaults = {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 7,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "n_jobs": -1,
            "random_state": seed,
        }
    else:
        raise ValueError("Model randomforest, lightgbm, catboost veya xgboost olmalıdır.")

    defaults.update(params)
    return cls(**defaults)
