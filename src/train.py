from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score

from .models import build_model
from .utils import ensure_columns, set_seed, utc_timestamp
from .validation import make_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid Up generic baseline trainer")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--metric", default=None, help="rmse, mae, auc veya accuracy")
    parser.add_argument("--model", choices=["randomforest", "lightgbm", "catboost", "xgboost"], default="randomforest")
    parser.add_argument("--cv", choices=["kfold", "stratified", "group", "timeseries"], default="kfold")
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--experiment-name", default="baseline")
    return parser.parse_args()


def get_metric(name: str, task: str) -> tuple[Callable, bool]:
    name = name or ("rmse" if task == "regression" else "auc")
    if name == "rmse":
        return lambda y, p: mean_squared_error(y, p) ** 0.5, False
    if name == "mae":
        return mean_absolute_error, False
    if name == "auc":
        return roc_auc_score, True
    if name == "accuracy":
        return lambda y, p: accuracy_score(y, (p >= 0.5).astype(int)), True
    raise ValueError("Metrik rmse, mae, auc veya accuracy olmalıdır.")


def prepare_tabular_features(train: pd.DataFrame, test: pd.DataFrame, target: str, id_column: str):
    feature_columns = [c for c in train.columns if c not in {target, id_column}]
    X = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    combined = pd.concat([X, X_test], axis=0, ignore_index=True)
    categorical = combined.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    combined = pd.get_dummies(combined, columns=categorical, dummy_na=True)

    numeric_cols = combined.select_dtypes(include=np.number).columns
    combined[numeric_cols] = combined[numeric_cols].replace([np.inf, -np.inf], np.nan)
    combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].median())

    X_out = combined.iloc[: len(X)].reset_index(drop=True)
    X_test_out = combined.iloc[len(X) :].reset_index(drop=True)
    return X_out, X_test_out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    ensure_columns(train, [args.target, args.id_column], "train")
    ensure_columns(test, [args.id_column], "test")
    if args.group_column:
        ensure_columns(train, [args.group_column], "train")

    y = train[args.target].reset_index(drop=True)
    test_ids = test[args.id_column].copy()
    groups = train[args.group_column].reset_index(drop=True) if args.group_column else None
    X, X_test = prepare_tabular_features(train, test, args.target, args.id_column)

    metric_fn, higher_is_better = get_metric(args.metric, args.task)
    oof = np.full(len(train), np.nan, dtype=float)
    test_predictions = np.zeros(len(test), dtype=float)
    fold_scores: list[float] = []
    models = []

    splits = make_splits(X, y, args.cv, args.n_splits, args.seed, groups)
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = build_model(args.model, args.task, args.seed + fold)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        if args.task == "classification":
            valid_pred = model.predict_proba(X.iloc[valid_idx])[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]
        else:
            valid_pred = model.predict(X.iloc[valid_idx])
            test_pred = model.predict(X_test)

        oof[valid_idx] = valid_pred
        test_predictions += test_pred / args.n_splits
        score = float(metric_fn(y.iloc[valid_idx], valid_pred))
        fold_scores.append(score)
        models.append(model)
        print(f"Fold {fold}: {score:.6f}")

    valid_oof_mask = np.isfinite(oof)
    if not valid_oof_mask.any():
        raise RuntimeError("Hiçbir OOF tahmini üretilemedi.")
    cv_score = float(metric_fn(y[valid_oof_mask], oof[valid_oof_mask]))
    print(f"OOF score: {cv_score:.6f} ({'yüksek iyi' if higher_is_better else 'düşük iyi'})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / f"{args.experiment_name}.csv"
    pd.DataFrame({args.id_column: test_ids, args.target: test_predictions}).to_csv(submission_path, index=False)

    model_dir = Path("models") / args.experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    for idx, model in enumerate(models, start=1):
        joblib.dump(model, model_dir / f"fold_{idx}.joblib")

    record = {
        "timestamp_utc": utc_timestamp(),
        "experiment": args.experiment_name,
        "model": args.model,
        "task": args.task,
        "metric": args.metric or ("rmse" if args.task == "regression" else "auc"),
        "cv": args.cv,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "fold_scores": fold_scores,
        "cv_score": cv_score,
        "submission": str(submission_path),
        "public_lb": "",
        "notes": "",
    }
    (model_dir / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    log_path = Path("experiments/experiments.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_row = pd.DataFrame([{**record, "fold_scores": json.dumps(fold_scores)}])
    log_row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    print(f"Submission: {submission_path}")


if __name__ == "__main__":
    main()
