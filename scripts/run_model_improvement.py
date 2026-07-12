from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_improvement import run_model_improvement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the rehearsal Random Forest with diagnosis-driven CatBoost."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--thread-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_model_improvement(
        output_root=Path("."),
        seed=args.seed,
        n_splits=args.n_splits,
        iterations=args.iterations,
        thread_count=args.thread_count,
    )

    print("\nMODEL IMPROVEMENT COMPLETE")
    print("=" * 80)
    print(f"Random Forest baseline RMSE:       {artifacts.baseline_rmse:.4f}")
    print(f"CatBoost raw OOF RMSE:             {artifacts.catboost_rmse:.4f}")
    print(f"CatBoost causal-calibrated RMSE:   {artifacts.causal_calibrated_rmse:.4f}")
    print(f"CatBoost raw bias:                 {artifacts.raw_bias:+.4f}")
    print(f"CatBoost causal-calibrated bias:   {artifacts.causal_calibrated_bias:+.4f}")
    print(f"Final calibrated test shift:       {artifacts.final_test_shift:+.4f} kWh")
    print("\nCatBoost fold scores:")
    for fold, score in enumerate(artifacts.fold_scores, start=1):
        print(f"  Fold {fold}: {score:.4f}")
    print("\nGenerated files:")
    print(f"  Results:                {artifacts.results_path}")
    print(f"  OOF comparison:         {artifacts.oof_path}")
    print(f"  Feature importance:     {artifacts.feature_importance_path}")
    print(f"  Raw submission:         {artifacts.raw_submission_path}")
    print(f"  Calibrated submission:  {artifacts.calibrated_submission_path}")
    print(f"  Report:                 {artifacts.report_path}")
    print(f"  Models:                 {artifacts.model_dir}")
    print(
        "\nCore lesson: change the model and features to attack a diagnosed failure mode, "
        "then compare on identical folds."
    )


if __name__ == "__main__":
    main()
