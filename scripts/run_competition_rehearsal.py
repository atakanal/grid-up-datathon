from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rehearsal import run_rehearsal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Grid Up end-to-end synthetic competition rehearsal."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_rehearsal(
        output_root=Path("."),
        seed=args.seed,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
    )

    print("\nCOMPETITION REHEARSAL COMPLETE")
    print("=" * 80)
    for fold, score in enumerate(artifacts.fold_scores, start=1):
        print(f"Fold {fold} RMSE: {score:.4f}")
    print(f"OOF RMSE:    {artifacts.cv_score:.4f}")
    print("\nGenerated files:")
    print(f"  Train:       {artifacts.train_path}")
    print(f"  Test:        {artifacts.test_path}")
    print(f"  Sample:      {artifacts.sample_submission_path}")
    print(f"  OOF:         {artifacts.oof_path}")
    print(f"  Results:     {artifacts.results_path}")
    print(f"  Importance:  {artifacts.feature_importance_path}")
    print(f"  Model:       {artifacts.model_path}")
    print(f"  Submission:  {artifacts.submission_path}")
    print("\nCore lesson: a valid submission is the final output of a reproducible pipeline, not a manually edited CSV.")


if __name__ == "__main__":
    main()
