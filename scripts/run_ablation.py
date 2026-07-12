from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation import compute_ablation_attribution, run_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate model, feature, calibration and meter-identity effects."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--thread-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_ablation(
        output_root=Path("."),
        seed=args.seed,
        n_splits=args.n_splits,
        iterations=args.iterations,
        thread_count=args.thread_count,
    )

    import pandas as pd

    results = pd.read_csv(artifacts.results_path)
    attribution = compute_ablation_attribution(results)

    print("\nABLATION COMPLETE")
    print("=" * 80)
    print(f"Random Forest baseline RMSE:          {artifacts.baseline_rmse:.4f}")
    print(f"CatBoost safe-feature RMSE:           {artifacts.safe_catboost_rmse:.4f}")
    print(f"CatBoost diagnosis-feature RMSE:      {artifacts.diagnosis_catboost_rmse:.4f}")
    print(f"CatBoost calibrated RMSE:             {artifacts.calibrated_rmse:.4f}")
    print(f"CatBoost without direct identity:     {artifacts.no_identity_rmse:.4f}")
    print("\nRMSE attribution:")
    print(f"  Model gain:                          {attribution['model_gain_rmse']:+.4f}")
    print(f"  Diagnosis-feature gain:              {attribution['feature_gain_rmse']:+.4f}")
    print(f"  Calibration gain:                    {attribution['calibration_gain_rmse']:+.4f}")
    print(f"  Direct meter identity value:         {attribution['direct_identity_value_rmse']:+.4f}")
    print(f"  Total gain:                          {attribution['total_gain_rmse']:+.4f} ({attribution['total_gain_pct']:.2f}%)")
    print("\nGenerated files:")
    print(f"  Results:              {artifacts.results_path}")
    print(f"  OOF comparison:       {artifacts.oof_path}")
    print(f"  Report:               {artifacts.report_path}")
    print(f"  Safe submission:      {artifacts.safe_submission_path}")
    print(f"  No-identity submission: {artifacts.no_identity_submission_path}")
    print("\nCore lesson: change one component at a time before crediting an improvement.")


if __name__ == "__main__":
    main()
