from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.error_analysis import run_error_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze out-of-fold errors from the Grid Up competition rehearsal."
    )
    parser.add_argument("--train-path", default="data/raw/rehearsal_train.csv")
    parser.add_argument("--oof-path", default="data/processed/rehearsal_oof.csv")
    parser.add_argument(
        "--feature-importance-path", default="experiments/rehearsal_feature_importance.csv"
    )
    parser.add_argument("--output-dir", default="experiments/error_analysis")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--top-n-errors", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_error_analysis(
        train_path=args.train_path,
        oof_path=args.oof_path,
        feature_importance_path=args.feature_importance_path,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        top_n_errors=args.top_n_errors,
    )

    print("\nERROR ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"OOF RMSE: {artifacts.overall_rmse:.4f}")
    print(f"OOF MAE:  {artifacts.overall_mae:.4f}")
    print(f"OOF bias: {artifacts.overall_bias:.4f} (prediction - actual)")
    print("\nGenerated files:")
    print(f"  Row diagnostics:  {artifacts.analysis_path}")
    print(f"  Segment summary:  {artifacts.segment_summary_path}")
    print(f"  Worst errors:     {artifacts.worst_errors_path}")
    print(f"  Markdown report:  {artifacts.report_path}")
    for path in artifacts.plot_paths:
        print(f"  Plot:             {path}")
    print(
        "\nCore lesson: improve the model against a diagnosed failure mode, not against a single aggregate score."
    )


if __name__ == "__main__":
    main()
