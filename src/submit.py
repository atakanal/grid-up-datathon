from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submission schema and value checker")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--sample", required=True)
    return parser.parse_args()


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    same_columns = list(submission.columns) == list(sample.columns)
    if not same_columns:
        errors.append(f"Sütunlar farklı. Beklenen: {list(sample.columns)}")
    if len(submission) != len(sample):
        errors.append(f"Satır sayısı farklı. Beklenen: {len(sample)}, bulunan: {len(submission)}")
    if submission.isna().any().any():
        errors.append("Submission içinde NaN var.")
    if submission.duplicated().any():
        errors.append("Submission içinde tamamen aynı yinelenen satırlar var.")

    if same_columns and len(submission) == len(sample) and len(sample.columns) >= 1:
        id_column = sample.columns[0]
        if not submission[id_column].reset_index(drop=True).equals(
            sample[id_column].reset_index(drop=True)
        ):
            errors.append(f"{id_column} değerleri veya sırası sample submission ile eşleşmiyor.")

        for prediction_column in sample.columns[1:]:
            numeric_values = pd.to_numeric(submission[prediction_column], errors="coerce")
            if numeric_values.isna().any():
                errors.append(f"{prediction_column} sütununda sayısal olmayan değer var.")
            elif not np.isfinite(numeric_values.to_numpy()).all():
                errors.append(f"{prediction_column} sütununda sonsuz değer var.")
    return errors


def main() -> None:
    args = parse_args()
    submission_path = Path(args.submission)
    sample_path = Path(args.sample)
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    errors = validate_submission(submission, sample)
    if errors:
        raise SystemExit("\n".join(f"HATA: {error}" for error in errors))
    print("Submission şeması, kimlik sırası ve temel değer kontrolleri başarılı.")


if __name__ == "__main__":
    main()
