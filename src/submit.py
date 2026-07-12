from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submission schema and value checker")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--sample", required=True)
    return parser.parse_args()


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if list(submission.columns) != list(sample.columns):
        errors.append(f"Sütunlar farklı. Beklenen: {list(sample.columns)}")
    if len(submission) != len(sample):
        errors.append(f"Satır sayısı farklı. Beklenen: {len(sample)}, bulunan: {len(submission)}")
    if submission.isna().any().any():
        errors.append("Submission içinde NaN var.")
    if submission.duplicated().any():
        errors.append("Submission içinde tamamen aynı yinelenen satırlar var.")
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
    print("Submission şeması ve temel değer kontrolleri başarılı.")


if __name__ == "__main__":
    main()
