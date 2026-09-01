"""Verify frozen leaderboard submissions against their manifest and sample order."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "leaderboard_results.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sample_ids(zip_path: Path | None) -> list[str] | None:
    if zip_path is None:
        return None
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("sample_submission.csv") as raw:
            rows = csv.DictReader(line.decode("utf-8-sig") for line in raw)
            if rows.fieldnames is None or "id" not in rows.fieldnames:
                raise AssertionError("sample_submission.csv has no id column")
            return [row["id"] for row in rows]


def verify_file(record: dict[str, object], expected_ids: list[str] | None) -> dict[str, object]:
    path = ROOT / str(record["file"])
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256(path)
    if digest != record["sha256"]:
        raise AssertionError(f"SHA-256 mismatch: {path}")

    ids: list[str] = []
    minimum = math.inf
    maximum = -math.inf
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "tuketim"]:
            raise AssertionError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            ids.append(row["id"])
            value = float(row["tuketim"])
            if not math.isfinite(value) or value < 0.0:
                raise AssertionError(f"Invalid prediction in {path}")
            minimum = min(minimum, value)
            maximum = max(maximum, value)

    expected_rows = int(record["rows"])
    unique_ids = len(set(ids))
    if len(ids) != expected_rows or unique_ids != int(record["unique_ids"]):
        raise AssertionError(f"Row or unique-ID mismatch: {path}")
    if expected_ids is not None and ids != expected_ids:
        raise AssertionError(f"Sample-submission order mismatch: {path}")
    return {
        "version": record["version"],
        "public_score": record["public_score"],
        "rows": len(ids),
        "unique_ids": unique_ids,
        "minimum": minimum,
        "maximum": maximum,
        "sample_order_checked": expected_ids is not None,
        "sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-zip", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_ids = sample_ids(args.competition_zip)
    results = [verify_file(record, expected_ids) for record in manifest["submissions"]]
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
