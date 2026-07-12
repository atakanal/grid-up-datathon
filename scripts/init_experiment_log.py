from pathlib import Path

import pandas as pd

COLUMNS = [
    "timestamp_utc",
    "experiment",
    "model",
    "task",
    "metric",
    "cv",
    "n_splits",
    "seed",
    "fold_scores",
    "cv_score",
    "submission",
    "public_lb",
    "notes",
]

path = Path("experiments/experiments.csv")
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
    print(f"Oluşturuldu: {path}")
else:
    print(f"Zaten mevcut: {path}")
