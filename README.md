# Grid Up Datathon

[![Verify submissions](https://github.com/atakanal/grid-up-datathon/actions/workflows/verify-submissions.yml/badge.svg)](https://github.com/atakanal/grid-up-datathon/actions/workflows/verify-submissions.yml)

End-to-end experimentation, validation, and submission archive for the
[Grid Up Datathon](https://www.kaggle.com/competitions/grid-up-datathon).

The project forecasts daily transformer electricity consumption under RMSLE.
Its final public-leaderboard solution treats transformers with historical
observations (warm series) separately from unseen transformers (cold series),
models predictions in `log1p` space, and applies stress-tested ensemble
selection.

## Results

| Public score | Submission | Rows | SHA-256 |
|---:|---|---:|---|
| **0.99430** | [`gridup_v108_uniform_endpoint_candidate.csv`](submissions/gridup_v108_uniform_endpoint_candidate.csv) | 714,688 | `E4EF38...1151` |
| **0.99676** | [`gridup_v96b_fresh_warm_challenger.csv`](submissions/gridup_v96b_fresh_warm_challenger.csv) | 714,688 | `FBCA47...CFFC` |

Both files preserve the official sample-submission order and pass checks for
unique IDs, finite values, and non-negative predictions. Full hashes and
machine-readable metadata are stored in
[`results/leaderboard_results.json`](results/leaderboard_results.json).

## Solution outline

- **Warm/cold routing:** 5,012 test transformers have train history; 2,024 do
  not. Models and validation protocols are separated accordingly.
- **Warm forecasting:** recent level, weekday seasonality, year-over-year
  profiles, trend, staleness, and horizon-aware combinations.
- **Cold forecasting:** power/location priors, natural-entry cohorts,
  lifecycle features, and hierarchical shrinkage.
- **External signals:** calendar, weather, water, electricity-system, and
  activity indicators were evaluated with source-specific controls.
- **Final selection:** predictions are combined in `log1p` space and tested
  across deterministic row, transformer, day, month, horizon, and model-family
  stress slices.

See [`docs/APPROACH.md`](docs/APPROACH.md) for the modeling narrative and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for verification steps.

## Repository layout

```text
.
├── data/           # local-only raw and processed data placeholders
├── docs/           # methodology and reproducibility notes
├── experiments/    # baseline experiment reports and diagnostics
├── notebooks/      # EDA and modeling notebooks
├── results/        # final leaderboard manifest
├── scripts/        # training, analysis, and verification utilities
├── src/            # reusable feature/model/validation modules
├── submissions/    # two best scored submissions
└── tests/           # pipeline and submission safety tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_environment.py
pytest -q
```

Verify the archived submissions without competition data:

```bash
python scripts/verify_final_submissions.py
```

To additionally verify their ID order against the official sample submission:

```bash
python scripts/verify_final_submissions.py \
  --competition-zip /path/to/grid-up-datathon.zip
```

## Data and reproducibility note

Competition data and third-party raw datasets are not redistributed. Some
late-stage public-leaderboard experiments, including the lineage of V108,
incorporated external observations dated after 31 March 2026. Their eligibility
must be reviewed against the competition's final external-data interpretation
before using the files in a judged notebook or final solution.

The repository is an experiment and submission archive, not production energy
forecasting software.
