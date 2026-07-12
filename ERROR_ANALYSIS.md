# Error Analysis Lab

This lab turns out-of-fold predictions into actionable model diagnostics.

## Why OOF errors?

Training errors are optimistic because the model has already seen those rows. This lab only uses rows predicted by a model that did not train on them.

Signed error is defined as:

```text
prediction - actual
```

- Positive bias: the model overpredicts.
- Negative bias: the model underpredicts.

## Run

```powershell
.\.venv\Scripts\python.exe .\scripts\run_error_analysis.py
```

The competition rehearsal must have been run first so that these files exist:

```text
data\raw\rehearsal_train.csv
data\processed\rehearsal_oof.csv
```

## Outputs

```text
experiments\error_analysis\rehearsal_error_rows.csv
experiments\error_analysis\rehearsal_error_segments.csv
experiments\error_analysis\rehearsal_worst_errors.csv
experiments\error_analysis\rehearsal_error_report.md
experiments\error_analysis\rehearsal_rmse_by_hour.png
experiments\error_analysis\rehearsal_hardest_meters.png
experiments\error_analysis\rehearsal_bias_by_consumption_band.png
experiments\error_analysis\rehearsal_error_distribution.png
```

## Questions to answer

1. Which fold is hardest, and is it the one with the least training history?
2. Does the model struggle at specific hours?
3. Are a few meters responsible for a disproportionate share of error?
4. Does the model flatten high-consumption peaks?
5. Is average bias near zero while segment-level bias remains large?
6. What do the largest individual errors have in common?

A useful next experiment should target one observed failure mode. Avoid changing five things at once.
