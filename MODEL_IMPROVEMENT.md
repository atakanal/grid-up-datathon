# Model Improvement Lab

This lab responds directly to the rehearsal error analysis:

- Random Forest regressed high-consumption meters toward the mean.
- Overall OOF bias was negative.
- Large errors clustered around meter-specific and nonlinear operating patterns.

The new experiment keeps the exact same timestamp-block folds and compares:

1. Existing Random Forest OOF predictions.
2. CatBoost with diagnosis-driven, target-independent features.
3. CatBoost with a causal fold-bias diagnostic.

## Why CatBoost?

CatBoost can model categorical identities and nonlinear interactions without expanding every category into a large dense one-hot matrix. This is useful for meter, region, customer type and meter-hour interactions.

## New features

- Absolute and squared distance from 20°C.
- Voltage drop from 232 V.
- One-step changes in weather and voltage inputs.
- Cyclical day-of-week features.
- Missing-value flags.
- Customer-hour, meter-hour and region-customer interactions.

None of these features reads the target column.

## Calibration rule

For honest OOF diagnostics, fold 1 receives no correction. Each later fold can use only residuals observed in earlier folds. The final calibrated test submission uses the mean residual from all OOF rows as a one-parameter shift. Keep both raw and calibrated submissions because real test distributions can drift.
