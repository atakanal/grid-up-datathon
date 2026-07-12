# Grid Up Ablation Lab

This lab separates four effects on identical temporal OOF rows:

1. **Model effect:** Random Forest to CatBoost while keeping the safe feature set.
2. **Feature effect:** Safe features to diagnosis-driven features with CatBoost fixed.
3. **Calibration effect:** Raw diagnosis predictions to causal fold calibration.
4. **Identity dependence:** Remove direct `meter_id` and `meter_hour` features.

## Run

```powershell
.\.venv\Scripts\python.exe .\scripts\run_ablation.py
```

The rehearsal and model-improvement stages must already have been run.

## Interpretation

- A large model gain means algorithm choice mattered more than feature engineering.
- A positive diagnosis-feature gain means the error-analysis features added value beyond CatBoost alone.
- A small or negative feature gain means the added complexity is not justified.
- A large no-identity deterioration means the temporal task benefits from knowing the meter. This is valid only when the real test contains the same meters.
- If test meters are unseen, validate with GroupKFold before keeping identity features.
