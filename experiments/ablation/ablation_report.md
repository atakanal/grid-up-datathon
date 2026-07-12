# Grid Up Ablation Report

## Same-fold results

| experiment | rmse | mae | bias | feature_count | purpose |
| --- | --- | --- | --- | --- | --- |
| RandomForest baseline | 6.3864 | 4.6732 | -1.7397 | 20 | reference |
| CatBoost safe features | 4.1390 | 3.2490 | -1.3553 | 20 | model effect |
| CatBoost diagnosis features | 3.8493 | 3.0434 | -1.3826 | 36 | feature effect |
| CatBoost diagnosis + causal calibration | 3.7574 | 2.9431 | -0.3219 | 36 | calibration effect |
| CatBoost diagnosis without direct meter identity | 10.4870 | 8.0546 | -1.4593 | 34 | identity stress test |

## Improvement attribution

- Model change (Random Forest -> CatBoost, same safe features): +2.2474 RMSE
- Diagnosis-driven feature engineering: +0.2897 RMSE
- Causal calibration: +0.0919 RMSE
- Total baseline-to-calibrated gain: +2.6290 RMSE (41.17%)

## Direct meter identity stress test

- Removing `meter_id` and `meter_hour` changes RMSE by +6.6378.
- A large deterioration means the temporal split rewards known-meter identity. It does not prove leakage because the same meters exist at inference time in this rehearsal.
- If the real test contains unseen meters, repeat this with GroupKFold before trusting identity features.

## Fold scores

- CatBoost safe features: [4.9524, 3.6891, 3.6411]
- CatBoost without direct identity: [10.8768, 10.2133, 10.3595]

## Interpretation rule

Ablation changes one component at a time. Prefer changes that improve the same validation rows and remain defensible under the real test-generation process.