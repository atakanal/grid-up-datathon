# Grid Up Model Improvement Report

## Same-fold comparison

| experiment | rmse | mae | bias |
| --- | --- | --- | --- |
| RandomForest baseline | 6.3864 | 4.6732 | -1.7397 |
| CatBoost diagnosis features | 3.8493 | 3.0434 | -1.3826 |
| CatBoost causal calibration | 3.7574 | 2.9431 | -0.3219 |

## CatBoost fold RMSE

- Fold 1: 4.4716
- Fold 2: 3.4461
- Fold 3: 3.5468

## Causal fold corrections

- Fold 1: +0.0000 kWh
- Fold 2: +1.8070 kWh
- Fold 3: +1.3753 kWh

## Final test calibration

- OOF-estimated shift applied to the calibrated test submission: +1.3826 kWh
- Keep both raw and calibrated submissions. Real competitions may react differently under distribution shift.

## Top features

| feature | importance |
| --- | --- |
| meter_id | 41.2016 |
| voltage_v_past_mean_4 | 8.2143 |
| customer_hour | 7.8302 |
| region | 6.7563 |
| voltage_v | 4.8607 |
| region_customer | 4.8550 |
| hour_cos | 3.6931 |
| hour | 3.0436 |
| planned_maintenance | 2.9083 |
| customer_type | 2.7234 |
| voltage_drop_from_232 | 1.9900 |
| temperature_distance_20 | 1.7515 |
| temperature_distance_20_sq | 1.7345 |
| temperature_c | 1.1527 |
| voltage_v_lag_1 | 1.0350 |

## Interpretation

CatBoost reduces regression-to-the-mean by modelling categorical identity and nonlinear interactions more effectively. The causal calibration diagnostic uses only earlier-fold residuals, while the final test shift is estimated from all available OOF residuals.