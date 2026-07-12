# Rehearsal Error Analysis Report

This report uses only out-of-fold rows that were never used to fit the model producing their prediction.
Signed error is defined as `prediction - actual`; positive bias means overprediction.

## Overall diagnostics

- Validated rows: **3240**
- RMSE: **6.3864 kWh**
- MAE: **4.6732 kWh**
- Mean bias: **-1.7397 kWh**; the model underpredicts on average.

## Hardest segments

- Fold **1** has the highest RMSE: **7.9258**.
- Hour **6:00** has the highest RMSE: **7.0317**.
- Meter **M013** has the highest RMSE: **10.0690**.
- Consumption band **Q5** has the highest RMSE: **8.3063**.

## Interpretation checklist

1. A bad fold can indicate limited history or distribution shift, not merely weak hyperparameters.
2. Meter-level errors can reveal identity effects or underrepresented operating profiles.
3. Bias by consumption band shows whether peaks are systematically flattened or exaggerated.
4. The largest individual errors should be inspected before adding model complexity.

## Top model features

- `numeric__voltage_v_past_mean_4`: 0.4919
- `numeric__voltage_v`: 0.0689
- `categorical__meter_id_M017`: 0.0450
- `categorical__meter_id_M004`: 0.0415
- `categorical__meter_id_M008`: 0.0384
- `categorical__meter_id_M015`: 0.0266
- `categorical__meter_id_M005`: 0.0252
- `numeric__hour_cos`: 0.0216
- `categorical__meter_id_M014`: 0.0201
- `categorical__meter_id_M009`: 0.0196
