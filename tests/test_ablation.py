from __future__ import annotations

import pandas as pd
import pytest

from src.ablation import compute_ablation_attribution, drop_direct_meter_identity


def test_drop_direct_meter_identity_removes_only_direct_columns() -> None:
    frame = pd.DataFrame(
        {
            "meter_id": ["M001"],
            "meter_hour": ["M001__6"],
            "voltage_v_past_mean_4": [229.1],
            "customer_hour": ["residential__6"],
        }
    )
    result = drop_direct_meter_identity(frame)
    assert "meter_id" not in result.columns
    assert "meter_hour" not in result.columns
    assert "voltage_v_past_mean_4" in result.columns
    assert "customer_hour" in result.columns


def test_ablation_attribution_changes_one_component_at_a_time() -> None:
    results = pd.DataFrame(
        {
            "experiment": [
                "RandomForest baseline",
                "CatBoost safe features",
                "CatBoost diagnosis features",
                "CatBoost diagnosis + causal calibration",
                "CatBoost diagnosis without direct meter identity",
            ],
            "rmse": [6.0, 4.5, 4.0, 3.8, 5.2],
        }
    )
    attribution = compute_ablation_attribution(results)
    assert attribution["model_gain_rmse"] == pytest.approx(1.5)
    assert attribution["feature_gain_rmse"] == pytest.approx(0.5)
    assert attribution["calibration_gain_rmse"] == pytest.approx(0.2)
    assert attribution["direct_identity_value_rmse"] == pytest.approx(1.2)
    assert attribution["total_gain_rmse"] == pytest.approx(2.2)


def test_ablation_attribution_rejects_missing_experiment() -> None:
    with pytest.raises(ValueError, match="Missing ablation experiments"):
        compute_ablation_attribution(
            pd.DataFrame({"experiment": ["RandomForest baseline"], "rmse": [6.0]})
        )
