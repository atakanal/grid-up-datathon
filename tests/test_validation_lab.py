from __future__ import annotations

import numpy as np

from scripts.run_validation_lab import generate_panel_data


def test_safe_rolling_feature_uses_only_previous_targets() -> None:
    df = generate_panel_data(seed=7)
    meter = df[df["meter_id"] == "M000"].sort_values("day_index").reset_index(drop=True)

    expected_safe = meter.loc[0:2, "target"].mean()
    expected_leaky = meter.loc[1:3, "target"].mean()

    assert np.isclose(meter.loc[3, "safe_roll_3"], expected_safe)
    assert np.isclose(meter.loc[3, "leaky_roll_3"], expected_leaky)


def test_generated_panel_has_future_and_multiple_groups() -> None:
    df = generate_panel_data(seed=7)

    assert df["day_index"].max() == 139
    assert df["meter_id"].nunique() == 45
    assert df["date"].is_monotonic_increasing
