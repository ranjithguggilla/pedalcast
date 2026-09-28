import pandas as pd

from pedalcast.app_logic import explain, predict_demand
from pedalcast.features import FEATURE_COLUMNS
from pedalcast.model import build_models

BASE = {
    "season": 3, "year": 1, "month": 7, "hour": 8, "holiday": 0, "weekday": 2,
    "workingday": 1, "weather": 1, "temp": 0.6, "atemp": 0.6, "humidity": 0.5,
    "windspeed": 0.15,
}


def _tiny_bundle():
    import numpy as np
    rng = np.random.default_rng(1)
    n = 300
    hour = rng.integers(0, 24, n)
    from pedalcast.features import build_feature_frame
    df = build_feature_frame(pd.DataFrame({
        "season": rng.integers(1, 5, n), "year": rng.integers(0, 2, n),
        "month": rng.integers(1, 13, n), "hour": hour, "holiday": 0,
        "weekday": rng.integers(0, 7, n), "workingday": rng.integers(0, 2, n),
        "weather": rng.integers(1, 4, n), "temp": rng.random(n), "atemp": rng.random(n),
        "humidity": rng.random(n), "windspeed": rng.random(n),
        "count": (hour * 6 + 5).astype(int),
    }))
    model = build_models()["LightGBM"]
    model.fit(df[FEATURE_COLUMNS], df["count"].to_numpy())
    return {"model": model, "feature_columns": FEATURE_COLUMNS, "model_name": "LightGBM"}


def test_predict_demand_returns_non_negative_int():
    bundle = _tiny_bundle()
    value = predict_demand(bundle, BASE)
    assert isinstance(value, int)
    assert value >= 0


def test_rush_hour_beats_night():
    bundle = _tiny_bundle()
    day = predict_demand(bundle, {**BASE, "hour": 8})
    night = predict_demand(bundle, {**BASE, "hour": 3})
    assert day > night


def test_explain_mentions_rush_hour():
    reasons = " ".join(explain(BASE))
    assert "rush hour" in reasons
