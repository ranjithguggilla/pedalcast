import numpy as np
import pandas as pd

from pedalcast.model import (
    build_models,
    chronological_split,
    regression_metrics,
    rmsle,
    seasonal_naive_predict,
)


def _frame(n=400):
    rng = np.random.default_rng(0)
    hour = rng.integers(0, 24, n)
    return pd.DataFrame({
        "date": pd.date_range("2011-01-01", periods=n, freq="h"),
        "season": rng.integers(1, 5, n), "year": rng.integers(0, 2, n),
        "month": rng.integers(1, 13, n), "hour": hour, "holiday": 0,
        "weekday": rng.integers(0, 7, n), "workingday": rng.integers(0, 2, n),
        "weather": rng.integers(1, 4, n), "temp": rng.random(n), "atemp": rng.random(n),
        "humidity": rng.random(n), "windspeed": rng.random(n),
        "count": (hour * 5 + rng.integers(0, 20, n)).astype(int),
    })


def test_split_is_chronological_and_disjoint():
    from pedalcast.features import build_feature_frame
    df = build_feature_frame(_frame())
    train, test = chronological_split(df, test_fraction=0.25)
    assert len(train) + len(test) == len(df)
    assert train["date"].max() <= test["date"].min()


def test_metrics_keys_and_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y, y)
    assert set(m) == {"rmse", "mae", "r2", "rmsle"}
    assert m["rmse"] == 0.0 and m["r2"] == 1.0


def test_rmsle_clips_negative_predictions():
    assert rmsle(np.array([5.0]), np.array([-3.0])) == rmsle(np.array([5.0]), np.array([0.0]))


def test_model_trains_and_predicts_non_negative():
    from pedalcast.features import FEATURE_COLUMNS, build_feature_frame
    df = build_feature_frame(_frame())
    train, test = chronological_split(df)
    model = build_models()["Ridge"]
    model.fit(train[FEATURE_COLUMNS], train["count"].to_numpy())
    preds = model.predict(test[FEATURE_COLUMNS])
    assert len(preds) == len(test)
    assert (preds >= 0).all()


def test_seasonal_naive_shapes():
    df = _frame()
    train, test = chronological_split(df)
    preds = seasonal_naive_predict(train, test)
    assert len(preds) == len(test)
