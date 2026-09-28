import pandas as pd

from pedalcast.features import FEATURE_COLUMNS, TARGET, build_feature_frame


def _sample():
    return pd.DataFrame({
        "season": [1, 3], "year": [0, 1], "month": [1, 7], "hour": [8, 3],
        "holiday": [0, 0], "weekday": [1, 6], "workingday": [1, 0], "weather": [1, 2],
        "temp": [0.3, 0.7], "atemp": [0.3, 0.7], "humidity": [0.5, 0.6],
        "windspeed": [0.1, 0.2], "count": [120, 4],
    })


def test_cyclical_features_are_bounded():
    out = build_feature_frame(_sample())
    for col in ["hour_sin", "hour_cos", "month_sin", "weekday_cos"]:
        assert out[col].between(-1, 1).all()


def test_commute_flags():
    out = build_feature_frame(_sample())
    assert out.loc[0, "is_rush_hour"] == 1  # 8am
    assert out.loc[1, "is_rush_hour"] == 0  # 3am
    assert out.loc[1, "is_weekend"] == 1    # Saturday


def test_feature_columns_present_and_no_target():
    out = build_feature_frame(_sample())
    for col in FEATURE_COLUMNS:
        assert col in out.columns
    assert TARGET not in FEATURE_COLUMNS
