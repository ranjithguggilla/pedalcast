"""Feature engineering. Nothing here touches the target, so it is leak-free and
can run identically at train time and at scoring time for a single row."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "count"

# Encoded as small integers in the source; treated as categories by the model.
CATEGORICAL = ["season", "weather", "year"]

# Numeric columns fed straight through the preprocessor.
NUMERIC = [
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "holiday",
    "workingday",
    "is_weekend",
    "is_rush_hour",
]

FEATURE_COLUMNS = CATEGORICAL + NUMERIC

# Morning and evening commute peaks on the DC system.
RUSH_HOURS = {7, 8, 9, 17, 18, 19}


def _cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2.0 * np.pi * values / period
    return np.sin(radians), np.cos(radians)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encodings and a couple of commute flags.

    Works on the full training frame or on a single-row scoring frame.
    """
    out = df.copy()

    out["hour_sin"], out["hour_cos"] = _cyclical(out["hour"], 24)
    out["month_sin"], out["month_cos"] = _cyclical(out["month"], 12)
    out["weekday_sin"], out["weekday_cos"] = _cyclical(out["weekday"], 7)

    out["is_weekend"] = out["weekday"].isin([0, 6]).astype(int)
    out["is_rush_hour"] = out["hour"].isin(RUSH_HOURS).astype(int)

    return out
