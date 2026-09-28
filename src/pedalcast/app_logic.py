"""Scoring logic kept separate from the Streamlit layer so it can be unit tested
without spinning up a server."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from pedalcast.config import MODEL_PATH
from pedalcast.features import RUSH_HOURS, build_feature_frame

WEATHER_LABELS = {
    1: "clear or partly cloudy",
    2: "misty or cloudy",
    3: "light rain or snow",
    4: "heavy rain or storm",
}


def load_bundle(path: str | Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def _raw_row(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "season": inputs["season"],
        "year": inputs.get("year", 1),
        "month": inputs["month"],
        "hour": inputs["hour"],
        "holiday": inputs.get("holiday", 0),
        "weekday": inputs["weekday"],
        "workingday": inputs.get("workingday", 1),
        "weather": inputs["weather"],
        "temp": inputs["temp"],
        "atemp": inputs.get("atemp", inputs["temp"]),
        "humidity": inputs["humidity"],
        "windspeed": inputs["windspeed"],
    }])


def predict_demand(bundle: dict, inputs: dict) -> int:
    """Return the predicted number of bikes rented in the described hour."""
    frame = build_feature_frame(_raw_row(inputs))
    model = bundle["model"]
    pred = float(model.predict(frame[bundle["feature_columns"]])[0])
    return int(round(max(pred, 0.0)))


def explain(inputs: dict) -> list[str]:
    """Plain-language reasons for the score, no model internals required."""
    reasons = []
    if inputs["hour"] in RUSH_HOURS:
        reasons.append("commute rush hour, which is the busiest part of the day")
    elif 0 <= inputs["hour"] <= 5:
        reasons.append("overnight window, when demand is usually near zero")
    if inputs["temp"] >= 0.6:
        reasons.append("warm weather, which lifts casual ridership")
    elif inputs["temp"] <= 0.3:
        reasons.append("cold weather, which suppresses ridership")
    if inputs["weather"] >= 3:
        reasons.append(f"poor conditions ({WEATHER_LABELS[inputs['weather']]})")
    if inputs.get("workingday", 1) == 0:
        reasons.append("non-working day, so the commute spikes flatten out")
    if not reasons:
        reasons.append("typical daytime conditions")
    return reasons
