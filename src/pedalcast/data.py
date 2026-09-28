"""Load and normalise the raw hourly bike-share records.

The raw file is the UCI Bike Sharing dataset (Capital Bikeshare, Washington DC,
2011-2012). Column names in the source are terse, so I rename them to something
readable and drop the two columns that would leak the target.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

# Source name -> readable name.
RENAME = {
    "dteday": "date",
    "yr": "year",
    "mnth": "month",
    "hr": "hour",
    "weathersit": "weather",
    "hum": "humidity",
    "cnt": "count",
}

# casual + registered sum exactly to count, so keeping either one leaks the label.
LEAKY_COLUMNS = ["instant", "casual", "registered"]

EXPECTED_RAW = {
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
}


def load_hourly(path: str | Path) -> pd.DataFrame:
    """Read the hourly CSV with polars, validate it, return a tidy pandas frame."""
    frame = pl.read_csv(str(path), try_parse_dates=True)

    missing = EXPECTED_RAW - set(frame.columns)
    if missing:
        raise ValueError(f"raw file is missing expected columns: {sorted(missing)}")

    df = frame.to_pandas()
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
    df = df.rename(columns=RENAME)
    df["date"] = pd.to_datetime(df["date"])

    if df["count"].isna().any():
        raise ValueError("target column 'count' contains nulls")
    if (df["count"] < 0).any():
        raise ValueError("target column 'count' contains negative values")

    return df.sort_values(["date", "hour"]).reset_index(drop=True)
