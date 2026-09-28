from pathlib import Path

from pedalcast.data import load_hourly

DATA = Path(__file__).resolve().parents[1] / "data" / "hour.csv"


def test_load_returns_expected_columns():
    df = load_hourly(DATA)
    for col in ["date", "year", "month", "hour", "weather", "humidity", "count"]:
        assert col in df.columns


def test_no_target_leakage_columns():
    df = load_hourly(DATA)
    assert "casual" not in df.columns
    assert "registered" not in df.columns


def test_target_is_clean_and_sorted():
    df = load_hourly(DATA)
    assert df["count"].notna().all()
    assert (df["count"] >= 0).all()
    assert df["date"].is_monotonic_increasing
    assert len(df) == 17379
