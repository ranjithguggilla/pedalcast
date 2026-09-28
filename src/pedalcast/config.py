"""Project paths and shared constants."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = REPO_ROOT / "data"
ASSETS_DIR = REPO_ROOT / "assets"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"

HOURLY_CSV = DATA_DIR / "hour.csv"
MODEL_PATH = MODELS_DIR / "pedalcast_model.joblib"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

# Fraction of the (chronologically ordered) history held out as the test window.
TEST_FRACTION = 0.2
RANDOM_STATE = 42

# Revenue/cost assumptions used only for the illustrative business-impact estimate.
# Stated openly so the number can be re-derived or challenged.
VALUE_PER_BIKE_ERROR_USD = 1.50
