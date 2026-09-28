"""Model zoo, chronological split, and evaluation.

The comparison deliberately starts from baselines an operations team would
actually use (a flat mean, and an hour-of-day seasonal average) so the fancier
models have to earn their place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pedalcast.config import RANDOM_STATE, TEST_FRACTION
from pedalcast.features import CATEGORICAL, FEATURE_COLUMNS, NUMERIC, TARGET


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Split by time: earliest rows train, latest rows test. No shuffling, so
    the model is always asked to predict the future from the past."""
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    cut = int(len(df) * (1.0 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )


def _wrap(estimator) -> TransformedTargetRegressor:
    """Fit every model on log1p(count) and invert on the way out. Demand counts
    are right-skewed, and this keeps predictions non-negative."""
    pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def build_models() -> dict:
    return {
        "Ridge": _wrap(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "RandomForest": _wrap(
            RandomForestRegressor(
                n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
            )
        ),
        "LightGBM": _wrap(
            LGBMRegressor(
                n_estimators=600,
                learning_rate=0.05,
                num_leaves=48,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        ),
    }


# --- baselines an ops team would otherwise rely on -------------------------

def naive_mean_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), train[TARGET].mean())


def seasonal_naive_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Average demand for the same hour on the same kind of day (working vs not)."""
    table = train.groupby(["hour", "workingday"])[TARGET].mean()
    global_mean = train[TARGET].mean()
    keys = list(zip(test["hour"], test["workingday"], strict=False))
    return np.array([table.get(k, global_mean) for k in keys])


def fit_ols(train: pd.DataFrame, test: pd.DataFrame):
    """Interpretable statsmodels OLS on the same features, fit on log1p(count)."""
    prep = build_preprocessor()
    x_train = prep.fit_transform(train[FEATURE_COLUMNS])
    x_test = prep.transform(test[FEATURE_COLUMNS])
    names = list(prep.get_feature_names_out())

    x_train_c = sm.add_constant(x_train, has_constant="add")
    x_test_c = sm.add_constant(x_test, has_constant="add")
    result = sm.OLS(np.log1p(train[TARGET].to_numpy()), x_train_c).fit()

    preds = np.clip(np.expm1(result.predict(x_test_c)), 0, None)
    coeffs = pd.DataFrame(
        {"feature": ["const", *names], "coef": result.params, "p_value": result.pvalues}
    )
    return preds, coeffs, result.rsquared


# --- metrics ---------------------------------------------------------------

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "rmsle": rmsle(y_true, y_pred),
    }
