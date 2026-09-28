"""End-to-end training run: load, split, fit every model, score on the future
window, save the best pipeline, metrics, and the README charts."""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.inspection import permutation_importance

from pedalcast.config import (
    ASSETS_DIR,
    HOURLY_CSV,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    VALUE_PER_BIKE_ERROR_USD,
)
from pedalcast.data import load_hourly
from pedalcast.features import FEATURE_COLUMNS, TARGET, build_feature_frame
from pedalcast.model import (
    build_models,
    chronological_split,
    fit_ols,
    naive_mean_predict,
    regression_metrics,
    seasonal_naive_predict,
)
from pedalcast.plots import (
    save_eda,
    save_importance,
    save_model_comparison,
    save_predictions,
)


def business_impact(test, best_mae: float, seasonal_mae: float) -> dict:
    """Turn the error gap over the seasonal baseline into an illustrative dollar
    figure, annualised. Assumptions live in config and are quoted in the README."""
    hours = len(test)
    bikes_saved = max(seasonal_mae - best_mae, 0.0) * hours
    annual_scale = 8760.0 / hours
    return {
        "test_hours": hours,
        "seasonal_naive_mae": round(seasonal_mae, 2),
        "best_model_mae": round(best_mae, 2),
        "mae_reduction_bikes_per_hour": round(seasonal_mae - best_mae, 2),
        "cumulative_error_bikes_avoided_test": round(bikes_saved, 0),
        "value_per_bike_error_usd": VALUE_PER_BIKE_ERROR_USD,
        "estimated_annual_value_usd": round(
            bikes_saved * VALUE_PER_BIKE_ERROR_USD * annual_scale, 2
        ),
    }


def main() -> dict:
    for d in (ASSETS_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = load_hourly(HOURLY_CSV)
    df = build_feature_frame(df)
    train, test = chronological_split(df)

    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET].to_numpy()
    x_test, y_test = test[FEATURE_COLUMNS], test[TARGET].to_numpy()

    results: dict[str, dict] = {}

    results["NaiveMean"] = regression_metrics(y_test, naive_mean_predict(train, test))
    results["SeasonalNaive"] = regression_metrics(y_test, seasonal_naive_predict(train, test))

    ols_pred, ols_coeffs, ols_r2 = fit_ols(train, test)
    results["LinearOLS"] = regression_metrics(y_test, ols_pred)

    fitted = {}
    predictions = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results[name] = regression_metrics(y_test, pred)
        fitted[name] = model
        predictions[name] = pred

    ml_names = [n for n in fitted]
    best_name = min(ml_names, key=lambda n: results[n]["rmse"])
    best_model = fitted[best_name]
    best_pred = predictions[best_name]

    impact = business_impact(test, results[best_name]["mae"], results["SeasonalNaive"]["mae"])

    perm = permutation_importance(
        best_model, x_test, y_test, n_repeats=5, random_state=0, n_jobs=-1
    )
    importances = perm.importances_mean

    save_eda(df, ASSETS_DIR / "eda_overview.png")
    save_model_comparison(results, ASSETS_DIR / "model_comparison.png")
    save_predictions(test, best_pred, ASSETS_DIR / "predictions.png")
    save_importance(list(FEATURE_COLUMNS), importances, ASSETS_DIR / "feature_importance.png")

    joblib.dump(
        {"model": best_model, "feature_columns": FEATURE_COLUMNS, "model_name": best_name},
        MODEL_PATH,
    )

    payload = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "target": TARGET,
        "mean_hourly_demand": round(float(df[TARGET].mean()), 2),
        "models": {n: {k: round(v, 4) for k, v in m.items()} for n, m in results.items()},
        "best_model": best_name,
        "ols_train_r2": round(float(ols_r2), 4),
        "top_features": [
            FEATURE_COLUMNS[i] for i in np.argsort(importances)[-8:][::-1]
        ],
        "business_impact": impact,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    ols_coeffs.to_csv(OUTPUTS_DIR / "ols_coefficients.csv", index=False)

    print(f"best model: {best_name}")
    for name, m in results.items():
        print(f"  {name:14s} rmse={m['rmse']:7.2f}  mae={m['mae']:6.2f}  r2={m['r2']:.3f}")
    print(f"estimated annual value: ${impact['estimated_annual_value_usd']:,.0f}")
    return payload


if __name__ == "__main__":
    main()
