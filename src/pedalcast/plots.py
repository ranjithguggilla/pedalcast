"""Chart helpers. Everything writes a PNG so the README and dossier can embed
the exact figures produced by the last training run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEASON_LABELS = {1: "winter", 2: "spring", 3: "summer", 4: "fall"}


def save_eda(df: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    by_hour = df.groupby("hour")["count"].mean()
    axes[0, 0].plot(by_hour.index, by_hour.values, marker="o", color="#2b6cb0")
    axes[0, 0].set_title("average demand by hour")
    axes[0, 0].set_xlabel("hour of day")
    axes[0, 0].set_ylabel("bikes rented")

    by_season = df.groupby("season")["count"].mean()
    axes[0, 1].bar(
        [SEASON_LABELS[s] for s in by_season.index], by_season.values, color="#dd6b20"
    )
    axes[0, 1].set_title("average demand by season")
    axes[0, 1].set_ylabel("bikes rented")

    axes[1, 0].scatter(df["temp"], df["count"], s=4, alpha=0.15, color="#2f855a")
    axes[1, 0].set_title("demand vs temperature")
    axes[1, 0].set_xlabel("normalised temperature")
    axes[1, 0].set_ylabel("bikes rented")

    axes[1, 1].hist(df["count"], bins=50, color="#6b46c1")
    axes[1, 1].set_title("distribution of hourly demand")
    axes[1, 1].set_xlabel("bikes rented in an hour")
    axes[1, 1].set_ylabel("number of hours")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_model_comparison(results: dict, path: str | Path) -> None:
    names = list(results.keys())
    rmse = [results[n]["rmse"] for n in names]
    mae = [results[n]["mae"] for n in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, rmse, width, label="RMSE", color="#2b6cb0")
    ax.bar(x + width / 2, mae, width, label="MAE", color="#dd6b20")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("error (bikes per hour)")
    ax.set_title("model comparison on the held-out test window")
    ax.legend()
    for i, v in enumerate(rmse):
        ax.text(i - width / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_predictions(test: pd.DataFrame, y_pred: np.ndarray, path: str | Path) -> None:
    slice_len = min(240, len(test))
    y_true = test["count"].to_numpy()[:slice_len]
    y_hat = np.asarray(y_pred)[:slice_len]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(range(slice_len), y_true, label="actual", color="#1a202c", linewidth=1)
    axes[0].plot(range(slice_len), y_hat, label="predicted", color="#e53e3e", linewidth=1)
    axes[0].set_title("actual vs predicted (first 10 test days)")
    axes[0].set_xlabel("hour index in test window")
    axes[0].set_ylabel("bikes rented")
    axes[0].legend()

    residuals = np.asarray(y_pred) - test["count"].to_numpy()
    axes[1].hist(residuals, bins=50, color="#718096")
    axes[1].axvline(0, color="#e53e3e", linewidth=1)
    axes[1].set_title("residuals on the full test window")
    axes[1].set_xlabel("predicted minus actual")
    axes[1].set_ylabel("number of hours")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_importance(names: list[str], importances: np.ndarray, path: str | Path) -> None:
    order = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([names[i] for i in order], importances[order], color="#2b6cb0")
    ax.set_title("permutation importance (top 15)")
    ax.set_xlabel("increase in error when the feature is shuffled")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
