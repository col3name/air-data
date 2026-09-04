from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from train1 import (
    TARGET,
    TIME_COL,
    build_features,
    create_model,
    get_departure_training_rows,
    rmse,
    train_model,
    _prepare_xy,
)

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

TRAIN_MONTH = 9
SEPTEMBER_FILE = "training_2025-09-01_2025-10-01.parquet"


def load_september_data() -> pd.DataFrame:
    file = DATA_DIR / SEPTEMBER_FILE
    if not file.exists():
        raise FileNotFoundError(f"Data file not found: {file}")
    print(f"Loading {file}")
    return pd.read_parquet(file)


def main():
    raw = load_september_data()

    print("\nBuilding features on ALL movements (arrivals + departures)...")
    all_movements = build_features(raw)
    print(f"All movement feature dataset: {all_movements.shape}")

    # NOTE: traffic/queue features for September are computed using ALL earlier
    # movements (including August), since the pipeline sorts globally first.
    # Strictly-earlier windows mean no future leakage.
    train_df = get_departure_training_rows(all_movements)
    print(f"Departure training dataset: {train_df.shape}")

    if "MONTH" not in train_df.columns:
        train_df["MONTH"] = train_df[TIME_COL].dt.month.astype("int16")

    monthly_counts = train_df["MONTH"].value_counts().sort_index()
    print("\nRows by month:")
    print(monthly_counts.to_string())

    train_df = train_df[train_df["MONTH"].eq(TRAIN_MONTH)].copy()
    print(
        f"\nTraining ONLY on month {TRAIN_MONTH}: {len(train_df):,} rows, "
        f"target mean={train_df[TARGET].mean():.2f}"
    )

    # Hold out last 15% of the month as a validation set.
    n = len(train_df)
    split = int(n * 0.85)
    fit_df = train_df.iloc[:split].reset_index(drop=True)
    valid_df = train_df.iloc[split:].reset_index(drop=True)
    print(f"Fit rows: {len(fit_df):,} | Valid rows: {len(valid_df):,}")

    model, score, features, best_iteration = train_model(fit_df, valid_df)

    csv_path = MODEL_DIR / "python2_1_cv.csv"
    pd.DataFrame(
        [{"month": TRAIN_MONTH, "rmse": score, "best_iteration": best_iteration}]
    ).to_csv(csv_path, index=False)
    print(f"Saved CV record: {csv_path}")

    print("\n" + "=" * 70)
    print(f"FINAL MODEL: MONTH {TRAIN_MONTH} ONLY")
    print("=" * 70)
    X, y, features, cat_features = _prepare_xy(train_df)
    final_model = create_model(iterations=best_iteration, use_early_stopping=False)
    final_model.fit(X, y, cat_features=cat_features)

    model_path = MODEL_DIR / "catboost_month9_final.cbm"
    final_model.save_model(str(model_path))

    metadata = {
        "features": features,
        "categorical_features": cat_features,
        "iterations": best_iteration,
        "target": TARGET,
        "train_monthes": [TRAIN_MONTH],
        "note": "Trained only on 9th month data.",
    }
    with open(MODEL_DIR / "features_month9.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    importance = pd.DataFrame({
        "feature": features,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(MODEL_DIR / "feature_importance_month9.csv", index=False)
    print("\nTop feature importance:")
    print(importance.head(30).to_string(index=False))
    print(f"\nValidation RMSE: {score:.4f}")
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()