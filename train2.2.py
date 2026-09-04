from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from train1 import (
    TARGET,
    TIME_COL,
    build_features,
    create_model,
    get_departure_training_rows,
    train_model,
    _prepare_xy,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# September 2025 training data
TRAIN_FILE = DATA_DIR / "training_2025-09-01_2025-10-01.parquet"

# Ranking / test movement dataset
RANKING_FILE = DATA_DIR / "ranking.parquet"

# Official submission template
SUBMISSION_TEMPLATE = DATA_DIR / "submitting.parquet"

# ============================================================
# IMPORTANT:
# Competition requires:
#
# kind-earthquake_v1.parquet
#
# Change VERSION to 2, 3, ... for later submissions.
# ============================================================

TEAM_NAME = "kind-earthquake"
VERSION = 1

SUBMISSION_OUTPUT = Path(
    f"{TEAM_NAME}_v{VERSION}.parquet"
)

TRAIN_MONTH = 9

# Last 15% of September = validation
VALIDATION_FRACTION = 0.15


# ============================================================
# DATA LOADING
# ============================================================

def load_september_data() -> pd.DataFrame:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"\nTraining file not found:\n"
            f"  {TRAIN_FILE}\n"
        )

    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)
    print(f"File: {TRAIN_FILE}")

    df = pd.read_parquet(TRAIN_FILE)

    print(f"Raw rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def load_ranking_data() -> pd.DataFrame:
    if not RANKING_FILE.exists():
        raise FileNotFoundError(
            "\nRanking dataset was not found.\n"
            f"Expected:\n"
            f"  {RANKING_FILE}\n\n"
            "Put the ranking movement parquet into data/ "
            "or change RANKING_FILE in train2.2.py."
        )

    print("\n" + "=" * 70)
    print("LOADING RANKING DATA")
    print("=" * 70)
    print(f"File: {RANKING_FILE}")

    df = pd.read_parquet(RANKING_FILE)

    print(f"Ranking raw rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def load_submission_template() -> pd.DataFrame:
    if not SUBMISSION_TEMPLATE.exists():
        raise FileNotFoundError(
            "\nSubmission template was not found.\n"
            f"Expected:\n"
            f"  {SUBMISSION_TEMPLATE}"
        )

    print("\n" + "=" * 70)
    print("LOADING SUBMISSION TEMPLATE")
    print("=" * 70)

    df = pd.read_parquet(SUBMISSION_TEMPLATE)

    print(f"Submission rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    required = {
        "MVT_ID_mvt",
        TARGET,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Submission template is missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_training_data(df: pd.DataFrame) -> None:
    required = {
        TIME_COL,
        TARGET,
        "PHASE_mvt",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Training data is missing columns: "
            f"{sorted(missing)}"
        )


def validate_ranking_data(df: pd.DataFrame) -> None:
    required = {
        TIME_COL,
        "MVT_ID_mvt",
        "PHASE_mvt",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Ranking data is missing columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# TRAINING
# ============================================================

def train_september_model():
    raw = load_september_data()

    validate_training_data(raw)

    print("\n" + "=" * 70)
    print("BUILDING TRAINING FEATURES")
    print("=" * 70)

    # IMPORTANT:
    # Build features on ALL movements:
    # arrivals + departures.
    all_movements = build_features(raw)

    print(
        f"All movement feature dataset: "
        f"{all_movements.shape}"
    )

    # Keep departures with known taxi-out target.
    train_df = get_departure_training_rows(
        all_movements
    )

    print(
        f"Departure training dataset: "
        f"{train_df.shape}"
    )

    if train_df.empty:
        raise ValueError(
            "No valid departure training rows found."
        )

    # Make sure month exists.
    if "MONTH" not in train_df.columns:
        train_df["MONTH"] = (
            train_df[TIME_COL]
            .dt.month
            .astype("int16")
        )

    print("\nRows by month:")
    print(
        train_df["MONTH"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ========================================================
    # September only
    # ========================================================

    train_df = train_df[
        train_df["MONTH"].eq(TRAIN_MONTH)
    ].copy()

    train_df = train_df.sort_values(
        TIME_COL
    ).reset_index(drop=True)

    if train_df.empty:
        raise ValueError(
            f"No training rows for month {TRAIN_MONTH}."
        )

    print("\n" + "=" * 70)
    print("SEPTEMBER TRAINING SET")
    print("=" * 70)

    print(f"Rows: {len(train_df):,}")
    print(
        f"Target mean:   "
        f"{train_df[TARGET].mean():.2f}"
    )
    print(
        f"Target median: "
        f"{train_df[TARGET].median():.2f}"
    )
    print(
        f"Target std:    "
        f"{train_df[TARGET].std():.2f}"
    )

    # ========================================================
    # Temporal validation
    # ========================================================

    n = len(train_df)

    split = int(
        n * (1.0 - VALIDATION_FRACTION)
    )

    fit_df = (
        train_df
        .iloc[:split]
        .reset_index(drop=True)
    )

    valid_df = (
        train_df
        .iloc[split:]
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("TEMPORAL VALIDATION")
    print("=" * 70)

    print(
        f"Fit rows:   {len(fit_df):,}"
    )
    print(
        f"Valid rows: {len(valid_df):,}"
    )

    if len(valid_df):
        print(
            f"Validation start: "
            f"{valid_df[TIME_COL].min()}"
        )

        print(
            f"Validation end:   "
            f"{valid_df[TIME_COL].max()}"
        )

    # ========================================================
    # CatBoost CV / early stopping
    # ========================================================

    model, score, features, best_iteration = train_model(
        fit_df,
        valid_df,
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    print(f"RMSE: {score:.4f}")
    print(
        f"Best iterations: "
        f"{best_iteration}"
    )

    # ========================================================
    # Save CV result
    # ========================================================

    cv_path = MODEL_DIR / "train2_2_cv.csv"

    pd.DataFrame(
        [
            {
                "month": TRAIN_MONTH,
                "rmse": score,
                "best_iteration": best_iteration,
                "fit_rows": len(fit_df),
                "valid_rows": len(valid_df),
            }
        ]
    ).to_csv(
        cv_path,
        index=False,
    )

    print(
        f"Saved CV record: {cv_path}"
    )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING FINAL SEPTEMBER MODEL")
    print("=" * 70)

    X, y, features, cat_features = _prepare_xy(
        train_df
    )

    print(
        f"Training rows: {len(X):,}"
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Categorical features: "
        f"{len(cat_features)}"
    )

    print(
        f"Iterations: {best_iteration}"
    )

    final_model = create_model(
        iterations=best_iteration,
        use_early_stopping=False,
    )

    final_model.fit(
        X,
        y,
        cat_features=cat_features,
    )

    # ========================================================
    # Save model
    # ========================================================

    model_path = (
            MODEL_DIR /
            "catboost_month9_final.cbm"
    )

    final_model.save_model(
        str(model_path)
    )

    print(
        f"\nSaved model: {model_path}"
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "team": TEAM_NAME,
        "submission_version": VERSION,
        "features": features,
        "categorical_features": cat_features,
        "iterations": int(best_iteration),
        "target": TARGET,
        "train_months": [TRAIN_MONTH],
        "validation_rmse": float(score),
        "train_rows": int(len(train_df)),
        "note": (
            "Final model trained only on "
            "September 2025 departures."
        ),
    }

    metadata_path = (
            MODEL_DIR /
            "features_month9.json"
    )

    with open(
            metadata_path,
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved metadata: {metadata_path}"
    )

    # ========================================================
    # Feature importance
    # ========================================================

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": (
                final_model
                .get_feature_importance()
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_path = (
            MODEL_DIR /
            "feature_importance_month9.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print(
        f"Saved feature importance: "
        f"{importance_path}"
    )

    print("\nTop 30 features:")
    print(
        importance
        .head(30)
        .to_string(index=False)
    )

    return (
        final_model,
        features,
        cat_features,
        train_df,
        all_movements,
        score,
        best_iteration,
    )


# ============================================================
# PREDICTION FEATURES
# ============================================================

def prepare_prediction_features(
        ranking_df: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("BUILDING RANKING FEATURES")
    print("=" * 70)

    ranking_df = ranking_df.copy()

    # build_features() expects TARGET to exist.
    # Ranking target is hidden, therefore NaN.
    if TARGET not in ranking_df.columns:
        ranking_df[TARGET] = np.nan

    print(
        f"Ranking movements: "
        f"{len(ranking_df):,}"
    )

    # IMPORTANT:
    # Use exactly the same feature builder
    # as during training.
    featured = build_features(
        ranking_df
    )

    print(
        f"Ranking feature dataset: "
        f"{featured.shape}"
    )

    return featured


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
        model,
        features: list[str],
        cat_features: list[str],
        ranking_df: pd.DataFrame,
) -> pd.DataFrame:

    ranking_features = (
        prepare_prediction_features(
            ranking_df
        )
    )

    # ========================================================
    # Select departures
    # ========================================================

    if "IS_DEPARTURE" not in ranking_features.columns:
        raise ValueError(
            "IS_DEPARTURE was not created "
            "in ranking dataset."
        )

    departures = ranking_features[
        ranking_features["IS_DEPARTURE"].eq(1)
    ].copy()

    print(
        f"Ranking departures: "
        f"{len(departures):,}"
    )

    if departures.empty:
        raise ValueError(
            "No departure movements found "
            "in ranking dataset."
        )

    # ========================================================
    # Check IDs
    # ========================================================

    if "MVT_ID_mvt" not in departures.columns:
        raise ValueError(
            "MVT_ID_mvt is missing from "
            "ranking dataset."
        )

    if departures["MVT_ID_mvt"].isna().any():
        raise ValueError(
            "Ranking dataset contains "
            "NaN MVT_ID_mvt values."
        )

    if departures["MVT_ID_mvt"].duplicated().any():
        duplicated = departures[
            departures["MVT_ID_mvt"]
            .duplicated(keep=False)
        ]

        raise ValueError(
            "Duplicate MVT_ID_mvt found "
            "in ranking departures.\n"
            f"{duplicated.head(20)}"
        )

    # ========================================================
    # Check model features
    # ========================================================

    missing_features = [
        f
        for f in features
        if f not in departures.columns
    ]

    if missing_features:
        raise ValueError(
            "Ranking dataset is missing "
            "model features:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing_features
            )
        )

    # ========================================================
    # Prepare X
    # ========================================================

    X_pred = departures[
        features
    ].copy()

    for col in cat_features:
        X_pred[col] = (
            X_pred[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    # ========================================================
    # Predict
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    predictions = model.predict(
        X_pred
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    # Taxi-out cannot be negative.
    predictions = np.maximum(
        predictions,
        0.0,
    )

    # Check finite values.
    if not np.isfinite(predictions).all():
        raise ValueError(
            "Model produced NaN or infinite predictions."
        )

    print(
        f"Prediction min:    "
        f"{predictions.min():.2f}"
    )

    print(
        f"Prediction max:    "
        f"{predictions.max():.2f}"
    )

    print(
        f"Prediction mean:   "
        f"{predictions.mean():.2f}"
    )

    print(
        f"Prediction median: "
        f"{np.median(predictions):.2f}"
    )

    predictions_df = pd.DataFrame(
        {
            "MVT_ID_mvt": departures[
                "MVT_ID_mvt"
            ].values,
            TARGET: predictions,
        }
    )

    return predictions_df


# ============================================================
# BUILD FINAL SUBMISSION
# ============================================================

def build_final_submission(
        predictions_df: pd.DataFrame,
) -> pd.DataFrame:

    template = load_submission_template()

    print("\n" + "=" * 70)
    print("MERGING PREDICTIONS INTO TEMPLATE")
    print("=" * 70)

    template_ids = template[
        "MVT_ID_mvt"
    ]

    prediction_ids = predictions_df[
        "MVT_ID_mvt"
    ]

    print(
        f"Template IDs:   {len(template_ids):,}"
    )

    print(
        f"Prediction IDs: {len(prediction_ids):,}"
    )

    # ========================================================
    # Check ID sets
    # ========================================================

    missing_predictions = template_ids[
        ~template_ids.isin(prediction_ids)
    ]

    extra_predictions = prediction_ids[
        ~prediction_ids.isin(template_ids)
    ]

    if len(missing_predictions):
        print("\n" + "!" * 70)
        print(
            f"ERROR: {len(missing_predictions):,} "
            "submission IDs have no prediction."
        )

        print("\nFirst missing IDs:")
        print(
            missing_predictions
            .head(20)
            .to_list()
        )

        raise ValueError(
            "Not all submission IDs have predictions. "
            "Submission was NOT created."
        )

    if len(extra_predictions):
        print("\n" + "!" * 70)
        print(
            f"ERROR: {len(extra_predictions):,} "
            "predictions are not present "
            "in submitting.parquet."
        )

        print("\nFirst extra IDs:")
        print(
            extra_predictions
            .head(20)
            .to_list()
        )

        raise ValueError(
            "Ranking contains IDs not present "
            "in submission template."
        )

    # ========================================================
    # Check exact ID count
    # ========================================================

    if len(template_ids) != len(prediction_ids):
        raise ValueError(
            "Template and prediction row counts differ."
        )

    # ========================================================
    # Map predictions to template
    # ========================================================

    pred_map = (
        predictions_df
        .set_index("MVT_ID_mvt")[TARGET]
    )

    result = template[
        ["MVT_ID_mvt"]
    ].copy()

    result[TARGET] = (
        result["MVT_ID_mvt"]
        .map(pred_map)
    )

    # ========================================================
    # Final checks
    # ========================================================

    if result[TARGET].isna().any():
        missing_count = int(
            result[TARGET].isna().sum()
        )

        raise ValueError(
            f"Final submission contains "
            f"{missing_count:,} NaN predictions."
        )

    result[TARGET] = pd.to_numeric(
        result[TARGET],
        errors="coerce",
    )

    result[TARGET] = np.maximum(
        result[TARGET],
        0.0,
    )

    if not np.isfinite(
            result[TARGET].to_numpy()
    ).all():

        raise ValueError(
            "Final submission contains "
            "non-finite predictions."
        )

    if len(result) != len(template):
        raise ValueError(
            "Submission row count changed."
        )

    if result[
        "MVT_ID_mvt"
    ].duplicated().any():

        raise ValueError(
            "Duplicate MVT_ID_mvt "
            "in final submission."
        )

    # ========================================================
    # EXACT submission columns
    # ========================================================

    result = result[
        [
            "MVT_ID_mvt",
            TARGET,
        ]
    ].copy()

    # ========================================================
    # Save parquet
    # ========================================================

    result.to_parquet(
        SUBMISSION_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SUBMISSION CREATED")
    print("=" * 70)

    print(
        f"File: {SUBMISSION_OUTPUT}"
    )

    print(
        f"Rows: {len(result):,}"
    )

    print(
        f"Columns: {list(result.columns)}"
    )

    print(
        f"Size: "
        f"{SUBMISSION_OUTPUT.stat().st_size / 1024:.1f} KB"
    )

    print(
        f"NaN: "
        f"{result[TARGET].isna().sum():,}"
    )

    print(
        f"Min: "
        f"{result[TARGET].min():.2f}"
    )

    print(
        f"Max: "
        f"{result[TARGET].max():.2f}"
    )

    print(
        f"Mean: "
        f"{result[TARGET].mean():.2f}"
    )

    print("\nFirst rows:")
    print(
        result
        .head(10)
        .to_string(index=False)
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PRC DATA CHALLENGE 2026")
    print("train2.2 — September Taxi-Out Model")
    print("=" * 70)

    print(
        f"\nTeam: {TEAM_NAME}"
    )

    print(
        f"Submission version: v{VERSION}"
    )

    print(
        f"Output: {SUBMISSION_OUTPUT}"
    )

    # ========================================================
    # 1. TRAIN
    # ========================================================

    (
        model,
        features,
        cat_features,
        train_df,
        _all_movements,
        validation_rmse,
        best_iteration,
    ) = train_september_model()

    # ========================================================
    # 2. LOAD RANKING DATA
    # ========================================================

    ranking_df = load_ranking_data()

    validate_ranking_data(
        ranking_df
    )

    # ========================================================
    # 3. GENERATE PREDICTIONS
    # ========================================================

    predictions_df = generate_predictions(
        model=model,
        features=features,
        cat_features=cat_features,
        ranking_df=ranking_df,
    )

    # ========================================================
    # 4. BUILD FINAL SUBMISSION
    # ========================================================

    submission = build_final_submission(
        predictions_df
    )

    # ========================================================
    # 5. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Validation RMSE: "
        f"{validation_rmse:.4f}"
    )

    print(
        f"Best iteration: "
        f"{best_iteration}"
    )

    print(
        f"Submission rows: "
        f"{len(submission):,}"
    )

    print(
        f"Output: "
        f"{SUBMISSION_OUTPUT}"
    )

    print("\nREADY FOR UPLOAD:")
    print(
        f"  {SUBMISSION_OUTPUT}"
    )

    print("\nSubmission format:")
    print(
        "  MVT_ID_mvt"
    )
    print(
        f"  {TARGET}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
