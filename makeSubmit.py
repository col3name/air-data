from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train1 import (
    TARGET,
    build_features,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "catboost_final.cbm"

# Ranking / test movement data
RANKING_FILE = DATA_DIR / "ranking.parquet"

# Official competition template
SUBMISSION_TEMPLATE = DATA_DIR / "submitting.parquet"

# Final competition file
SUBMISSION_OUTPUT = Path(
    "kind-earthquake_v1.parquet"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model() -> CatBoostRegressor:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n  {MODEL_PATH}\n\n"
            "Run train1.py first."
        )

    print("=" * 70)
    print("LOADING FINAL MODEL")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")

    model = CatBoostRegressor()

    model.load_model(
        str(MODEL_PATH)
    )

    print(
        f"Model loaded successfully."
    )

    return model


# ============================================================
# LOAD RANKING DATA
# ============================================================

def load_ranking_data() -> pd.DataFrame:

    if not RANKING_FILE.exists():
        raise FileNotFoundError(
            f"\nRanking file not found:\n"
            f"  {RANKING_FILE}\n\n"
            "Put the ranking/test parquet into "
            "the data/ directory."
        )

    print("\n" + "=" * 70)
    print("LOADING RANKING DATA")
    print("=" * 70)

    print(
        f"File: {RANKING_FILE}"
    )

    df = pd.read_parquet(
        RANKING_FILE
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    required = {
        "MVT_ID_mvt",
        "PHASE_mvt",
    }

    missing = (
            required -
            set(df.columns)
    )

    if missing:
        raise ValueError(
            "Ranking file is missing columns:\n"
            + "\n".join(
                f"  - {x}"
                for x in sorted(missing)
            )
        )

    return df


# ============================================================
# LOAD SUBMISSION TEMPLATE
# ============================================================

def load_submission_template() -> pd.DataFrame:

    if not SUBMISSION_TEMPLATE.exists():
        raise FileNotFoundError(
            f"\nSubmission template not found:\n"
            f"  {SUBMISSION_TEMPLATE}"
        )

    print("\n" + "=" * 70)
    print("LOADING SUBMISSION TEMPLATE")
    print("=" * 70)

    template = pd.read_parquet(
        SUBMISSION_TEMPLATE
    )

    print(
        f"Template rows: "
        f"{len(template):,}"
    )

    print(
        f"Template columns: "
        f"{list(template.columns)}"
    )

    required = {
        "MVT_ID_mvt",
        TARGET,
    }

    missing = (
            required -
            set(template.columns)
    )

    if missing:
        raise ValueError(
            "Submission template is missing:\n"
            + "\n".join(
                f"  - {x}"
                for x in sorted(missing)
            )
        )

    if template[
        "MVT_ID_mvt"
    ].duplicated().any():

        raise ValueError(
            "Submission template contains "
            "duplicate MVT_ID_mvt."
        )

    return template


# ============================================================
# BUILD FEATURES
# ============================================================

def build_prediction_features(
        ranking_df: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("BUILDING PREDICTION FEATURES")
    print("=" * 70)

    df = ranking_df.copy()

    # build_features() expects target column.
    # Target is hidden for ranking/test data.
    if TARGET not in df.columns:
        df[TARGET] = np.nan

    print(
        f"Input movements: "
        f"{len(df):,}"
    )

    features_df = build_features(
        df
    )

    print(
        f"Feature dataset: "
        f"{features_df.shape}"
    )

    return features_df


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features(
        model: CatBoostRegressor,
) -> tuple[list[str], list[str]]:

    features = model.feature_names_

    if not features:
        raise ValueError(
            "Model does not contain feature names."
        )

    # CatBoost stores categorical feature indexes.
    cat_indexes = (
        model.get_cat_feature_indices()
    )

    cat_features = [
        features[i]
        for i in cat_indexes
    ]

    print("\n" + "=" * 70)
    print("MODEL FEATURES")
    print("=" * 70)

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Categorical: "
        f"{len(cat_features)}"
    )

    return features, cat_features


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
        model: CatBoostRegressor,
        ranking_features: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # Select departures
    # --------------------------------------------------------

    if "IS_DEPARTURE" not in ranking_features.columns:
        raise ValueError(
            "IS_DEPARTURE was not created "
            "by build_features()."
        )

    departures = ranking_features[
        ranking_features["IS_DEPARTURE"].eq(1)
    ].copy()

    print(
        f"Departures: "
        f"{len(departures):,}"
    )

    if departures.empty:
        raise ValueError(
            "No departures found."
        )

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    if "MVT_ID_mvt" not in departures.columns:
        raise ValueError(
            "MVT_ID_mvt missing."
        )

    if departures[
        "MVT_ID_mvt"
    ].isna().any():

        raise ValueError(
            "Ranking data contains "
            "NaN MVT_ID_mvt."
        )

    if departures[
        "MVT_ID_mvt"
    ].duplicated().any():

        raise ValueError(
            "Duplicate MVT_ID_mvt "
            "in ranking departures."
        )

    # --------------------------------------------------------
    # Get features directly from model
    # --------------------------------------------------------

    features, cat_features = (
        get_model_features(model)
    )

    missing_features = [
        feature
        for feature in features
        if feature not in departures.columns
    ]

    if missing_features:
        raise ValueError(
            "Ranking data is missing model "
            "features:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing_features
            )
        )

    X = departures[
        features
    ].copy()

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    for col in cat_features:

        X[col] = (
            X[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(X)

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    # Taxi-out cannot be negative.
    predictions = np.maximum(
        predictions,
        0.0,
    )

    if not np.isfinite(
            predictions
    ).all():

        raise ValueError(
            "Model produced NaN "
            "or infinite predictions."
        )

    print(
        f"Prediction min: "
        f"{predictions.min():.2f}"
    )

    print(
        f"Prediction max: "
        f"{predictions.max():.2f}"
    )

    print(
        f"Prediction mean: "
        f"{predictions.mean():.2f}"
    )

    print(
        f"Prediction median: "
        f"{np.median(predictions):.2f}"
    )

    return pd.DataFrame(
        {
            "MVT_ID_mvt": departures[
                "MVT_ID_mvt"
            ].values,
            TARGET: predictions,
        }
    )


# ============================================================
# BUILD SUBMISSION
# ============================================================

def make_submission(
        predictions: pd.DataFrame,
        template: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("BUILDING FINAL SUBMISSION")
    print("=" * 70)

    print(
        f"Template rows: "
        f"{len(template):,}"
    )

    print(
        f"Prediction rows: "
        f"{len(predictions):,}"
    )

    # --------------------------------------------------------
    # Compare IDs
    # --------------------------------------------------------

    template_ids = template[
        "MVT_ID_mvt"
    ]

    prediction_ids = predictions[
        "MVT_ID_mvt"
    ]

    missing_ids = template_ids[
        ~template_ids.isin(
            prediction_ids
        )
    ]

    extra_ids = prediction_ids[
        ~prediction_ids.isin(
            template_ids
        )
    ]

    if len(missing_ids):

        print("\nMissing predictions:")
        print(
            missing_ids
            .head(20)
            .to_list()
        )

        raise ValueError(
            f"{len(missing_ids):,} template IDs "
            "do not have predictions."
        )

    if len(extra_ids):

        print("\nExtra predictions:")
        print(
            extra_ids
            .head(20)
            .to_list()
        )

        raise ValueError(
            f"{len(extra_ids):,} predictions "
            "are not present in template."
        )

    # --------------------------------------------------------
    # Map predictions
    # --------------------------------------------------------

    prediction_map = (
        predictions
        .set_index("MVT_ID_mvt")[TARGET]
    )

    result = template[
        ["MVT_ID_mvt"]
    ].copy()

    result[TARGET] = (
        result[
            "MVT_ID_mvt"
        ].map(prediction_map)
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if result[TARGET].isna().any():

        raise ValueError(
            "Final submission contains NaN."
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
            "Duplicate IDs in submission."
        )

    # EXACTLY two columns
    result = result[
        [
            "MVT_ID_mvt",
            TARGET,
        ]
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_parquet(
        SUBMISSION_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL SUBMISSION CREATED")
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

    print("\nFirst 10 rows:")
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
    print("MAKE SUBMISSION")
    print("=" * 70)

    print(
        f"\nOutput file: "
        f"{SUBMISSION_OUTPUT}"
    )

    # 1. Load trained model
    model = load_model()

    # 2. Load ranking/test movements
    ranking_df = load_ranking_data()

    # 3. Build exactly the same features
    ranking_features = (
        build_prediction_features(
            ranking_df
        )
    )

    # 4. Generate taxi-out predictions
    predictions = generate_predictions(
        model=model,
        ranking_features=ranking_features,
    )

    # 5. Load official submission template
    template = load_submission_template()

    # 6. Merge predictions into template
    submission = make_submission(
        predictions=predictions,
        template=template,
    )

    print("\n" + "=" * 70)
    print("READY FOR UPLOAD")
    print("=" * 70)

    print(
        f"\n{SUBMISSION_OUTPUT}"
    )

    print(
        "\nUpload this file to:"
    )

    print(
        "prc-2026-kind-earthquake"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
