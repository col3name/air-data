from __future__ import annotations
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
TARGET = "TAXITIME_SEC_mvt"


def safe_str(s):
    return s.fillna("__MISSING__").astype(str)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_training_data():
    files = sorted(glob.glob(str(TRAIN_DIR / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {TRAIN_DIR}")
    frames = []
    for file in files:
        print(f"Loading {file}")
        df = pd.read_parquet(file)
        if TARGET in df.columns:
            df = df[df[TARGET].notna()].copy()
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"Total rows: {len(data):,}")
    return data


def prepare_data(df):
    df = df.copy()

    time_columns = [
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
        "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
    ]
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    categorical_columns = [
        "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt",
        "RUNWAY_mvt", "STAND_mvt", "FLIGHT_RULE_mvt",
        "ADEP_flt", "ADES_flt", "ADES_FILED_flt",
        "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
        "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt",
        "AIRCRAFT_OPERATOR_flt", "CALLSIGN_flt",
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = safe_str(df[col])

    df["AIRPORT"] = safe_str(df["ADEP_mvt"])
    if "PHASE_mvt" in df.columns:
        phase = df["PHASE_mvt"].astype(str).str.upper()
        df = df[phase.isin(["DEPARTURE", "DEP", "D"])].copy()

    df["AIRLINE"] = safe_str(
        df["AIRCRAFT_OPERATOR_flt"]
        if "AIRCRAFT_OPERATOR_flt" in df.columns
        else pd.Series("__UNKNOWN__", index=df.index)
    )
    df["AIRCRAFT"] = safe_str(df["AIRCRAFT_TYPE_mvt"])
    df["DESTINATION"] = safe_str(df["ADES_mvt"])
    return df


def add_time_features(df):
    df = df.copy()
    t = df["MVT_TIME_UTC_mvt"]

    df["YEAR"] = t.dt.year
    df["MONTH"] = t.dt.month
    df["DAY"] = t.dt.day
    df["DAY_OF_YEAR"] = t.dt.dayofyear
    df["WEEKDAY"] = t.dt.dayofweek
    df["HOUR"] = t.dt.hour
    df["MINUTE"] = t.dt.minute

    df["TIME_15M"] = (t.dt.hour * 4 + t.dt.minute // 15).astype(str)
    df["TIME_30M"] = (t.dt.hour * 2 + t.dt.minute // 30).astype(str)
    df["TIME_60M"] = t.dt.hour.astype(str)

    df["WEEKDAY_TIME"] = (
        df["WEEKDAY"].astype(str) + "_" + df["TIME_30M"]
    )
    df["AIRPORT_TIME"] = df["AIRPORT"] + "_" + df["TIME_30M"]
    df["AIRPORT_WEEKDAY"] = (
        df["AIRPORT"] + "_" + df["WEEKDAY"].astype(str)
    )

    def period(hour):
        if 5 <= hour < 8:
            return "EARLY_MORNING"
        if 8 <= hour < 11:
            return "MORNING_BANK"
        if 11 <= hour < 14:
            return "MIDDAY"
        if 14 <= hour < 18:
            return "AFTERNOON"
        if 18 <= hour < 22:
            return "EVENING_BANK"
        return "NIGHT"

    df["TRAFFIC_BANK"] = df["HOUR"].map(period)
    df["IS_WEEKEND"] = (df["WEEKDAY"] >= 5).astype(str)
    return df


def add_categorical_interactions(df):
    df = df.copy()

    def interaction(*cols):
        result = safe_str(df[cols[0]])
        for col in cols[1:]:
            result = result + "__" + safe_str(df[col])
        return result

    for cols, name in [
        (("AIRPORT", "AIRLINE"), "AIRPORT_AIRLINE"),
        (("AIRPORT", "AIRCRAFT"), "AIRPORT_AIRCRAFT"),
        (("AIRPORT", "DESTINATION"), "AIRPORT_DESTINATION"),
        (("AIRPORT", "RUNWAY_mvt"), "AIRPORT_RUNWAY"),
        (("AIRPORT", "STAND_mvt"), "AIRPORT_STAND"),
        (("AIRLINE", "DESTINATION"), "AIRLINE_DESTINATION"),
        (("AIRLINE", "AIRCRAFT"), "AIRLINE_AIRCRAFT"),
        (("AIRPORT", "AIRLINE", "AIRCRAFT"), "AIRPORT_AIRLINE_AIRCRAFT"),
        (("AIRPORT", "WEEKDAY", "TIME_30M"), "AIRPORT_WEEKDAY_TIME"),
    ]:
        if all(c in df.columns for c in cols):
            df[name] = interaction(*cols)

    return df


def add_traffic_features(df):
    df = df.sort_values(["AIRPORT", "MVT_TIME_UTC_mvt"])
    result = []

    for _, g in df.groupby("AIRPORT", sort=False):
        g = g.copy().set_index("MVT_TIME_UTC_mvt")
        dep = pd.Series(1, index=g.index)

        g["DEP_COUNT_15M"] = dep.rolling(
            "15min", closed="left"
        ).sum().fillna(0)
        g["DEP_COUNT_30M"] = dep.rolling(
            "30min", closed="left"
        ).sum().fillna(0)
        g["DEP_COUNT_60M"] = dep.rolling(
            "60min", closed="left"
        ).sum().fillna(0)

        if "PHASE_mvt" in g.columns:
            phase = g["PHASE_mvt"].astype(str).str.upper()
            arr = phase.isin(["ARRIVAL", "ARR", "A"]).astype(int)
            arr.index = g.index

            g["ARR_COUNT_15M"] = arr.rolling(
                "15min", closed="left"
            ).sum().fillna(0)
            g["ARR_COUNT_30M"] = arr.rolling(
                "30min", closed="left"
            ).sum().fillna(0)
            g["ARR_COUNT_60M"] = arr.rolling(
                "60min", closed="left"
            ).sum().fillna(0)
        else:
            g["ARR_COUNT_15M"] = 0
            g["ARR_COUNT_30M"] = 0
            g["ARR_COUNT_60M"] = 0

        result.append(g.reset_index())
    return pd.concat(result, ignore_index=True)


def add_runway_features(df):
    if "RUNWAY_mvt" not in df.columns:
        df["RUNWAY_LOAD_15M"] = 0
        df["RUNWAY_LOAD_30M"] = 0
        return df

    df = df.sort_values(
        ["AIRPORT", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]
    )
    result = []

    for _, g in df.groupby(
        ["AIRPORT", "RUNWAY_mvt"], sort=False
    ):
        g = g.copy().set_index("MVT_TIME_UTC_mvt")
        activity = pd.Series(1, index=g.index)

        g["RUNWAY_LOAD_15M"] = activity.rolling(
            "15min", closed="left"
        ).sum().fillna(0)
        g["RUNWAY_LOAD_30M"] = activity.rolling(
            "30min", closed="left"
        ).sum().fillna(0)

        result.append(g.reset_index())
    return pd.concat(result, ignore_index=True)


def add_queue_features(df):
    for col in [
        "PREV_TAXI_MEAN_15M",
        "PREV_TAXI_MEAN_30M",
        "PREV_TAXI_MEDIAN_30M",
        "PREV_TAXI_P90_60M",
        "PREV_TAXI_COUNT_30M",
    ]:
        df[col] = np.nan

    if TARGET not in df.columns:
        return df

    df = df.sort_values(["AIRPORT", "MVT_TIME_UTC_mvt"])
    result = []

    for _, g in df.groupby("AIRPORT", sort=False):
        g = g.copy().set_index("MVT_TIME_UTC_mvt")
        taxi = pd.to_numeric(g[TARGET], errors="coerce")
        taxi_previous = taxi.shift(1)

        g["PREV_TAXI_MEAN_15M"] = taxi_previous.rolling(
            "15min", closed="left"
        ).mean()
        g["PREV_TAXI_MEAN_30M"] = taxi_previous.rolling(
            "30min", closed="left"
        ).mean()
        g["PREV_TAXI_MEDIAN_30M"] = taxi_previous.rolling(
            "30min", closed="left"
        ).median()
        g["PREV_TAXI_P90_60M"] = taxi_previous.rolling(
            "60min", closed="left"
        ).quantile(0.90)
        g["PREV_TAXI_COUNT_30M"] = taxi_previous.rolling(
            "30min", closed="left"
        ).count()

        result.append(g.reset_index())
    return pd.concat(result, ignore_index=True)


def add_regimes(df):
    df = df.copy()

    traffic = df["DEP_COUNT_30M"] + df["ARR_COUNT_30M"]
    df["TRAFFIC_REGIME"] = pd.cut(
        traffic,
        [-np.inf, 5, 15, 30, 50, np.inf],
        labels=["QUIET", "NORMAL", "BUSY", "VERY_BUSY", "SATURATED"],
    ).astype(str)

    df["DEP_DEMAND_REGIME"] = pd.cut(
        df["DEP_COUNT_30M"],
        [-np.inf, 3, 8, 15, 25, np.inf],
        labels=["LOW", "NORMAL", "HIGH", "VERY_HIGH", "EXTREME"],
    ).astype(str)

    df["ARR_DEMAND_REGIME"] = pd.cut(
        df["ARR_COUNT_30M"],
        [-np.inf, 3, 8, 15, 25, np.inf],
        labels=["LOW", "NORMAL", "HIGH", "VERY_HIGH", "EXTREME"],
    ).astype(str)

    df["RUNWAY_LOAD_REGIME"] = pd.cut(
        df["RUNWAY_LOAD_15M"],
        [-np.inf, 2, 5, 8, np.inf],
        labels=["LOW", "NORMAL", "HIGH", "SATURATED"],
    ).astype(str)

    df["QUEUE_REGIME"] = pd.cut(
        df["PREV_TAXI_MEDIAN_30M"],
        [-np.inf, 300, 600, 900, 1200, np.inf],
        labels=["FREE", "NORMAL", "BUILDING", "CONGESTED", "SEVERE"],
    ).astype(str)

    diff = (
        df["PREV_TAXI_MEAN_15M"]
        - df["PREV_TAXI_MEAN_30M"]
    )
    df["QUEUE_TREND"] = pd.cut(
        diff,
        [-np.inf, -120, -30, 30, 120, np.inf],
        labels=[
            "RECOVERING_FAST", "RECOVERING", "STABLE",
            "BUILDING", "BUILDING_FAST"
        ],
    ).astype(str)

    return df


def build_features(df):
    df = prepare_data(df)
    df = add_time_features(df)
    df = add_categorical_interactions(df)
    df = add_traffic_features(df)
    df = add_runway_features(df)
    df = add_queue_features(df)
    df = add_regimes(df)
    return df


DROP_COLUMNS = {
    TARGET,
    "MVT_ID_mvt", "FLIGHT_ID_mvt",
    "FLIGHT_mvt", "CALLSIGN_flt",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt",
    "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
}


def get_features(df):
    return [c for c in df.columns if c not in DROP_COLUMNS]


def get_cat_features(df, features):
    return [
        c for c in features
        if df[c].dtype == "object"
        or str(df[c].dtype).startswith("category")
        or df[c].dtype == "str"
        or str(df[c].dtype).startswith("str")
    ]


def temporal_split(df, validation_month):
    month = df["MVT_TIME_UTC_mvt"].dt.month
    return (
        df[month < validation_month].copy(),
        df[month == validation_month].copy(),
    )


def train_model(train_df, valid_df):
    features = get_features(train_df)
    cat_features = get_cat_features(train_df, features)

    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()

    for col in cat_features:
        X_train[col] = X_train[col].fillna("__MISSING__").astype(str)
        X_valid[col] = X_valid[col].fillna("__MISSING__").astype(str)

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=3000,
        learning_rate=0.035,
        depth=8,
        l2_leaf_reg=5,
        random_seed=42,
        random_strength=1,
        bagging_temperature=1,
        od_type="Iter",
        od_wait=150,
        verbose=100,
    )

    model.fit(
        X_train,
        train_df[TARGET],
        cat_features=cat_features,
        eval_set=(X_valid, valid_df[TARGET]),
        use_best_model=True,
    )

    pred = np.maximum(model.predict(X_valid), 0)
    score = rmse(valid_df[TARGET], pred)
    print(f"Validation RMSE: {score:.4f}")

    return model, score, features


def run_cv(df):
    scores = []
    models = []

    for month in [9, 10, 11, 12]:
        print("\n" + "=" * 70)
        print(f"VALIDATION MONTH: {month}")
        print("=" * 70)

        train_df, valid_df = temporal_split(df, month)
        if train_df.empty or valid_df.empty:
            continue

        model, score, features = train_model(train_df, valid_df)
        scores.append({"month": month, "rmse": score})
        models.append(model)

        model.save_model(
            MODEL_DIR / f"catboost_month_{month}.cbm"
        )

    scores_df = pd.DataFrame(scores)
    print("\nCV RESULTS")
    print(scores_df)
    if not scores_df.empty:
        print("Mean RMSE:", scores_df["rmse"].mean())
    return models, scores_df


def train_final_model(df):
    train_df = df[df["MVT_TIME_UTC_mvt"].dt.month <= 11].copy()
    valid_df = df[df["MVT_TIME_UTC_mvt"].dt.month == 12].copy()

    model, score, features = train_model(train_df, valid_df)

    model.save_model(MODEL_DIR / "catboost_final.cbm")
    pd.Series(features).to_json(MODEL_DIR / "features.json")

    print("Final model saved.")
    return model, features


if __name__ == "__main__":
    df = load_training_data()
    df = build_features(df)

    print("Feature dataset:", df.shape)

    run_cv(df)
    train_final_model(df)
