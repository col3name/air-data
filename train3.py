from __future__ import annotations

import glob
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_GLOB = "training_*.parquet"
TARGET = "TAXITIME_SEC_mvt"
TIME_COL = "MVT_TIME_UTC_mvt"
RANDOM_SEED = 42

# More robust than train1: several temporal folds, deeper trees, more iterations.
CV_MONTHS = [9, 10, 11, 12]
CV_ITERATIONS = 1400
CV_DEPTH = 8
CV_LEARNING_RATE = 0.035
CV_OD_WAIT = 150

TIMESTAMP_COLUMNS = [
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt",
    "AOBT_3_flt", "ARVT_3_flt",
]

CATEGORICAL_COLUMNS = [
    "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
    "STAND_mvt", "FLIGHT_RULE_mvt", "ADEP_flt", "ADES_flt",
    "ADES_FILED_flt", "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt", "CALLSIGN_flt", "PHASE_mvt",
]

DERIVED_CATEGORICAL_COLUMNS = [
    "AIRPORT", "AIRLINE", "AIRCRAFT", "DESTINATION",
    "TIME_15M", "TIME_30M", "TIME_60M", "TRAFFIC_BANK",
    "DELAY_REGIME", "AIRPORT_AIRLINE", "AIRPORT_AIRCRAFT",
    "AIRPORT_DESTINATION", "AIRPORT_RUNWAY", "AIRPORT_STAND",
    "AIRPORT_STAND_RUNWAY", "AIRPORT_TIME15", "AIRPORT_TIME30",
    "AIRPORT_WEEKDAY_TIME", "TRAFFIC_REGIME", "RUNWAY_LOAD_REGIME",
]

HISTORICAL_GROUPS = [
    ("AIRPORT", "HIST_AIRPORT_MEAN", 20.0),
    ("AIRPORT", "HIST_AIRPORT_MEDIAN_PROXY", 20.0),
    ("AIRPORT_AIRLINE", "HIST_AIRPORT_AIRLINE_MEAN", 10.0),
    ("AIRPORT_RUNWAY", "HIST_AIRPORT_RUNWAY_MEAN", 10.0),
    ("AIRPORT_TIME30", "HIST_AIRPORT_TIME30_MEAN", 10.0),
    ("AIRPORT_WEEKDAY_TIME", "HIST_AIRPORT_WEEKDAY_TIME_MEAN", 10.0),
]

FEATURES = [
    "AIRPORT", "AIRLINE", "AIRCRAFT", "DESTINATION",
    "RUNWAY_mvt", "STAND_mvt", "FLIGHT_RULE_mvt",
    "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "MONTH", "DAY_OF_YEAR", "WEEKDAY", "HOUR", "MINUTE",
    "IS_WEEKEND", "MINUTE_OF_DAY", "HOUR_SIN", "HOUR_COS",
    "WEEKDAY_SIN", "WEEKDAY_COS",
    "TIME_15M", "TIME_30M", "TIME_60M", "TRAFFIC_BANK",
    "MVT_MINUS_SCHED_MIN", "MVT_MINUS_EOBT_MIN",
    "AOBT_MINUS_EOBT_MIN", "EOBT_MINUS_IOBT_MIN",
    "SCHED_DELAY_ABS", "SCHED_DELAY_POS", "EOBT_DELAY_ABS",
    "EOBT_DELAY_POS", "DELAY_REGIME",
    "AIRPORT_AIRLINE", "AIRPORT_AIRCRAFT", "AIRPORT_DESTINATION",
    "AIRPORT_RUNWAY", "AIRPORT_STAND", "AIRPORT_STAND_RUNWAY",
    "AIRPORT_TIME15", "AIRPORT_TIME30", "AIRPORT_WEEKDAY_TIME",
    "DEP_COUNT_5M", "DEP_COUNT_10M", "DEP_COUNT_15M",
    "DEP_COUNT_30M", "DEP_COUNT_60M", "DEP_COUNT_90M", "DEP_COUNT_120M",
    "ARR_COUNT_5M", "ARR_COUNT_10M", "ARR_COUNT_15M",
    "ARR_COUNT_30M", "ARR_COUNT_60M", "ARR_COUNT_90M", "ARR_COUNT_120M",
    "TOTAL_TRAFFIC_5M", "TOTAL_TRAFFIC_15M", "TOTAL_TRAFFIC_30M",
    "TOTAL_TRAFFIC_60M", "TOTAL_TRAFFIC_120M",
    "DEP_RATE_15M", "DEP_RATE_60M", "TRAFFIC_ACCELERATION",
    "DEP_ARR_RATIO_15M", "DEP_ARR_RATIO_60M",
    "RUNWAY_LOAD_5M", "RUNWAY_LOAD_15M", "RUNWAY_LOAD_30M",
    "RUNWAY_LOAD_60M",
    "RUNWAY_DEP_15M", "RUNWAY_ARR_15M",
    "TRAFFIC_REGIME", "RUNWAY_LOAD_REGIME",
] + [x[1] for x in HISTORICAL_GROUPS]


def safe_str(s: pd.Series) -> pd.Series:
    return s.fillna("__MISSING__").astype(str).str.strip().str.upper()


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_training_data2() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_DIR / TRAIN_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files matching {DATA_DIR / TRAIN_GLOB}")
    frames = []
    for file in files:
        print(f"Loading {file}")
        frames.append(pd.read_parquet(file))
    data = pd.concat(frames, ignore_index=True)
    print(f"Total raw movement rows: {len(data):,}")
    return data

def load_training_data() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_DIR / TRAIN_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files matching {DATA_DIR / TRAIN_GLOB}")

    frames = []
    for file in files:
        print(f"Loading {file}")
        frames.append(pd.read_parquet(file))

    data = pd.concat(frames, ignore_index=True)
    print(f"Total raw movement rows: {len(data):,}")
    return data
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TIMESTAMP_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = safe_str(df[col])

    df["AIRPORT"] = safe_str(df["ADEP_mvt"])
    df["DESTINATION"] = safe_str(df["ADES_mvt"])

    aircraft = df.get(
        "AIRCRAFT_TYPE_mvt",
        pd.Series("__MISSING__", index=df.index),
    ).replace("__MISSING__", np.nan)
    if "AIRCRAFT_TYPE_flt" in df.columns:
        aircraft = aircraft.fillna(df["AIRCRAFT_TYPE_flt"])
    df["AIRCRAFT"] = safe_str(aircraft)

    df["AIRLINE"] = safe_str(
        df.get("AIRCRAFT_OPERATOR_flt", pd.Series("__MISSING__", index=df.index))
    )

    phase = safe_str(df.get("PHASE_mvt", pd.Series("", index=df.index)))
    df["IS_DEPARTURE"] = (
        phase.str.contains("DEP", na=False)
        | phase.isin(["D", "DEPARTURE"])
    ).astype(np.int8)
    df["IS_ARRIVAL"] = (
        phase.str.contains("ARR", na=False)
        | phase.isin(["A", "ARRIVAL"])
    ).astype(np.int8)

    df = df[df[TIME_COL].notna()].copy()
    return df.sort_values(TIME_COL).reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df[TIME_COL]
    df["MONTH"] = t.dt.month.astype(np.int8)
    df["DAY_OF_YEAR"] = t.dt.dayofyear.astype(np.int16)
    df["WEEKDAY"] = t.dt.dayofweek.astype(np.int8)
    df["HOUR"] = t.dt.hour.astype(np.int8)
    df["MINUTE"] = t.dt.minute.astype(np.int8)
    df["IS_WEEKEND"] = (df["WEEKDAY"] >= 5).astype(np.int8)

    minute_of_day = df["HOUR"] * 60 + df["MINUTE"]
    df["MINUTE_OF_DAY"] = minute_of_day.astype(np.int16)
    df["HOUR_SIN"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["HOUR_COS"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    df["WEEKDAY_SIN"] = np.sin(2 * np.pi * df["WEEKDAY"] / 7.0)
    df["WEEKDAY_COS"] = np.cos(2 * np.pi * df["WEEKDAY"] / 7.0)

    df["TIME_15M"] = (df["HOUR"] * 4 + df["MINUTE"] // 15).astype(str)
    df["TIME_30M"] = (df["HOUR"] * 2 + df["MINUTE"] // 30).astype(str)
    df["TIME_60M"] = df["HOUR"].astype(str)

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
    return df


def minutes_diff(a, b):
    return (a - b).dt.total_seconds() / 60.0


def add_schedule_features(df):
    df = df.copy()
    if "SCHED_TIME_UTC_mvt" in df.columns:
        d = minutes_diff(df[TIME_COL], df["SCHED_TIME_UTC_mvt"])
        df["MVT_MINUS_SCHED_MIN"] = d
        df["SCHED_DELAY_ABS"] = d.abs()
        df["SCHED_DELAY_POS"] = d.clip(lower=0)

    if "EOBT_1_flt" in df.columns:
        d = minutes_diff(df[TIME_COL], df["EOBT_1_flt"])
        df["MVT_MINUS_EOBT_MIN"] = d
        df["EOBT_DELAY_ABS"] = d.abs()
        df["EOBT_DELAY_POS"] = d.clip(lower=0)

    if "AOBT_3_flt" in df.columns and "EOBT_1_flt" in df.columns:
        df["AOBT_MINUS_EOBT_MIN"] = minutes_diff(
            df["AOBT_3_flt"], df["EOBT_1_flt"]
        )

    if "IOBT_flt" in df.columns and "EOBT_1_flt" in df.columns:
        df["EOBT_MINUS_IOBT_MIN"] = minutes_diff(
            df["EOBT_1_flt"], df["IOBT_flt"]
        )

    delay = df.get("MVT_MINUS_EOBT_MIN", pd.Series(np.nan, index=df.index))
    df["DELAY_REGIME"] = pd.cut(
        delay,
        [-np.inf, -30, -5, 5, 15, 30, 60, np.inf],
        labels=["VERY_EARLY", "EARLY", "ON_TIME", "SMALL_DELAY",
                "DELAY", "BIG_DELAY", "SEVERE_DELAY"],
    ).astype(str)
    return df


def _interaction(df, *cols):
    result = safe_str(df[cols[0]])
    for col in cols[1:]:
        result = result + "__" + safe_str(df[col])
    return result


def add_categorical_interactions(df):
    df = df.copy()
    specs = [
        (("AIRPORT", "AIRLINE"), "AIRPORT_AIRLINE"),
        (("AIRPORT", "AIRCRAFT"), "AIRPORT_AIRCRAFT"),
        (("AIRPORT", "DESTINATION"), "AIRPORT_DESTINATION"),
        (("AIRPORT", "RUNWAY_mvt"), "AIRPORT_RUNWAY"),
        (("AIRPORT", "STAND_mvt"), "AIRPORT_STAND"),
        (("AIRPORT", "STAND_mvt", "RUNWAY_mvt"), "AIRPORT_STAND_RUNWAY"),
        (("AIRPORT", "TIME_15M"), "AIRPORT_TIME15"),
        (("AIRPORT", "TIME_30M"), "AIRPORT_TIME30"),
        (("AIRPORT", "WEEKDAY", "TIME_30M"), "AIRPORT_WEEKDAY_TIME"),
    ]
    for cols, name in specs:
        if all(c in df.columns for c in cols):
            df[name] = _interaction(df, *cols)
    return df


def _rolling(g, value_col, window):
    s = g.set_index(TIME_COL)[value_col]
    return s.rolling(window, closed="left").sum().fillna(0).to_numpy()


def add_traffic_features(df):
    df = df.copy()
    windows = (5, 10, 15, 30, 60, 90, 120)
    for m in windows:
        df[f"DEP_COUNT_{m}M"] = 0.0
        df[f"ARR_COUNT_{m}M"] = 0.0

    work = df.sort_values(["AIRPORT", TIME_COL])
    for _, g in work.groupby("AIRPORT", sort=False):
        idx = g.index
        for m in windows:
            df.loc[idx, f"DEP_COUNT_{m}M"] = _rolling(g, "IS_DEPARTURE", f"{m}min")
            df.loc[idx, f"ARR_COUNT_{m}M"] = _rolling(g, "IS_ARRIVAL", f"{m}min")

    for m in windows:
        df[f"TOTAL_TRAFFIC_{m}M"] = (
            df[f"DEP_COUNT_{m}M"] + df[f"ARR_COUNT_{m}M"]
        )

    df["DEP_RATE_15M"] = df["DEP_COUNT_15M"] / 0.25
    df["DEP_RATE_60M"] = df["DEP_COUNT_60M"]
    df["TRAFFIC_ACCELERATION"] = (
        df["TOTAL_TRAFFIC_15M"] / 15.0
        - df["TOTAL_TRAFFIC_60M"] / 60.0
    )
    df["DEP_ARR_RATIO_15M"] = (
        df["DEP_COUNT_15M"] + 1.0
    ) / (df["ARR_COUNT_15M"] + 1.0)
    df["DEP_ARR_RATIO_60M"] = (
        df["DEP_COUNT_60M"] + 1.0
    ) / (df["ARR_COUNT_60M"] + 1.0)
    return df


def add_runway_features(df):
    df = df.copy()
    for m in (5, 15, 30, 60):
        df[f"RUNWAY_LOAD_{m}M"] = 0.0
        df[f"RUNWAY_DEP_{m}M"] = 0.0
        df[f"RUNWAY_ARR_{m}M"] = 0.0

    if "RUNWAY_mvt" not in df.columns:
        return df

    valid = ~safe_str(df["RUNWAY_mvt"]).isin(
        ["__MISSING__", "UNKNOWN", ""]
    )
    work = df.loc[valid].sort_values(["AIRPORT", "RUNWAY_mvt", TIME_COL])

    for _, g in work.groupby(["AIRPORT", "RUNWAY_mvt"], sort=False):
        idx = g.index
        for m in (5, 15, 30, 60):
            dt_idx = pd.DatetimeIndex(g[TIME_COL])
            dep = pd.Series(g["IS_DEPARTURE"].to_numpy(), index=dt_idx)
            arr = pd.Series(g["IS_ARRIVAL"].to_numpy(), index=dt_idx)
            df.loc[idx, f"RUNWAY_DEP_{m}M"] = dep.rolling(
                f"{m}min", closed="left"
            ).sum().fillna(0).to_numpy()
            df.loc[idx, f"RUNWAY_ARR_{m}M"] = arr.rolling(
                f"{m}min", closed="left"
            ).sum().fillna(0).to_numpy()
            df.loc[idx, f"RUNWAY_LOAD_{m}M"] = (
                df.loc[idx, f"RUNWAY_DEP_{m}M"].to_numpy()
                + df.loc[idx, f"RUNWAY_ARR_{m}M"].to_numpy()
            )
    return df


def add_regimes(df):
    df = df.copy()
    df["TRAFFIC_REGIME"] = pd.cut(
        df["TOTAL_TRAFFIC_30M"],
        [-np.inf, 5, 15, 30, 50, np.inf],
        labels=["QUIET", "NORMAL", "BUSY", "VERY_BUSY", "SATURATED"],
    ).astype(str)
    df["RUNWAY_LOAD_REGIME"] = pd.cut(
        df["RUNWAY_LOAD_15M"],
        [-np.inf, 2, 5, 8, np.inf],
        labels=["LOW", "NORMAL", "HIGH", "SATURATED"],
    ).astype(str)
    return df


def build_features(df):
    df = prepare_data(df)
    df = add_time_features(df)
    df = add_schedule_features(df)
    df = add_categorical_interactions(df)
    df = add_traffic_features(df)
    df = add_runway_features(df)
    df = add_regimes(df)
    return df


def get_departure_training_rows(df):
    mask = (
        df["IS_DEPARTURE"].eq(1)
        & pd.to_numeric(df[TARGET], errors="coerce").notna()
    )
    out = df.loc[mask].copy()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    return out[out[TARGET].between(30, 7200)].sort_values(TIME_COL).reset_index(drop=True)


def _target_stats(frame, key):
    y = pd.to_numeric(frame[TARGET], errors="coerce")
    tmp = pd.DataFrame({"key": frame[key].astype(str), "y": y})
    return tmp.groupby("key")["y"].agg(["sum", "count", "mean", "median"])


def add_historical_target_features(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame | None = None,
    train_mode: bool = True,
) -> pd.DataFrame:
    """
    Target statistics with no validation leakage.

    train_mode=True:
      creates leave-one-out means for training rows.

    train_mode=False:
      computes statistics only from train_df and applies them to apply_df.
    """
    keys = [
        ("AIRPORT", "HIST_AIRPORT_MEAN", 20.0),
        ("AIRPORT", "HIST_AIRPORT_MEDIAN_PROXY", 20.0),
        ("AIRPORT_AIRLINE", "HIST_AIRPORT_AIRLINE_MEAN", 10.0),
        ("AIRPORT_RUNWAY", "HIST_AIRPORT_RUNWAY_MEAN", 10.0),
        ("AIRPORT_TIME30", "HIST_AIRPORT_TIME30_MEAN", 10.0),
        ("AIRPORT_WEEKDAY_TIME", "HIST_AIRPORT_WEEKDAY_TIME_MEAN", 10.0),
    ]

    out = train_df.copy() if apply_df is None else apply_df.copy()
    global_mean = pd.to_numeric(train_df[TARGET], errors="coerce").mean()

    for key, out_col, smoothing in keys:
        stats = _target_stats(train_df, key)

        if train_mode:
            key_values = train_df[key].astype(str)
            sums = key_values.map(stats["sum"]).to_numpy(dtype=float)
            counts = key_values.map(stats["count"]).to_numpy(dtype=float)
            y = pd.to_numeric(train_df[TARGET], errors="coerce").to_numpy(dtype=float)
            loo_count = np.maximum(counts - 1.0, 0.0)
            loo_sum = sums - y
            values = np.where(
                loo_count > 0,
                (loo_sum + smoothing * global_mean) / (loo_count + smoothing),
                global_mean,
            )
            out[out_col] = values
        else:
            means = stats["mean"].fillna(global_mean)
            values = out[key].astype(str).map(means).fillna(global_mean)
            out[out_col] = values.astype(float)

    return out


def get_features(df):
    return [c for c in FEATURES if c in df.columns]


def get_cat_features(df, features):
    explicit = set(CATEGORICAL_COLUMNS) | set(DERIVED_CATEGORICAL_COLUMNS)
    return [c for c in features if c in explicit]


def prepare_xy(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
):
    features = get_features(train_df)
    cat_features = get_cat_features(train_df, features)

    X_train = train_df[features].copy()
    for col in cat_features:
        X_train[col] = X_train[col].fillna("__MISSING__").astype(str)

    if valid_df is None:
        return X_train, train_df[TARGET], features, cat_features

    X_valid = valid_df[features].copy()
    for col in cat_features:
        X_valid[col] = X_valid[col].fillna("__MISSING__").astype(str)

    return (
        X_train,
        train_df[TARGET],
        X_valid,
        valid_df[TARGET],
        features,
        cat_features,
    )


def temporal_split(df, validation_month):
    month = df[TIME_COL].dt.month
    return df[month < validation_month].copy(), df[month == validation_month].copy()


def enrich_fold(train_df, valid_df):
    train_enriched = add_historical_target_features(
        train_df, train_mode=True
    )
    valid_enriched = add_historical_target_features(
        train_df, apply_df=valid_df, train_mode=False
    )
    return train_enriched, valid_enriched


def create_model(iterations=CV_ITERATIONS, use_early_stopping=True):
    params = dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=max(100, int(iterations)),
        learning_rate=CV_LEARNING_RATE,
        depth=CV_DEPTH,
        l2_leaf_reg=10,
        random_seed=RANDOM_SEED,
        random_strength=0.35,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )
    if use_early_stopping:
        params.update(od_type="Iter", od_wait=CV_OD_WAIT)
    return CatBoostRegressor(**params)


def print_airport_rmse(valid_df, pred):
    report = valid_df[["AIRPORT", TARGET]].copy()
    report["PRED"] = pred
    rows = []
    for airport, g in report.groupby("AIRPORT"):
        rows.append((airport, len(g), rmse(g[TARGET], g["PRED"])))
    rows.sort(key=lambda x: x[2], reverse=True)
    print("\nRMSE by airport:")
    for airport, n, score in rows[:30]:
        print(f"  {airport:5s} rows={n:8,d} rmse={score:8.2f}")


def train_model(train_df, valid_df, month):
    train_df, valid_df = enrich_fold(train_df, valid_df)
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        features,
        cat_features,
    ) = prepare_xy(train_df, valid_df)

    print(f"Train rows: {len(train_df):,} | Valid rows: {len(valid_df):,}")
    print(f"Features: {len(features)} | Categorical: {len(cat_features)}")
    print(f"Categorical columns: {cat_features}")

    model = create_model()
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    pred = np.maximum(model.predict(X_valid), 0)
    score = rmse(y_valid, pred)
    best_iteration = model.get_best_iteration()
    if best_iteration is None or best_iteration < 0:
        best_iteration = model.tree_count_ - 1

    print(f"Validation RMSE: {score:.4f}")
    print(f"Best iteration: {best_iteration}")
    print_airport_rmse(valid_df, pred)

    model.save_model(str(MODEL_DIR / f"catboost_month_{month}.cbm"))
    return score, int(best_iteration) + 1


def run_cv(df):
    scores = []
    best_iterations = []

    for month in CV_MONTHS:
        print("\n" + "=" * 70)
        print(f"VALIDATION MONTH: {month}")
        print("=" * 70)

        train_df, valid_df = temporal_split(df, month)
        if train_df.empty or valid_df.empty:
            print("Skipping empty fold")
            continue

        score, best_iteration = train_model(train_df, valid_df, month)
        scores.append({
            "month": month,
            "rmse": score,
            "best_iteration": best_iteration,
        })
        best_iterations.append(best_iteration)

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(MODEL_DIR / "cv_scores_train3.csv", index=False)

    print("\nCV RESULTS")
    if not scores_df.empty:
        print(scores_df.to_string(index=False))
        print(f"Mean RMSE: {scores_df['rmse'].mean():.4f}")
        print(f"Median RMSE: {scores_df['rmse'].median():.4f}")

    return scores_df, best_iterations


def train_final_model(df, best_iterations):
    final_iterations = (
        int(np.median(best_iterations))
        if best_iterations else CV_ITERATIONS
    )
    final_iterations = max(300, final_iterations)

    enriched = add_historical_target_features(df, train_mode=True)
    X, y, features, cat_features = prepare_xy(enriched)

    print("\n" + "=" * 70)
    print("FINAL MODEL: ALL 2025 DEPARTURES")
    print("=" * 70)
    print(f"Rows: {len(enriched):,} | Features: {len(features)}")
    print(f"Iterations from CV median: {final_iterations}")
    print(f"Categorical: {len(cat_features)}")

    model = create_model(
        iterations=final_iterations,
        use_early_stopping=False,
    )
    model.fit(X, y, cat_features=cat_features)

    model_path = MODEL_DIR / "catboost_train3_final.cbm"
    model.save_model(str(model_path))

    metadata = {
        "features": features,
        "categorical_features": cat_features,
        "iterations": final_iterations,
        "target": TARGET,
        "cv_months": CV_MONTHS,
        "version": "train3",
    }
    with open(MODEL_DIR / "train3_features.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save training target-stat maps for the later submission/prediction script.
    hist = {}
    for key, _, _ in HISTORICAL_GROUPS:
        stats = _target_stats(df, key)
        hist[key] = stats[["mean", "median", "count"]].to_dict(orient="index")

    with open(MODEL_DIR / "train3_historical_stats.pkl", "wb") as f:
        pickle.dump(
            {
                "global_mean": float(pd.to_numeric(df[TARGET], errors="coerce").mean()),
                "stats": hist,
            },
            f,
        )

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(
        MODEL_DIR / "train3_feature_importance.csv",
        index=False,
    )

    print("\nTop feature importance:")
    print(importance.head(35).to_string(index=False))
    print(f"\nSaved: {model_path}")
    return model, features


def main():
    raw = load_training_data()

    print("\nBuilding leakage-safe features on ALL movements...")
    all_movements = build_features(raw)
    print(f"All movement feature dataset: {all_movements.shape}")

    train_df = get_departure_training_rows(all_movements)
    print(f"Departure training dataset: {train_df.shape}")
    print(
        f"Target mean={train_df[TARGET].mean():.2f}, "
        f"median={train_df[TARGET].median():.2f}"
    )

    del raw, all_movements

    scores_df, best_iterations = run_cv(train_df)
    train_final_model(train_df, best_iterations)

    print("\nTrain3 training complete.")
    if not scores_df.empty:
        print(f"CV mean RMSE: {scores_df['rmse'].mean():.4f}")


if __name__ == "__main__":
    main()
