from __future__ import annotations

import glob
import json
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

# Fast CV configuration. Final model is retrained on all 2025 rows using
# the median best_iteration discovered by CV.
CV_MONTHS = [9, 10, 11, 12]
CV_ITERATIONS = 800
CV_DEPTH = 7
CV_LEARNING_RATE = 0.05
CV_OD_WAIT = 100


def safe_str(s: pd.Series) -> pd.Series:
    return s.fillna("__MISSING__").astype(str).str.strip().str.upper()


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_training_data() -> pd.DataFrame:
    """Load ONLY the 2025 training parquet files.

    ranking.parquet and submitting.parquet are deliberately excluded.
    """
    files = sorted(glob.glob(str(DATA_DIR / TRAIN_GLOB)))
    if not files:
        raise FileNotFoundError(
            f"No training parquet files found matching {DATA_DIR / TRAIN_GLOB}"
        )

    frames = []
    for file in files:
        print(f"Loading {file}")
        frames.append(pd.read_parquet(file))

    data = pd.concat(frames, ignore_index=True)
    print(f"Total raw movement rows: {len(data):,}")
    return data


TIMESTAMP_COLUMNS = [
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
]

CATEGORICAL_COLUMNS = [
    "ADEP_mvt",
    "ADES_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "FLIGHT_RULE_mvt",
    "ADEP_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "CALLSIGN_flt",
    "PHASE_mvt",
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw movements WITHOUT filtering arrivals out."""
    df = df.copy()

    for col in TIMESTAMP_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = safe_str(df[col])

    df["AIRPORT"] = safe_str(df["ADEP_mvt"])
    df["DESTINATION"] = safe_str(df["ADES_mvt"])

    if "AIRCRAFT_TYPE_mvt" in df.columns:
        aircraft = df["AIRCRAFT_TYPE_mvt"].replace("__MISSING__", np.nan)
    else:
        aircraft = pd.Series(np.nan, index=df.index)
    if "AIRCRAFT_TYPE_flt" in df.columns:
        aircraft = aircraft.fillna(df["AIRCRAFT_TYPE_flt"])
    df["AIRCRAFT"] = safe_str(aircraft)

    if "AIRCRAFT_OPERATOR_flt" in df.columns:
        df["AIRLINE"] = safe_str(df["AIRCRAFT_OPERATOR_flt"])
    else:
        df["AIRLINE"] = "__MISSING__"

    phase = safe_str(df.get("PHASE_mvt", pd.Series("", index=df.index)))
    df["IS_DEPARTURE"] = (
        phase.str.contains("DEP", na=False) | phase.isin(["D", "DEPARTURE"])
    ).astype(np.int8)
    df["IS_ARRIVAL"] = (
        phase.str.contains("ARR", na=False) | phase.isin(["A", "ARRIVAL"])
    ).astype(np.int8)

    df = df[df[TIME_COL].notna()].copy()
    return df.sort_values(TIME_COL).reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df[TIME_COL]

    df["MONTH"] = t.dt.month.astype("int16")
    df["DAY"] = t.dt.day.astype("int16")
    df["DAY_OF_YEAR"] = t.dt.dayofyear.astype("int16")
    df["WEEKDAY"] = t.dt.dayofweek.astype("int8")
    df["HOUR"] = t.dt.hour.astype("int8")
    df["MINUTE"] = t.dt.minute.astype("int8")
    df["IS_WEEKEND"] = (df["WEEKDAY"] >= 5).astype("int8")

    df["TIME_15M"] = (df["HOUR"] * 4 + df["MINUTE"] // 15).astype(str)
    df["TIME_30M"] = (df["HOUR"] * 2 + df["MINUTE"] // 30).astype(str)
    df["TIME_60M"] = df["HOUR"].astype(str)

    def period(hour: int) -> str:
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


def minutes_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds() / 60.0


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "SCHED_TIME_UTC_mvt" in df.columns:
        df["MVT_MINUS_SCHED_MIN"] = minutes_diff(
            df[TIME_COL], df["SCHED_TIME_UTC_mvt"]
        )
    if "EOBT_1_flt" in df.columns:
        df["MVT_MINUS_EOBT_MIN"] = minutes_diff(df[TIME_COL], df["EOBT_1_flt"])
    if "AOBT_3_flt" in df.columns and "EOBT_1_flt" in df.columns:
        df["AOBT_MINUS_EOBT_MIN"] = minutes_diff(
            df["AOBT_3_flt"], df["EOBT_1_flt"]
        )
    if "IOBT_flt" in df.columns and "EOBT_1_flt" in df.columns:
        df["EOBT_MINUS_IOBT_MIN"] = minutes_diff(
            df["EOBT_1_flt"], df["IOBT_flt"]
        )
    return df


def _interaction(df: pd.DataFrame, *cols: str) -> pd.Series:
    result = safe_str(df[cols[0]])
    for col in cols[1:]:
        result = result + "__" + safe_str(df[col])
    return result


def add_categorical_interactions(df: pd.DataFrame) -> pd.DataFrame:
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
        if all(col in df.columns for col in cols):
            df[name] = _interaction(df, *cols)
    return df


def _rolling_sum_by_time(g: pd.DataFrame, value_col: str, window: str) -> np.ndarray:
    # closed='left' makes the current movement unavailable to itself.
    temp = g.set_index(TIME_COL)[value_col]
    return temp.rolling(window, closed="left").sum().fillna(0).to_numpy()


def add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute arrival/departure congestion using ALL movements."""
    df = df.copy()
    for minutes in (15, 30, 60):
        df[f"DEP_COUNT_{minutes}M"] = 0.0
        df[f"ARR_COUNT_{minutes}M"] = 0.0

    for _, g in df.sort_values(["AIRPORT", TIME_COL]).groupby("AIRPORT", sort=False):
        idx = g.index
        for minutes in (15, 30, 60):
            df.loc[idx, f"DEP_COUNT_{minutes}M"] = _rolling_sum_by_time(
                g, "IS_DEPARTURE", f"{minutes}min"
            )
            df.loc[idx, f"ARR_COUNT_{minutes}M"] = _rolling_sum_by_time(
                g, "IS_ARRIVAL", f"{minutes}min"
            )

    for minutes in (15, 30, 60):
        df[f"TOTAL_TRAFFIC_{minutes}M"] = (
            df[f"DEP_COUNT_{minutes}M"] + df[f"ARR_COUNT_{minutes}M"]
        )
    df["DEP_ARR_RATIO_30M"] = (df["DEP_COUNT_30M"] + 1.0) / (
        df["ARR_COUNT_30M"] + 1.0
    )
    return df


def add_runway_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for minutes in (15, 30):
        df[f"RUNWAY_LOAD_{minutes}M"] = 0.0

    if "RUNWAY_mvt" not in df.columns:
        return df

    valid = ~safe_str(df["RUNWAY_mvt"]).isin(["__MISSING__", "UNKNOWN", ""])
    work = df.loc[valid].sort_values(["AIRPORT", "RUNWAY_mvt", TIME_COL])

    for _, g in work.groupby(["AIRPORT", "RUNWAY_mvt"], sort=False):
        idx = g.index
        temp = pd.Series(1.0, index=pd.DatetimeIndex(g[TIME_COL]))
        for minutes in (15, 30):
            df.loc[idx, f"RUNWAY_LOAD_{minutes}M"] = (
                temp.rolling(f"{minutes}min", closed="left").sum().fillna(0).to_numpy()
            )
    return df


def add_queue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Historical taxi-out statistics, using strictly earlier known departures."""
    df = df.copy()
    cols = [
        "PREV_TAXI_MEAN_15M",
        "PREV_TAXI_MEAN_30M",
        "PREV_TAXI_MEAN_60M",
        "PREV_TAXI_MEDIAN_30M",
        "PREV_TAXI_P90_60M",
        "PREV_TAXI_COUNT_30M",
    ]
    for col in cols:
        df[col] = np.nan

    if TARGET not in df.columns:
        return df

    for _, g in df.sort_values(["AIRPORT", TIME_COL]).groupby("AIRPORT", sort=False):
        idx = g.index
        known_departure_target = pd.to_numeric(g[TARGET], errors="coerce").where(
            g["IS_DEPARTURE"].eq(1)
        )
        hist = pd.DataFrame(
            {"taxi": known_departure_target.to_numpy()},
            index=pd.DatetimeIndex(g[TIME_COL]),
        )

        # closed='left' is enough; no shift is required and avoids accidentally
        # excluding two rows. Only movements strictly before current timestamp count.
        taxi = hist["taxi"]
        df.loc[idx, "PREV_TAXI_MEAN_15M"] = taxi.rolling(
            "15min", closed="left"
        ).mean().to_numpy()
        df.loc[idx, "PREV_TAXI_MEAN_30M"] = taxi.rolling(
            "30min", closed="left"
        ).mean().to_numpy()
        df.loc[idx, "PREV_TAXI_MEAN_60M"] = taxi.rolling(
            "60min", closed="left"
        ).mean().to_numpy()
        df.loc[idx, "PREV_TAXI_MEDIAN_30M"] = taxi.rolling(
            "30min", closed="left"
        ).median().to_numpy()
        df.loc[idx, "PREV_TAXI_P90_60M"] = taxi.rolling(
            "60min", closed="left"
        ).quantile(0.90).to_numpy()
        df.loc[idx, "PREV_TAXI_COUNT_30M"] = taxi.rolling(
            "30min", closed="left"
        ).count().to_numpy()

    return df


def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    traffic = df["TOTAL_TRAFFIC_30M"]
    df["TRAFFIC_REGIME"] = pd.cut(
        traffic,
        [-np.inf, 5, 15, 30, 50, np.inf],
        labels=["QUIET", "NORMAL", "BUSY", "VERY_BUSY", "SATURATED"],
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
    queue_delta = df["PREV_TAXI_MEAN_15M"] - df["PREV_TAXI_MEAN_60M"]
    df["QUEUE_TREND"] = pd.cut(
        queue_delta,
        [-np.inf, -120, -30, 30, 120, np.inf],
        labels=["RECOVERING_FAST", "RECOVERING", "STABLE", "BUILDING", "BUILDING_FAST"],
    ).astype(str)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_data(df)
    df = add_time_features(df)
    df = add_schedule_features(df)
    df = add_categorical_interactions(df)
    df = add_traffic_features(df)
    df = add_runway_features(df)
    df = add_queue_features(df)
    df = add_regimes(df)
    return df


def get_departure_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["IS_DEPARTURE"].eq(1) & pd.to_numeric(df[TARGET], errors="coerce").notna()
    out = df.loc[mask].copy()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    # Remove only clearly invalid/extreme records. Increase upper bound if challenge docs
    # explicitly state legitimate values above two hours.
    out = out[out[TARGET].between(30, 7200)].copy()
    return out.sort_values(TIME_COL).reset_index(drop=True)


# Explicit feature list: keeps CatBoost memory usage under control and prevents
# accidental leakage from future/raw timestamp columns.
FEATURES = [
    "AIRPORT", "AIRLINE", "AIRCRAFT", "DESTINATION",
    "RUNWAY_mvt", "STAND_mvt", "FLIGHT_RULE_mvt",
    "MARKET_SEGMENT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt",
    "MONTH", "DAY_OF_YEAR", "WEEKDAY", "HOUR", "MINUTE", "IS_WEEKEND",
    "TIME_15M", "TIME_30M", "TRAFFIC_BANK",
    "MVT_MINUS_SCHED_MIN", "MVT_MINUS_EOBT_MIN", "AOBT_MINUS_EOBT_MIN", "EOBT_MINUS_IOBT_MIN",
    "AIRPORT_AIRLINE", "AIRPORT_AIRCRAFT", "AIRPORT_DESTINATION",
    "AIRPORT_RUNWAY", "AIRPORT_STAND", "AIRPORT_STAND_RUNWAY",
    "AIRPORT_TIME15", "AIRPORT_TIME30", "AIRPORT_WEEKDAY_TIME",
    "DEP_COUNT_15M", "DEP_COUNT_30M", "DEP_COUNT_60M",
    "ARR_COUNT_15M", "ARR_COUNT_30M", "ARR_COUNT_60M",
    "TOTAL_TRAFFIC_15M", "TOTAL_TRAFFIC_30M", "TOTAL_TRAFFIC_60M", "DEP_ARR_RATIO_30M",
    "RUNWAY_LOAD_15M", "RUNWAY_LOAD_30M",
    "PREV_TAXI_MEAN_15M", "PREV_TAXI_MEAN_30M", "PREV_TAXI_MEAN_60M",
    "PREV_TAXI_MEDIAN_30M", "PREV_TAXI_P90_60M", "PREV_TAXI_COUNT_30M",
    "TRAFFIC_REGIME", "RUNWAY_LOAD_REGIME", "QUEUE_REGIME", "QUEUE_TREND",
]


def get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURES if c in df.columns]


def get_cat_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [
        c for c in features
        if pd.api.types.is_string_dtype(df[c])
        or pd.api.types.is_object_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def temporal_split(df: pd.DataFrame, validation_month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    month = df[TIME_COL].dt.month
    return df[month < validation_month].copy(), df[month == validation_month].copy()


def create_model(iterations: int = CV_ITERATIONS, use_early_stopping: bool = True) -> CatBoostRegressor:
    params = dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=max(50, int(iterations)),
        learning_rate=CV_LEARNING_RATE,
        depth=CV_DEPTH,
        l2_leaf_reg=8,
        random_seed=RANDOM_SEED,
        random_strength=0.5,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )
    if use_early_stopping:
        params.update(od_type="Iter", od_wait=CV_OD_WAIT)
    return CatBoostRegressor(**params)


def _prepare_xy(train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None):
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

    return X_train, train_df[TARGET], X_valid, valid_df[TARGET], features, cat_features


def print_airport_rmse(valid_df: pd.DataFrame, pred: np.ndarray) -> None:
    report = valid_df[["AIRPORT", TARGET]].copy()
    report["PRED"] = pred
    rows = []
    for airport, g in report.groupby("AIRPORT"):
        rows.append((airport, len(g), rmse(g[TARGET], g["PRED"])))
    rows.sort(key=lambda x: x[2], reverse=True)
    print("\nRMSE by airport:")
    for airport, n, score in rows:
        print(f"  {airport:5s} rows={n:8,d} rmse={score:8.2f}")


def train_model(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    X_train, y_train, X_valid, y_valid, features, cat_features = _prepare_xy(train_df, valid_df)

    print(f"Train rows: {len(train_df):,} | Valid rows: {len(valid_df):,}")
    print(f"Features: {len(features)} | Categorical: {len(cat_features)}")

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

    return model, score, features, int(best_iteration) + 1


def run_cv(df: pd.DataFrame):
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

        model, score, _, best_iteration = train_model(train_df, valid_df)
        scores.append({"month": month, "rmse": score, "best_iteration": best_iteration})
        best_iterations.append(best_iteration)
        model.save_model(str(MODEL_DIR / f"catboost_month_{month}.cbm"))
        del model, train_df, valid_df

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(MODEL_DIR / "cv_scores.csv", index=False)
    print("\nCV RESULTS")
    print(scores_df.to_string(index=False))
    if not scores_df.empty:
        print(f"Mean RMSE: {scores_df['rmse'].mean():.4f}")

    return scores_df, best_iterations


def train_final_model(df: pd.DataFrame, best_iterations: list[int]):
    if best_iterations:
        final_iterations = int(np.median(best_iterations))
    else:
        final_iterations = CV_ITERATIONS
    final_iterations = max(200, final_iterations)

    X, y, features, cat_features = _prepare_xy(df)
    print("\n" + "=" * 70)
    print("FINAL MODEL: ALL 2025 DEPARTURES")
    print("=" * 70)
    print(f"Rows: {len(df):,} | Features: {len(features)}")
    print(f"Iterations from CV median: {final_iterations}")

    model = create_model(iterations=final_iterations, use_early_stopping=False)
    model.fit(X, y, cat_features=cat_features)

    model_path = MODEL_DIR / "catboost_final.cbm"
    model.save_model(str(model_path))

    metadata = {
        "features": features,
        "categorical_features": cat_features,
        "iterations": final_iterations,
        "target": TARGET,
    }
    with open(MODEL_DIR / "features.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    print("\nTop feature importance:")
    print(importance.head(30).to_string(index=False))
    print(f"\nSaved: {model_path}")
    return model, features


def main():
    raw = load_training_data()

    print("\nBuilding features on ALL movements (arrivals + departures)...")
    all_movements = build_features(raw)
    print(f"All movement feature dataset: {all_movements.shape}")

    train_df = get_departure_training_rows(all_movements)
    print(f"Departure training dataset: {train_df.shape}")
    print(
        f"Target mean={train_df[TARGET].mean():.2f}, "
        f"median={train_df[TARGET].median():.2f}"
    )

    # Free the large all-movements frame before CatBoost CV.
    del raw, all_movements

    scores_df, best_iterations = run_cv(train_df)
    train_final_model(train_df, best_iterations)

    print("\nTraining complete.")
    if not scores_df.empty:
        print(f"CV mean RMSE: {scores_df['rmse'].mean():.4f}")


if __name__ == "__main__":
    main()
