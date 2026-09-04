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
MODEL_DIR = Path("models_train2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_GLOB = "training_*.parquet"
TARGET = "TAXITIME_SEC_mvt"
TIME_COL = "MVT_TIME_UTC_mvt"
RANDOM_SEED = 42

# Stronger CV / CatBoost configuration.
CV_MONTHS = [9, 10, 11, 12]
CV_ITERATIONS = 2500
CV_DEPTH = 8
CV_LEARNING_RATE = 0.03
CV_OD_WAIT = 200

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


def safe_str(s: pd.Series) -> pd.Series:
    return s.fillna("__MISSING__").astype(str).str.strip().str.upper()


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_training_data() -> pd.DataFrame:
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

    df["MONTH"] = t.dt.month.astype("int16")
    df["DAY"] = t.dt.day.astype("int16")
    df["DAY_OF_YEAR"] = t.dt.dayofyear.astype("int16")
    df["WEEKDAY"] = t.dt.dayofweek.astype("int8")
    df["HOUR"] = t.dt.hour.astype("int8")
    df["MINUTE"] = t.dt.minute.astype("int8")
    df["IS_WEEKEND"] = (df["WEEKDAY"] >= 5).astype("int8")

    # Cyclic time representation.
    hour_float = df["HOUR"] + df["MINUTE"] / 60.0
    df["HOUR_SIN"] = np.sin(2 * np.pi * hour_float / 24.0)
    df["HOUR_COS"] = np.cos(2 * np.pi * hour_float / 24.0)

    df["DOY_SIN"] = np.sin(2 * np.pi * df["DAY_OF_YEAR"] / 365.25)
    df["DOY_COS"] = np.cos(2 * np.pi * df["DAY_OF_YEAR"] / 365.25)

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
        df["MVT_MINUS_EOBT_MIN"] = minutes_diff(
            df[TIME_COL], df["EOBT_1_flt"]
        )

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
        (("AIRPORT", "HOUR"), "AIRPORT_HOUR"),
        (("AIRPORT", "WEEKDAY", "HOUR"), "AIRPORT_WEEKDAY_HOUR"),
        (("AIRPORT", "RUNWAY_mvt", "HOUR"), "AIRPORT_RUNWAY_HOUR"),
        (("AIRPORT", "AIRLINE", "HOUR"), "AIRPORT_AIRLINE_HOUR"),
    ]

    for cols, name in specs:
        if all(col in df.columns for col in cols):
            df[name] = _interaction(df, *cols)

    return df


def _rolling_sum(g: pd.DataFrame, value_col: str, window: str) -> np.ndarray:
    temp = g.set_index(TIME_COL)[value_col]
    return (
        temp.rolling(window, closed="left")
        .sum()
        .fillna(0)
        .to_numpy()
    )


def _rolling_mean(g: pd.DataFrame, value_col: str, window: str) -> np.ndarray:
    temp = g.set_index(TIME_COL)[value_col]
    return temp.rolling(window, closed="left").mean().to_numpy()


def add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traffic features use only movements strictly before the current timestamp.
    Windows are intentionally short as well as long because airport congestion
    can change rapidly.
    """
    df = df.copy()
    windows = (5, 10, 15, 30, 60)

    for minutes in windows:
        df[f"DEP_COUNT_{minutes}M"] = 0.0
        df[f"ARR_COUNT_{minutes}M"] = 0.0

    for _, g in df.sort_values(["AIRPORT", TIME_COL]).groupby(
        "AIRPORT", sort=False
    ):
        idx = g.index

        for minutes in windows:
            window = f"{minutes}min"

            df.loc[idx, f"DEP_COUNT_{minutes}M"] = _rolling_sum(
                g, "IS_DEPARTURE", window
            )
            df.loc[idx, f"ARR_COUNT_{minutes}M"] = _rolling_sum(
                g, "IS_ARRIVAL", window
            )

    for minutes in windows:
        df[f"TOTAL_TRAFFIC_{minutes}M"] = (
            df[f"DEP_COUNT_{minutes}M"]
            + df[f"ARR_COUNT_{minutes}M"]
        )

    df["DEP_ARR_RATIO_5M"] = (
        df["DEP_COUNT_5M"] + 1.0
    ) / (df["ARR_COUNT_5M"] + 1.0)

    df["DEP_ARR_RATIO_10M"] = (
        df["DEP_COUNT_10M"] + 1.0
    ) / (df["ARR_COUNT_10M"] + 1.0)

    df["DEP_ARR_RATIO_30M"] = (
        df["DEP_COUNT_30M"] + 1.0
    ) / (df["ARR_COUNT_30M"] + 1.0)

    # Acceleration of traffic / queue build-up.
    df["TRAFFIC_ACCEL_10_30"] = (
        df["TOTAL_TRAFFIC_10M"] * 3.0
        - df["TOTAL_TRAFFIC_30M"]
    )

    df["DEP_ACCEL_10_30"] = (
        df["DEP_COUNT_10M"] * 3.0
        - df["DEP_COUNT_30M"]
    )

    return df


def add_runway_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runway-specific traffic, strictly historical.
    """
    df = df.copy()
    windows = (5, 10, 15, 30)

    for minutes in windows:
        for prefix in ("RUNWAY_LOAD", "RUNWAY_DEP", "RUNWAY_ARR"):
            df[f"{prefix}_{minutes}M"] = 0.0

    if "RUNWAY_mvt" not in df.columns:
        return df

    runway = safe_str(df["RUNWAY_mvt"])
    valid = ~runway.isin(["__MISSING__", "UNKNOWN", ""])

    work = df.loc[valid].sort_values(
        ["AIRPORT", "RUNWAY_mvt", TIME_COL]
    )

    for _, g in work.groupby(
        ["AIRPORT", "RUNWAY_mvt"], sort=False
    ):
        idx = g.index
        time_index = pd.DatetimeIndex(g[TIME_COL])

        dep = pd.Series(
            g["IS_DEPARTURE"].to_numpy(dtype=float),
            index=time_index,
        )
        arr = pd.Series(
            g["IS_ARRIVAL"].to_numpy(dtype=float),
            index=time_index,
        )
        total = pd.Series(
            np.ones(len(g), dtype=float),
            index=time_index,
        )

        for minutes in windows:
            window = f"{minutes}min"

            df.loc[idx, f"RUNWAY_LOAD_{minutes}M"] = (
                total.rolling(window, closed="left")
                .sum()
                .fillna(0)
                .to_numpy()
            )

            df.loc[idx, f"RUNWAY_DEP_{minutes}M"] = (
                dep.rolling(window, closed="left")
                .sum()
                .fillna(0)
                .to_numpy()
            )

            df.loc[idx, f"RUNWAY_ARR_{minutes}M"] = (
                arr.rolling(window, closed="left")
                .sum()
                .fillna(0)
                .to_numpy()
            )

    return df


def add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time since previous movement / departure / arrival at the airport and
    previous movement on the same runway.
    """
    df = df.copy()

    for col in (
        "MINUTES_SINCE_PREV_MOVEMENT",
        "MINUTES_SINCE_PREV_DEP",
        "MINUTES_SINCE_PREV_ARR",
        "MINUTES_SINCE_PREV_RUNWAY_MOVEMENT",
    ):
        df[col] = np.nan

    sorted_df = df.sort_values(["AIRPORT", TIME_COL])

    # Previous movement of any kind.
    prev = sorted_df.groupby("AIRPORT", sort=False)[TIME_COL].shift(1)
    df.loc[sorted_df.index, "MINUTES_SINCE_PREV_MOVEMENT"] = (
        (sorted_df[TIME_COL].to_numpy() - prev.to_numpy())
        / np.timedelta64(1, "m")
    )

    # Previous departure.
    dep_times = sorted_df[TIME_COL].where(
        sorted_df["IS_DEPARTURE"].eq(1)
    )
    prev_dep = dep_times.groupby(
        sorted_df["AIRPORT"], sort=False
    ).ffill().groupby(
        sorted_df["AIRPORT"], sort=False
    ).shift(1)

    df.loc[sorted_df.index, "MINUTES_SINCE_PREV_DEP"] = (
        (
            sorted_df[TIME_COL].to_numpy()
            - prev_dep.to_numpy()
        ) / np.timedelta64(1, "m")
    )

    # Previous arrival.
    arr_times = sorted_df[TIME_COL].where(
        sorted_df["IS_ARRIVAL"].eq(1)
    )
    prev_arr = arr_times.groupby(
        sorted_df["AIRPORT"], sort=False
    ).ffill().groupby(
        sorted_df["AIRPORT"], sort=False
    ).shift(1)

    df.loc[sorted_df.index, "MINUTES_SINCE_PREV_ARR"] = (
        (
            sorted_df[TIME_COL].to_numpy()
            - prev_arr.to_numpy()
        ) / np.timedelta64(1, "m")
    )

    # Previous movement on same runway.
    if "RUNWAY_mvt" in sorted_df.columns:
        valid_runway = ~safe_str(sorted_df["RUNWAY_mvt"]).isin(
            ["__MISSING__", "UNKNOWN", ""]
        )

        runway_times = sorted_df[TIME_COL].where(valid_runway)
        prev_runway = runway_times.groupby(
            [
                sorted_df["AIRPORT"],
                safe_str(sorted_df["RUNWAY_mvt"]),
            ],
            sort=False,
        ).shift(1)

        df.loc[sorted_df.index, "MINUTES_SINCE_PREV_RUNWAY_MOVEMENT"] = (
            (
                sorted_df[TIME_COL].to_numpy()
                - prev_runway.to_numpy()
            ) / np.timedelta64(1, "m")
        )

    return df


def add_queue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical taxi-out statistics from strictly earlier departures.
    """
    df = df.copy()

    for col in [
        "PREV_TAXI_MEAN_5M",
        "PREV_TAXI_MEAN_10M",
        "PREV_TAXI_MEAN_15M",
        "PREV_TAXI_MEAN_30M",
        "PREV_TAXI_MEAN_60M",
        "PREV_TAXI_MEDIAN_10M",
        "PREV_TAXI_MEDIAN_30M",
        "PREV_TAXI_P90_30M",
        "PREV_TAXI_P90_60M",
        "PREV_TAXI_STD_30M",
        "PREV_TAXI_COUNT_10M",
        "PREV_TAXI_COUNT_30M",
        "PREV_TAXI_COUNT_60M",
    ]:
        df[col] = np.nan

    if TARGET not in df.columns:
        return df

    for _, g in df.sort_values(
        ["AIRPORT", TIME_COL]
    ).groupby("AIRPORT", sort=False):
        idx = g.index

        known = pd.to_numeric(
            g[TARGET], errors="coerce"
        ).where(g["IS_DEPARTURE"].eq(1))

        hist = pd.Series(
            known.to_numpy(dtype=float),
            index=pd.DatetimeIndex(g[TIME_COL]),
            name="taxi",
        )

        for minutes in (5, 10, 15, 30, 60):
            df.loc[idx, f"PREV_TAXI_MEAN_{minutes}M"] = (
                hist.rolling(
                    f"{minutes}min", closed="left"
                ).mean().to_numpy()
            )

        for minutes in (10, 30):
            df.loc[idx, f"PREV_TAXI_MEDIAN_{minutes}M"] = (
                hist.rolling(
                    f"{minutes}min", closed="left"
                ).median().to_numpy()
            )

        for minutes in (30, 60):
            df.loc[idx, f"PREV_TAXI_P90_{minutes}M"] = (
                hist.rolling(
                    f"{minutes}min", closed="left"
                ).quantile(0.90).to_numpy()
            )

        df.loc[idx, "PREV_TAXI_STD_30M"] = (
            hist.rolling(
                "30min", closed="left"
            ).std().to_numpy()
        )

        for minutes in (10, 30, 60):
            df.loc[idx, f"PREV_TAXI_COUNT_{minutes}M"] = (
                hist.rolling(
                    f"{minutes}min", closed="left"
                ).count().to_numpy()
            )

    # Short-vs-long taxi trend.
    df["PREV_TAXI_TREND_15_60"] = (
        df["PREV_TAXI_MEAN_15M"]
        - df["PREV_TAXI_MEAN_60M"]
    )

    df["PREV_TAXI_TREND_10_30"] = (
        df["PREV_TAXI_MEAN_10M"]
        - df["PREV_TAXI_MEAN_30M"]
    )

    return df


def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    traffic = df["TOTAL_TRAFFIC_30M"]

    df["TRAFFIC_REGIME"] = pd.cut(
        traffic,
        [-np.inf, 5, 15, 30, 50, np.inf],
        labels=[
            "QUIET",
            "NORMAL",
            "BUSY",
            "VERY_BUSY",
            "SATURATED",
        ],
    ).astype(str)

    df["RUNWAY_LOAD_REGIME"] = pd.cut(
        df["RUNWAY_LOAD_15M"],
        [-np.inf, 2, 5, 8, np.inf],
        labels=["LOW", "NORMAL", "HIGH", "SATURATED"],
    ).astype(str)

    df["QUEUE_REGIME"] = pd.cut(
        df["PREV_TAXI_MEDIAN_30M"],
        [-np.inf, 300, 600, 900, 1200, np.inf],
        labels=[
            "FREE",
            "NORMAL",
            "BUILDING",
            "CONGESTED",
            "SEVERE",
        ],
    ).astype(str)

    queue_delta = (
        df["PREV_TAXI_MEAN_15M"]
        - df["PREV_TAXI_MEAN_60M"]
    )

    df["QUEUE_TREND"] = pd.cut(
        queue_delta,
        [-np.inf, -120, -30, 30, 120, np.inf],
        labels=[
            "RECOVERING_FAST",
            "RECOVERING",
            "STABLE",
            "BUILDING",
            "BUILDING_FAST",
        ],
    ).astype(str)

    # Extra traffic regime based on short-term acceleration.
    df["TRAFFIC_TREND_REGIME"] = pd.cut(
        df["TRAFFIC_ACCEL_10_30"],
        [-np.inf, -5, 0, 5, 15, np.inf],
        labels=[
            "FALLING",
            "STABLE",
            "RISING",
            "RISING_FAST",
            "SURGING",
        ],
    ).astype(str)

    return df


def add_numeric_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TRAFFIC_PER_MIN_10M"] = (
        df["TOTAL_TRAFFIC_10M"] / 10.0
    )

    df["DEP_PER_MIN_10M"] = (
        df["DEP_COUNT_10M"] / 10.0
    )

    df["RUNWAY_TRAFFIC_SHARE_15M"] = (
        df["RUNWAY_LOAD_15M"] + 1.0
    ) / (
        df["TOTAL_TRAFFIC_15M"] + 1.0
    )

    df["RUNWAY_DEP_SHARE_15M"] = (
        df["RUNWAY_DEP_15M"] + 1.0
    ) / (
        df["DEP_COUNT_15M"] + 1.0
    )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_data(df)
    df = add_time_features(df)
    df = add_schedule_features(df)
    df = add_categorical_interactions(df)
    df = add_traffic_features(df)
    df = add_runway_features(df)
    df = add_recency_features(df)
    df = add_queue_features(df)
    df = add_regimes(df)
    df = add_numeric_interactions(df)
    return df


def get_departure_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["IS_DEPARTURE"].eq(1)
        & pd.to_numeric(df[TARGET], errors="coerce").notna()
    )

    out = df.loc[mask].copy()
    out[TARGET] = pd.to_numeric(
        out[TARGET], errors="coerce"
    )

    # Keep the same conservative validity range as train1.
    out = out[out[TARGET].between(30, 7200)].copy()

    return out.sort_values(TIME_COL).reset_index(drop=True)


FEATURES = [
    "AIRPORT",
    "AIRLINE",
    "AIRCRAFT",
    "DESTINATION",
    "RUNWAY_mvt",
    "STAND_mvt",
    "FLIGHT_RULE_mvt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt",
    "WK_TBL_CAT_flt",

    "MONTH",
    "DAY_OF_YEAR",
    "WEEKDAY",
    "HOUR",
    "MINUTE",
    "IS_WEEKEND",
    "HOUR_SIN",
    "HOUR_COS",
    "DOY_SIN",
    "DOY_COS",

    "TIME_15M",
    "TIME_30M",
    "TIME_60M",
    "TRAFFIC_BANK",

    "MVT_MINUS_SCHED_MIN",
    "MVT_MINUS_EOBT_MIN",
    "AOBT_MINUS_EOBT_MIN",
    "EOBT_MINUS_IOBT_MIN",

    "AIRPORT_AIRLINE",
    "AIRPORT_AIRCRAFT",
    "AIRPORT_DESTINATION",
    "AIRPORT_RUNWAY",
    "AIRPORT_STAND",
    "AIRPORT_STAND_RUNWAY",
    "AIRPORT_TIME15",
    "AIRPORT_TIME30",
    "AIRPORT_WEEKDAY_TIME",
    "AIRPORT_HOUR",
    "AIRPORT_WEEKDAY_HOUR",
    "AIRPORT_RUNWAY_HOUR",
    "AIRPORT_AIRLINE_HOUR",

    "DEP_COUNT_5M",
    "DEP_COUNT_10M",
    "DEP_COUNT_15M",
    "DEP_COUNT_30M",
    "DEP_COUNT_60M",

    "ARR_COUNT_5M",
    "ARR_COUNT_10M",
    "ARR_COUNT_15M",
    "ARR_COUNT_30M",
    "ARR_COUNT_60M",

    "TOTAL_TRAFFIC_5M",
    "TOTAL_TRAFFIC_10M",
    "TOTAL_TRAFFIC_15M",
    "TOTAL_TRAFFIC_30M",
    "TOTAL_TRAFFIC_60M",

    "DEP_ARR_RATIO_5M",
    "DEP_ARR_RATIO_10M",
    "DEP_ARR_RATIO_30M",
    "TRAFFIC_ACCEL_10_30",
    "DEP_ACCEL_10_30",

    "RUNWAY_LOAD_5M",
    "RUNWAY_LOAD_10M",
    "RUNWAY_LOAD_15M",
    "RUNWAY_LOAD_30M",
    "RUNWAY_DEP_5M",
    "RUNWAY_DEP_10M",
    "RUNWAY_DEP_15M",
    "RUNWAY_DEP_30M",
    "RUNWAY_ARR_5M",
    "RUNWAY_ARR_10M",
    "RUNWAY_ARR_15M",
    "RUNWAY_ARR_30M",

    "MINUTES_SINCE_PREV_MOVEMENT",
    "MINUTES_SINCE_PREV_DEP",
    "MINUTES_SINCE_PREV_ARR",
    "MINUTES_SINCE_PREV_RUNWAY_MOVEMENT",

    "PREV_TAXI_MEAN_5M",
    "PREV_TAXI_MEAN_10M",
    "PREV_TAXI_MEAN_15M",
    "PREV_TAXI_MEAN_30M",
    "PREV_TAXI_MEAN_60M",
    "PREV_TAXI_MEDIAN_10M",
    "PREV_TAXI_MEDIAN_30M",
    "PREV_TAXI_P90_30M",
    "PREV_TAXI_P90_60M",
    "PREV_TAXI_STD_30M",
    "PREV_TAXI_COUNT_10M",
    "PREV_TAXI_COUNT_30M",
    "PREV_TAXI_COUNT_60M",
    "PREV_TAXI_TREND_15_60",
    "PREV_TAXI_TREND_10_30",

    "TRAFFIC_REGIME",
    "RUNWAY_LOAD_REGIME",
    "QUEUE_REGIME",
    "QUEUE_TREND",
    "TRAFFIC_TREND_REGIME",

    "TRAFFIC_PER_MIN_10M",
    "DEP_PER_MIN_10M",
    "RUNWAY_TRAFFIC_SHARE_15M",
    "RUNWAY_DEP_SHARE_15M",
]


def get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURES if c in df.columns]


def get_cat_features(
    df: pd.DataFrame,
    features: list[str],
) -> list[str]:
    return [
        c
        for c in features
        if pd.api.types.is_string_dtype(df[c])
        or pd.api.types.is_object_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]


def temporal_split(
    df: pd.DataFrame,
    validation_month: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    month = df[TIME_COL].dt.month
    return (
        df[month < validation_month].copy(),
        df[month == validation_month].copy(),
    )


def create_model(
    iterations: int = CV_ITERATIONS,
    use_early_stopping: bool = True,
) -> CatBoostRegressor:
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
        params.update(
            od_type="Iter",
            od_wait=CV_OD_WAIT,
        )

    return CatBoostRegressor(**params)


def _prepare_xy(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
):
    features = get_features(train_df)
    cat_features = get_cat_features(
        train_df, features
    )

    X_train = train_df[features].copy()

    for col in cat_features:
        X_train[col] = (
            X_train[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    if valid_df is None:
        return (
            X_train,
            train_df[TARGET],
            features,
            cat_features,
        )

    X_valid = valid_df[features].copy()

    for col in cat_features:
        X_valid[col] = (
            X_valid[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return (
        X_train,
        train_df[TARGET],
        X_valid,
        valid_df[TARGET],
        features,
        cat_features,
    )


def print_airport_rmse(
    valid_df: pd.DataFrame,
    pred: np.ndarray,
) -> None:
    report = valid_df[
        ["AIRPORT", TARGET]
    ].copy()

    report["PRED"] = pred
    rows = []

    for airport, g in report.groupby("AIRPORT"):
        rows.append(
            (
                airport,
                len(g),
                rmse(g[TARGET], g["PRED"]),
                float(
                    g["PRED"].mean()
                ),
                float(
                    g[TARGET].mean()
                ),
            )
        )

    rows.sort(key=lambda x: x[2], reverse=True)

    print("\nRMSE by airport:")
    for airport, n, score, pred_mean, target_mean in rows:
        print(
            f"  {airport:5s}"
            f" rows={n:8,d}"
            f" rmse={score:8.2f}"
            f" pred_mean={pred_mean:8.2f}"
            f" target_mean={target_mean:8.2f}"
        )


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
):
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        features,
        cat_features,
    ) = _prepare_xy(
        train_df,
        valid_df,
    )

    print(
        f"Train rows: {len(train_df):,}"
        f" | Valid rows: {len(valid_df):,}"
    )
    print(
        f"Features: {len(features)}"
        f" | Categorical: {len(cat_features)}"
    )

    model = create_model()

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    pred = np.maximum(
        model.predict(X_valid),
        0,
    )

    score = rmse(y_valid, pred)

    best_iteration = model.get_best_iteration()

    if best_iteration is None or best_iteration < 0:
        best_iteration = (
            model.tree_count_ - 1
        )

    print(
        f"Validation RMSE: {score:.4f}"
    )
    print(
        f"Best iteration: {best_iteration}"
    )

    print_airport_rmse(
        valid_df,
        pred,
    )

    return (
        model,
        score,
        features,
        int(best_iteration) + 1,
    )


def run_cv(df: pd.DataFrame):
    scores = []
    best_iterations = []

    for month in CV_MONTHS:
        print("\n" + "=" * 70)
        print(
            f"VALIDATION MONTH: {month}"
        )
        print("=" * 70)

        train_df, valid_df = temporal_split(
            df,
            month,
        )

        if train_df.empty or valid_df.empty:
            print("Skipping empty fold")
            continue

        (
            model,
            score,
            _,
            best_iteration,
        ) = train_model(
            train_df,
            valid_df,
        )

        scores.append(
            {
                "month": month,
                "rmse": score,
                "best_iteration": best_iteration,
            }
        )

        best_iterations.append(
            best_iteration
        )

        model.save_model(
            str(
                MODEL_DIR
                / f"catboost_month_{month}.cbm"
            )
        )

        del model
        del train_df
        del valid_df

    scores_df = pd.DataFrame(scores)

    scores_df.to_csv(
        MODEL_DIR / "cv_scores.csv",
        index=False,
    )

    print("\nCV RESULTS")
    print(
        scores_df.to_string(
            index=False
        )
    )

    if not scores_df.empty:
        print(
            f"Mean RMSE: "
            f"{scores_df['rmse'].mean():.4f}"
        )
        print(
            f"Median RMSE: "
            f"{scores_df['rmse'].median():.4f}"
        )

    return (
        scores_df,
        best_iterations,
    )


def train_final_model(
    df: pd.DataFrame,
    best_iterations: list[int],
):
    if best_iterations:
        final_iterations = int(
            np.median(best_iterations)
        )
    else:
        final_iterations = CV_ITERATIONS

    final_iterations = max(
        300,
        final_iterations,
    )

    (
        X,
        y,
        features,
        cat_features,
    ) = _prepare_xy(df)

    print("\n" + "=" * 70)
    print(
        "FINAL MODEL: ALL 2025 DEPARTURES"
    )
    print("=" * 70)

    print(
        f"Rows: {len(df):,}"
        f" | Features: {len(features)}"
    )

    print(
        f"Iterations from CV median:"
        f" {final_iterations}"
    )

    model = create_model(
        iterations=final_iterations,
        use_early_stopping=False,
    )

    model.fit(
        X,
        y,
        cat_features=cat_features,
    )

    model_path = (
        MODEL_DIR
        / "catboost_final.cbm"
    )

    model.save_model(
        str(model_path)
    )

    metadata = {
        "features": features,
        "categorical_features": cat_features,
        "iterations": final_iterations,
        "target": TARGET,
        "cv_months": CV_MONTHS,
        "cv_depth": CV_DEPTH,
        "cv_learning_rate": CV_LEARNING_RATE,
    }

    with open(
        MODEL_DIR / "features.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": (
                model.get_feature_importance()
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        MODEL_DIR
        / "feature_importance.csv",
        index=False,
    )

    print(
        "\nTop feature importance:"
    )
    print(
        importance.head(40)
        .to_string(index=False)
    )

    print(
        f"\nSaved: {model_path}"
    )

    return model, features


def main():
    raw = load_training_data()

    print(
        "\nBuilding enhanced features "
        "on ALL movements "
        "(arrivals + departures)..."
    )

    all_movements = build_features(raw)

    print(
        f"All movement feature dataset:"
        f" {all_movements.shape}"
    )

    train_df = get_departure_training_rows(
        all_movements
    )

    print(
        f"Departure training dataset:"
        f" {train_df.shape}"
    )

    print(
        f"Target mean="
        f"{train_df[TARGET].mean():.2f}, "
        f"median="
        f"{train_df[TARGET].median():.2f}"
    )

    del raw
    del all_movements

    scores_df, best_iterations = run_cv(
        train_df
    )

    train_final_model(
        train_df,
        best_iterations,
    )

    print(
        "\nTraining complete."
    )

    if not scores_df.empty:
        print(
            f"CV mean RMSE: "
            f"{scores_df['rmse'].mean():.4f}"
        )
        print(
            f"CV median RMSE: "
            f"{scores_df['rmse'].median():.4f}"
        )


if __name__ == "__main__":
    main()
