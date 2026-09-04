import pandas as pd
import numpy as np

from train4 import add_time_features, add_leakage_safe_decay_features


def test_time_features():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2025-09-01 10:00", "2025-09-01 10:30"])
    })
    out = add_time_features(df.copy(), "ts")
    assert "hour" in out
    assert "dow" in out
    assert "hour_sin" in out
    assert np.isfinite(out["hour_sin"]).all()


def test_decay_features_never_use_current_target():
    df = pd.DataFrame({
        "ts": pd.to_datetime([
            "2025-09-01 10:00",
            "2025-09-01 10:10",
            "2025-09-01 10:20",
        ]),
        "airport": ["JFK", "JFK", "JFK"],
        "target": [100.0, 200.0, 300.0],
    })
    out = add_leakage_safe_decay_features(
        df.copy(),
        target_col="target",
        time_col="ts",
        group_cols=["airport"],
        half_lives_minutes=[30],
    )

    assert pd.isna(out.loc[0, "decay_mean_airport_30m"])
    assert np.isclose(out.loc[1, "decay_mean_airport_30m"], 100.0)
    assert 100.0 < out.loc[2, "decay_mean_airport_30m"] < 200.0
'''

train4_code = r'''
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "TAXITIME_SEC_mvt"

# Resourceful-Quiver-inspired ideas:
# - aircraft/physical proxies when present
# - rich temporal/trajectory features
# - airport traffic/congestion
# - strictly historical, exponentially decayed target statistics ("expiry")
# - chronological month CV
#
# The script is intentionally schema-adaptive because the 2026 dataset can
# contain slightly different column names than the 2025 challenge.

TIME_CANDIDATES = [
    "timestamp", "time", "datetime", "ts", "date_time", "event_time",
    "start_time", "start", "mvt_time", "movement_time", "departure_time",
    "firstseen", "lastseen", "begin", "dt",
]
ID_CANDIDATES = ["flight_id", "flightId", "fid", "icao24", "callsign"]
AIRPORT_CANDIDATES = [
    "airport", "departure_airport", "origin", "origin_airport",
    "apt", "airport_icao", "departure_apt", "adep",
]
AIRCRAFT_CANDIDATES = [
    "aircraft_type", "aircraftType", "icao_type", "typecode",
    "ac_type", "aircraft", "type",
]
RUNWAY_CANDIDATES = ["runway", "departure_runway", "rwy", "runway_id"]


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_data(path: str | None = None) -> pd.DataFrame:
    if path:
        files = [Path(path)]
    else:
        patterns = [
            str(DATA_DIR / "*.parquet"),
            str(DATA_DIR / "**" / "*.parquet"),
        ]
        files = []
        for p in patterns:
            files.extend(Path(x) for x in glob.glob(p, recursive=True))

        # Avoid accidentally reading generated outputs.
        files = [
            p for p in files
            if "submission" not in p.name.lower()
            and "prediction" not in p.name.lower()
            and "feature" not in p.name.lower()
        ]

    if not files:
        raise FileNotFoundError(
            "No parquet files found. Put the competition parquet files under ./data "
            "or pass --input PATH."
        )

    # Prefer files containing the target.
    scored = []
    for p in files:
        try:
            cols = pd.read_parquet(p, engine="pyarrow").columns
            scored.append((TARGET in cols, p))
        except Exception:
            continue

    target_files = [p for has_target, p in scored if has_target]
    if target_files:
        files = target_files

    print("Input files:")
    for p in files:
        print("  ", p)

    frames = []
    for p in files:
        d = pd.read_parquet(p)
        if TARGET in d.columns:
            frames.append(d)

    if not frames:
        raise ValueError(f"Could not find {TARGET!r} in any input parquet.")

    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    print(f"Raw shape: {df.shape}")
    return df


def find_time_column(df: pd.DataFrame) -> str:
    col = first_existing(df, TIME_CANDIDATES)
    if col:
        return col

    # Last-resort search: choose a column with a high datetime parse rate.
    best, best_score = None, 0.0
    for c in df.columns:
        if c == TARGET:
            continue
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            score = parsed.notna().mean()
            if score > best_score and score >= 0.70:
                best, best_score = c, score
        except Exception:
            pass

    if best is None:
        raise ValueError(
            "Could not identify a timestamp column. "
            f"Available columns: {list(df.columns)}"
        )
    return best


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df["_event_time"] = t

    minute = t.dt.hour * 60 + t.dt.minute + t.dt.second / 60.0
    dow = t.dt.dayofweek

    df["hour"] = t.dt.hour.fillna(-1).astype("int16")
    df["minute"] = t.dt.minute.fillna(-1).astype("int16")
    df["dow"] = dow.fillna(-1).astype("int16")
    df["day"] = t.dt.day.fillna(-1).astype("int16")
    df["month"] = t.dt.month.fillna(-1).astype("int16")
    df["weekofyear"] = t.dt.isocalendar().week.astype("float32")
    df["is_weekend"] = (dow >= 5).astype("int8")
    df["is_peak"] = (
        ((t.dt.hour >= 6) & (t.dt.hour < 10))
        | ((t.dt.hour >= 16) & (t.dt.hour < 20))
    ).astype("int8")

    # Circular time representation.
    df["minute_of_day"] = minute
    df["hour_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    df["hour_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow.fillna(0) / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow.fillna(0) / 7.0)

    # Useful interaction buckets.
    df["hour_dow"] = (
        df["dow"].astype(str) + "_" + df["hour"].astype(str)
    )
    return df


def add_generic_numeric_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pull in Resourceful-Quiver-style aircraft/trajectory information if the
    current dataset already contains it. No external aircraft database is
    required; missing fields are simply skipped.
    """
    aliases = {
        "mtow": ["MTOW", "mtow", "max_takeoff_weight", "maximum_takeoff_weight"],
        "mlw": ["MLW", "mlw", "max_landing_weight", "maximum_landing_weight"],
        "engines": ["Num_Engines", "num_engines", "engines", "engine_count"],
        "length": ["ac_L", "aircraft_length", "length_m"],
        "span": ["ac_B2", "aircraft_span", "wingspan_m"],
        "altitude": ["altitude", "mean_altitude", "mean_altitude_ft"],
        "tas": ["TAS", "tas", "mean_TAS", "mean_tas", "tas_kt"],
        "gs": ["GS", "gs", "groundspeed", "mean_groundspeed", "gs_kt"],
        "roc": ["ROC", "roc", "vertical_rate", "mean_vertical_rate"],
        "distance": ["distance", "distance_km", "flight_distance", "route_distance"],
        "bearing": ["bearing", "route_bearing"],
        "duration": [
            "flight_duration", "segment_duration", "duration",
            "fuel_segment_duration", "duration_sec"
        ],
    }

    for out, candidates in aliases.items():
        c = first_existing(df, candidates)
        if c is not None and pd.api.types.is_numeric_dtype(df[c]):
            df[f"rq_{out}"] = pd.to_numeric(df[c], errors="coerce")

    # Useful physics proxies.
    if "rq_mtw" in df:
        df["rq_mtw_log"] = np.log1p(df["rq_mtw"].clip(lower=0))
    if "rq_tas" in df and "rq_gs" in df:
        df["rq_gs_minus_tas"] = df["rq_gs"] - df["rq_tas"]
    if "rq_altitude" in df and "rq_distance" in df:
        df["rq_alt_per_distance"] = df["rq_altitude"] / (df["rq_distance"].abs() + 1.0)
    if "rq_duration" in df and "rq_distance" in df:
        df["rq_distance_per_sec"] = df["rq_distance"] / (df["rq_duration"].abs() + 1.0)

    return df


def add_aircraft_features(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    ac = first_existing(df, AIRCRAFT_CANDIDATES)
    if ac:
        df["aircraft_type_cat"] = df[ac].astype("string").fillna("__MISSING__").astype(str)
    return df, ac


def add_lag_congestion_features(
    df: pd.DataFrame,
    time_col: str,
    airport_col: str | None,
) -> pd.DataFrame:
    """
    Features based only on event timestamps. These do not look at future rows.
    """
    if "_event_time" not in df:
        df["_event_time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    valid = df["_event_time"].notna()
    work = df.loc[valid].copy()

    if airport_col:
        group_cols = [airport_col]
        work["_airport_key"] = work[airport_col].astype("string").fillna("__MISSING__")
        group_cols = ["_airport_key"]
    else:
        work["_airport_key"] = "__ALL__"
        group_cols = ["_airport_key"]

    work = work.sort_values("_event_time", kind="mergesort")
    idx = work.index

    # Number of previous movements inside windows.
    for minutes in [5, 10, 15, 30, 60]:
        seconds = minutes * 60
        times_ns = work["_event_time"].astype("int64").to_numpy()
        left = np.searchsorted(times_ns, times_ns - seconds * 1_000_000_000, side="left")
        work[f"traffic_prev_{minutes}m"] = np.arange(len(work)) - left

    # Previous same-airport event gap.
    work["prev_event_gap_sec"] = (
        work.groupby(group_cols)["_event_time"]
        .diff()
        .dt.total_seconds()
    )

    # Approximate queue intensity: recent movements per minute.
    work["traffic_rate_15m"] = work["traffic_prev_15m"] / 15.0
    work["traffic_rate_60m"] = work["traffic_prev_60m"] / 60.0

    # Restore original order.
    df.loc[idx, work.columns] = work
    return df


def add_leakage_safe_decay_features(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    group_cols: list[str],
    half_lives_minutes: list[int] | None = None,
) -> pd.DataFrame:
    """
    Exponentially-decayed historical target statistics.

    Crucially, the current target is excluded:
        history = all rows strictly before current row

    For each group, if y_i happened at t_i and current event is t:
        weight_i = 2 ** (-(t - t_i) / half_life)

    This is the "expiry" idea: stale observations naturally lose influence.
    """
    if half_lives_minutes is None:
        half_lives_minutes = [15, 30, 60, 180, 720, 1440, 4320, 10080]

    if not group_cols:
        group_cols = ["_all_group"]
        df["_all_group"] = 0

    work = df.copy()
    work["_event_time"] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    work["_target_num"] = pd.to_numeric(work[target_col], errors="coerce")
    work["_row_order"] = np.arange(len(work))

    # Missing target rows are allowed in inference data; they simply don't
    # contribute to historical statistics.
    work = work.sort_values("_event_time", kind="mergesort")

    # Stable numeric timestamp in seconds.
    ts = work["_event_time"].astype("int64") / 1e9
    work["_ts_sec"] = ts

    for hl in half_lives_minutes:
        means = np.full(len(work), np.nan, dtype="float64")
        counts = np.zeros(len(work), dtype="float64")

        # A Python group loop is intentional here: it is substantially safer
        # than a rolling implementation because every value is explicitly
        # shifted behind the current observation.
        for _, pos in work.groupby(group_cols, sort=False, dropna=False).groups.items():
            positions = np.asarray(list(pos), dtype=np.int64)
            # groupby groups contain index labels; map labels to positional rows.
            positions = work.index.get_indexer(positions)
            positions.sort()

            group_ts = work["_ts_sec"].to_numpy()[positions]
            group_y = work["_target_num"].to_numpy()[positions]

            # Maintain weighted sums incrementally.
            sum_w = 0.0
            sum_yw = 0.0
            n_hist = 0

            last_time = None
            for j, p in enumerate(positions):
                cur_t = group_ts[j]

                if not np.isfinite(cur_t):
                    continue

                if last_time is not None:
                    # Decay all previously accumulated mass to current time.
                    decay = 2.0 ** (-(cur_t - last_time) / (hl * 60.0))
                    sum_w *= decay
                    sum_yw *= decay

                if sum_w > 1e-12:
                    means[p] = sum_yw / sum_w
                counts[p] = n_hist

                y = group_y[j]
                if np.isfinite(y):
                    sum_w += 1.0
                    sum_yw += float(y)
                    n_hist += 1

                last_time = cur_t

        prefix = "_".join(group_cols)
        prefix = re.sub(r"[^A-Za-z0-9_]+", "_", prefix)
        df[f"decay_mean_{prefix}_{hl}m"] = means
        df[f"decay_count_{prefix}_{hl}m"] = counts

    return df


def add_historical_features(
    df: pd.DataFrame,
    time_col: str,
    airport_col: str | None,
    aircraft_col: str | None,
    runway_col: str | None,
) -> pd.DataFrame:
    # The strongest useful grouping levels first.
    candidate_groups = []

    def add_group(cols: list[str], name: str):
        if all(c in df.columns for c in cols):
            df[f"_grp_{name}"] = (
                df[cols].astype("string").fillna("__MISSING__").agg("|".join, axis=1)
            )
            candidate_groups.append(f"_grp_{name}")

    if airport_col:
        add_group([airport_col], "airport")
    if aircraft_col:
        add_group([aircraft_col], "aircraft")
    if airport_col and aircraft_col:
        add_group([airport_col, aircraft_col], "airport_aircraft")
    if airport_col and "hour" in df:
        add_group([airport_col, "hour"], "airport_hour")
    if airport_col and "dow" in df:
        add_group([airport_col, "dow"], "airport_dow")
    if airport_col and "hour" in df and "dow" in df:
        add_group([airport_col, "hour", "dow"], "airport_hour_dow")
    if airport_col and runway_col:
        add_group([airport_col, runway_col], "airport_runway")

    # We use decay stats on a limited set of high-value groups to keep memory
    # and runtime reasonable.
    for g in candidate_groups:
        df = add_leakage_safe_decay_features(
            df,
            target_col=TARGET,
            time_col=time_col,
            group_cols=[g],
            half_lives_minutes=[15, 60, 180, 1440, 4320, 10080],
        )

    # A global historical taxi-time prior is very useful for cold-start groups.
    df["_global_group"] = 0
    df = add_leakage_safe_decay_features(
        df,
        target_col=TARGET,
        time_col=time_col,
        group_cols=["_global_group"],
        half_lives_minutes=[30, 180, 1440, 10080],
    )
    return df


def clean_for_catboost(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    drop = {
        target_col,
        "_event_time",
        "_target_num",
        "_row_order",
        "_ts_sec",
        "_all_group",
        "_global_group",
    }
    drop.update(c for c in df.columns if c.startswith("_grp_"))

    # IDs can overfit badly and are often unavailable/meaningless in rank.
    for c in ID_CANDIDATES:
        if c in df.columns:
            drop.add(c)

    # Raw timestamps are not useful as CatBoost categorical strings.
    drop.add(time_col)

    features = [c for c in df.columns if c not in drop]

    X = df[features].copy()

    cat_cols = []
    for c in X.columns:
        if (
            pd.api.types.is_object_dtype(X[c])
            or pd.api.types.is_string_dtype(X[c])
            or pd.api.types.is_categorical_dtype(X[c])
        ):
            X[c] = X[c].fillna("__MISSING__").astype(str)
            cat_cols.append(c)
        elif pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("int8")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # CatBoost handles numeric NaNs, but not an all-NaN column.
    keep = [c for c in X.columns if not X[c].isna().all()]
    X = X[keep]
    cat_cols = [c for c in cat_cols if c in X.columns]

    return X, cat_cols


def month_splits(df: pd.DataFrame, time_col: str):
    months = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.to_period("M")
    unique = sorted(months.dropna().unique())
    if len(unique) < 2:
        # Fallback chronological 80/20 split.
        order = np.argsort(pd.to_datetime(df[time_col], errors="coerce").astype("int64"))
        cut = int(len(order) * 0.8)
        yield order[:cut], order[cut:], "holdout"
        return

    # Expanding-window monthly CV.
    for i in range(1, len(unique)):
        train_months = set(unique[:i])
        valid_month = unique[i]
        train_idx = np.flatnonzero(months.isin(train_months).to_numpy())
        valid_idx = np.flatnonzero((months == valid_month).to_numpy())
        if len(train_idx) and len(valid_idx):
            yield train_idx, valid_idx, str(valid_month)


def fit_model(X_train, y_train, X_valid, y_valid, cat_cols, seed=42):
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=5000,
        learning_rate=0.035,
        depth=8,
        l2_leaf_reg=8.0,
        random_strength=0.8,
        bagging_temperature=0.4,
        border_count=254,
        random_seed=seed,
        od_type="Iter",
        od_wait=250,
        verbose=250,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )
    return model


def train(args):
    df = load_data(args.input)

    time_col = find_time_column(df)
    airport_col = first_existing(df, AIRPORT_CANDIDATES)
    aircraft_col = first_existing(df, AIRCRAFT_CANDIDATES)
    runway_col = first_existing(df, RUNWAY_CANDIDATES)

    print(f"Time column:      {time_col}")
    print(f"Airport column:   {airport_col}")
    print(f"Aircraft column:  {aircraft_col}")
    print(f"Runway column:    {runway_col}")

    # Keep only rows with an observed training target for model fitting.
    target_num = pd.to_numeric(df[TARGET], errors="coerce")
    train_mask = target_num.notna() & np.isfinite(target_num) & (target_num >= 0)
    df[TARGET] = target_num
    df = df.loc[train_mask].copy()

    # Sort once. All historical features below are explicitly past-only.
    df["_event_time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.sort_values("_event_time", kind="mergesort").reset_index(drop=True)

    df = add_time_features(df, time_col)
    df = add_generic_numeric_physics(df)
    df, aircraft_col = add_aircraft_features(df)
    df = add_lag_congestion_features(df, time_col, airport_col)
    df = add_historical_features(
        df, time_col, airport_col, aircraft_col, runway_col
    )

    # A few robust interactions.
    if "traffic_prev_15m" in df and "traffic_prev_60m" in df:
        df["traffic_acceleration"] = (
            df["traffic_prev_15m"] * 4.0 - df["traffic_prev_60m"]
        )
    if "prev_event_gap_sec" in df:
        df["prev_event_gap_log"] = np.log1p(
            df["prev_event_gap_sec"].clip(lower=0)
        )

    X, cat_cols = clean_for_catboost(df, TARGET, time_col)
    y = df[TARGET].astype("float32")

    print(f"Training rows: {len(X):,}")
    print(f"Features:       {X.shape[1]}")
    print(f"Categoricals:   {len(cat_cols)}")
    print(f"Target mean:    {y.mean():.2f}")
    print(f"Target median:  {y.median():.2f}")

    fold_results = []
    models = []

    for fold, (tr_idx, va_idx, label) in enumerate(month_splits(df, time_col), 1):
        print("\n" + "=" * 80)
        print(f"VALIDATION: {label}")
        print(f"Train rows: {len(tr_idx):,}")
        print(f"Valid rows: {len(va_idx):,}")

        model = fit_model(
            X.iloc[tr_idx],
            y.iloc[tr_idx],
            X.iloc[va_idx],
            y.iloc[va_idx],
            cat_cols,
            seed=42 + fold,
        )

        pred = np.clip(model.predict(X.iloc[va_idx]), 0, None)
        rmse = math.sqrt(mean_squared_error(y.iloc[va_idx], pred))

        fold_results.append({"fold": label, "rmse": rmse})
        print(f"RMSE: {rmse:.4f}")

        model.save_model(MODEL_DIR / f"train4_fold_{fold}_{label}.cbm")
        models.append(model)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(OUTPUT_DIR / "train4_cv.csv", index=False)

    print("\n" + "=" * 80)
    print("CV RESULTS")
    print(results_df.to_string(index=False))
    print(f"Mean RMSE: {results_df.rmse.mean():.4f}")
    print(f"Best RMSE: {results_df.rmse.min():.4f}")

    # Feature importance from the last model.
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": models[-1].get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "train4_feature_importance.csv", index=False)
    print("\nTop features:")
    print(importance.head(30).to_string(index=False))

    # Train a final model on all available training data.
    print("\nTraining final model on all rows...")
    final_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=max(1200, int(np.median([m.tree_count_ for m in models]))),
        learning_rate=0.035,
        depth=8,
        l2_leaf_reg=8.0,
        random_strength=0.8,
        bagging_temperature=0.4,
        border_count=254,
        random_seed=2026,
        verbose=250,
        allow_writing_files=False,
    )
    final_model.fit(X, y, cat_features=cat_cols)
    final_model.save_model(MODEL_DIR / "train4_final.cbm")

    metadata = {
        "target": TARGET,
        "time_col": time_col,
        "airport_col": airport_col,
        "aircraft_col": aircraft_col,
        "runway_col": runway_col,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "categorical_features": cat_cols,
        "features": list(X.columns),
    }
    (MODEL_DIR / "train4_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )

    print("\nSaved:")
    print("  models/train4_final.cbm")
    print("  models/train4_metadata.json")
    print("  outputs/train4_cv.csv")
    print("  outputs/train4_feature_importance.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=None,
        help="Optional parquet file. Otherwise all suitable parquet files in ./data are used.",
    )
    args = parser.parse_args()
    train(args)
