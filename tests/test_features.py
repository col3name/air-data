import pandas as pd
from train import build_features


def test_core_features_are_created():
    df = pd.DataFrame({
        "MVT_TIME_UTC_mvt": pd.to_datetime(
            ["2025-01-01 10:00:00", "2025-01-01 10:05:00"],
            utc=True,
        ),
        "ADEP_mvt": ["EDDF", "EDDF"],
        "ADES_mvt": ["EGLL", "EHAM"],
        "AIRCRAFT_TYPE_mvt": ["A320", "B738"],
        "RUNWAY_mvt": ["07C", "07C"],
        "STAND_mvt": ["A10", "A11"],
        "AIRCRAFT_OPERATOR_flt": ["DLH", "KLM"],
        "PHASE_mvt": ["DEPARTURE", "DEPARTURE"],
        "TAXITIME_SEC_mvt": [500, 700],
    })

    result = build_features(df)

    for col in [
        "AIRPORT_AIRLINE",
        "AIRPORT_DESTINATION",
        "TIME_15M",
        "DEP_COUNT_15M",
        "QUEUE_REGIME",
        "RUNWAY_LOAD_REGIME",
    ]:
        assert col in result.columns
