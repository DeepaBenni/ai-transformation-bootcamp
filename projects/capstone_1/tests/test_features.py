import pandas as pd
import pytest

from app.features import build
from app.features.forbidden import FORBIDDEN


def _raw():
    return pd.DataFrame({
        "FlightDate": pd.to_datetime(
            ["2026-01-05", "2026-01-05", "2026-02-10", "2026-03-15"]
        ),
        "period": ["2026-01", "2026-01", "2026-02", "2026-03"],
        "Reporting_Airline": ["DL", "AA", "DL", "DL"],
        "Tail_Number": ["N1", "N2", "N1", "N1"],
        "Origin": ["ATL", "ATL", "ATL", "ORD"],
        "Dest": ["ORD", "ORD", "ORD", "ATL"],
        "CRSDepTime": [1730, 630, 1730, 2230],
        "CRSArrTime": [1925, 825, 1925, 35],
        "CRSElapsedTime": [115.0, 115.0, 115.0, 125.0],
        "Distance": [606.0, 606.0, 606.0, 606.0],
        "DistanceGroup": [3, 3, 3, 3],
        "DayOfWeek": [1, 1, 2, 7],
        "Month": [1, 1, 2, 3],
        "Quarter": [1, 1, 1, 1],
        "label": [1, 0, 1, 0],
    })


def test_feature_list_is_twenty_four_long():
    assert len(build.FEATURE_COLUMNS) == 24


def test_feature_list_has_no_duplicates():
    assert len(set(build.FEATURE_COLUMNS)) == len(build.FEATURE_COLUMNS)


def test_feature_list_contains_no_forbidden_column():
    assert not FORBIDDEN & set(build.FEATURE_COLUMNS)


def test_feature_order_is_the_documented_contract():
    """Order is the contract - a reordered matrix scores silent nonsense."""
    assert build.FEATURE_COLUMNS[0] == "Distance"
    assert build.FEATURE_COLUMNS[-1] == "route_late_rate_n"
    assert build.FEATURE_COLUMNS[6] == "dep_hour"


def test_dep_hour_is_extracted_from_hhmm():
    out = build.add_schedule_features(_raw())
    assert out.dep_hour.tolist() == [17, 6, 17, 22]


def test_dep_minute_of_day_combines_hour_and_minute():
    out = build.add_schedule_features(_raw())
    assert out.dep_minute_of_day.iloc[0] == 17 * 60 + 30


def test_red_eye_flags_only_departures_before_six():
    out = build.add_schedule_features(_raw())
    assert out.is_red_eye.tolist() == [0, 0, 0, 0]


def test_weekend_flags_sunday():
    out = build.add_schedule_features(_raw())
    assert out.is_weekend.tolist() == [0, 0, 0, 1]


def test_scheduled_speed_is_clipped_to_a_physical_range():
    frame = _raw()
    frame.loc[0, "CRSElapsedTime"] = 1.0      # absurd: 36,360 mph
    out = build.add_schedule_features(frame)
    assert out.sched_speed_mph.max() <= 700
    assert out.sched_speed_mph.min() >= 100


def test_tail_leg_number_counts_legs_within_a_day():
    out = build.add_schedule_features(_raw())
    same_day_same_tail = out[(out.Tail_Number == "N1") & (out.period == "2026-01")]
    assert same_day_same_tail.tail_leg_number.tolist() == [1]


def test_build_produces_every_feature_column():
    out = build.build(_raw())
    missing = set(build.FEATURE_COLUMNS) - set(out.columns)
    assert not missing, f"missing: {sorted(missing)}"


def test_build_output_carries_no_forbidden_column():
    out = build.build(_raw())
    assert not FORBIDDEN & set(out.columns)


def test_build_fills_non_rate_feature_nan_with_median():
    """RULING 2: every feature is filled, not only the _rate columns.

    CRSElapsedTime carries real nulls in the cleaned parquet (from
    prepare.clean_frame), and they propagate into sched_speed_mph.
    LogisticRegression raises on any remaining NaN, so a fillna scoped to
    `_rate` columns is not enough.
    """
    frame = _raw()
    frame.loc[0, "CRSElapsedTime"] = float("nan")
    out = build.build(frame)
    assert out[build.FEATURE_COLUMNS].isna().sum().sum() == 0


def test_days_to_holiday_matches_a_manual_nearest_holiday_search():
    """RULING 1: the running-minimum loop must match a brute-force search."""
    frame = _raw()
    out = build.add_schedule_features(frame)

    holidays = [h.date() for h in build.HOLIDAYS]
    expected = [
        min(abs((d.date() - h).days) for h in holidays)
        for d in frame.FlightDate
    ]
    expected = [min(gap, 7) for gap in expected]
    assert out.days_to_holiday.tolist() == expected
