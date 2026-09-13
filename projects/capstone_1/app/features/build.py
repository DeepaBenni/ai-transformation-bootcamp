"""The 24 features, all available 24 hours before departure.

Every feature answers one question: would a dispatcher have known this
yesterday? That question is the whole leakage defence, and anything failing it
belongs in :mod:`app.features.forbidden` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from app.config import FEATURES_PARQUET, PRIOR
from app.data.seal import development_frame
from app.features.forbidden import assert_no_leakage
from app.features.history import expanding_rate

# US federal holidays plus the peak travel days around them, for the 12-month
# window. Known from the calendar, so available at any prediction time.
HOLIDAYS: Final[list[pd.Timestamp]] = [
    pd.Timestamp(d)
    for d in (
        "2025-07-04",
        "2025-09-01",
        "2025-11-27",
        "2025-11-28",
        "2025-12-24",
        "2025-12-25",
        "2025-12-31",
        "2026-01-01",
        "2026-01-19",
        "2026-02-16",
        "2026-05-25",
        "2026-07-03",
        "2026-07-04",
    )
]

FEATURE_COLUMNS: Final[list[str]] = [
    # raw, straight from the timetable
    "Distance",
    "CRSElapsedTime",
    "DayOfWeek",
    "Month",
    "Quarter",
    "DistanceGroup",
    # derived from the schedule
    "dep_hour",
    "dep_minute_of_day",
    "arr_hour",
    "is_red_eye",
    "is_weekend",
    "sched_speed_mph",
    "days_to_holiday",
    "is_holiday_window",
    # derived from the timetable's shape
    "origin_bank_size",
    "dest_bank_size",
    "tail_leg_number",
    # derived from prior months only
    "route_late_rate",
    "carrier_late_rate",
    "origin_late_rate",
    "dest_late_rate",
    "carrier_origin_late_rate",
    "origin_hour_late_rate",
    "route_late_rate_n",
]


def _hour(hhmm: pd.Series) -> pd.Series:
    """Convert a BTS HHMM integer to an hour, mapping 2400 to 0."""
    return (hhmm // 100).mod(24).astype("int16")


def _minute(hhmm: pd.Series) -> pd.Series:
    """Convert a BTS HHMM integer to a minute-of-hour."""
    return (hhmm % 100).astype("int16")


def add_schedule_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add everything derivable from the published timetable.

    Args:
        frame: The cleaned flights, carrying ``CRSDepTime`` and ``CRSArrTime``.

    Returns:
        ``frame`` with the schedule-derived columns added.
    """
    frame["dep_hour"] = _hour(frame.CRSDepTime)
    frame["arr_hour"] = _hour(frame.CRSArrTime)
    frame["dep_minute_of_day"] = frame.dep_hour * 60 + _minute(frame.CRSDepTime)

    frame["is_red_eye"] = frame.dep_hour.between(0, 5).astype("int8")
    frame["is_weekend"] = frame.DayOfWeek.isin([6, 7]).astype("int8")

    # Schedule padding proxy. Clipped because BTS ships impossible schedules.
    hours = frame.CRSElapsedTime / 60.0
    frame["sched_speed_mph"] = (frame.Distance / hours).clip(100, 700)

    # Holiday proximity, capped at a week - beyond that it carries no signal.
    #
    # RULING 1: a brute-force `np.abs(dates[:, None] - holidays[None, :])`
    # broadcast builds a (rows x holidays) temporary - 5.7M x 13 is ~1.23 GB
    # peak on this machine's ~6.8 GB free. A running minimum over the 13
    # holidays instead touches one (rows,) array per holiday, verified with
    # np.array_equal to produce identical output at 6.7x lower peak memory.
    dates = frame.FlightDate.to_numpy(dtype="datetime64[D]")
    nearest = np.full(len(frame), np.iinfo(np.int32).max, dtype="int32")
    for holiday in HOLIDAYS:
        gap = np.abs(dates - np.datetime64(holiday.date(), "D")).astype("int32")
        np.minimum(nearest, gap, out=nearest)
    frame["days_to_holiday"] = np.clip(nearest, 0, 7)
    frame["is_holiday_window"] = (nearest <= 2).astype("int8")

    # Congestion: how many flights share this airport and hour that day.
    frame["origin_bank_size"] = (
        frame.groupby(["FlightDate", "Origin", "dep_hour"], observed=True)
        .Origin.transform("size")
        .astype("int16")
    )
    frame["dest_bank_size"] = (
        frame.groupby(["FlightDate", "Dest", "arr_hour"], observed=True)
        .Dest.transform("size")
        .astype("int16")
    )

    # Rotation: which leg of this aircraft's day. Known from the timetable.
    frame = frame.sort_values(["Tail_Number", "FlightDate", "CRSDepTime"])
    frame["tail_leg_number"] = (
        frame.groupby(["Tail_Number", "FlightDate"], observed=True).cumcount() + 1
    ).astype("int16")
    return frame.sort_index()


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the six backward-looking rates, plus the route evidence count.

    Args:
        frame: Must carry ``period``, ``label`` and the grouping columns.

    Returns:
        ``frame`` with the history columns added.
    """
    frame["route"] = frame.Origin + "-" + frame.Dest

    frame = expanding_rate(frame, ["route"], "route_late_rate")
    frame = expanding_rate(frame, ["Reporting_Airline"], "carrier_late_rate")
    frame = expanding_rate(frame, ["Origin"], "origin_late_rate")
    frame = expanding_rate(frame, ["Dest"], "dest_late_rate")
    frame = expanding_rate(frame, ["Reporting_Airline", "Origin"], "carrier_origin_late_rate")
    frame = expanding_rate(frame, ["Origin", "dep_hour"], "origin_hour_late_rate")

    # Drop the counts nothing consumes; route_late_rate_n is a feature in its
    # own right, because it says how much evidence the rates rest on.
    drop = [
        "carrier_late_rate_n",
        "origin_late_rate_n",
        "dest_late_rate_n",
        "carrier_origin_late_rate_n",
        "origin_hour_late_rate_n",
    ]
    return frame.drop(columns=[c for c in drop if c in frame.columns])


def build(frame: pd.DataFrame) -> pd.DataFrame:
    """Build every feature and assert the result is leakage-free.

    RULING 2: every feature column is filled, not only the ``_rate`` ones.
    ``CRSElapsedTime`` carries a handful of real nulls in the cleaned data
    (3 rows in the full development set) that propagate into
    ``sched_speed_mph``; LogisticRegression raises ``ValueError`` on any
    remaining NaN, and it is one of three model families a later task must
    compare. Non-rate columns are filled with their own median computed over
    the whole frame passed in. Note this is a mild look-ahead for the
    earliest folds when called on the full development set (the median at
    fold-scoring time could differ slightly from the median over all prior
    months) - it is immaterial here (3 affected rows out of ~5.7M) and is
    reported in the model card rather than hidden.

    Args:
        frame: The cleaned development flights.

    Returns:
        ``frame`` with all 24 features present and no NaNs among them.

    Raises:
        LeakageError: If any forbidden column survived.
    """
    frame = add_schedule_features(frame)
    frame = add_history_features(frame)

    rate_columns = [c for c in FEATURE_COLUMNS if c.endswith("_rate")]
    frame[rate_columns] = frame[rate_columns].fillna(PRIOR)
    other_columns = [c for c in FEATURE_COLUMNS if c not in rate_columns]
    frame[other_columns] = frame[other_columns].fillna(frame[other_columns].median())

    assert_no_leakage(frame.columns)
    return frame


def run() -> Path:
    """Build features over the development set and write them to parquet.

    Returns:
        The path written.
    """
    frame = development_frame()
    frame = build(frame)

    keep = [
        "row_key",
        "period",
        "label",
        "FlightDate",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "route",
        *FEATURE_COLUMNS,
    ]
    frame = frame[keep]
    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FEATURES_PARQUET, index=False)
    return FEATURES_PARQUET
