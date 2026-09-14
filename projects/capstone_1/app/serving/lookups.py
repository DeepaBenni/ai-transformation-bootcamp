"""Historical late-rate lookup tables needed at score time (R9).

Six of the model's 24 features are backward-looking rates a caller cannot
supply directly - they must be looked up by airport, airline and route from
the same feature dataset training used. Without this module the serving
layer cannot build a feature row at all.
"""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from app.config import FEATURES_PARQUET, LOOKUP_PATH

_lookups: dict[str, Any] | None = None

LOOKUP_COLUMNS: list[str] = [
    "period",
    "Origin",
    "Dest",
    "Reporting_Airline",
    "route",
    "dep_hour",
    "origin_hour_late_rate",
    "route_late_rate",
    "carrier_late_rate",
    "origin_late_rate",
    "dest_late_rate",
    "carrier_origin_late_rate",
    "route_late_rate_n",
]


def build_lookups() -> dict[str, Any]:
    """Derive every lookup table from the most recent month of feature data.

    Returns:
        A mapping with keys ``origin_hour``, ``route``, ``carrier``, ``origin``,
        ``dest``, ``carrier_origin`` and ``route_n``.
    """
    frame = pd.read_parquet(FEATURES_PARQUET, columns=LOOKUP_COLUMNS)
    latest = frame[frame.period == frame.period.max()]

    return {
        "origin_hour": latest.groupby(["Origin", "dep_hour"], observed=True)
        .origin_hour_late_rate.mean()
        .to_dict(),
        "route": latest.groupby("route", observed=True).route_late_rate.mean().to_dict(),
        "carrier": latest.groupby("Reporting_Airline", observed=True)
        .carrier_late_rate.mean()
        .to_dict(),
        "origin": latest.groupby("Origin", observed=True).origin_late_rate.mean().to_dict(),
        "dest": latest.groupby("Dest", observed=True).dest_late_rate.mean().to_dict(),
        "carrier_origin": latest.groupby(["Reporting_Airline", "Origin"], observed=True)
        .carrier_origin_late_rate.mean()
        .to_dict(),
        "route_n": latest.groupby("route", observed=True).route_late_rate_n.mean().to_dict(),
    }


def load_lookups() -> dict[str, Any]:
    """Load the lookup tables, cached first in memory then on disk.

    Returns:
        The lookup mapping - possibly empty if neither the cache nor the
        feature dataset is available, in which case callers fall back to
        training medians.
    """
    global _lookups
    if _lookups is not None:
        return _lookups

    if LOOKUP_PATH.exists():
        _lookups = joblib.load(LOOKUP_PATH)
        return _lookups

    if not FEATURES_PARQUET.exists():
        _lookups = {}
        return _lookups

    _lookups = build_lookups()
    LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_lookups, LOOKUP_PATH, compress=3)
    return _lookups


def reset_cache() -> None:
    """Clear the in-memory cache. For tests only."""
    global _lookups
    _lookups = None
