"""The columns that must never reach the feature matrix.

Track A's defining trap is leaking departure-delay information that would not
exist at T-24h. This list is the whole defence, so it is a module with a test,
not a comment.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final


class LeakageError(AssertionError):
    """Raised when a column unavailable at prediction time reaches the matrix."""


FORBIDDEN: Final[frozenset[str]] = frozenset(
    {
        # departure actuals
        "DepTime",
        "DepDelay",
        "DepDelayMinutes",
        "DepDel15",
        "DepartureDelayGroups",
        # in-flight actuals
        "TaxiOut",
        "WheelsOff",
        "WheelsOn",
        "TaxiIn",
        "AirTime",
        "ActualElapsedTime",
        # arrival actuals
        "ArrTime",
        "ArrDelay",
        "ArrDelayMinutes",
        "ArrivalDelayGroups",
        # post-hoc cause attribution
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
        # outcome flags
        "Cancelled",
        "CancellationCode",
        "Diverted",
        "DivAirportLandings",
        # gate-return detail
        "FirstDepTime",
        "TotalAddGTime",
        "LongestAddGTime",
    }
)


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Raise if any column is unknown 24 hours before departure.

    Args:
        columns: The feature matrix's column names.

    Raises:
        LeakageError: Naming every offending column.
    """
    leaked = FORBIDDEN & set(columns)
    if leaked:
        raise LeakageError(
            f"Leakage: {sorted(leaked)} are not known at T-24h. "
            f"A dispatcher could not have seen these yesterday."
        )
