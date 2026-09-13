"""The columns that must never reach the feature matrix.

Track A's defining trap is leaking departure-delay information that would not
exist at T-24h. This list is the whole defence, so it is a module with a test,
not a comment.

The guard is two-layered: an explicit set of named columns, plus a prefix rule
for the ``Div*`` diversion-detail family (``Div1Airport``, ``Div2TailNum``, and
so on - 45 columns as of the current BTS schema). Enumerating all 45 by name
risks missing one, and silently under-guards the day BTS adds a ``Div6*``
field; a prefix ban degrades safely instead. ``DivAirportLandings`` stays in
the explicit set too - belt and braces is the point of a defence-in-depth
guard.
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

# Diversion detail: every Div1..Div5 column (airport, wheels-on/off, tail
# number, elapsed time, ...) is post-hoc actual data, unknown at T-24h. A
# prefix rule covers the whole family without enumerating all 45 - no
# legitimate feature starts with "Div" (the closest is "DistanceGroup", which
# starts "Dis").
FORBIDDEN_PREFIXES: Final[tuple[str, ...]] = ("Div",)


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Raise if any column is unknown 24 hours before departure.

    Args:
        columns: The feature matrix's column names.

    Raises:
        LeakageError: Naming every offending column.
    """
    column_set = set(columns)
    leaked = FORBIDDEN & column_set
    leaked |= {column for column in column_set if column.startswith(FORBIDDEN_PREFIXES)}
    if leaked:
        raise LeakageError(
            f"Leakage: {sorted(leaked)} are not known at T-24h. "
            f"A dispatcher could not have seen these yesterday."
        )
