"""The pure parts of the acquisition script: month maths, URLs and the column whitelist.

No network. The download path is exercised by running the real command; what is tested here is the
logic that would otherwise fail silently after a 25-minute download.
"""

from __future__ import annotations

import pytest

from app.data.acquire import (
    KEEP_COLUMNS,
    AcquisitionError,
    month_key,
    month_range,
    month_url,
    parse_month,
)


def test_full_year_range_is_twelve_months() -> None:
    months = month_range(parse_month("2025-07"), parse_month("2026-06"))

    assert len(months) == 12
    assert month_key(*months[0]) == "2025-07"
    assert month_key(*months[-1]) == "2026-06"


def test_range_crosses_the_year_boundary() -> None:
    months = [month_key(*m) for m in month_range((2025, 11), (2026, 2))]

    assert months == ["2025-11", "2025-12", "2026-01", "2026-02"]


def test_single_month_range_is_one_month() -> None:
    assert month_range((2026, 6), (2026, 6)) == [(2026, 6)]


def test_reversed_range_is_rejected() -> None:
    with pytest.raises(AcquisitionError, match="before"):
        month_range((2026, 6), (2025, 7))


@pytest.mark.parametrize("text", ["2025-13", "2025-00", "July", "2025/07", "", "202507"])
def test_malformed_month_is_rejected(text: str) -> None:
    with pytest.raises(AcquisitionError, match="YYYY-MM"):
        parse_month(text)


def test_url_omits_the_leading_zero_on_the_month() -> None:
    """BTS files June as ``_2026_6``, not ``_2026_06`` - a leading zero 404s."""
    assert month_url(2026, 6).endswith("_2026_6.zip")


def test_no_banned_column_is_ever_read() -> None:
    """The leakage defence at its cheapest point: these are never loaded into memory."""
    banned = {
        "DepDelay",
        "DepDel15",
        "DepTime",
        "TaxiOut",
        "WheelsOff",
        "AirTime",
        "ActualElapsedTime",
        "ArrTime",
        "ArrDelay",
        "ArrDelayMinutes",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "LateAircraftDelay",
    }

    assert not banned & set(KEEP_COLUMNS)


def test_the_row_key_columns_are_all_present() -> None:
    """R2's sealed holdout hashes this composite; a missing part would corrupt the seal."""
    row_key = {
        "FlightDate",
        "Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "Origin",
        "Dest",
        "CRSDepTime",
    }

    assert row_key <= set(KEEP_COLUMNS)


def test_the_label_and_its_two_flags_are_present() -> None:
    """``ArrDel15`` is the label; Cancelled and Diverted decide how to treat a blank one."""
    assert {"ArrDel15", "Cancelled", "Diverted"} <= set(KEEP_COLUMNS)


def test_keep_columns_has_no_duplicates() -> None:
    assert len(KEEP_COLUMNS) == len(set(KEEP_COLUMNS))
