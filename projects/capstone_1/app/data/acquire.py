"""One-command acquisition of the BTS on-time performance data (requirement R1).

R1 asks that a stranger can clone this repository, run a single command, and end up with the
dataset. No browser, no Kaggle account, no manual step::

    python -m src.data.acquire --start 2025-07 --end 2026-06

Expected result: 12 Parquet files under ``data/interim/``, roughly 7.3 M rows in total, and a
``data/manifest.json`` recording URL, SHA-256, retrieval timestamp, row count and column count per
file. Re-running is cheap: a month already recorded in the manifest with its Parquet file still on
disk is skipped.

**The boundary of this module.** It downloads, extracts and stores. It does not clean, label,
engineer or model - those live in the notebooks. Most importantly it reads only ``KEEP_COLUMNS``,
24 of the source file's 110. That is the leakage defence at its cheapest point: ``DepDelay`` cannot
reach a feature matrix if it was never loaded into memory. Filtering here is stronger than dropping
later, because there is no intermediate frame in which someone could reach for it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Final

import pandas as pd
import requests

from app.config import DATA_DIR, INTERIM_DIR, MANIFEST_PATH

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

# ------------------------------------------------------------------------- the source --
BASE_URL: Final[str] = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

SOURCE_NAME: Final[str] = (
    "Bureau of Transportation Statistics, On-Time Reporting Carrier On-Time Performance "
    "(1987-present)"
)
SOURCE_LICENCE: Final[str] = (
    "US federal government work, public domain (17 U.S.C. 105). No registration required."
)

# BTS serves the zip only to something that looks like a browser.
USER_AGENT: Final[str] = "Mozilla/5.0"

# ------------------------------------------------------------------- transfer settings --
CONNECT_TIMEOUT_S: Final[float] = 30.0
READ_TIMEOUT_S: Final[float] = 120.0
CHUNK_BYTES: Final[int] = 1024 * 1024
MAX_ATTEMPTS: Final[int] = 4
RETRY_BACKOFF_S: Final[float] = 5.0

# Every zip begins with these two bytes. A 404 from BTS arrives as a ~108 KB HTML page rather
# than an empty body, so the magic bytes are a second line of defence behind the status code.
ZIP_MAGIC: Final[bytes] = b"PK"

# The CSV inside is latin-1. UTF-8 raises on carrier and city names.
CSV_ENCODING: Final[str] = "latin-1"

# ---------------------------------------------------------------------------- columns --
# 24 of 110. Everything omitted is either unused or, far more importantly, not known at T-24h.
# The banned list is in 01_SignalCraft_Requirements_Understanding.md section 3.
KEEP_COLUMNS: Final[list[str]] = [
    # identity - the composite key for the R2 sealed holdout
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    # calendar
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    # schedule, all known at T-24h
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "DistanceGroup",
    "DepTimeBlk",
    "ArrTimeBlk",
    "Tail_Number",
    "OriginCityName",
    "OriginState",
    "DestCityName",
    "DestState",
    # label, and the two flags needed to decide how to treat it
    "ArrDel15",
    "Cancelled",
    "Diverted",
]

DATE_COLUMNS: Final[list[str]] = ["FlightDate"]

MONTH_FORMAT: Final[str] = "%Y-%m"


# ------------------------------------------------------------------------- exceptions --
class AcquisitionError(RuntimeError):
    """Raised when a month cannot be acquired, with the reason and the fix in the message."""


# ------------------------------------------------------------------------ the record --
@dataclass(frozen=True)
class MonthlyFile:
    """One month's provenance, as written to the manifest.

    Attributes:
        url: The exact URL the bytes came from.
        sha256: Checksum of the downloaded zip, so a re-run can prove it has the same file.
        retrieved_utc: ISO 8601 UTC timestamp of the download.
        rows: Rows in the Parquet file.
        columns: Columns kept, out of the source file's 110.
        zip_bytes: Size of the downloaded zip.
        path: Parquet path, relative to the repository root.
    """

    url: str
    sha256: str
    retrieved_utc: str
    rows: int
    columns: int
    zip_bytes: int
    path: str


# ------------------------------------------------------------------------ month maths --
def month_url(year: int, month: int) -> str:
    """Return the download URL for one month.

    Args:
        year: Four-digit year.
        month: 1-12. BTS formats this without a leading zero.
    """
    return BASE_URL.format(year=year, month=month)


def month_key(year: int, month: int) -> str:
    """Return the ``YYYY-MM`` key this month is filed under in the manifest."""
    return f"{year:04d}-{month:02d}"


def parse_month(text: str) -> tuple[int, int]:
    """Parse a ``YYYY-MM`` argument into a year and month.

    Args:
        text: The command-line value, e.g. ``"2025-07"``.

    Returns:
        A ``(year, month)`` pair.

    Raises:
        AcquisitionError: If the value is not a valid ``YYYY-MM`` month.
    """
    try:
        parsed = datetime.strptime(text.strip(), MONTH_FORMAT)
    except ValueError as exc:
        raise AcquisitionError(
            f"Expected a month as YYYY-MM, got {text!r}. Example: --start 2025-07"
        ) from exc
    return parsed.year, parsed.month


def month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """List every month from ``start`` to ``end`` inclusive.

    Raises:
        AcquisitionError: If ``end`` falls before ``start``.
    """
    start_index = start[0] * 12 + (start[1] - 1)
    end_index = end[0] * 12 + (end[1] - 1)
    if end_index < start_index:
        raise AcquisitionError(
            f"--end {month_key(*end)} is before --start {month_key(*start)}. Swap them."
        )
    return [(index // 12, index % 12 + 1) for index in range(start_index, end_index + 1)]


# -------------------------------------------------------------------------- the fetch --
def _download(url: str) -> bytes:
    """Stream one zip into memory, proving it arrived whole.

    Three checks, because a partial download is worse than a failed one: a failure is loud, while a
    truncated zip unzips to a partial CSV and quietly costs a third of a month's rows.

    Args:
        url: The month's zip URL.

    Returns:
        The complete zip as bytes.

    Raises:
        AcquisitionError: If every attempt failed, or the month does not exist.
    """
    last_problem = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                stream=True,
                timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
            ) as response:
                # Check the status code, never the response size: BTS answers a month that does
                # not exist with a large HTML page rather than an empty body.
                if response.status_code == 404:
                    raise AcquisitionError(
                        f"{url}\nreturned 404 - that month is not published yet. "
                        "BTS lags the current month by several months."
                    )
                response.raise_for_status()

                expected = response.headers.get("Content-Length")
                expected_bytes = int(expected) if expected and expected.isdigit() else None

                buffer = io.BytesIO()
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    buffer.write(chunk)
                payload = buffer.getvalue()

            if expected_bytes is not None and len(payload) != expected_bytes:
                raise AcquisitionError(
                    f"Truncated download: got {len(payload):,} bytes, "
                    f"Content-Length said {expected_bytes:,}"
                )
            if not payload.startswith(ZIP_MAGIC):
                raise AcquisitionError(
                    f"Response is not a zip (first bytes {payload[:16]!r}). "
                    "BTS may have served an error page."
                )
            return payload

        except AcquisitionError as exc:
            if "404" in str(exc):
                raise
            last_problem = str(exc)
        except requests.RequestException as exc:
            last_problem = f"{type(exc).__name__}: {exc}"

        LOGGER.warning("Attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, last_problem)
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_S * attempt)

    raise AcquisitionError(
        f"Gave up on {url} after {MAX_ATTEMPTS} attempts. Last problem: {last_problem}"
    )


def _extract_csv(payload: bytes) -> bytes:
    """Return the single CSV member of the downloaded zip.

    Raises:
        AcquisitionError: If the archive is unreadable or does not hold exactly one CSV.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise AcquisitionError(
                    f"Expected exactly one .csv in the zip, found {csv_names or 'none'}"
                )
            return archive.read(csv_names[0])
    except zipfile.BadZipFile as exc:
        raise AcquisitionError(f"Downloaded bytes are not a readable zip: {exc}") from exc


def _to_frame(csv_bytes: bytes) -> pd.DataFrame:
    """Read the CSV, keeping only the columns known at T-24h.

    Raises:
        AcquisitionError: If the source file is missing a column this project depends on, which
            would mean BTS changed the reporting standard.
    """
    try:
        return pd.read_csv(
            io.BytesIO(csv_bytes),
            encoding=CSV_ENCODING,
            usecols=KEEP_COLUMNS,
            parse_dates=DATE_COLUMNS,
            low_memory=False,
        )
    except ValueError as exc:
        raise AcquisitionError(
            f"Could not select the expected columns: {exc}\n"
            "If BTS has renamed or dropped a field, update KEEP_COLUMNS and say so in the README - "
            "a changed reporting standard is one of the model's stated switch-off conditions."
        ) from exc


# ----------------------------------------------------------------------- the manifest --
def load_manifest() -> dict[str, Any]:
    """Return the manifest, or an empty one if this is the first run."""
    if not MANIFEST_PATH.exists():
        return {
            "source": SOURCE_NAME,
            "licence": SOURCE_LICENCE,
            "url_pattern": BASE_URL,
            "files": {},
        }
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, Any]) -> None:
    """Write the manifest, sorted by month so its diffs stay readable."""
    manifest["files"] = dict(sorted(manifest["files"].items()))
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------ one month --
def acquire_month(year: int, month: int, *, force: bool = False) -> MonthlyFile:
    """Download, convert and record one month.

    Args:
        year: Four-digit year.
        month: 1-12.
        force: Re-download even when the manifest already has this month.

    Returns:
        The manifest record for the month.

    Raises:
        AcquisitionError: If the download or the conversion failed.
    """
    key = month_key(year, month)
    url = month_url(year, month)
    parquet_path = INTERIM_DIR / f"ontime_{year:04d}_{month:02d}.parquet"

    manifest = load_manifest()
    recorded = manifest["files"].get(key)
    if recorded and parquet_path.exists() and not force:
        LOGGER.info("%s already acquired (%s rows) - skipping", key, f"{recorded['rows']:,}")
        return MonthlyFile(**recorded)

    LOGGER.info("%s downloading %s", key, url)
    payload = _download(url)
    digest = hashlib.sha256(payload).hexdigest()

    frame = _to_frame(_extract_csv(payload))

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)

    record = MonthlyFile(
        url=url,
        sha256=digest,
        retrieved_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        rows=len(frame),
        columns=frame.shape[1],
        zip_bytes=len(payload),
        path=str(parquet_path.relative_to(DATA_DIR.parent)).replace("\\", "/"),
    )

    manifest["files"][key] = asdict(record)
    save_manifest(manifest)

    LOGGER.info(
        "%s done - %s rows, %.1f MB zip, %s",
        key,
        f"{record.rows:,}",
        record.zip_bytes / 1e6,
        parquet_path.name,
    )
    return record


# ------------------------------------------------------------------------ entry point --
def main(argv: list[str] | None = None) -> int:
    """Acquire every month in the requested range. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Download BTS on-time performance data and convert it to Parquet.",
    )
    parser.add_argument("--start", required=True, help="First month, YYYY-MM (e.g. 2025-07)")
    parser.add_argument("--end", required=True, help="Last month, YYYY-MM (e.g. 2026-06)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download months already in the manifest",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        months = month_range(parse_month(args.start), parse_month(args.end))
    except AcquisitionError as exc:
        LOGGER.error("%s", exc)
        return 2

    LOGGER.info("Acquiring %d month(s) into %s", len(months), INTERIM_DIR)

    total_rows = 0
    for year, month in months:
        try:
            record = acquire_month(year, month, force=args.force)
        except AcquisitionError as exc:
            LOGGER.error("%s failed: %s", month_key(year, month), exc)
            return 1
        total_rows += record.rows

    LOGGER.info(
        "Finished: %d month(s), %s rows, manifest at %s",
        len(months),
        f"{total_rows:,}",
        MANIFEST_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
