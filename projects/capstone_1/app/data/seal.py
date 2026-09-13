"""The sealed holdout (R2).

The final two months are carved out before any modelling and their row keys
hashed. :func:`development_frame` is the only sanctioned loader for every stage
downstream, so no stage can see the holdout by accident. Opening it is explicit,
one-way, and recorded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd
import pyarrow.parquet as pq

from app.config import CLEAN_PARQUET, HOLDOUT_MONTHS, SEAL_PATH


class SealViolation(RuntimeError):  # noqa: N818 - name fixed by the task-3 interface spec
    """Raised when the holdout hash does not match, or it is opened twice."""


def holdout_hash(row_keys: Iterable[str]) -> str:
    """Return the SHA-256 of the sorted row keys, newline-separated.

    Sorting makes the hash independent of row order, so a reordered frame does
    not look like a tampered one.

    Args:
        row_keys: The holdout's identifiers.

    Returns:
        A 64-character hex digest.
    """
    return hashlib.sha256("\n".join(sorted(row_keys)).encode("utf-8")).hexdigest()


def _read_seal() -> dict:
    return json.loads(SEAL_PATH.read_text(encoding="utf-8"))


def _require_clean_parquet() -> None:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"flights_clean.parquet not found at {CLEAN_PARQUET}.\n"
            f"Run:  python -m app.cli prepare"
        )


def _read_clean(columns: list[str] | None = None) -> pd.DataFrame:
    """Read the cleaned parquet, deriving ``period`` when the file predates it.

    The parquet built by the original notebook carries ``FlightDate`` but no
    ``period``. Deriving it here reproduces exactly what ``prepare.clean_frame``
    writes, so the sealed split is identical either way.

    Args:
        columns: Optional column subset. ``period`` (or ``FlightDate``, when the
            file has no ``period``) is always included.

    Returns:
        The cleaned frame with a ``period`` column present.
    """
    _require_clean_parquet()
    if columns is not None:
        available = set(pq.ParquetFile(CLEAN_PARQUET).schema_arrow.names)
        wanted = set(columns) | ({"period"} if "period" in available else {"FlightDate"})
        columns = [c for c in wanted if c in available]
    frame = pd.read_parquet(CLEAN_PARQUET, columns=columns)
    if "period" not in frame.columns:
        frame["period"] = frame.FlightDate.dt.strftime("%Y-%m")
    return frame


def development_frame(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the cleaned data with every holdout month removed.

    This is the only loader `features`, `train` and `evaluate` may use.

    Args:
        columns: Optional column subset; ``period`` is always included.

    Returns:
        The development rows only.

    Raises:
        FileNotFoundError: If the cleaned parquet has not been built.
    """
    frame = _read_clean(columns)
    return frame[~frame.period.isin(HOLDOUT_MONTHS)].copy()


def holdout_frame(*, unseal: bool = False) -> pd.DataFrame:
    """Open the sealed holdout. Once, deliberately, and recorded (R2).

    Args:
        unseal: Must be True. The keyword exists so the call site reads as a
            decision rather than an ordinary load.

    Returns:
        The holdout rows.

    Raises:
        SealViolation: If not explicitly unsealed, or if already opened.
    """
    if not unseal:
        raise SealViolation(
            "The holdout is sealed (R2). It opens exactly once, on day 20. "
            "Pass unseal=True only when that is what you are doing."
        )
    seal = _read_seal()
    if seal.get("opened"):
        raise SealViolation(
            f"The holdout was already opened at {seal.get('opened_utc')}. "
            f"R2 allows exactly one opening; a second would not be a holdout."
        )
    frame = _read_clean()
    out = frame[frame.period.isin(HOLDOUT_MONTHS)].copy()

    seal["opened"] = True
    seal["opened_utc"] = datetime.now(UTC).isoformat()
    SEAL_PATH.write_text(json.dumps(seal, indent=2), encoding="utf-8")
    return out


def verify() -> bool:
    """Check the holdout's row keys still hash to the sealed value.

    Returns:
        True when the hash matches.

    Raises:
        SealViolation: On mismatch - the data or the split changed, and either
            invalidates every result. There is no force option.
    """
    seal = _read_seal()
    frame = _read_clean(["row_key", "period"])
    keys = frame.loc[frame.period.isin(HOLDOUT_MONTHS), "row_key"]

    actual = holdout_hash(keys)
    if actual != seal["holdout_sha256"]:
        raise SealViolation(
            f"Holdout hash mismatch.\n"
            f"  sealed: {seal['holdout_sha256']}\n"
            f"  actual: {actual}\n"
            f"  rows sealed: {seal['holdout_rows']:,}, found: {len(keys):,}\n"
            f"The data or the split changed. Every downstream result is void."
        )
    return True
