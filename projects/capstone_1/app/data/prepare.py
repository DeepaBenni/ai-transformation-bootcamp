"""Turn the monthly BTS parquet files into one cleaned modelling frame.

This is notebook 01's logic as a module. It drops flights that never operated,
because a cancelled flight has no arrival delay to predict, and repairs the one
impossible schedule BTS ships.

Read/write asymmetry: ``config.CLEAN_PARQUET`` may resolve into
``capstone-1-signalcraft/`` - the user's own, read-only reference data - because
``config.resolve_data_dir`` falls back there when this project has not built its
own copy yet. :func:`run` must never write there, so it always writes to
``LOCAL_CLEAN_PARQUET``, a project-local path, guarded so a write inside the
read-only folder fails loudly instead of silently overwriting 141 MB someone
else owns. Because ``resolve_data_dir`` prefers the project-local directory once
it holds a processed parquet, the *next* process to import ``app.config`` reads
back exactly what this module just wrote - the asymmetry self-heals after one
run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from app.config import INTERIM_DIR, PROJECT_ROOT

LOCAL_CLEAN_PARQUET: Final[Path] = PROJECT_ROOT / "data" / "processed" / "flights_clean.parquet"


def add_row_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the composite identifier the sealed holdout is hashed over."""
    frame["row_key"] = (
        frame.FlightDate.dt.strftime("%Y-%m-%d")
        + "|"
        + frame.Reporting_Airline
        + "|"
        + frame.Flight_Number_Reporting_Airline.astype(str)
        + "|"
        + frame.Origin
        + "|"
        + frame.Dest
        + "|"
        + frame.CRSDepTime.astype(str)
    )
    return frame


def clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop flights that never operated and repair impossible schedules.

    Args:
        raw: The concatenated monthly files.

    Returns:
        Operated flights only, with ``row_key``, ``period`` and ``label``.
    """
    frame = add_row_key(raw)
    frame["period"] = frame.FlightDate.dt.strftime("%Y-%m")

    operated = frame.Cancelled.eq(0) & frame.Diverted.eq(0) & frame.ArrDel15.notna()
    frame = frame.loc[operated].copy()

    frame["label"] = frame.ArrDel15.astype("int8")
    frame = frame.drop(columns=["ArrDel15", "Cancelled", "Diverted"])

    # One row in the 12-month download schedules a negative duration.
    frame.loc[frame.CRSElapsedTime <= 0, "CRSElapsedTime"] = np.nan
    return frame


def run() -> Path:
    """Build the cleaned parquet from every monthly file.

    Writes to :data:`LOCAL_CLEAN_PARQUET`, never to ``config.CLEAN_PARQUET`` -
    see the module docstring for why those can differ.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If no monthly files exist.
        RuntimeError: If the destination would land inside the read-only
            ``capstone-1-signalcraft`` source folder.
    """
    files = sorted(INTERIM_DIR.glob("ontime_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No monthly files in {INTERIM_DIR}.\n"
            f"Run:  python -m app.cli acquire --start 2025-07 --end 2026-06"
        )
    raw = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    frame = clean_frame(raw)

    if "capstone-1-signalcraft" in LOCAL_CLEAN_PARQUET.parts:
        raise RuntimeError(
            f"Refusing to write inside the read-only source folder: {LOCAL_CLEAN_PARQUET}"
        )

    LOCAL_CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(LOCAL_CLEAN_PARQUET, index=False)
    return LOCAL_CLEAN_PARQUET
