"""Paths and constants for the SignalCraft pipeline.

Data location is configurable so the project can read the parquet prepared in
``capstone-1-signalcraft/`` without duplicating 141 MB. Set
``SIGNALCRAFT_DATA_DIR`` to point somewhere else - running
``python -m app.cli acquire`` populates a local tree under ``data/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parents[1]

# The sibling learning folder that already holds a prepared parquet.
FALLBACK_DATA_DIR: Final[Path] = REPO_ROOT / "capstone-1-signalcraft" / "data"

DATA_ENV_VAR: Final[str] = "SIGNALCRAFT_DATA_DIR"


def resolve_data_dir() -> Path:
    """Return the data directory, honouring ``SIGNALCRAFT_DATA_DIR``.

    Returns:
        The configured directory, else the project's own ``data/`` if it holds a
        prepared parquet, else the sibling folder's ``data/``.
    """
    override = os.environ.get(DATA_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    local = PROJECT_ROOT / "data"
    if (local / "processed" / "flights_clean.parquet").exists():
        return local
    return FALLBACK_DATA_DIR


DATA_DIR: Final[Path] = resolve_data_dir()
INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
MANIFEST_PATH: Final[Path] = DATA_DIR / "manifest.json"

CLEAN_PARQUET: Final[Path] = PROCESSED_DIR / "flights_clean.parquet"
FEATURES_PARQUET: Final[Path] = PROJECT_ROOT / "data" / "processed" / "flights_features.parquet"

MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
MODEL_PATH: Final[Path] = MODEL_DIR / "flight_delay_model.pkl"
LOOKUP_PATH: Final[Path] = MODEL_DIR / "lookup_tables.pkl"
METRICS_PATH: Final[Path] = MODEL_DIR / "metrics.json"

FIGURES_DIR: Final[Path] = PROJECT_ROOT / "reports" / "figures"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
SEAL_PATH: Final[Path] = PROJECT_ROOT / "holdout_seal.json"

# --- the sealed holdout, R2 -------------------------------------------------
HOLDOUT_MONTHS: Final[tuple[str, str]] = ("2026-05", "2026-06")

DEV_MONTHS: Final[tuple[str, ...]] = (
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
)

# History features are expanding means over prior months, so the first five
# months are training-only - validating there would score mostly-null features.
FOLD_VALIDATION_MONTHS: Final[tuple[str, ...]] = (
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
)

# --- modelling constants ----------------------------------------------------
PRIOR: Final[float] = 0.2242
ALPHA: Final[int] = 20
LATE_THRESHOLD_MIN: Final[int] = 15
PREDICTION_TIME: Final[str] = "T-24h"
RANDOM_STATE: Final[int] = 42
