# Capstone 1 SignalCraft (Track A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a served, explained flight late-arrival risk model in `projects/capstone_1` that meets R1–R11 of the Capstone 1 brief, porting what is reusable from `capstone-1-signalcraft/` and building the modelling half that does not exist there.

**Architecture:** A stage-based CLI (`acquire | prepare | seal | features | train | evaluate | report`) over an `app/` package, with file artefacts between stages so an expensive middle step is never repeated after a late failure. Serving is Streamlit over a `predict.py` that loads a pickled artefact plus lookup tables. Notebooks are generated last, executed, importing from `app/` — never in the execution path.

**Tech Stack:** Python 3.12, pandas, pyarrow, scikit-learn, LightGBM, SHAP, Streamlit, joblib, pytest, ruff, black.

**Spec:** `docs/superpowers/specs/2026-09-13-capstone-1-signalcraft-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.12.** `target-version = "py312"`.
- **Line length 100.** `ruff` lint select `["E", "F", "I", "N", "UP", "B", "D", "ANN"]`, ignore `["D203", "D213"]`, Google docstring convention. `tests/*` ignores `["D", "ANN"]`.
- **`HOLDOUT_MONTHS = ("2026-05", "2026-06")` is never read** by `features`, `train` or `evaluate`. Only `app/data/seal.py::development_frame()` loads data for those stages, and it filters those months out. The seal file must stay `"opened": false`.
- **Feature order is a contract.** `artefact["features"]` is the ordered list; any frame handed to the model is reindexed to it. A model given columns in another order returns confident nonsense with no error.
- **Every score is reported as a lift multiple**, never bare. Baseline is recomputed per fold because the positive rate moves across months.
- **Nothing raw or generated is committed** — `data/`, `models/*.pkl`, `reports/figures/*.png` are gitignored.
- **Fixed constants:** smoothing `ALPHA = 20`, global prior `PRIOR = 0.2242`, late threshold 15 minutes, prediction time T-24h.
- **All tests run on synthetic fixtures.** No test reads `flights_clean.parquet` or requires a trained model.
- **Commit after every task.** No Claude co-author trailers in commit messages.
- **Source folder `capstone-1-signalcraft/` is read-only.** Copy out of it; never write into it.

---

### Task 1: Project scaffold and configuration

**Files:**
- Create: `projects/capstone_1/app/__init__.py`, `app/data/__init__.py`, `app/features/__init__.py`, `app/models/__init__.py`, `app/explain/__init__.py`, `app/serving/__init__.py`
- Create: `projects/capstone_1/app/config.py`
- Create: `projects/capstone_1/pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`
- Create: `projects/capstone_1/tests/__init__.py`, `tests/conftest.py`
- Test: `projects/capstone_1/tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `app.config` exposing `DATA_DIR: Path`, `INTERIM_DIR: Path`, `PROCESSED_DIR: Path`, `MODEL_DIR: Path`, `FIGURES_DIR: Path`, `CLEAN_PARQUET: Path`, `FEATURES_PARQUET: Path`, `MODEL_PATH: Path`, `LOOKUP_PATH: Path`, `SEAL_PATH: Path`, `HOLDOUT_MONTHS: tuple[str, str]`, `PRIOR: float`, `ALPHA: int`, `DEV_MONTHS: tuple[str, ...]`, `FOLD_VALIDATION_MONTHS: tuple[str, ...]`

- [ ] **Step 1: Create the directory tree**

```bash
cd "projects/capstone_1"
mkdir -p app/data app/features app/models app/explain app/serving
mkdir -p tests docs notebooks reports/figures models data/interim data/processed
for d in app app/data app/features app/models app/explain app/serving tests; do
  touch "$d/__init__.py"
done
touch reports/figures/.gitkeep models/.gitkeep
```

- [ ] **Step 2: Write the failing test**

Create `projects/capstone_1/tests/test_config.py`:

```python
import os
from pathlib import Path

import pytest

from app import config


def test_holdout_months_are_the_sealed_pair():
    assert config.HOLDOUT_MONTHS == ("2026-05", "2026-06")


def test_development_months_exclude_the_holdout():
    assert set(config.DEV_MONTHS).isdisjoint(config.HOLDOUT_MONTHS)
    assert len(config.DEV_MONTHS) == 10


def test_fold_validation_months_are_the_last_five_development_months():
    assert config.FOLD_VALIDATION_MONTHS == (
        "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    )


def test_prior_and_alpha_are_the_documented_constants():
    assert config.PRIOR == 0.2242
    assert config.ALPHA == 20


def test_data_dir_honours_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNALCRAFT_DATA_DIR", str(tmp_path))
    reloaded = config.resolve_data_dir()
    assert reloaded == tmp_path


def test_data_dir_falls_back_when_env_var_absent(monkeypatch):
    monkeypatch.delenv("SIGNALCRAFT_DATA_DIR", raising=False)
    assert isinstance(config.resolve_data_dir(), Path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd projects/capstone_1 && python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 4: Write `app/config.py`**

```python
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
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04",
)

# History features are expanding means over prior months, so the first five
# months are training-only - validating there would score mostly-null features.
FOLD_VALIDATION_MONTHS: Final[tuple[str, ...]] = (
    "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
)

# --- modelling constants ----------------------------------------------------
PRIOR: Final[float] = 0.2242
ALPHA: Final[int] = 20
LATE_THRESHOLD_MIN: Final[int] = 15
PREDICTION_TIME: Final[str] = "T-24h"
RANDOM_STATE: Final[int] = 42
```

- [ ] **Step 5: Write `tests/conftest.py`**

```python
"""Puts the project root on the import path so ``import app`` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd projects/capstone_1 && python -m pytest tests/test_config.py -v`
Expected: 6 passed

- [ ] **Step 7: Write `pyproject.toml`**

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "D", "ANN"]
ignore = ["D203", "D213"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D", "ANN"]
"notebooks/*" = ["D", "ANN", "E402"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.black]
line-length = 100
target-version = ["py312"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 8: Write `.gitignore`**

```
# R1: no raw or derived data is committed. The manifest is the evidence.
/data/*
!/data/manifest.json

# trained artefacts - rebuild with: python -m app.cli train
models/*.pkl
models/metrics.json

# generated figures
reports/figures/*.png

# secrets
.env

# python
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
```

- [ ] **Step 9: Write `.env.example`**

```
# Where the prepared flight data lives. Leave unset to use the sibling
# capstone-1-signalcraft/data folder, or run `python -m app.cli acquire`
# to build a local copy under projects/capstone_1/data.
SIGNALCRAFT_DATA_DIR=
```

- [ ] **Step 10: Write `requirements.txt` and pin it**

Start from this list, install, then freeze:

```
pandas
pyarrow
numpy
requests
scikit-learn
lightgbm
shap
matplotlib
streamlit
joblib
pytest
ruff
black
```

Then:

```bash
cd projects/capstone_1
python -m pip install -r requirements.txt
python -m pip freeze > requirements.txt
```

R11 requires pinned versions. Unpinned, this file is a shopping list, not a lock.

- [ ] **Step 11: Commit**

```bash
git add projects/capstone_1
git commit -m "Scaffold capstone_1: package layout, config, tooling"
```

---

### Task 2: Port the acquisition module (R1)

**Files:**
- Create: `projects/capstone_1/app/data/acquire.py` (ported)
- Create: `projects/capstone_1/tests/test_acquire.py` (ported)
- Copy: `capstone-1-signalcraft/data/manifest.json` → `projects/capstone_1/data/manifest.json`

**Interfaces:**
- Consumes: `app.config.INTERIM_DIR`, `app.config.MANIFEST_PATH`
- Produces: `app.data.acquire` exposing `month_range(start: tuple[int,int], end: tuple[int,int]) -> list[tuple[int,int]]`, `month_url(year: int, month: int) -> str`, `parse_month(text: str) -> tuple[int,int]`, `acquire_month(year: int, month: int, *, force: bool = False) -> MonthlyFile`, `KEEP_COLUMNS: list[str]`, `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Copy the module and its tests**

```bash
cd "D:/AI Transformation Bootcamp Project/ai-transformation-bootcamp"
cp capstone-1-signalcraft/src/data/acquire.py projects/capstone_1/app/data/acquire.py
cp capstone-1-signalcraft/tests/test_acquire.py projects/capstone_1/tests/test_acquire.py
cp capstone-1-signalcraft/data/manifest.json projects/capstone_1/data/manifest.json
```

- [ ] **Step 2: Repoint the module's paths at `app.config`**

In `app/data/acquire.py`, delete these three lines:

```python
DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data"
INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
MANIFEST_PATH: Final[Path] = DATA_DIR / "manifest.json"
```

and replace with:

```python
from app.config import INTERIM_DIR, MANIFEST_PATH  # noqa: E402
```

placed with the other imports at the top of the file.

- [ ] **Step 3: Repoint the test's import**

In `tests/test_acquire.py`, change every `from src.data.acquire import ...` to `from app.data.acquire import ...`.

- [ ] **Step 4: Run the ported tests**

Run: `cd projects/capstone_1 && python -m pytest tests/test_acquire.py -v`
Expected: 10 passed — month ranges, year-boundary crossing, reversed range rejected, malformed month rejected, URL omits the leading zero, no banned column read, row-key columns present, label flags present, no duplicate keep-columns

- [ ] **Step 5: Verify the CLI entry point responds**

Run: `cd projects/capstone_1 && python -m app.data.acquire --help`
Expected: argparse usage text, exit 0. Does **not** download anything.

- [ ] **Step 6: Commit**

```bash
git add projects/capstone_1
git commit -m "Port BTS acquisition module and its tests (R1)"
```

---

### Task 3: Prepare and seal (R1, R2)

**Files:**
- Create: `projects/capstone_1/app/data/prepare.py`
- Create: `projects/capstone_1/app/data/seal.py`
- Create: `projects/capstone_1/holdout_seal.json` (copied from the reference)
- Test: `projects/capstone_1/tests/test_seal.py`

**Interfaces:**
- Consumes: `app.config` paths and `HOLDOUT_MONTHS`
- Produces:
  - `app.data.prepare.clean_frame(raw: pd.DataFrame) -> pd.DataFrame` — adds `row_key`, `period`, `label`; drops cancelled/diverted/unlabelled; sets non-positive `CRSElapsedTime` to NaN
  - `app.data.prepare.run() -> Path` — reads `INTERIM_DIR/ontime_*.parquet`, writes `CLEAN_PARQUET`
  - `app.data.seal.holdout_hash(row_keys: Iterable[str]) -> str` — SHA-256 over sorted keys
  - `app.data.seal.verify() -> bool`
  - `app.data.seal.development_frame(columns: list[str] | None = None) -> pd.DataFrame` — **the only sanctioned loader**
  - `app.data.seal.SealViolation(RuntimeError)`

- [ ] **Step 1: Copy the seal file**

```bash
cp capstone-1-signalcraft/holdout_seal.reference.json projects/capstone_1/holdout_seal.json
```

- [ ] **Step 2: Write the failing test**

Create `projects/capstone_1/tests/test_seal.py`:

```python
import pandas as pd
import pytest

from app import config
from app.data import seal


def _frame():
    return pd.DataFrame({
        "row_key": ["a", "b", "c", "d"],
        "period": ["2026-04", "2026-04", "2026-05", "2026-06"],
        "label": [0, 1, 1, 0],
    })


def test_hash_is_stable_regardless_of_input_order():
    assert seal.holdout_hash(["b", "a"]) == seal.holdout_hash(["a", "b"])


def test_hash_changes_when_a_key_changes():
    assert seal.holdout_hash(["a", "b"]) != seal.holdout_hash(["a", "c"])


def test_development_frame_never_returns_a_holdout_month(tmp_path, monkeypatch):
    parquet = tmp_path / "flights_clean.parquet"
    _frame().to_parquet(parquet, index=False)
    monkeypatch.setattr(config, "CLEAN_PARQUET", parquet)
    monkeypatch.setattr(seal, "CLEAN_PARQUET", parquet)

    out = seal.development_frame()

    assert set(out.period) == {"2026-04"}
    assert not set(out.period) & set(config.HOLDOUT_MONTHS)


def test_development_frame_drops_every_sealed_row(tmp_path, monkeypatch):
    parquet = tmp_path / "flights_clean.parquet"
    _frame().to_parquet(parquet, index=False)
    monkeypatch.setattr(seal, "CLEAN_PARQUET", parquet)

    assert len(seal.development_frame()) == 2


def test_missing_parquet_names_the_command_that_builds_it(tmp_path, monkeypatch):
    monkeypatch.setattr(seal, "CLEAN_PARQUET", tmp_path / "absent.parquet")
    with pytest.raises(FileNotFoundError, match="app.cli prepare"):
        seal.development_frame()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd projects/capstone_1 && python -m pytest tests/test_seal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.data.seal'`

- [ ] **Step 4: Write `app/data/seal.py`**

```python
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

from app.config import CLEAN_PARQUET, HOLDOUT_MONTHS, SEAL_PATH


class SealViolation(RuntimeError):
    """Raised when the holdout hash does not match, or it is opened twice."""


def holdout_hash(row_keys: Iterable[str]) -> str:
    """Return the SHA-256 of the sorted row keys.

    Sorting makes the hash independent of row order, so a reordered frame does
    not look like a tampered one.

    Args:
        row_keys: The holdout's identifiers.

    Returns:
        A 64-character hex digest.
    """
    digest = hashlib.sha256()
    for key in sorted(row_keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_seal() -> dict:
    return json.loads(SEAL_PATH.read_text(encoding="utf-8"))


def _require_clean_parquet() -> None:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"flights_clean.parquet not found at {CLEAN_PARQUET}.\n"
            f"Run:  python -m app.cli prepare"
        )


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
    _require_clean_parquet()
    if columns is not None and "period" not in columns:
        columns = [*columns, "period"]
    frame = pd.read_parquet(CLEAN_PARQUET, columns=columns)
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
    _require_clean_parquet()
    frame = pd.read_parquet(CLEAN_PARQUET)
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
    _require_clean_parquet()
    seal = _read_seal()
    frame = pd.read_parquet(CLEAN_PARQUET, columns=["row_key", "period"])
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd projects/capstone_1 && python -m pytest tests/test_seal.py -v`
Expected: 5 passed

- [ ] **Step 6: Write `app/data/prepare.py`**

```python
"""Turn the monthly BTS parquet files into one cleaned modelling frame.

This is notebook 01's logic as a module. It drops flights that never operated,
because a cancelled flight has no arrival delay to predict, and repairs the one
impossible schedule BTS ships.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import CLEAN_PARQUET, INTERIM_DIR, PROCESSED_DIR


def add_row_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the composite identifier the sealed holdout is hashed over."""
    frame["row_key"] = (
        frame.FlightDate.dt.strftime("%Y-%m-%d")
        + "|" + frame.Reporting_Airline
        + "|" + frame.Flight_Number_Reporting_Airline.astype(str)
        + "|" + frame.Origin
        + "|" + frame.Dest
        + "|" + frame.CRSDepTime.astype(str)
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

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If no monthly files exist.
    """
    files = sorted(INTERIM_DIR.glob("ontime_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No monthly files in {INTERIM_DIR}.\n"
            f"Run:  python -m app.cli acquire --start 2025-07 --end 2026-06"
        )
    raw = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    frame = clean_frame(raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CLEAN_PARQUET, index=False)
    return CLEAN_PARQUET
```

Add `from pathlib import Path` to the imports.

- [ ] **Step 7: Verify the seal against the real data**

Run:

```bash
cd projects/capstone_1
python -c "from app.data import seal; print('seal verified:', seal.verify())"
```

Expected: `seal verified: True`.

If it raises `SealViolation`, **stop and report it**. It means the parquet does not match what was sealed, and nothing downstream is trustworthy until that is understood.

- [ ] **Step 8: Commit**

```bash
git add projects/capstone_1
git commit -m "Add prepare and sealed-holdout modules (R1, R2)"
```

---

### Task 4: The leakage guard and history encodings (R5)

**Files:**
- Create: `projects/capstone_1/app/features/forbidden.py`
- Create: `projects/capstone_1/app/features/history.py`
- Test: `projects/capstone_1/tests/test_leakage.py`

**Interfaces:**
- Consumes: `app.config.ALPHA`, `app.config.PRIOR`
- Produces:
  - `app.features.forbidden.FORBIDDEN: frozenset[str]` — 28 column names
  - `app.features.forbidden.assert_no_leakage(columns: Iterable[str]) -> None` — raises `LeakageError`
  - `app.features.forbidden.LeakageError(AssertionError)`
  - `app.features.history.expanding_rate(frame: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame` — adds `<name>` and `<name>_n`, computed over **strictly prior periods only**

- [ ] **Step 1: Write the failing test — leakage is the headline risk**

Create `projects/capstone_1/tests/test_leakage.py`:

```python
import pandas as pd
import pytest

from app.features import history
from app.features.forbidden import FORBIDDEN, LeakageError, assert_no_leakage


def _tiny_frame():
    """Four periods, one route, alternating labels - hand-checkable."""
    return pd.DataFrame({
        "period": ["2026-01", "2026-02", "2026-03", "2026-04"],
        "route": ["ATL-ORD"] * 4,
        "label": [1, 1, 0, 0],
    })


def test_forbidden_set_is_not_empty():
    """A guard that guards nothing is worse than none, because it reassures."""
    assert len(FORBIDDEN) >= 27


def test_departure_actuals_are_forbidden():
    for column in ("DepDelay", "DepDel15", "ArrDelay", "TaxiOut", "CarrierDelay"):
        assert column in FORBIDDEN


def test_assert_no_leakage_rejects_a_forbidden_column():
    with pytest.raises(LeakageError, match="DepDelay"):
        assert_no_leakage(["Distance", "DepDelay", "dep_hour"])


def test_assert_no_leakage_accepts_a_clean_column_list():
    assert_no_leakage(["Distance", "dep_hour", "route_late_rate"]) is None


def test_history_feature_ignores_future_months():
    """A past row's history must not move when a future row's label flips.

    This is the property that separates a fold-safe expanding mean from a
    whole-dataset group mean. A group mean fails it immediately.
    """
    frame = _tiny_frame()
    before = history.expanding_rate(frame.copy(), ["route"], "route_late_rate")

    tampered = _tiny_frame()
    tampered.loc[tampered.period == "2026-04", "label"] = 1
    after = history.expanding_rate(tampered, ["route"], "route_late_rate")

    past = before.period < "2026-04"
    pd.testing.assert_series_equal(
        before.loc[past, "route_late_rate"],
        after.loc[past, "route_late_rate"],
    )


def test_first_period_has_no_history():
    """Nothing precedes the first period, so its rate is the prior and n is 0."""
    out = history.expanding_rate(_tiny_frame(), ["route"], "route_late_rate")
    first = out[out.period == "2026-01"].iloc[0]
    assert first.route_late_rate_n == 0
    assert first.route_late_rate == pytest.approx(0.2242)


def test_rate_uses_only_prior_periods():
    """Period 3 sees periods 1-2 only: 2 flights, both late, smoothed by alpha."""
    out = history.expanding_rate(_tiny_frame(), ["route"], "route_late_rate")
    third = out[out.period == "2026-03"].iloc[0]

    expected = (2 * 1.0 + 20 * 0.2242) / (2 + 20)
    assert third.route_late_rate == pytest.approx(expected)
    assert third.route_late_rate_n == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd projects/capstone_1 && python -m pytest tests/test_leakage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.features.forbidden'`

- [ ] **Step 3: Write `app/features/forbidden.py`**

```python
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


FORBIDDEN: Final[frozenset[str]] = frozenset({
    # departure actuals
    "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15", "DepartureDelayGroups",
    # in-flight actuals
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn", "AirTime", "ActualElapsedTime",
    # arrival actuals
    "ArrTime", "ArrDelay", "ArrDelayMinutes", "ArrivalDelayGroups",
    # post-hoc cause attribution
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    # outcome flags
    "Cancelled", "CancellationCode", "Diverted", "DivAirportLandings",
    # gate-return detail
    "FirstDepTime", "TotalAddGTime", "LongestAddGTime",
})


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
```

- [ ] **Step 4: Write `app/features/history.py`**

```python
"""Backward-looking rate features, computed without look-ahead.

The naive version - a group mean over the whole dataset - leaks the label into
its own predictor. These are expanding means over strictly prior periods, so a
row's feature depends only on months that had already happened when a dispatcher
would have needed it.

Thin groups are smoothed toward the global prior: a route with three flights
should not claim a 0.0 late rate.
"""

from __future__ import annotations

import pandas as pd

from app.config import ALPHA, PRIOR


def expanding_rate(
    frame: pd.DataFrame,
    keys: list[str],
    name: str,
    *,
    alpha: int = ALPHA,
    prior: float = PRIOR,
) -> pd.DataFrame:
    """Add a smoothed late rate over strictly prior periods, and its count.

    Args:
        frame: Must carry ``period`` and ``label``, plus every column in ``keys``.
        keys: The grouping columns, e.g. ``["route"]``.
        name: Output column name; the count lands in ``f"{name}_n"``.
        alpha: Smoothing weight toward ``prior``.
        prior: The global positive rate a thin group is shrunk toward.

    Returns:
        ``frame`` with two columns added.
    """
    periods = sorted(frame.period.unique())
    rate_col = pd.Series(prior, index=frame.index, dtype="float64")
    count_col = pd.Series(0.0, index=frame.index, dtype="float64")

    running_sum: dict[tuple, float] = {}
    running_count: dict[tuple, float] = {}

    for period in periods:
        mask = frame.period == period

        # Score this period from what was known BEFORE it. Nothing from
        # `period` itself has been folded in yet.
        if running_count:
            group_keys = list(
                frame.loc[mask, keys].itertuples(index=False, name=None)
            )
            rates, counts = [], []
            for key in group_keys:
                n = running_count.get(key, 0.0)
                s = running_sum.get(key, 0.0)
                rates.append((s + alpha * prior) / (n + alpha))
                counts.append(n)
            rate_col.loc[mask] = rates
            count_col.loc[mask] = counts

        # Only now does this period join the history.
        grouped = frame.loc[mask].groupby(keys, observed=True).label.agg(["sum", "count"])
        for key, row in grouped.iterrows():
            key_tuple = key if isinstance(key, tuple) else (key,)
            running_sum[key_tuple] = running_sum.get(key_tuple, 0.0) + float(row["sum"])
            running_count[key_tuple] = running_count.get(key_tuple, 0.0) + float(row["count"])

    frame[name] = rate_col
    frame[f"{name}_n"] = count_col
    return frame
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd projects/capstone_1 && python -m pytest tests/test_leakage.py -v`
Expected: 7 passed

The two that matter most: `test_history_feature_ignores_future_months` and `test_rate_uses_only_prior_periods`. If either fails, do not proceed — every downstream number would be inflated by leakage.

- [ ] **Step 6: Commit**

```bash
git add projects/capstone_1
git commit -m "Add leakage guard and look-ahead-free history encodings (R5)"
```

---

### Task 5: The 24 features (R5)

**Files:**
- Create: `projects/capstone_1/app/features/build.py`
- Test: `projects/capstone_1/tests/test_features.py`

**Interfaces:**
- Consumes: `app.features.history.expanding_rate`, `app.features.forbidden.assert_no_leakage`, `app.data.seal.development_frame`
- Produces:
  - `app.features.build.FEATURE_COLUMNS: list[str]` — the 24 names **in order**
  - `app.features.build.HOLIDAYS: list[pd.Timestamp]`
  - `app.features.build.add_schedule_features(frame: pd.DataFrame) -> pd.DataFrame`
  - `app.features.build.add_history_features(frame: pd.DataFrame) -> pd.DataFrame`
  - `app.features.build.build(frame: pd.DataFrame) -> pd.DataFrame`
  - `app.features.build.run() -> Path` — writes `FEATURES_PARQUET`

**The exact 24, in order.** This list is the artefact contract and `predict.py` depends on these names verbatim:

```
Distance, CRSElapsedTime, DayOfWeek, Month, Quarter, DistanceGroup,
dep_hour, dep_minute_of_day, arr_hour, is_red_eye, is_weekend,
sched_speed_mph, days_to_holiday, is_holiday_window,
origin_bank_size, dest_bank_size, tail_leg_number,
route_late_rate, carrier_late_rate, origin_late_rate, dest_late_rate,
carrier_origin_late_rate, origin_hour_late_rate, route_late_rate_n
```

- [ ] **Step 1: Write the failing test**

Create `projects/capstone_1/tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd projects/capstone_1 && python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.features.build'`

- [ ] **Step 3: Write `app/features/build.py`**

```python
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
    pd.Timestamp(d) for d in (
        "2025-07-04", "2025-09-01", "2025-11-27", "2025-11-28", "2025-12-24",
        "2025-12-25", "2025-12-31", "2026-01-01", "2026-01-19", "2026-02-16",
        "2026-05-25", "2026-07-03", "2026-07-04",
    )
]

FEATURE_COLUMNS: Final[list[str]] = [
    # raw, straight from the timetable
    "Distance", "CRSElapsedTime", "DayOfWeek", "Month", "Quarter", "DistanceGroup",
    # derived from the schedule
    "dep_hour", "dep_minute_of_day", "arr_hour", "is_red_eye", "is_weekend",
    "sched_speed_mph", "days_to_holiday", "is_holiday_window",
    # derived from the timetable's shape
    "origin_bank_size", "dest_bank_size", "tail_leg_number",
    # derived from prior months only
    "route_late_rate", "carrier_late_rate", "origin_late_rate", "dest_late_rate",
    "carrier_origin_late_rate", "origin_hour_late_rate", "route_late_rate_n",
]


def _hour(hhmm: pd.Series) -> pd.Series:
    """Convert a BTS HHMM integer to an hour, mapping 2400 to 0."""
    return (hhmm // 100).mod(24).astype("int16")


def _minute(hhmm: pd.Series) -> pd.Series:
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
    dates = frame.FlightDate.to_numpy(dtype="datetime64[D]")
    holidays = np.array([h.to_datetime64() for h in HOLIDAYS], dtype="datetime64[D]")
    gaps = np.abs(dates[:, None] - holidays[None, :]).astype("timedelta64[D]")
    nearest = gaps.min(axis=1).astype("int32")
    frame["days_to_holiday"] = np.clip(nearest, 0, 7)
    frame["is_holiday_window"] = (nearest <= 2).astype("int8")

    # Congestion: how many flights share this airport and hour that day.
    frame["origin_bank_size"] = frame.groupby(
        ["FlightDate", "Origin", "dep_hour"], observed=True
    ).Origin.transform("size").astype("int16")
    frame["dest_bank_size"] = frame.groupby(
        ["FlightDate", "Dest", "arr_hour"], observed=True
    ).Dest.transform("size").astype("int16")

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
        "carrier_late_rate_n", "origin_late_rate_n", "dest_late_rate_n",
        "carrier_origin_late_rate_n", "origin_hour_late_rate_n",
    ]
    return frame.drop(columns=[c for c in drop if c in frame.columns])


def build(frame: pd.DataFrame) -> pd.DataFrame:
    """Build every feature and assert the result is leakage-free.

    Args:
        frame: The cleaned development flights.

    Returns:
        ``frame`` with all 24 features present.

    Raises:
        LeakageError: If any forbidden column survived.
    """
    frame = add_schedule_features(frame)
    frame = add_history_features(frame)
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].fillna(
        {c: PRIOR for c in FEATURE_COLUMNS if c.endswith("_rate")}
    )
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
        "row_key", "period", "label", "FlightDate", "Reporting_Airline",
        "Origin", "Dest", "route", *FEATURE_COLUMNS,
    ]
    FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].to_parquet(FEATURES_PARQUET, index=False)
    return FEATURES_PARQUET
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/capstone_1 && python -m pytest tests/test_features.py tests/test_leakage.py -v`
Expected: 19 passed

- [ ] **Step 5: Build the real feature set**

Run:

```bash
cd projects/capstone_1
python -c "
from app.features import build
p = build.run()
import pandas as pd
f = pd.read_parquet(p, columns=['period','label','route_late_rate_n'])
print(f'{len(f):,} rows written to {p.name}')
print('periods:', sorted(f.period.unique()))
print(f'rows with prior route history: {(f.route_late_rate_n > 0).mean():.3%}')
"
```

Expected: ~5,696,742 rows, ten periods, **none of them 2026-05 or 2026-06**, and a history-coverage figure near 91%.

- [ ] **Step 6: Commit**

```bash
git add projects/capstone_1
git commit -m "Build the 24 T-24h features with leakage assertion (R5)"
```

---

### Task 6: Metrics, baseline and threshold (R4, R7)

**Files:**
- Create: `projects/capstone_1/app/models/metrics.py`
- Create: `projects/capstone_1/app/models/threshold.py`
- Test: `projects/capstone_1/tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks beyond `app.config`
- Produces:
  - `app.models.metrics.baseline_pr_auc(y: np.ndarray) -> float` — the positive rate
  - `app.models.metrics.score(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]` — keys `pr_auc`, `baseline`, `lift`, `recall`, `precision`, `roc_auc`, `brier`
  - `app.models.threshold.sweep(y_true: np.ndarray, probabilities: np.ndarray, grid: Iterable[float]) -> pd.DataFrame` — columns `threshold`, `recall`, `precision`, `flagged_share`
  - `app.models.threshold.threshold_for_recall(y_true: np.ndarray, probabilities: np.ndarray, target_recall: float) -> float`

- [ ] **Step 1: Write the failing test**

Create `projects/capstone_1/tests/test_metrics.py`:

```python
import numpy as np
import pytest

from app.models import metrics, threshold


def test_baseline_is_the_positive_rate():
    y = np.array([0, 0, 0, 1])
    assert metrics.baseline_pr_auc(y) == pytest.approx(0.25)


def test_lift_is_pr_auc_over_baseline():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])          # perfect ranking
    out = metrics.score(y, p, threshold=0.5)
    assert out["baseline"] == pytest.approx(0.5)
    assert out["lift"] == pytest.approx(out["pr_auc"] / out["baseline"])
    assert out["lift"] > 1.0


def test_a_useless_ranker_lifts_about_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    p = rng.random(20_000)                       # no signal at all
    out = metrics.score(y, p, threshold=0.5)
    assert out["lift"] == pytest.approx(1.0, abs=0.05)


def test_recall_is_measured_at_the_given_threshold():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.3, 0.2, 0.1])
    assert metrics.score(y, p, threshold=0.5)["recall"] == pytest.approx(0.5)
    assert metrics.score(y, p, threshold=0.25)["recall"] == pytest.approx(1.0)


def test_threshold_for_recall_achieves_at_least_the_target():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.7, 0.4, 0.2, 0.35, 0.25, 0.15, 0.05])
    cut = threshold.threshold_for_recall(y, p, 0.75)
    achieved = (p >= cut)[y == 1].mean()
    assert achieved >= 0.75


def test_lower_target_recall_gives_a_higher_threshold():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.7, 0.4, 0.2, 0.35, 0.25, 0.15, 0.05])
    assert threshold.threshold_for_recall(y, p, 0.5) >= threshold.threshold_for_recall(y, p, 1.0)


def test_sweep_reports_the_share_of_schedule_flagged():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    table = threshold.sweep(y, p, [0.5])
    assert table.flagged_share.iloc[0] == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd projects/capstone_1 && python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.metrics'`

- [ ] **Step 3: Write `app/models/metrics.py`**

```python
"""Scoring, always relative to a baseline that moves.

R4 forbids reporting a bare score. The positive rate shifts month to month, so
the same PR-AUC means different things in different folds; only the multiple
over that fold's own baseline compares.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def baseline_pr_auc(y: np.ndarray) -> float:
    """Return the PR-AUC a no-skill ranker achieves: the positive rate.

    Args:
        y: Binary labels.

    Returns:
        The proportion of positives.
    """
    return float(np.mean(y))


def score(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Score predictions against this fold's own baseline.

    Args:
        y_true: Binary labels.
        probabilities: Predicted P(late).
        threshold: The operating point recall and precision are measured at.

    Returns:
        A mapping with pr_auc, baseline, lift, recall, precision, roc_auc, brier.
    """
    flagged = (probabilities >= threshold).astype(int)
    base = baseline_pr_auc(y_true)
    pr_auc = float(average_precision_score(y_true, probabilities))

    return {
        "pr_auc": pr_auc,
        "baseline": base,
        "lift": pr_auc / base if base else float("nan"),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }
```

- [ ] **Step 4: Write `app/models/threshold.py`**

```python
"""Choosing the operating point.

The operator states the recall the operation needs; the threshold follows. Doing
it the other way round - taking 0.5 because it is the default, then reporting
whatever recall falls out - produced 2.9% recall in the source analysis.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def sweep(
    y_true: np.ndarray, probabilities: np.ndarray, grid: Iterable[float]
) -> pd.DataFrame:
    """Tabulate recall, precision and how much of the schedule each cut flags.

    Args:
        y_true: Binary labels.
        probabilities: Predicted P(late).
        grid: Thresholds to evaluate.

    Returns:
        One row per threshold.
    """
    rows = []
    positives = y_true.sum()
    for cut in grid:
        flagged = probabilities >= cut
        hits = int((flagged & (y_true == 1)).sum())
        rows.append({
            "threshold": float(cut),
            "recall": hits / positives if positives else 0.0,
            "precision": hits / flagged.sum() if flagged.sum() else 0.0,
            "flagged_share": float(flagged.mean()),
        })
    return pd.DataFrame(rows)


def threshold_for_recall(
    y_true: np.ndarray, probabilities: np.ndarray, target_recall: float
) -> float:
    """Return the highest threshold still achieving ``target_recall``.

    Args:
        y_true: Binary labels.
        probabilities: Predicted P(late).
        target_recall: The recall the operation requires, 0-1.

    Returns:
        The threshold. Falls to the minimum probability if the target is
        unreachable, which flags everything and is reported as such.
    """
    positive_scores = np.sort(probabilities[y_true == 1])[::-1]
    if positive_scores.size == 0:
        return float(probabilities.min())

    index = int(np.ceil(target_recall * positive_scores.size)) - 1
    index = min(max(index, 0), positive_scores.size - 1)
    return float(positive_scores[index])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd projects/capstone_1 && python -m pytest tests/test_metrics.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add projects/capstone_1
git commit -m "Add baseline-relative metrics and threshold selection (R4, R7)"
```

---

### Task 7: Expanding-window CV and the model zoo (R6)

**Files:**
- Create: `projects/capstone_1/app/models/cv.py`
- Create: `projects/capstone_1/app/models/zoo.py`
- Test: `projects/capstone_1/tests/test_cv.py`

**Interfaces:**
- Consumes: `app.config.FOLD_VALIDATION_MONTHS`, `app.config.RANDOM_STATE`
- Produces:
  - `app.models.cv.Fold` — a frozen dataclass with `train_periods: tuple[str, ...]`, `validation_period: str`
  - `app.models.cv.expanding_folds(periods: Iterable[str], validation_periods: Iterable[str]) -> list[Fold]`
  - `app.models.zoo.model_zoo() -> dict[str, object]` — keys `linear`, `forest`, `boosted`

- [ ] **Step 1: Write the failing test**

Create `projects/capstone_1/tests/test_cv.py`:

```python
import pytest

from app.models import cv, zoo

PERIODS = [
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04",
]
VALIDATION = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]


def test_there_are_five_folds():
    assert len(cv.expanding_folds(PERIODS, VALIDATION)) == 5


def test_no_fold_trains_on_a_month_later_than_it_validates():
    """Random k-fold on a time series is a defect, not a shortcut (R6)."""
    for fold in cv.expanding_folds(PERIODS, VALIDATION):
        assert max(fold.train_periods) < fold.validation_period


def test_the_training_window_expands():
    folds = cv.expanding_folds(PERIODS, VALIDATION)
    sizes = [len(f.train_periods) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_the_first_fold_trains_on_the_burn_in_months():
    first = cv.expanding_folds(PERIODS, VALIDATION)[0]
    assert first.validation_period == "2025-12"
    assert first.train_periods == ("2025-07", "2025-08", "2025-09", "2025-10", "2025-11")


def test_validation_periods_never_overlap_training():
    for fold in cv.expanding_folds(PERIODS, VALIDATION):
        assert fold.validation_period not in fold.train_periods


def test_a_validation_period_absent_from_the_data_is_rejected():
    with pytest.raises(ValueError, match="2027-01"):
        cv.expanding_folds(PERIODS, ["2027-01"])


def test_zoo_offers_exactly_three_families():
    models = zoo.model_zoo()
    assert set(models) == {"linear", "forest", "boosted"}


def test_zoo_returns_unfitted_estimators():
    for model in zoo.model_zoo().values():
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd projects/capstone_1 && python -m pytest tests/test_cv.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.cv'`

- [ ] **Step 3: Write `app/models/cv.py`**

```python
"""Expanding-window cross-validation.

A random fold would train on June to predict January. Worse here than usual:
the history features are expanding means over prior months, so shuffled folds
would let a model see rates computed from its own validation rows. Every fold
trains strictly on the past.

The five windows also serve as the rolling backtest - stability across them is
reported rather than averaged away.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """One expanding window: everything before, validated on one month."""

    train_periods: tuple[str, ...]
    validation_period: str


def expanding_folds(
    periods: Iterable[str], validation_periods: Iterable[str]
) -> list[Fold]:
    """Build folds whose training window grows and never wraps.

    Args:
        periods: Every period present, ``YYYY-MM``.
        validation_periods: The periods to validate on, in order.

    Returns:
        One fold per validation period.

    Raises:
        ValueError: If a validation period is absent, or nothing precedes it.
    """
    ordered = sorted(set(periods))
    folds: list[Fold] = []

    for period in sorted(validation_periods):
        if period not in ordered:
            raise ValueError(f"validation period {period} is not in the data")
        train = tuple(p for p in ordered if p < period)
        if not train:
            raise ValueError(f"nothing precedes {period}; it cannot be validated")
        folds.append(Fold(train_periods=train, validation_period=period))
    return folds
```

- [ ] **Step 4: Write `app/models/zoo.py`**

```python
"""Three model families, so complexity has to justify itself (R6).

The random forest runs deliberately smaller than a tuned configuration would.
It is the slowest family and loses to boosting on every metric that matters, so
it is present to be compared, not to be shipped - and its reduced settings are
reported alongside its score rather than hidden.
"""

from __future__ import annotations

from typing import Final

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import RANDOM_STATE

# Reported in the results table so the comparison is not silently unfair.
FOREST_CONFIG: Final[dict[str, int]] = {"n_estimators": 100, "max_depth": 10}


def model_zoo() -> dict[str, object]:
    """Return one unfitted estimator per family.

    Returns:
        ``linear`` - the interpretable floor; if boosting cannot beat it,
        complexity is not earning its keep.
        ``forest`` - bagging, captures interactions without boosting.
        ``boosted`` - LightGBM, which handles the categorical explosion natively.
    """
    return {
        "linear": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "forest": RandomForestClassifier(
            n_estimators=FOREST_CONFIG["n_estimators"],
            max_depth=FOREST_CONFIG["max_depth"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "boosted": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd projects/capstone_1 && python -m pytest tests/test_cv.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add projects/capstone_1
git commit -m "Add expanding-window folds and the three-family zoo (R6)"
```

---

### Task 8: Train, cross-validate, export the artefact (R4, R6, R7)

**Files:**
- Create: `projects/capstone_1/app/models/train.py`
- Create: `projects/capstone_1/app/cli.py`
- Test: `projects/capstone_1/tests/test_train.py`

**Interfaces:**
- Consumes: `app.models.cv.expanding_folds`, `app.models.zoo.model_zoo`, `app.models.metrics.score`, `app.models.threshold.threshold_for_recall`, `app.features.build.FEATURE_COLUMNS`
- Produces:
  - `app.models.train.run_cv(frame: pd.DataFrame, *, validation_periods: list[str] | None = None, threshold: float = 0.30) -> pd.DataFrame` — one row per model×fold; `validation_periods` defaults to `config.FOLD_VALIDATION_MONTHS` and is overridden in tests
  - `app.models.train.summarise(results: pd.DataFrame) -> pd.DataFrame`
  - `app.models.train.fit_final(frame: pd.DataFrame, family: str) -> tuple[object, dict]`
  - `app.models.train.export(model, frame, cv_results, *, family: str = "boosted", target_recall: float = 0.60) -> Path`
  - `app.cli.main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Create `projects/capstone_1/tests/test_train.py`:

```python
import numpy as np
import pandas as pd
import pytest

from app.features.build import FEATURE_COLUMNS
from app.models import train


def _synthetic(rows_per_period=400):
    """Small frame with real signal, so a model can actually learn something."""
    rng = np.random.default_rng(7)
    periods = ["2026-01", "2026-02", "2026-03"]
    frames = []
    for period in periods:
        n = rows_per_period
        data = {col: rng.random(n) for col in FEATURE_COLUMNS}
        # A learnable rule: high origin_hour_late_rate means late.
        signal = data["origin_hour_late_rate"]
        data["label"] = (signal + rng.normal(0, 0.15, n) > 0.55).astype(int)
        data["period"] = period
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def test_run_cv_returns_one_row_per_model_and_fold():
    out = train.run_cv(_synthetic(), validation_periods=["2026-02", "2026-03"])
    assert len(out) == 3 * 2
    assert set(out.model) == {"linear", "forest", "boosted"}


def test_every_result_carries_a_lift_not_a_bare_score():
    out = train.run_cv(_synthetic(), validation_periods=["2026-03"])
    assert {"pr_auc", "baseline", "lift"} <= set(out.columns)
    assert (out.lift > 0).all()


def test_a_model_with_signal_beats_the_baseline():
    out = train.run_cv(_synthetic(), validation_periods=["2026-03"])
    assert out.lift.max() > 1.0


def test_summarise_reports_the_standard_deviation():
    out = train.run_cv(_synthetic(), validation_periods=["2026-02", "2026-03"])
    summary = train.summarise(out)
    assert "pr_auc_sd" in summary.columns
    assert len(summary) == 3


def test_export_writes_an_artefact_carrying_the_feature_order(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "model.pkl")
    frame = _synthetic()
    results = train.run_cv(frame, validation_periods=["2026-03"])
    model, _ = train.fit_final(frame, "boosted")

    path = train.export(model, frame, results)

    import joblib
    artefact = joblib.load(path)
    assert artefact["features"] == FEATURE_COLUMNS
    assert 0.0 < artefact["threshold"] < 1.0
    assert artefact["baseline_pr_auc"] > 0
    assert set(artefact["defaults"]) == set(FEATURE_COLUMNS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd projects/capstone_1 && python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.train'`

- [ ] **Step 3: Write `app/models/train.py`**

```python
"""Cross-validate the three families, fit the winner, export the artefact."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import FOLD_VALIDATION_MONTHS, MODEL_PATH, PREDICTION_TIME
from app.features.build import FEATURE_COLUMNS
from app.models.cv import expanding_folds
from app.models.metrics import baseline_pr_auc, score
from app.models.threshold import threshold_for_recall
from app.models.zoo import FOREST_CONFIG, model_zoo


def run_cv(
    frame: pd.DataFrame,
    *,
    validation_periods: list[str] | None = None,
    threshold: float = 0.30,
) -> pd.DataFrame:
    """Fit every family on every expanding fold.

    Args:
        frame: Features plus ``period`` and ``label``.
        validation_periods: Defaults to the five configured months.
        threshold: The operating point recall is measured at during CV.

    Returns:
        One row per model and fold, carrying pr_auc, baseline, lift and fit time.
    """
    validation_periods = list(validation_periods or FOLD_VALIDATION_MONTHS)
    folds = expanding_folds(frame.period.unique(), validation_periods)
    rows: list[dict[str, Any]] = []

    for fold in folds:
        train_mask = frame.period.isin(fold.train_periods)
        valid_mask = frame.period == fold.validation_period

        x_train = frame.loc[train_mask, FEATURE_COLUMNS]
        y_train = frame.loc[train_mask, "label"].to_numpy()
        x_valid = frame.loc[valid_mask, FEATURE_COLUMNS]
        y_valid = frame.loc[valid_mask, "label"].to_numpy()

        for name, model in model_zoo().items():
            started = time.perf_counter()
            model.fit(x_train, y_train)
            fit_s = time.perf_counter() - started

            probabilities = model.predict_proba(x_valid)[:, 1]
            result = score(y_valid, probabilities, threshold)
            rows.append({
                "model": name,
                "period": fold.validation_period,
                "train_months": len(fold.train_periods),
                "fit_s": round(fit_s, 1),
                **result,
            })
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Average each family across folds, keeping the spread.

    The standard deviation is not decoration: if one family's spread overlaps
    another's mean, the ranking between them is not established.

    Args:
        results: Output of :func:`run_cv`.

    Returns:
        One row per family.
    """
    grouped = results.groupby("model")
    summary = grouped.agg(
        pr_auc=("pr_auc", "mean"),
        pr_auc_sd=("pr_auc", "std"),
        lift=("lift", "mean"),
        recall=("recall", "mean"),
        precision=("precision", "mean"),
        roc_auc=("roc_auc", "mean"),
        brier=("brier", "mean"),
        fit_s=("fit_s", "mean"),
    )
    return summary.sort_values("lift", ascending=False).round(4)


def fit_final(frame: pd.DataFrame, family: str) -> tuple[Any, dict[str, Any]]:
    """Fit one family on every development row.

    Args:
        frame: Features plus ``period`` and ``label``.
        family: A key of :func:`app.models.zoo.model_zoo`.

    Returns:
        The fitted model and its training metadata.
    """
    model = model_zoo()[family]
    started = time.perf_counter()
    model.fit(frame[FEATURE_COLUMNS], frame.label.to_numpy())

    meta = {
        "rows": int(len(frame)),
        "period_start": str(frame.period.min()),
        "period_end": str(frame.period.max()),
        "fit_s": round(time.perf_counter() - started, 1),
    }
    return model, meta


def export(
    model: Any,
    frame: pd.DataFrame,
    cv_results: pd.DataFrame,
    *,
    family: str = "boosted",
    target_recall: float = 0.60,
) -> Path:
    """Write the model plus everything needed to use it six weeks from now.

    A model alone is a trap: nobody remembers which columns it wants, in what
    order, or which threshold was chosen.

    Args:
        model: The fitted estimator.
        frame: The development frame it was fitted on.
        cv_results: Output of :func:`run_cv`.
        family: Which family ``model`` belongs to.
        target_recall: The recall the operating point is chosen for.

    Returns:
        The artefact path.
    """
    last_period = frame.period.max()
    holdback = frame[frame.period == last_period]
    probabilities = model.predict_proba(holdback[FEATURE_COLUMNS])[:, 1]
    y_true = holdback.label.to_numpy()

    chosen = threshold_for_recall(y_true, probabilities, target_recall)
    family_rows = cv_results[cv_results.model == family]

    artefact = {
        "model": model,
        "model_family": family,
        "features": list(FEATURE_COLUMNS),
        "threshold": round(float(chosen), 4),
        "baseline_pr_auc": round(baseline_pr_auc(frame.label.to_numpy()), 4),
        "forest_config": FOREST_CONFIG,
        "validation": {
            "pr_auc": round(float(family_rows.pr_auc.mean()), 4),
            "pr_auc_sd": round(float(family_rows.pr_auc.std()), 4),
            "lift": round(float(family_rows.lift.mean()), 4),
            "folds": family_rows.period.tolist(),
        },
        "training": {
            "rows": int(len(frame)),
            "period_start": str(frame.period.min()),
            "period_end": str(last_period),
        },
        "defaults": {
            col: float(np.nanmedian(frame[col])) for col in FEATURE_COLUMNS
        },
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "notes": f"Predicts P(arrival 15+ min late) at {PREDICTION_TIME}.",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, MODEL_PATH, compress=3)
    return MODEL_PATH
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/capstone_1 && python -m pytest tests/test_train.py -v`
Expected: 5 passed

- [ ] **Step 5: Write `app/cli.py`**

```python
"""One entry point for every pipeline stage.

Each stage reads a file and writes a file, so a failure late in the run never
costs an expensive earlier step.
"""

from __future__ import annotations

import argparse
import sys

STAGES = ("acquire", "prepare", "seal", "features", "train", "evaluate", "report")


def main(argv: list[str] | None = None) -> int:
    """Dispatch a pipeline stage.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="SignalCraft: flight late-arrival risk (Track A).",
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--start", help="acquire: first month, YYYY-MM")
    parser.add_argument("--end", help="acquire: last month, YYYY-MM")
    parser.add_argument(
        "--target-recall", type=float, default=0.60,
        help="train: the recall the operating point is chosen for",
    )
    args = parser.parse_args(argv)

    if args.stage == "acquire":
        from app.data import acquire
        return acquire.main(["--start", args.start, "--end", args.end])

    if args.stage == "prepare":
        from app.data import prepare
        print(f"wrote {prepare.run()}")
        return 0

    if args.stage == "seal":
        from app.data import seal
        print(f"seal verified: {seal.verify()}")
        return 0

    if args.stage == "features":
        from app.features import build
        print(f"wrote {build.run()}")
        return 0

    if args.stage == "train":
        import pandas as pd

        from app.config import FEATURES_PARQUET
        from app.models import train

        if not FEATURES_PARQUET.exists():
            print(
                f"flights_features.parquet not found at {FEATURES_PARQUET}.\n"
                f"Run:  python -m app.cli features",
                file=sys.stderr,
            )
            return 1

        frame = pd.read_parquet(FEATURES_PARQUET)
        results = train.run_cv(frame)
        print(train.summarise(results).to_string())

        best = train.summarise(results).index[0]
        model, meta = train.fit_final(frame, best)
        path = train.export(model, frame, results, family=best,
                            target_recall=args.target_recall)
        print(f"\nshipped {best}: {path}  ({meta['rows']:,} rows)")
        return 0

    if args.stage == "evaluate":
        from app.explain import report
        return report.main()

    if args.stage == "report":
        from app.explain import docs
        return docs.main()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Verify the CLI dispatches**

Run: `cd projects/capstone_1 && python -m app.cli --help`
Expected: usage text listing all seven stages, exit 0.

- [ ] **Step 7: Run the real training**

```bash
cd projects/capstone_1
python -m app.cli train --target-recall 0.60 2>&1 | tee models/train_log.txt
```

Expected: a summary table with three rows, every `lift` above 1.0, and `shipped boosted: models/flight_delay_model.pkl`.

**Record the real numbers.** They will differ from the source documents' 0.3417 / 1.53×, which came from a run whose code is not available. Report what this run measured.

If any family's lift is at or below 1.0, that is a finding, not a failure — report it plainly rather than tuning until it looks better.

- [ ] **Step 8: Commit**

```bash
git add projects/capstone_1
git commit -m "Add CV training, artefact export and the stage CLI (R4, R6, R7)"
```

---

### Task 9: Serving — lookups, prediction, Streamlit (R9)

**Files:**
- Create: `projects/capstone_1/app/serving/lookups.py`
- Create: `projects/capstone_1/app/serving/predict.py` (ported)
- Create: `projects/capstone_1/streamlit_app.py` (ported)
- Test: `projects/capstone_1/tests/test_predict_validation.py` (ported), `tests/test_cold_start.py`

**Interfaces:**
- Consumes: `app.config.LOOKUP_PATH`, `app.config.FEATURES_PARQUET`, the exported artefact
- Produces:
  - `app.serving.lookups.build_lookups(frame: pd.DataFrame) -> dict[str, dict]` — keys `origin_hour`, `route`, `carrier`, `origin`, `dest`, `carrier_origin`, `route_n`
  - `app.serving.lookups.load() -> dict[str, dict]`
  - `app.serving.predict.predict_flight(...) -> Prediction`
  - `app.serving.predict.InvalidFlight`, `ModelNotFound`, `Prediction`, `model_info()`

- [ ] **Step 1: Write the failing test**

Create `projects/capstone_1/tests/test_cold_start.py`:

```python
import pandas as pd
import pytest

from app.config import PRIOR
from app.serving import lookups


def _features():
    return pd.DataFrame({
        "period": ["2026-03", "2026-03", "2026-04", "2026-04"],
        "Origin": ["ATL", "ATL", "ATL", "ORD"],
        "Dest": ["ORD", "ORD", "ORD", "ATL"],
        "route": ["ATL-ORD", "ATL-ORD", "ATL-ORD", "ORD-ATL"],
        "Reporting_Airline": ["DL", "DL", "DL", "AA"],
        "dep_hour": [17, 6, 17, 22],
        "origin_hour_late_rate": [0.31, 0.09, 0.33, 0.40],
        "route_late_rate": [0.28, 0.28, 0.30, 0.22],
        "carrier_late_rate": [0.18, 0.18, 0.19, 0.27],
        "origin_late_rate": [0.29, 0.29, 0.30, 0.29],
        "dest_late_rate": [0.29, 0.29, 0.29, 0.21],
        "carrier_origin_late_rate": [0.20, 0.20, 0.21, 0.26],
        "route_late_rate_n": [500.0, 500.0, 762.0, 300.0],
    })


def test_lookups_expose_the_seven_keys_predict_expects():
    out = lookups.build_lookups(_features())
    assert set(out) == {
        "origin_hour", "route", "carrier", "origin", "dest",
        "carrier_origin", "route_n",
    }


def test_lookups_are_built_from_the_latest_period_only():
    """A new flight should use the most recent backward-looking rates."""
    out = lookups.build_lookups(_features())
    assert out["route"]["ATL-ORD"] == pytest.approx(0.30)


def test_an_unseen_route_is_absent_so_the_caller_falls_back_to_the_prior():
    out = lookups.build_lookups(_features())
    assert out["route"].get("XXX-YYY") is None
    assert out["route"].get("XXX-YYY", PRIOR) == PRIOR


def test_an_unseen_route_has_no_evidence_count():
    out = lookups.build_lookups(_features())
    assert out["route_n"].get("XXX-YYY", 0.0) == 0.0


def test_a_known_route_reports_its_evidence_count():
    out = lookups.build_lookups(_features())
    assert out["route_n"]["ATL-ORD"] == pytest.approx(762.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd projects/capstone_1 && python -m pytest tests/test_cold_start.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.serving.lookups'`

- [ ] **Step 3: Write `app/serving/lookups.py`**

```python
"""The historical rate tables the interface needs at score time.

Six of the 24 features are averages over past flights. A duty manager types a
carrier and a route, not a rate, so those rates are looked up here. Without
this the strongest feature would be a constant and the app would be a toy.

An unseen route returns nothing, and the caller falls back to the global prior
with an evidence count of zero. That degrades honestly, because
``route_late_rate_n`` is itself a trained feature - the model already learned to
discount rates resting on thin evidence.
"""

from __future__ import annotations

import joblib
import pandas as pd

from app.config import FEATURES_PARQUET, LOOKUP_PATH

_cache: dict[str, dict] | None = None


def build_lookups(frame: pd.DataFrame) -> dict[str, dict]:
    """Derive the seven lookup tables from the latest period's rates.

    Args:
        frame: The feature dataset, carrying ``period`` and the rate columns.

    Returns:
        A mapping with keys origin_hour, route, carrier, origin, dest,
        carrier_origin and route_n.
    """
    latest = frame[frame.period == frame.period.max()]
    return {
        "origin_hour": latest.groupby(["Origin", "dep_hour"], observed=True)
                             .origin_hour_late_rate.mean().to_dict(),
        "route": latest.groupby("route", observed=True)
                       .route_late_rate.mean().to_dict(),
        "carrier": latest.groupby("Reporting_Airline", observed=True)
                         .carrier_late_rate.mean().to_dict(),
        "origin": latest.groupby("Origin", observed=True)
                        .origin_late_rate.mean().to_dict(),
        "dest": latest.groupby("Dest", observed=True)
                      .dest_late_rate.mean().to_dict(),
        "carrier_origin": latest.groupby(["Reporting_Airline", "Origin"], observed=True)
                                .carrier_origin_late_rate.mean().to_dict(),
        "route_n": latest.groupby("route", observed=True)
                         .route_late_rate_n.mean().to_dict(),
    }


def load() -> dict[str, dict]:
    """Return the lookup tables, building and caching them on first use.

    Returns:
        The seven tables, or empty mappings if neither cache nor dataset exists -
        degraded but still functional, and reported honestly in the explanation.
    """
    global _cache
    if _cache is not None:
        return _cache
    if LOOKUP_PATH.exists():
        _cache = joblib.load(LOOKUP_PATH)
        return _cache
    if not FEATURES_PARQUET.exists():
        _cache = {}
        return _cache

    frame = pd.read_parquet(FEATURES_PARQUET)
    _cache = build_lookups(frame)
    LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_cache, LOOKUP_PATH, compress=3)
    return _cache
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/capstone_1 && python -m pytest tests/test_cold_start.py -v`
Expected: 5 passed

- [ ] **Step 5: Port `predict.py` and its tests**

```bash
cd "D:/AI Transformation Bootcamp Project/ai-transformation-bootcamp"
cp capstone-1-signalcraft/predict.py projects/capstone_1/app/serving/predict.py
cp capstone-1-signalcraft/streamlit_app.py projects/capstone_1/streamlit_app.py
cp capstone-1-signalcraft/tests/test_predict_validation.py projects/capstone_1/tests/
```

In `app/serving/predict.py`:
- Replace the `MODEL_PATH`, `LOOKUP_PATH`, `FEATURES_PARQUET`, `ROOT` definitions with `from app.config import MODEL_PATH, LOOKUP_PATH, FEATURES_PARQUET`
- Replace the body of `_load_lookups()` with `return lookups.load()` and import `from app.serving import lookups`
- Import `HOLIDAYS` from `app.features.build` rather than redefining it

In `streamlit_app.py`, change `from predict import ...` to `from app.serving.predict import ...`.

In `tests/test_predict_validation.py`, change `from predict import ...` to `from app.serving.predict import ...`.

- [ ] **Step 6: Build the lookup cache and verify a real prediction**

```bash
cd projects/capstone_1
python -c "
from app.serving.predict import predict_flight, model_info
info = model_info()
print('threshold', info['threshold'], '| baseline', info['baseline_pr_auc'])

r = predict_flight(carrier='DL', origin='ATL', dest='ORD',
                   departure_hour=17, day_of_week=5, month=7,
                   distance_miles=606, scheduled_minutes=115, leg_number=3)
print(f'{r.probability:.1%} flagged={r.flagged} band={r.risk_band}')
for reason in r.top_reasons:
    print(' -', reason)
"
```

Expected: a probability, a risk band, and plain-language reasons.

- [ ] **Step 7: Verify graceful failure on bad input**

```bash
cd projects/capstone_1
python -c "
from app.serving.predict import predict_flight, InvalidFlight
for kwargs in [
    dict(departure_hour=99), dict(origin='TOOLONG'), dict(distance_miles=-5),
]:
    base = dict(carrier='DL', origin='ATL', dest='ORD', departure_hour=17,
                day_of_week=5, month=7, distance_miles=606,
                scheduled_minutes=115, leg_number=3)
    base.update(kwargs)
    try:
        predict_flight(**base)
        print('NO ERROR - this is a bug')
    except InvalidFlight as exc:
        print('ok:', exc)
"
```

Expected: three readable messages, each naming the field, the rule and the value. **No traceback.**

- [ ] **Step 8: Run the whole suite and launch the app**

Run: `cd projects/capstone_1 && python -m pytest tests/ -v`
Expected: all tests pass.

Then: `streamlit run streamlit_app.py` — confirm it opens and scores a flight in under two seconds.

- [ ] **Step 9: Commit**

```bash
git add projects/capstone_1
git commit -m "Add serving layer: lookups, prediction, Streamlit app (R9)"
```

---

### Task 10: Explainability and error analysis (R8)

**Files:**
- Create: `projects/capstone_1/app/explain/shap_report.py`
- Create: `projects/capstone_1/app/explain/errors.py`
- Create: `projects/capstone_1/app/explain/report.py`
- Test: `projects/capstone_1/tests/test_errors.py`

**Interfaces:**
- Consumes: the exported artefact, `app.config.FIGURES_DIR`, `app.config.METRICS_PATH`
- Produces:
  - `app.explain.shap_report.global_importance(model, sample: pd.DataFrame) -> pd.DataFrame`
  - `app.explain.shap_report.save_summary_plot(model, sample: pd.DataFrame) -> Path`
  - `app.explain.errors.worst_predictions(frame, probabilities, y_true, n=20) -> pd.DataFrame`
  - `app.explain.errors.categorise_failures(worst: pd.DataFrame) -> pd.DataFrame`
  - `app.explain.report.main() -> int`

- [ ] **Step 1: Write the failing test**

Create `projects/capstone_1/tests/test_errors.py`:

```python
import numpy as np
import pandas as pd

from app.explain import errors


def _context():
    return pd.DataFrame({
        "dep_hour": [6, 7, 17, 19, 5],
        "route": ["A-B", "C-D", "E-F", "G-H", "I-J"],
        "Reporting_Airline": ["DL", "AA", "UA", "WN", "B6"],
        "route_late_rate_n": [500.0, 400.0, 600.0, 700.0, 0.0],
    })


def test_worst_predictions_returns_the_requested_count():
    y = np.array([1, 1, 1, 0, 1])
    p = np.array([0.02, 0.05, 0.10, 0.90, 0.04])
    out = errors.worst_predictions(_context(), p, y, n=3)
    assert len(out) == 3


def test_worst_predictions_are_ordered_by_loss():
    y = np.array([1, 1, 1, 0, 1])
    p = np.array([0.02, 0.05, 0.10, 0.90, 0.04])
    out = errors.worst_predictions(_context(), p, y, n=5)
    assert out.loss.is_monotonic_decreasing


def test_a_confident_miss_is_labelled_a_false_negative():
    y = np.array([1, 1, 1, 0, 1])
    p = np.array([0.02, 0.05, 0.10, 0.90, 0.04])
    out = errors.worst_predictions(_context(), p, y, n=1)
    assert out.error_type.iloc[0] == "false negative - missed a late arrival"


def test_a_confident_false_alarm_is_labelled_a_false_positive():
    y = np.array([0, 1])
    p = np.array([0.97, 0.60])
    context = _context().head(2)
    out = errors.worst_predictions(context, p, y, n=1)
    assert out.error_type.iloc[0] == "false positive - flagged an on-time flight"


def test_early_departures_are_grouped_as_their_own_failure_category():
    y = np.array([1, 1, 1, 0, 1])
    p = np.array([0.02, 0.05, 0.10, 0.90, 0.04])
    worst = errors.worst_predictions(_context(), p, y, n=5)
    grouped = errors.categorise_failures(worst)
    assert "early departure" in " ".join(grouped.failure_category).lower()


def test_categorise_reports_a_count_and_a_mean_prediction():
    y = np.array([1, 1, 1, 0, 1])
    p = np.array([0.02, 0.05, 0.10, 0.90, 0.04])
    worst = errors.worst_predictions(_context(), p, y, n=5)
    grouped = errors.categorise_failures(worst)
    assert {"n", "mean_pred"} <= set(grouped.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd projects/capstone_1 && python -m pytest tests/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.explain.errors'`

- [ ] **Step 3: Write `app/explain/errors.py`**

```python
"""The twenty worst predictions, and what they have in common.

R8 says the error analysis carries more marks than the headline score, and it is
the section that tells an operator where not to trust the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARLY_HOUR = 9

FALSE_NEGATIVE = "false negative - missed a late arrival"
FALSE_POSITIVE = "false positive - flagged an on-time flight"

CATEGORY_EARLY = "early departure - little accumulated delay, so a late one is anomalous"
CATEGORY_THIN = "thin route history - the rates rest on little evidence"
CATEGORY_UNEXPLAINED = "unexplained by current features - candidate for weather or NAS data"


def worst_predictions(
    context: pd.DataFrame, probabilities: np.ndarray, y_true: np.ndarray, n: int = 20
) -> pd.DataFrame:
    """Return the n predictions with the largest log loss.

    Args:
        context: Row-aligned descriptive columns for each prediction.
        probabilities: Predicted P(late).
        y_true: Binary labels.
        n: How many to return.

    Returns:
        The worst n, ordered by loss descending.
    """
    clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
    loss = -(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))

    out = context.copy().reset_index(drop=True)
    out["label"] = y_true
    out["pred"] = probabilities
    out["loss"] = loss
    out["error_type"] = np.where(y_true == 1, FALSE_NEGATIVE, FALSE_POSITIVE)
    return out.nlargest(n, "loss").reset_index(drop=True)


def _category(row: pd.Series) -> str:
    if row.get("route_late_rate_n", 1) == 0:
        return CATEGORY_THIN
    if row.dep_hour < EARLY_HOUR:
        return CATEGORY_EARLY
    return CATEGORY_UNEXPLAINED


def categorise_failures(worst: pd.DataFrame) -> pd.DataFrame:
    """Group the worst predictions into named failure categories.

    Args:
        worst: Output of :func:`worst_predictions`.

    Returns:
        One row per category, with a count and the mean prediction.
    """
    tagged = worst.copy()
    tagged["failure_category"] = tagged.apply(_category, axis=1)

    grouped = tagged.groupby("failure_category").agg(
        n=("loss", "size"),
        mean_loss=("loss", "mean"),
        mean_pred=("pred", "mean"),
    )
    return grouped.sort_values("n", ascending=False).round(4).reset_index()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/capstone_1 && python -m pytest tests/test_errors.py -v`
Expected: 6 passed

- [ ] **Step 5: Write `app/explain/shap_report.py`**

```python
"""SHAP global importance (R8a).

SHAP cannot run on 6.9 million rows, so it runs on a sample. The sample size is
reported alongside the result, because an importance ranking from 20,000 rows is
a different claim from one over the whole set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from app.config import FIGURES_DIR, RANDOM_STATE  # noqa: E402

SAMPLE_ROWS = 20_000


def sample_for_shap(frame: pd.DataFrame, rows: int = SAMPLE_ROWS) -> pd.DataFrame:
    """Take a reproducible sample small enough for SHAP to explain."""
    if len(frame) <= rows:
        return frame
    return frame.sample(rows, random_state=RANDOM_STATE)


def global_importance(model: object, sample: pd.DataFrame) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value.

    Args:
        model: The fitted estimator.
        sample: Feature rows, already sampled.

    Returns:
        Columns ``feature`` and ``mean_abs_shap``, most important first.
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[1]

    importance = pd.DataFrame({
        "feature": sample.columns,
        "mean_abs_shap": abs(values).mean(axis=0),
    })
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_summary_plot(model: object, sample: pd.DataFrame) -> Path:
    """Write the SHAP summary plot to the figures directory.

    Returns:
        The path written.
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[1]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "shap_summary.png"

    plt.figure()
    shap.summary_plot(values, sample, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path
```

- [ ] **Step 6: Write `app/explain/report.py`**

```python
"""The ``evaluate`` stage: threshold sweep, SHAP, error analysis, figures."""

from __future__ import annotations

import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app.config import (  # noqa: E402
    FEATURES_PARQUET,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
)
from app.explain import errors, shap_report  # noqa: E402
from app.models.metrics import score  # noqa: E402
from app.models.threshold import sweep  # noqa: E402

GRID = [0.10, 0.15, 0.20, 0.24, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def main() -> int:
    """Produce every artefact R7 and R8 ask for.

    Returns:
        A process exit code.
    """
    if not MODEL_PATH.exists():
        print(
            f"No model at {MODEL_PATH}.\nRun:  python -m app.cli train",
            file=sys.stderr,
        )
        return 1

    artefact = joblib.load(MODEL_PATH)
    model, features = artefact["model"], artefact["features"]

    frame = pd.read_parquet(FEATURES_PARQUET)
    latest = frame[frame.period == frame.period.max()]
    x_valid, y_valid = latest[features], latest.label.to_numpy()

    probabilities = model.predict_proba(x_valid)[:, 1]

    headline = score(y_valid, probabilities, artefact["threshold"])
    table = sweep(y_valid, probabilities, GRID)
    print(table.round(4).to_string(index=False))

    # --- R8a: SHAP ---------------------------------------------------------
    sample = shap_report.sample_for_shap(x_valid)
    importance = shap_report.global_importance(model, sample)
    figure = shap_report.save_summary_plot(model, sample)
    print(f"\nSHAP over {len(sample):,} rows -> {figure}")
    print(importance.head(10).to_string(index=False))

    # --- R8b: the twenty worst --------------------------------------------
    context = latest[["dep_hour", "route", "Reporting_Airline", "route_late_rate_n"]]
    worst = errors.worst_predictions(context, probabilities, y_valid, n=20)
    grouped = errors.categorise_failures(worst)
    print("\n" + grouped.to_string(index=False))

    worst.to_csv(MODEL_PATH.parent / "worst_predictions.csv", index=False)

    # --- the precision-recall curve ---------------------------------------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(table.recall, table.precision, marker="o")
    plt.axhline(headline["baseline"], ls="--", c="r", label="baseline")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall.png", dpi=120)
    plt.close()

    METRICS_PATH.write_text(
        json.dumps({
            "headline": headline,
            "threshold_sweep": table.to_dict("records"),
            "shap_top10": importance.head(10).to_dict("records"),
            "failure_categories": grouped.to_dict("records"),
            "shap_sample_rows": int(len(sample)),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {METRICS_PATH}")
    return 0
```

- [ ] **Step 7: Run the evaluation**

Run: `cd projects/capstone_1 && python -m app.cli evaluate`

Expected: the threshold sweep table, SHAP top-10, failure categories, and two figures written.

**Record what the failure categories actually say.** The source documents found 18 of 20 were early departures. If this run finds something different, report what it found — the honest categorisation is the deliverable, not a match to a previous result.

- [ ] **Step 8: Commit**

```bash
git add projects/capstone_1
git commit -m "Add SHAP importance and worst-prediction error analysis (R8)"
```

---

### Task 11: The six documents and the README (R3, R10, R11)

**Files:**
- Create: `projects/capstone_1/app/explain/docs.py`
- Create: `projects/capstone_1/docs/EDA.md`, `BASELINE.md`, `METRICS.md`, `ERROR_ANALYSIS.md`, `DECISION.md`, `MODEL_CARD.md`
- Create: `projects/capstone_1/README.md`

**Interfaces:**
- Consumes: `app.config.METRICS_PATH`, the exported artefact
- Produces: `app.explain.docs.main() -> int` — renders the numeric tables into `docs/`

- [ ] **Step 1: Write `app/explain/docs.py`**

```python
"""The ``report`` stage: render measured numbers into the documents.

This writes tables, never prose. R3 asks for assumptions recorded before looking
at the data and at least one contradicted - that is a human judgement, not a
generated artefact.
"""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd

from app.config import DOCS_DIR, METRICS_PATH, MODEL_PATH


def main() -> int:
    """Write the numeric sections of the documents.

    Returns:
        A process exit code.
    """
    if not METRICS_PATH.exists():
        print(
            f"No metrics at {METRICS_PATH}.\nRun:  python -m app.cli evaluate",
            file=sys.stderr,
        )
        return 1

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    artefact = joblib.load(MODEL_PATH)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    sweep = pd.DataFrame(metrics["threshold_sweep"])
    categories = pd.DataFrame(metrics["failure_categories"])
    shap_top = pd.DataFrame(metrics["shap_top10"])

    (DOCS_DIR / "_generated_tables.md").write_text(
        "<!-- generated by: python -m app.cli report -->\n\n"
        "## Threshold sweep\n\n" + sweep.round(4).to_markdown(index=False) + "\n\n"
        "## SHAP global importance (top 10)\n\n"
        + shap_top.round(4).to_markdown(index=False) + "\n\n"
        "## Failure categories\n\n" + categories.to_markdown(index=False) + "\n\n"
        f"## Artefact\n\n"
        f"- threshold: {artefact['threshold']}\n"
        f"- baseline PR-AUC: {artefact['baseline_pr_auc']}\n"
        f"- validation lift: {artefact['validation']['lift']}\n"
        f"- training rows: {artefact['training']['rows']:,}\n",
        encoding="utf-8",
    )
    print(f"wrote {DOCS_DIR / '_generated_tables.md'}")
    return 0
```

- [ ] **Step 2: Run the report stage**

Run: `cd projects/capstone_1 && python -m app.cli report`
Expected: `docs/_generated_tables.md` written, containing real numbers.

- [ ] **Step 3: Write `docs/EDA.md` (R3)**

Copy the five findings and the recorded assumptions from
`capstone-1-signalcraft/03_Data_Prep_And_EDA.md` (Cells 39–45), keeping the structure of the
template in `capstone-1-signalcraft/docs/EDA.md`. The five findings, with the numbers to verify
against this run:

| # | Finding | Number |
|---|---|---|
| 1 | Departure hour is the dominant signal | 8.32% late at 05:00 vs 32.46% at 19:00 — 3.9× |
| 2 | Distance barely matters — **assumption contradicted** | 0.2051–0.2433 across bands, 3.8 pp, non-monotonic |
| 3 | Carriers differ, but far less than the hour | DL 0.1838 – B6 0.2727, 1.48× |
| 4 | Sunday is the worst day, not Friday — **contradicted** | Tue 0.1874 – Sun 0.2649, 1.41× |
| 5 | September is best and July beats December — **partly contradicted** | Sep 0.1663 – Jul 0.2889, 1.74× |

Record the assumptions **first**, then the findings. At least one must be marked as contradicted;
findings 2 and 4 are the clearest.

- [ ] **Step 4: Write `docs/BASELINE.md` (R4)**

Fill the template's table from this run:

| Quantity | Where it comes from |
|---|---|
| Positive rate (development set) | `artefact["baseline_pr_auc"]` |
| Majority-class accuracy | `1 - positive_rate` |
| Baseline PR-AUC | equal to the positive rate |
| Imbalance ratio | `(1 - p) / p` |

Then state the prediction recorded **before** any model was fitted, and whether it held.

- [ ] **Step 5: Write `docs/METRICS.md` (R7)**

The justification paragraph must cover: why PR-AUC is the headline (the positives are the rare,
expensive class and PR-AUC ignores the easy true negatives); why ROC-AUC is not (it flatters a
model on an imbalanced problem by rewarding it for ranking the majority correctly); why accuracy is
not (77.58% is achievable by predicting "never late" and is useless); why recall at a working
threshold sits beside it; and the cost asymmetry that sets the threshold — **stated as an
assumption**, roughly 5:1.

Paste the threshold sweep table from `docs/_generated_tables.md`.

- [ ] **Step 6: Write `docs/ERROR_ANALYSIS.md` (R8)**

Sections: the SHAP summary plot with a written reading; **three plain-language explanations** with
no feature names and no unitless numbers, readable by an operations manager without a glossary; the
failure-category table from this run; and the honest conclusion.

If this run reproduces the source's finding — that the worst errors are overwhelmingly confident
false negatives on early departures — say so and state the operational consequence: **a low score
on an early-morning flight is less trustworthy than a low score on an afternoon flight, and the
interface must not present them as equivalent.**

- [ ] **Step 7: Write `docs/DECISION.md` (R10)**

One page, four parts:

1. **Who consumes it** — the station duty manager at a hub, reviewing tomorrow's departure bank the previous afternoon.
2. **Which decision changes** — standby ground crew are assigned to the top third of tomorrow's departures instead of spread evenly. Nothing is cancelled, nothing re-timed: **the model reallocates attention, not aircraft.**
3. **The threshold and its cost reasoning** — the operating point from `artefact["threshold"]`, the recall and precision it achieves, the share of schedule it flags, and the ~5:1 cost asymmetry **stated as an assumption**. Note that the flagged share is bounded by what a standby pool can cover, so the threshold is bounded by the operation and not only by the curve.
4. **Switch-off conditions** — five, each measurable:
   - Rolling PR-AUC lift falls below 1.25× on the trailing month
   - Recall at the deployed threshold moves more than 15 points month over month
   - A schedule change affecting more than 10% of routes
   - BTS changes the reporting standard or the 15-minute definition
   - Any operational decision starts being taken automatically

- [ ] **Step 8: Write `docs/MODEL_CARD.md` (R11)**

Fill every row from the artefact: model family, prediction time T-24h, target `ArrDel15`, training
period and rows, feature count, baseline PR-AUC, validation PR-AUC and lift, operating threshold.
Leave **holdout PR-AUC blank and marked "sealed — opens day 20"**. Then intended use, out of scope,
known weaknesses, and the switch-off conditions.

- [ ] **Step 9: Write `README.md` (R11)**

Must include, in this order:

1. **The headline, as a lift** — never a bare score — with the sealed-holdout status stated up front. The self-honesty clause goes here: if the holdout later comes in materially below validation, **this paragraph says so**.
2. **Track A justification** against the three criteria: genuinely public and no login wall; supports a real operator decision; large enough that evaluation means something.
3. **Quickstart a stranger can follow unaided** — clone, `pip install -r requirements.txt`, set `SIGNALCRAFT_DATA_DIR` or run `python -m app.cli acquire --start 2025-07 --end 2026-06`, then `prepare`, `seal`, `features`, `train`, `evaluate`, then `streamlit run streamlit_app.py`.
4. **The full 24-feature table** — feature, kind, why it should carry signal, available at T-24h. This is R5's acceptance criterion and it belongs where a grader will find it.
5. **Data provenance** — source, licence, retrieval date, row count, from `data/manifest.json`.
6. **Links** to the six documents.

- [ ] **Step 10: Verify a stranger can run it**

Read the README as if you have never seen the project. Follow every command literally. Any step
that needs knowledge not on the page is a defect — fix the README, not your memory of it.

Run: `cd projects/capstone_1 && python -m pytest tests/ -v && python -m ruff check . && python -m black --check .`
Expected: tests pass, no lint errors, formatting clean.

- [ ] **Step 11: Commit**

```bash
git add projects/capstone_1
git commit -m "Add the six capstone documents and the README (R3, R10, R11)"
```

---

### Task 12: Notebooks as executed evidence

**Files:**
- Create: `projects/capstone_1/notebooks/01_data_and_eda.ipynb`
- Create: `projects/capstone_1/notebooks/02_features.ipynb`
- Create: `projects/capstone_1/notebooks/03_model_and_errors.ipynb`

**Interfaces:**
- Consumes: everything in `app/`
- Produces: nothing other code depends on — these are evidence, not execution path

- [ ] **Step 1: Write notebook 01**

Cells that **import from `app/` and add only plots and commentary**. No logic may be redefined here;
a second definition of a feature is a second definition that can drift.

```python
import sys; sys.path.insert(0, "..")
import pandas as pd
from app.data.seal import development_frame
from app.config import PRIOR

frame = development_frame(columns=["period", "label", "dep_hour", "Distance",
                                   "DayOfWeek", "Month", "Reporting_Airline"])
print(f"{len(frame):,} development rows, positive rate {frame.label.mean():.4f}")
```

Then one cell per EDA finding, each producing the number and a plot: late rate by departure hour;
by distance band; by carrier; by day of week; by month.

- [ ] **Step 2: Write notebook 02**

```python
import sys; sys.path.insert(0, "..")
import pandas as pd
from app.config import FEATURES_PARQUET
from app.features.build import FEATURE_COLUMNS
from app.features.forbidden import FORBIDDEN

frame = pd.read_parquet(FEATURES_PARQUET)
print(f"{frame.shape[0]:,} rows x {len(FEATURE_COLUMNS)} features")
assert not FORBIDDEN & set(frame.columns), "leakage"
frame[FEATURE_COLUMNS].corrwith(frame.label).sort_values(key=abs, ascending=False).round(4)
```

Plus the correlation heatmap and the history-coverage figure.

- [ ] **Step 3: Write notebook 03**

```python
import sys; sys.path.insert(0, "..")
import joblib, json
import pandas as pd
from app.config import MODEL_PATH, METRICS_PATH

artefact = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text())

print(f"{artefact['model_family']}: lift {artefact['validation']['lift']}x "
      f"over baseline {artefact['baseline_pr_auc']}")
pd.DataFrame(metrics["threshold_sweep"]).round(4)
```

Plus the lift-by-fold plot (the rolling backtest), the SHAP summary, and the failure-category table.

- [ ] **Step 4: Execute every notebook and store the outputs**

```bash
cd projects/capstone_1
python -m pip install nbconvert ipykernel
for nb in notebooks/*.ipynb; do
  python -m jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

- [ ] **Step 5: Verify the outputs are actually stored**

```bash
cd projects/capstone_1
python -c "
import json, pathlib
for p in sorted(pathlib.Path('notebooks').glob('*.ipynb')):
    nb = json.load(open(p, encoding='utf-8'))
    code = [c for c in nb['cells'] if c['cell_type'] == 'code']
    with_out = [c for c in code if c.get('outputs')]
    print(f'{p.name}: {len(with_out)}/{len(code)} code cells have output')
    assert with_out, f'{p.name} has no stored outputs - it is a scaffold, not evidence'
"
```

Expected: every notebook reports stored outputs on most cells.

This check exists because notebooks 02 and 03 in the source folder are headings over empty cells
while being marked "Done, executed". A notebook committed with empty cells is worse than no
notebook, because it looks like evidence without being any.

- [ ] **Step 6: Final full verification**

```bash
cd projects/capstone_1
python -m pytest tests/ -v
python -m ruff check .
python -m black --check .
python -c "import json; s=json.load(open('holdout_seal.json')); assert s['opened'] is False; print('holdout still sealed:', not s['opened'])"
```

Expected: all tests pass, lint clean, formatting clean, and **the holdout still sealed**.

- [ ] **Step 7: Commit**

```bash
git add projects/capstone_1
git commit -m "Add executed notebooks as evidence, importing from app/"
```

---

## Self-Review

**Spec coverage.** Every numbered section of the spec maps to a task: §4 architecture → Task 1; §5 data flow → Tasks 2, 3, 5, 8; holdout enforcement → Task 3; folds → Task 7; artefacts → Tasks 8, 9; §6 error handling → Tasks 3, 9; cold start → Task 9; the caveat → Task 9 (ported in `predict.py`); §7 testing → every task's test step, with the leakage property test in Task 4; §8 R1–R11 → Tasks 2, 3, 4, 5, 6, 7, 8, 9, 10, 11; §9 build order → Tasks 1–12 in sequence.

**Type consistency.** `FEATURE_COLUMNS` is defined once in Task 5 and consumed by Tasks 8, 9, 10, 12. The seven lookup keys in Task 9 match `predict.py`'s `_build_feature_row` verbatim. `expanding_rate` produces `<name>` and `<name>_n`, and Task 5 drops exactly the five counts nothing consumes, leaving `route_late_rate_n` as the 24th feature. `Fold.train_periods` / `Fold.validation_period` are used identically in Tasks 7 and 8.

**Known gap, deliberate.** `app/explain/docs.py` renders tables only; the prose in the six documents is authored by hand in Task 11, as the spec requires — R3's "assumption contradicted" cannot be generated.
