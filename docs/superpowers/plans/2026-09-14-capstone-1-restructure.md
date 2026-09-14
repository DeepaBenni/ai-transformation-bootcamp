# Capstone 1 Flat-Layout Restructure + R8/R9 Build-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `projects/capstone_1/app/` into `projects/capstone_1/src/` with a flat entry-point surface, and build the two subsystems that were never completed — R8 (SHAP global importance + 20 worst errors) and R9 (serving: predict + lookups + Streamlit UI) — porting what already exists in the read-only `capstone-1-signalcraft/` source folder and writing fresh, tested code for what doesn't.

**Architecture:** `app/` becomes `src/`; the existing tested submodules (`data/`, `features/`, `models/`) move unchanged apart from import paths. Six thin composer files (`data_pipeline.py`, `feature_engineering.py`, `train_model.py`, `evaluate.py`, `predict.py`, `model_utils.py`) sit at `src/` root, each re-exporting the real logic underneath — none contain new logic themselves. `src/serving/` (predict, lookups) and `src/explain/` (shap_report, errors) are built fresh: serving is ported from `capstone-1-signalcraft/predict.py` and `streamlit_app.py` (working code, adapted to import from `src.config`), explain is new code driven by SHAP 0.52 and the already-trained model artifact.

**Tech Stack:** Python 3.12, pandas, scikit-learn, LightGBM, SHAP 0.52, Streamlit, pytest, ruff, black, jupyter/nbconvert (added by this plan).

**Spec:** `docs/superpowers/specs/2026-09-14-capstone-1-restructure-design.md` (this plan implements it; §4 config.json schema, §5 composer pattern, §6 notebooks, §7 validation sequence are load-bearing). Also relevant: `docs/superpowers/specs/2026-09-13-capstone-1-signalcraft-design.md` §6 (error handling / cold-start / caveat rules) and §7 (testing philosophy) — R8/R9 behavior must match what's already documented there.

## Global Constraints

- Line length 100, Google docstrings, `ruff` rules `E F I N UP B D ANN` (from `pyproject.toml`) — every new file must pass `ruff check`.
- `tests/*` and `notebooks/*` are exempt from `D`/`ANN` (already configured in `pyproject.toml`).
- No retraining: `app/models/zoo.py`'s hyperparameters carry over unchanged (100/10 forest, unregularized boosted).
- No change to holdout sealing (`seal.py`) or leakage-guard (`forbidden.py`) *behavior* — only their import path moves.
- Every new/moved module composing another module must use `from src.<path> import <name>` — never `from app.<path>`.
- All existing tests must stay green throughout; a task that breaks the suite is not done.
- The already-trained `models/flight_delay_model.pkl` and `data/processed/flights_features.parquet` are reused as-is — do not delete or regenerate them.

---

### Task 1: Rename `app/` → `src/`, add `config.json`, update `config.py`

**Files:**
- Move: `projects/capstone_1/app/` → `projects/capstone_1/src/` (entire tree: `__init__.py`, `config.py`, `cli.py`, `data/`, `features/`, `models/`, `explain/`, `serving/`)
- Create: `projects/capstone_1/config.json`
- Modify: `projects/capstone_1/src/config.py`
- Modify: `projects/capstone_1/tests/conftest.py`, and every file under `projects/capstone_1/tests/*.py` (import path only)
- Test: `projects/capstone_1/tests/test_config.py`

**Interfaces:**
- Produces: `src.config.load_config(path: Path | None = None) -> dict[str, Any]`, and every existing `src.config` constant (`PROJECT_ROOT`, `DATA_DIR`, `FEATURES_PARQUET`, `MODEL_PATH`, `LOOKUP_PATH`, `HOLDOUT_MONTHS`, `RANDOM_STATE`, etc.) unchanged in name and type — every later task imports from `src.config`, not `app.config`.

- [ ] **Step 1: Confirm the baseline is green before moving anything**

Run: `cd projects/capstone_1 && python -m pytest -q`
Expected: all existing tests PASS (this is the safety net for the move).

- [ ] **Step 2: Move the package with git, preserving history**

```bash
cd projects/capstone_1
git mv app src
```

- [ ] **Step 3: Rewrite every `app.` import reference to `src.`**

```bash
grep -rl '\bapp\.' src tests | xargs sed -i 's/\bapp\./src./g'
```

Also fix the two doc-comment references inside `src/config.py`'s module docstring and `src/data/prepare.py`'s docstring that mention `python -m app.cli` (the sed above already rewrites these since they match `app.cli`).

- [ ] **Step 4: Add `config.json` at the project root**

```json
{
  "project_name": "capstone_1",
  "target_column": "ArrDel15",
  "label_column": "label",
  "data_dir": "data",
  "processed_parquet": "data/processed/flights_features.parquet",
  "model_artifact": "models/flight_delay_model.pkl",
  "streamlit_app": "src/streamlit_app.py",
  "holdout_months": ["2026-05", "2026-06"],
  "random_state": 42
}
```

- [ ] **Step 5: Write the failing test for `load_config`**

Add to `tests/test_config.py`:

```python
import json

from src import config


def test_load_config_reads_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"random_state": 7}), encoding="utf-8")
    assert config.load_config(path) == {"random_state": 7}


def test_load_config_returns_empty_dict_when_missing(tmp_path):
    assert config.load_config(tmp_path / "nope.json") == {}
```

- [ ] **Step 6: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: module 'src.config' has no attribute 'load_config'`

- [ ] **Step 7: Add `load_config` and wire the constants to read it, in `src/config.py`**

Replace the top of `src/config.py` (through the `DATA_ENV_VAR` line) with:

```python
"""Paths and constants for the SignalCraft pipeline.

Defaults come from ``config.json`` at the project root; ``SIGNALCRAFT_DATA_DIR``
still overrides the data location for anyone pointing at a prepared parquet
without a re-download - see :func:`resolve_data_dir`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = PROJECT_ROOT.parents[1]

CONFIG_JSON_PATH: Final[Path] = PROJECT_ROOT / "config.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read config.json, or an empty mapping if it does not exist.

    Args:
        path: Override for testing; defaults to :data:`CONFIG_JSON_PATH`.

    Returns:
        The parsed JSON object, or ``{}`` when the file is absent.
    """
    target = path if path is not None else CONFIG_JSON_PATH
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


_CFG: Final[dict[str, Any]] = load_config()

# The sibling learning folder that already holds a prepared parquet.
FALLBACK_DATA_DIR: Final[Path] = REPO_ROOT / "capstone-1-signalcraft" / "data"

DATA_ENV_VAR: Final[str] = "SIGNALCRAFT_DATA_DIR"
```

Then replace `resolve_data_dir`'s body's `local = PROJECT_ROOT / "data"` line with:

```python
    local = PROJECT_ROOT / _CFG.get("data_dir", "data")
```

Then replace the `FEATURES_PARQUET` and `MODEL_PATH` assignments with:

```python
FEATURES_PARQUET: Final[Path] = PROJECT_ROOT / _CFG.get(
    "processed_parquet", "data/processed/flights_features.parquet"
)

MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
MODEL_PATH: Final[Path] = PROJECT_ROOT / _CFG.get(
    "model_artifact", "models/flight_delay_model.pkl"
)
```

(Keep `MODEL_DIR` where it already was relative to `MODEL_PATH` — just source `MODEL_PATH` from config.)

Then replace the `HOLDOUT_MONTHS` assignment with:

```python
HOLDOUT_MONTHS: Final[tuple[str, ...]] = tuple(
    _CFG.get("holdout_months", ["2026-05", "2026-06"])
)
```

Then replace the `RANDOM_STATE` assignment with:

```python
RANDOM_STATE: Final[int] = int(_CFG.get("random_state", 42))
```

Everything else in `src/config.py` (`DEV_MONTHS`, `FOLD_VALIDATION_MONTHS`, `PRIOR`, `ALPHA`, `LATE_THRESHOLD_MIN`, `PREDICTION_TIME`, `INTERIM_DIR`, `PROCESSED_DIR`, `MANIFEST_PATH`, `CLEAN_PARQUET`, `LOOKUP_PATH`, `METRICS_PATH`, `FIGURES_DIR`, `DOCS_DIR`, `SEAL_PATH`, `resolve_data_dir`, `DATA_DIR`) stays exactly as it was.

- [ ] **Step 8: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 9: Run the full suite to confirm nothing broke in the move**

Run: `python -m pytest -q`
Expected: same pass count as Step 1, all green, now importing from `src.*`.

- [ ] **Step 10: Commit**

```bash
git add projects/capstone_1/src projects/capstone_1/tests projects/capstone_1/config.json
git commit -m "Restructure app/ into src/, add config.json (E1, E4)"
```

---

### Task 2: Composer files — `data_pipeline.py`, `feature_engineering.py`, `train_model.py`, `model_utils.py`

**Files:**
- Create: `projects/capstone_1/src/data_pipeline.py`
- Create: `projects/capstone_1/src/feature_engineering.py`
- Create: `projects/capstone_1/src/train_model.py`
- Create: `projects/capstone_1/src/model_utils.py`
- Test: `projects/capstone_1/tests/test_composers.py`

**Interfaces:**
- Consumes: `src.data.prepare.run`, `src.data.seal.{development_frame,holdout_frame,verify}`, `src.features.forbidden.assert_no_leakage`, `src.features.history.expanding_rate`, `src.features.build.{FEATURE_COLUMNS,build,run}`, `src.models.cv.expanding_folds`, `src.models.zoo.model_zoo`, `src.models.metrics.{baseline_pr_auc,score}`, `src.models.threshold.{sweep,threshold_for_recall}`, `src.models.train.{run_cv,summarise,fit_final,export}`
- Produces: `src.model_utils.load_artefact(path: Path | None = None) -> dict[str, Any]` — later used by Task 6/7's explain modules and Task 4's serving module instead of each hand-rolling a `joblib.load`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_composers.py
"""Each composer just re-exports its submodule's public names - verify the wiring."""

from src import data_pipeline, feature_engineering, model_utils, train_model


def test_data_pipeline_exposes_prepare_and_seal():
    assert callable(data_pipeline.run)
    assert callable(data_pipeline.development_frame)
    assert callable(data_pipeline.verify)


def test_feature_engineering_exposes_build_and_columns():
    assert callable(feature_engineering.build)
    assert callable(feature_engineering.run)
    assert len(feature_engineering.FEATURE_COLUMNS) == 24


def test_train_model_exposes_the_training_pipeline():
    assert callable(train_model.model_zoo)
    assert callable(train_model.expanding_folds)
    assert callable(train_model.run_cv)
    assert callable(train_model.summarise)
    assert callable(train_model.fit_final)
    assert callable(train_model.export)


def test_model_utils_load_artefact_reads_a_joblib_dump(tmp_path):
    path = tmp_path / "artefact.pkl"
    import joblib

    joblib.dump({"model_family": "boosted"}, path)
    assert model_utils.load_artefact(path) == {"model_family": "boosted"}


def test_model_utils_load_artefact_raises_clearly_when_missing(tmp_path):
    from src.model_utils import ArtefactNotFound

    with pytest.raises(ArtefactNotFound, match="python -m src.cli train"):
        model_utils.load_artefact(tmp_path / "missing.pkl")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_composers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data_pipeline'`

- [ ] **Step 3: Write `src/data_pipeline.py`**

```python
"""Data pipeline entry point (row keys, monthly-to-clean, the sealed split).

Thin composer - logic lives in src/data/prepare.py and src/data/seal.py.
"""

from __future__ import annotations

from src.data.prepare import add_row_key, clean_frame, run
from src.data.seal import (
    SealViolation,
    development_frame,
    holdout_frame,
    holdout_hash,
    verify,
)

__all__ = [
    "add_row_key",
    "clean_frame",
    "run",
    "SealViolation",
    "development_frame",
    "holdout_frame",
    "holdout_hash",
    "verify",
]
```

- [ ] **Step 4: Write `src/feature_engineering.py`**

```python
"""The 24 engineered features - speed, distance, schedule, geography, calendar.

Thin composer - logic lives in src/features/{forbidden,history,build}.py.
"""

from __future__ import annotations

from src.features.build import (
    FEATURE_COLUMNS,
    add_history_features,
    add_schedule_features,
    build,
    run,
)
from src.features.forbidden import FORBIDDEN, LeakageError, assert_no_leakage
from src.features.history import expanding_rate

__all__ = [
    "FEATURE_COLUMNS",
    "add_history_features",
    "add_schedule_features",
    "build",
    "run",
    "FORBIDDEN",
    "LeakageError",
    "assert_no_leakage",
    "expanding_rate",
]
```

- [ ] **Step 5: Write `src/train_model.py`**

```python
"""Model training entry point (R6, R7): CV, the family zoo, thresholding, export.

Thin composer - logic lives in src/models/{cv,zoo,threshold,train}.py.
"""

from __future__ import annotations

from src.models.cv import Fold, expanding_folds
from src.models.threshold import sweep, threshold_for_recall
from src.models.train import export, fit_final, run_cv, summarise
from src.models.zoo import FOREST_CONFIG, model_zoo

__all__ = [
    "Fold",
    "expanding_folds",
    "sweep",
    "threshold_for_recall",
    "export",
    "fit_final",
    "run_cv",
    "summarise",
    "FOREST_CONFIG",
    "model_zoo",
]
```

- [ ] **Step 6: Write `src/model_utils.py`**

```python
"""Shared helpers: artefact loading, path resolution, preprocessing glue.

Everything here is small and cross-cutting enough that it doesn't belong to
any single stage - the model artefact loader in particular is used by both
serving (src/serving/predict.py) and explainability (src/explain/*).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.config import MODEL_PATH


class ArtefactNotFound(FileNotFoundError):
    """Raised when the trained model artefact is missing, naming the fix."""


def load_artefact(path: Path | None = None) -> dict[str, Any]:
    """Load the trained model artefact produced by ``src.models.train.export``.

    Args:
        path: Override for testing; defaults to :data:`src.config.MODEL_PATH`.

    Returns:
        The artefact dict: model, features, threshold, defaults, validation,
        training metadata.

    Raises:
        ArtefactNotFound: If the pickle does not exist, naming the training command.
    """
    target = path if path is not None else MODEL_PATH
    if not target.exists():
        raise ArtefactNotFound(
            f"No model at {target}.\nRun:  python -m src.cli train"
        )
    return joblib.load(target)
```

- [ ] **Step 7: Run to verify it passes**

Run: `python -m pytest tests/test_composers.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add projects/capstone_1/src/data_pipeline.py projects/capstone_1/src/feature_engineering.py \
        projects/capstone_1/src/train_model.py projects/capstone_1/src/model_utils.py \
        projects/capstone_1/tests/test_composers.py
git commit -m "Add flat entry-point composers for data/features/models (E2)"
```

---

### Task 3: `src/serving/lookups.py` — the history lookup tables

**Files:**
- Create: `projects/capstone_1/src/serving/lookups.py`
- Test: `projects/capstone_1/tests/test_lookups.py`

**Interfaces:**
- Consumes: `src.config.{FEATURES_PARQUET,LOOKUP_PATH}`
- Produces: `src.serving.lookups.load_lookups() -> dict[str, dict]`, `src.serving.lookups.build_lookups() -> dict[str, dict]`, `src.serving.lookups.reset_cache() -> None` — Task 4's `predict.py` imports `load_lookups`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lookups.py
import pandas as pd
import pytest

from src.serving import lookups


@pytest.fixture(autouse=True)
def _reset_cache():
    lookups.reset_cache()
    yield
    lookups.reset_cache()


def _features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["2026-04", "2026-04", "2026-04"],
            "Origin": ["ATL", "ATL", "ORD"],
            "Dest": ["ORD", "ORD", "ATL"],
            "Reporting_Airline": ["DL", "DL", "AA"],
            "route": ["ATL-ORD", "ATL-ORD", "ORD-ATL"],
            "dep_hour": [17, 17, 9],
            "origin_hour_late_rate": [0.3, 0.3, 0.1],
            "route_late_rate": [0.25, 0.25, 0.15],
            "carrier_late_rate": [0.22, 0.22, 0.18],
            "origin_late_rate": [0.2, 0.2, 0.12],
            "dest_late_rate": [0.19, 0.19, 0.21],
            "carrier_origin_late_rate": [0.24, 0.24, 0.17],
            "route_late_rate_n": [500, 500, 40],
        }
    )


def test_build_lookups_groups_by_route(monkeypatch, tmp_path):
    parquet = tmp_path / "features.parquet"
    _features_frame().to_parquet(parquet)
    monkeypatch.setattr(lookups, "FEATURES_PARQUET", parquet)

    out = lookups.build_lookups()

    assert out["route"]["ATL-ORD"] == pytest.approx(0.25)
    assert out["route_n"]["ORD-ATL"] == pytest.approx(40)


def test_load_lookups_returns_empty_when_no_parquet(monkeypatch, tmp_path):
    monkeypatch.setattr(lookups, "FEATURES_PARQUET", tmp_path / "missing.parquet")
    monkeypatch.setattr(lookups, "LOOKUP_PATH", tmp_path / "lookup_tables.pkl")
    assert lookups.load_lookups() == {}


def test_load_lookups_caches_across_calls(monkeypatch, tmp_path):
    parquet = tmp_path / "features.parquet"
    _features_frame().to_parquet(parquet)
    monkeypatch.setattr(lookups, "FEATURES_PARQUET", parquet)
    monkeypatch.setattr(lookups, "LOOKUP_PATH", tmp_path / "lookup_tables.pkl")

    first = lookups.load_lookups()
    parquet.unlink()  # prove the second call doesn't re-read the parquet
    second = lookups.load_lookups()
    assert first is second
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_lookups.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.serving.lookups'`

- [ ] **Step 3: Write `src/serving/lookups.py`**

```python
"""Historical late-rate lookup tables needed at score time (R9).

Six of the model's 24 features are backward-looking rates a caller cannot
supply directly - they must be looked up by airport, airline and route from
the same feature dataset training used. Without this module the serving
layer cannot build a feature row at all (the "missing join" the 2026-09-13
spec's architecture section names).
"""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.config import FEATURES_PARQUET, LOOKUP_PATH

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

    The latest period carries the most complete backward-looking rates, which
    is what a new flight should be scored against.

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

    If neither the cache nor the feature dataset is available, an empty
    mapping is returned - the caller falls back to training medians, which is
    degraded but still functional (see the 2026-09-13 spec's cold-start table).

    Returns:
        The lookup mapping - possibly empty.
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_lookups.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add projects/capstone_1/src/serving/lookups.py projects/capstone_1/tests/test_lookups.py
git commit -m "Add serving/lookups.py, the history-rate tables (R9)"
```

---

### Task 4: `src/serving/predict.py` — ported and adapted from `capstone-1-signalcraft/predict.py`

**Files:**
- Create: `projects/capstone_1/src/serving/predict.py`
- Test: `projects/capstone_1/tests/test_predict_validation.py`

**Interfaces:**
- Consumes: `src.config.{MODEL_PATH,FEATURES_PARQUET}`, `src.model_utils.{load_artefact,ArtefactNotFound}`, `src.serving.lookups.load_lookups`
- Produces: `src.serving.predict.{InvalidFlight,ModelNotFound,Prediction,predict_flight,model_info,known_carriers,known_airports}` — Task 5's `src/predict.py` composer re-exports these.

- [ ] **Step 1: Write the failing tests (ported from the source folder's `tests/test_predict_validation.py`, import path changed)**

```python
# tests/test_predict_validation.py
"""R9: malformed input must produce a readable message, never a stack trace.

These run without a trained model, because ``predict_flight`` validates before
it touches the artefact. That is deliberate: input validation is testable on a
clean checkout, and graders send junk deliberately.
"""

from __future__ import annotations

import pytest

from src.serving.predict import InvalidFlight, predict_flight

VALID: dict[str, object] = dict(
    carrier="DL", origin="ATL", dest="ORD", departure_hour=18,
    day_of_week=5, month=7, distance_miles=606, scheduled_minutes=115,
    leg_number=3,
)


def _with(**overrides: object) -> dict[str, object]:
    return {**VALID, **overrides}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("departure_hour", 25),
        ("departure_hour", -1),
        ("day_of_week", 0),
        ("day_of_week", 8),
        ("month", 13),
        ("distance_miles", -50),
        ("distance_miles", 99_999),
        ("scheduled_minutes", 0),
        ("leg_number", 0),
        ("leg_number", 99),
        ("origin", "ATLANTA"),
        ("origin", "A1L"),
        ("dest", "OR"),
        ("carrier", "DELTA"),
    ],
)
def test_out_of_range_input_raises_invalid_flight(field: str, value: object) -> None:
    with pytest.raises(InvalidFlight) as caught:
        predict_flight(**_with(**{field: value}))
    assert field in str(caught.value)


def test_origin_equal_to_dest_is_rejected() -> None:
    with pytest.raises(InvalidFlight):
        predict_flight(**_with(dest="ATL"))


def test_error_message_reports_the_value_received() -> None:
    with pytest.raises(InvalidFlight) as caught:
        predict_flight(**_with(departure_hour=25))
    assert "25" in str(caught.value)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_predict_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.serving.predict'`

- [ ] **Step 3: Write `src/serving/predict.py`**

Port `capstone-1-signalcraft/predict.py` verbatim with these adaptations:

1. Drop the `ROOT`, `MODEL_PATH`, `LOOKUP_PATH`, `FEATURES_PARQUET` path constants; instead:
   ```python
   from src.config import MODEL_PATH
   from src.model_utils import ArtefactNotFound, load_artefact
   from src.serving.lookups import load_lookups
   ```
2. Replace `_load_artefact()`'s body with a thin wrapper over `load_artefact`, keeping the same in-module cache and the same `ModelNotFound` exception name for backward-compatible error handling in `streamlit_app.py`:
   ```python
   _artefact: dict[str, Any] | None = None


   def _load_artefact() -> dict[str, Any]:
       """Load and cache the trained model artefact."""
       global _artefact
       if _artefact is None:
           try:
               _artefact = load_artefact()
           except ArtefactNotFound as exc:
               raise ModelNotFound(str(exc)) from exc
       return _artefact
   ```
3. Delete the entire `_load_lookups` function and the `_lookups` global (now in `src/serving/lookups.py`); every call site that referenced `_load_lookups()` calls `load_lookups()` instead.
4. Keep everything else unchanged: `HOLIDAYS`, `PLAIN_NAMES`, `DAY_NAMES`, `InvalidFlight`, `ModelNotFound`, `Prediction`, `model_info`, `known_carriers`, `known_airports`, `_validate`, `_build_feature_row`, `_risk_band`, `_top_reasons`, `_caveat`, `predict_flight`, `_demo`, and the `if __name__ == "__main__":` block.

The full adapted file:

```python
"""Score a single flight for late-arrival risk.

This is the serving layer (requirement R9). It loads the artefact produced by
``src.models.train.export`` and turns a plain description of one flight into a
probability, a recommendation and a plain-English explanation.

Design rules, and why each one is here:

* **The model is loaded once**, lazily, at first use. Loading takes about a
  second; doing it per call would blow the two-second response budget R9 sets.
* **Every input is validated** before it reaches the model. R9 asks for graceful
  behaviour on malformed or out-of-range input, so bad values raise
  :class:`InvalidFlight` with a message naming the field, the rule and the value
  received - never a stack trace.
* **The public signature takes plain Python values.** No pandas object crosses
  the boundary; the DataFrame is assembled inside.
* **It runs standalone.** ``python -m src.serving.predict`` prints a worked
  example and three deliberately invalid inputs.

Usage::

    from src.serving.predict import predict_flight

    result = predict_flight(
        carrier="DL", origin="ATL", dest="ORD",
        departure_hour=17, day_of_week=5, month=7,
        distance_miles=606, scheduled_minutes=115, leg_number=3,
    )
    print(result.probability, result.flagged)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from src.config import MODEL_PATH
from src.model_utils import ArtefactNotFound, load_artefact
from src.serving.lookups import load_lookups

# US federal holidays covering the training period, used to derive the two
# holiday features. Extend this list when the model is retrained on new months.
HOLIDAYS: Final[pd.DatetimeIndex] = pd.to_datetime([
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-11-28", "2025-12-24",
    "2025-12-25", "2025-12-31", "2026-01-01", "2026-01-19", "2026-02-16",
    "2026-05-25",
])

# Readable names for every feature, so explanations contain no jargon. An
# operations manager should be able to read the output without a glossary.
PLAIN_NAMES: Final[dict[str, str]] = {
    "origin_hour_late_rate": "how often the departure airport runs late at this hour",
    "route_late_rate": "how often this route runs late",
    "carrier_origin_late_rate": "how often this airline runs late at this airport",
    "dest_late_rate": "how often the arrival airport runs late",
    "carrier_late_rate": "how often this airline runs late",
    "origin_late_rate": "how often the departure airport runs late",
    "route_late_rate_n": "how much history exists for this route",
    "dep_hour": "the scheduled departure hour",
    "dep_minute_of_day": "the scheduled departure time",
    "arr_hour": "the scheduled arrival hour",
    "tail_leg_number": "which leg of the aircraft's day this is",
    "sched_speed_mph": "how tight the scheduled block time is",
    "Month": "the month",
    "DayOfWeek": "the day of the week",
    "Distance": "the flight distance",
    "CRSElapsedTime": "the scheduled flight time",
    "days_to_holiday": "how close this is to a holiday",
    "is_holiday_window": "falling inside a holiday travel peak",
    "is_red_eye": "being an overnight departure",
    "is_weekend": "being a weekend flight",
    "origin_bank_size": "how many departures leave this airport this hour",
    "dest_bank_size": "how many arrivals land at the destination this hour",
    "Quarter": "the quarter",
    "DistanceGroup": "the distance band",
}

DAY_NAMES: Final[dict[int, str]] = {
    1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
    5: "Friday", 6: "Saturday", 7: "Sunday",
}


# ---------------------------------------------------------------- exceptions --
class InvalidFlight(ValueError):
    """Raised when a caller's flight description cannot be scored.

    Carries a message naming the offending field, the rule it broke and the
    value received, so a user interface can show it directly.
    """


class ModelNotFound(FileNotFoundError):
    """Raised when the trained artefact is missing, with the command that fixes it."""


# -------------------------------------------------------------------- result --
@dataclass(frozen=True)
class Prediction:
    """The answer to one scoring request.

    Attributes:
        probability: Chance the flight arrives 15+ minutes late, between 0 and 1.
        flagged: True when ``probability`` reaches the model's operating threshold.
        threshold: The operating threshold the model was shipped with.
        risk_band: ``"low"``, ``"moderate"`` or ``"high"`` - a coarse label for a UI.
        top_reasons: Plain-English drivers of this prediction, strongest first.
        caveat: A warning when this flight falls in a region where the model is
            known to be unreliable, otherwise ``None``.
        inputs_used: The full feature row that was scored, for audit and debugging.
    """

    probability: float
    flagged: bool
    threshold: float
    risk_band: str
    top_reasons: list[str] = field(default_factory=list)
    caveat: str | None = None
    inputs_used: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        verdict = "FLAG for attention" if self.flagged else "no action"
        return f"{self.probability:.0%} chance of arriving 15+ min late - {verdict}"


# ------------------------------------------------------------- artefact load --
_artefact: dict[str, Any] | None = None


def _load_artefact() -> dict[str, Any]:
    """Load and cache the trained model artefact.

    Returns:
        The dictionary saved by ``src.models.train.export``: model, feature
        order, threshold, defaults and provenance.

    Raises:
        ModelNotFound: if the pickle does not exist, naming the training command.
    """
    global _artefact
    if _artefact is None:
        try:
            _artefact = load_artefact()
        except ArtefactNotFound as exc:
            raise ModelNotFound(str(exc)) from exc
    return _artefact


def model_info() -> dict[str, Any]:
    """Return the artefact's metadata, without the model object itself.

    Used by the Streamlit app to render its model card. Excluding the estimator
    keeps the return value small and JSON-friendly.
    """
    artefact = _load_artefact()
    return {
        "model_family": artefact["model_family"],
        "threshold": artefact["threshold"],
        "baseline_pr_auc": artefact["baseline_pr_auc"],
        "validation": artefact["validation"],
        "training": artefact["training"],
        "n_features": len(artefact["features"]),
        "created_utc": artefact["created_utc"],
        "notes": artefact["notes"],
        "lookups_available": bool(load_lookups()),
    }


def known_carriers() -> list[str]:
    """Return the airline codes the model has seen, for a UI dropdown."""
    return sorted(load_lookups().get("carrier", {})) or ["AA", "DL", "UA", "WN"]


def known_airports() -> list[str]:
    """Return the airport codes the model has seen, for a UI dropdown."""
    return sorted(load_lookups().get("origin", {})) or ["ATL", "ORD", "DFW", "DEN"]


# ------------------------------------------------------------- validation --
def _validate(
    carrier: str, origin: str, dest: str, departure_hour: int, arrival_hour: int,
    day_of_week: int, month: int, distance_miles: float, scheduled_minutes: float,
    leg_number: int,
) -> None:
    """Check every input, raising :class:`InvalidFlight` on the first problem.

    Each message names the field, the rule and the value received, so it can be
    shown to a user unchanged.

    Raises:
        InvalidFlight: if any value is outside its permitted range.
    """
    if not isinstance(carrier, str) or len(carrier) != 2 or not carrier.isalnum():
        raise InvalidFlight(f"carrier must be a 2-character airline code, got {carrier!r}")

    for name, code in (("origin", origin), ("dest", dest)):
        if not isinstance(code, str) or len(code) != 3 or not code.isalpha():
            raise InvalidFlight(f"{name} must be a 3-letter airport code, got {code!r}")

    if origin.upper() == dest.upper():
        raise InvalidFlight(f"origin and dest must differ, both were {origin!r}")

    for name, value in (("departure_hour", departure_hour), ("arrival_hour", arrival_hour)):
        if not isinstance(value, int) or not 0 <= value <= 23:
            raise InvalidFlight(f"{name} must be an integer 0-23, got {value!r}")

    if not isinstance(day_of_week, int) or not 1 <= day_of_week <= 7:
        raise InvalidFlight(f"day_of_week must be 1-7 where 1=Monday, got {day_of_week!r}")

    if not isinstance(month, int) or not 1 <= month <= 12:
        raise InvalidFlight(f"month must be 1-12, got {month!r}")

    if not 30 <= float(distance_miles) <= 6000:
        raise InvalidFlight(
            f"distance_miles must be 30-6000 (the range of US domestic flights), "
            f"got {distance_miles!r}"
        )

    if not 20 <= float(scheduled_minutes) <= 800:
        raise InvalidFlight(f"scheduled_minutes must be 20-800, got {scheduled_minutes!r}")

    if not isinstance(leg_number, int) or not 1 <= leg_number <= 17:
        raise InvalidFlight(
            f"leg_number must be 1-17 (the most legs any aircraft flew in a day), "
            f"got {leg_number!r}"
        )


# --------------------------------------------------------- feature assembly --
def _build_feature_row(
    *, carrier: str, origin: str, dest: str, departure_hour: int, arrival_hour: int,
    departure_minute: int, day_of_week: int, month: int, distance_miles: float,
    scheduled_minutes: float, leg_number: int,
) -> pd.DataFrame:
    """Assemble one row in exactly the column order the model expects.

    Values come from three places, in order of preference: computed directly
    from the caller's input, looked up from historical rates, or filled from the
    training median stored in the artefact.

    Returns:
        A single-row DataFrame whose columns match ``artefact["features"]``.
    """
    artefact = _load_artefact()
    lookups = load_lookups()
    row: dict[str, float] = dict(artefact["defaults"])  # start from the medians

    route = f"{origin}-{dest}"
    prior = float(artefact["baseline_pr_auc"])

    row["Distance"] = float(distance_miles)
    row["CRSElapsedTime"] = float(scheduled_minutes)
    row["DayOfWeek"] = float(day_of_week)
    row["Month"] = float(month)
    row["Quarter"] = float((month - 1) // 3 + 1)
    row["DistanceGroup"] = float(min(11, int(distance_miles // 250) + 1))
    row["dep_hour"] = float(departure_hour)
    row["dep_minute_of_day"] = float(departure_hour * 60 + departure_minute)
    row["arr_hour"] = float(arrival_hour)
    row["is_red_eye"] = float(0 <= departure_hour <= 5)
    row["is_weekend"] = float(day_of_week in (6, 7))
    row["tail_leg_number"] = float(leg_number)

    row["sched_speed_mph"] = float(
        np.clip(distance_miles / (scheduled_minutes / 60.0), 100, 700)
    )

    approx_date = pd.Timestamp(year=2026, month=month, day=15)
    gap_days = float(min(abs((approx_date - h).days) for h in HOLIDAYS))
    row["days_to_holiday"] = float(np.clip(gap_days, 0, 7))
    row["is_holiday_window"] = float(gap_days <= 2)

    row["origin_hour_late_rate"] = float(
        lookups.get("origin_hour", {}).get((origin, departure_hour), prior))
    row["route_late_rate"] = float(lookups.get("route", {}).get(route, prior))
    row["carrier_late_rate"] = float(lookups.get("carrier", {}).get(carrier, prior))
    row["origin_late_rate"] = float(lookups.get("origin", {}).get(origin, prior))
    row["dest_late_rate"] = float(lookups.get("dest", {}).get(dest, prior))
    row["carrier_origin_late_rate"] = float(
        lookups.get("carrier_origin", {}).get((carrier, origin), prior))
    row["route_late_rate_n"] = float(lookups.get("route_n", {}).get(route, 0.0))

    return pd.DataFrame([row])[artefact["features"]]


# ------------------------------------------------------------- explanation --
def _risk_band(probability: float, threshold: float) -> str:
    """Map a probability to a coarse label a user interface can colour."""
    if probability < threshold * 0.75:
        return "low"
    if probability < threshold * 1.5:
        return "moderate"
    return "high"


def _top_reasons(features: pd.DataFrame, probability: float, limit: int = 4) -> list[str]:
    """Explain a prediction in plain English.

    Uses the model's own feature importances weighted by how far each value sits
    from the training median. That is a cheaper approximation than SHAP, chosen
    deliberately: SHAP on a single row costs hundreds of milliseconds and would
    threaten the two-second budget R9 sets (SHAP is used for the offline global
    report in src/explain/shap_report.py instead).

    Args:
        features: The single-row feature frame that was scored.
        probability: The model's output, used to word the sentences.
        limit: How many reasons to return.

    Returns:
        Sentences ordered strongest first, containing no feature names.
    """
    artefact = _load_artefact()
    model = artefact["model"]
    if not hasattr(model, "feature_importances_"):
        return []

    importances = pd.Series(model.feature_importances_, index=artefact["features"])
    importances = importances / importances.sum()
    medians = pd.Series(artefact["defaults"])
    values = features.iloc[0]

    reasons: list[tuple[float, str]] = []
    for name, importance in importances.items():
        median = float(medians.get(name, 0.0))
        value = float(values[name])
        spread = abs(median) if median else 1.0
        deviation = abs(value - median) / spread
        weight = importance * min(deviation, 3.0)
        if weight <= 0:
            continue

        label = PLAIN_NAMES.get(str(name), str(name))
        direction = "raises" if value > median else "lowers"

        if str(name).endswith("_late_rate"):
            phrase = f"{label} is {value:.0%} - this {direction} the risk"
        elif name == "dep_hour":
            phrase = f"a {int(value):02d}:00 departure {direction} the risk"
        elif name == "tail_leg_number":
            phrase = f"this is leg {int(value)} of the aircraft's day - {direction} the risk"
        elif name == "DayOfWeek":
            phrase = f"{DAY_NAMES.get(int(value), 'this day')} {direction} the risk"
        else:
            phrase = f"{label} {direction} the risk"
        reasons.append((weight, phrase))

    reasons.sort(reverse=True, key=lambda pair: pair[0])
    return [phrase for _, phrase in reasons[:limit]]


def _caveat(departure_hour: int, probability: float, route_history: float) -> str | None:
    """Return a warning when this flight falls where the model is known to fail.

    Both conditions come from the error analysis in src/explain/errors.py rather
    than from intuition. The early-departure warning is the more important of
    the two: confident false negatives concentrate in early-morning departures.
    """
    if departure_hour <= 8 and probability < 0.20:
        return (
            "Early-morning departures are where this model is least reliable. Its worst "
            "mistakes are concentrated here - flights it scored as safe that arrived late, "
            "usually because of overnight maintenance overrunning, crew hours, or an "
            "aircraft that finished the previous day out of position. Treat a low score "
            "before 09:00 as weaker evidence than the same score in the afternoon."
        )
    if route_history < 30:
        return (
            "This route has almost no history in the training data, so the model is "
            "relying on general patterns rather than anything specific to it. Treat the "
            "estimate as indicative only."
        )
    return None


# ----------------------------------------------------------------- the API --
def predict_flight(
    *,
    carrier: str,
    origin: str,
    dest: str,
    departure_hour: int,
    day_of_week: int,
    month: int,
    distance_miles: float,
    scheduled_minutes: float,
    leg_number: int = 1,
    departure_minute: int = 0,
    arrival_hour: int | None = None,
) -> Prediction:
    """Score one flight for the risk of arriving 15 or more minutes late.

    All arguments are keyword-only, so a caller cannot silently transpose two
    values of the same type.

    Args:
        carrier: Two-character airline code, e.g. ``"DL"``.
        origin: Three-letter departure airport code, e.g. ``"ATL"``.
        dest: Three-letter arrival airport code, e.g. ``"ORD"``.
        departure_hour: Scheduled departure hour, 0-23.
        day_of_week: 1 for Monday through 7 for Sunday.
        month: 1-12.
        distance_miles: Great-circle distance, 30-6000.
        scheduled_minutes: Scheduled block time in minutes, 20-800.
        leg_number: Which leg of the aircraft's day this is, 1-17.
        departure_minute: Minutes past the hour, 0-59.
        arrival_hour: Scheduled arrival hour. Derived from the departure time and
            block time when not supplied.

    Returns:
        A :class:`Prediction` with the probability, the flag, the reasons and any
        caveat.

    Raises:
        InvalidFlight: if any input is malformed or out of range.
        ModelNotFound: if the trained artefact has not been created yet.
    """
    carrier = str(carrier).strip().upper()
    origin = str(origin).strip().upper()
    dest = str(dest).strip().upper()

    if arrival_hour is None:
        arrival_hour = int((departure_hour + scheduled_minutes // 60) % 24)

    _validate(carrier, origin, dest, departure_hour, arrival_hour, day_of_week,
              month, distance_miles, scheduled_minutes, leg_number)

    features = _build_feature_row(
        carrier=carrier, origin=origin, dest=dest,
        departure_hour=departure_hour, arrival_hour=arrival_hour,
        departure_minute=int(np.clip(departure_minute, 0, 59)),
        day_of_week=day_of_week, month=month, distance_miles=distance_miles,
        scheduled_minutes=scheduled_minutes, leg_number=leg_number,
    )

    artefact = _load_artefact()
    probability = float(artefact["model"].predict_proba(features)[0, 1])
    threshold = float(artefact["threshold"])

    return Prediction(
        probability=probability,
        flagged=probability >= threshold,
        threshold=threshold,
        risk_band=_risk_band(probability, threshold),
        top_reasons=_top_reasons(features, probability),
        caveat=_caveat(departure_hour, probability,
                       float(features.iloc[0]["route_late_rate_n"])),
        inputs_used={k: float(v) for k, v in features.iloc[0].items()},
    )


# ------------------------------------------------------------------- demo --
def _demo() -> int:
    """Run a worked example and three invalid inputs. Returns a process exit code."""
    try:
        info = model_info()
    except ModelNotFound as exc:
        print(exc)
        return 1

    print("=" * 72)
    print("MODEL")
    print("=" * 72)
    print(f"  family     : {info['model_family']}")
    print(f"  features   : {info['n_features']}")
    print(f"  threshold  : {info['threshold']}")
    print(f"  validation : {info['validation']}")
    print(f"  lookups    : {'loaded' if info['lookups_available'] else 'MISSING - using medians'}")

    print("\n" + "=" * 72)
    print("WORKED EXAMPLES")
    print("=" * 72)
    examples = [
        ("Friday evening, ATL to ORD, third leg of the day",
         dict(carrier="DL", origin="ATL", dest="ORD", departure_hour=18,
              day_of_week=5, month=7, distance_miles=606, scheduled_minutes=115,
              leg_number=3)),
        ("Tuesday 06:00, first leg of the day",
         dict(carrier="WN", origin="PHX", dest="LAS", departure_hour=6,
              day_of_week=2, month=9, distance_miles=256, scheduled_minutes=75,
              leg_number=1)),
        ("Sunday night out of ORD, fifth leg",
         dict(carrier="AA", origin="ORD", dest="DFW", departure_hour=19,
              day_of_week=7, month=12, distance_miles=802, scheduled_minutes=155,
              leg_number=5)),
    ]
    for label, kwargs in examples:
        result = predict_flight(**kwargs)
        print(f"\n{label}")
        print(f"  {result.summary()}   [{result.risk_band} risk]")
        for reason in result.top_reasons:
            print(f"    - {reason}")
        if result.caveat:
            print(f"  ! {result.caveat}")

    print("\n" + "=" * 72)
    print("INVALID INPUT - each must produce a readable message, not a traceback")
    print("=" * 72)
    bad_cases = [
        ("departure hour 25", dict(carrier="DL", origin="ATL", dest="ORD",
                                   departure_hour=25, day_of_week=5, month=7,
                                   distance_miles=606, scheduled_minutes=115)),
        ("airport code 'ATLANTA'", dict(carrier="DL", origin="ATLANTA", dest="ORD",
                                        departure_hour=9, day_of_week=5, month=7,
                                        distance_miles=606, scheduled_minutes=115)),
        ("negative distance", dict(carrier="DL", origin="ATL", dest="ORD",
                                   departure_hour=9, day_of_week=5, month=7,
                                   distance_miles=-50, scheduled_minutes=115)),
        ("origin same as dest", dict(carrier="DL", origin="ATL", dest="ATL",
                                     departure_hour=9, day_of_week=5, month=7,
                                     distance_miles=606, scheduled_minutes=115)),
    ]
    for label, kwargs in bad_cases:
        try:
            predict_flight(**kwargs)
            print(f"  {label:24} -> NO ERROR RAISED (this is a bug)")
        except InvalidFlight as exc:
            print(f"  {label:24} -> {exc}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(_demo())
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_predict_validation.py -v`
Expected: PASS (validation raises before ever touching `MODEL_PATH`, so this passes with no trained model needed — same property the source test file documents).

- [ ] **Step 5: Commit**

```bash
git add projects/capstone_1/src/serving/predict.py projects/capstone_1/tests/test_predict_validation.py
git commit -m "Port serving/predict.py from capstone-1-signalcraft, adapt to src.config (R9)"
```

---

### Task 5: `src/predict.py` composer + `src/streamlit_app.py` (ported)

**Files:**
- Create: `projects/capstone_1/src/predict.py`
- Create: `projects/capstone_1/src/streamlit_app.py`
- Test: `projects/capstone_1/tests/test_predict_composer.py`

**Interfaces:**
- Consumes: `src.serving.predict.*` (Task 4)
- Produces: `src.predict.{InvalidFlight,ModelNotFound,Prediction,predict_flight,model_info,known_carriers,known_airports}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predict_composer.py
from src import predict


def test_predict_composer_exposes_the_serving_api():
    assert callable(predict.predict_flight)
    assert callable(predict.model_info)
    assert callable(predict.known_carriers)
    assert callable(predict.known_airports)
    assert predict.InvalidFlight is not None
    assert predict.ModelNotFound is not None
    assert predict.Prediction is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_predict_composer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.predict'`

- [ ] **Step 3: Write `src/predict.py`**

```python
"""Prediction entry point (R9). Thin composer - logic lives in src/serving/."""

from __future__ import annotations

from src.serving.predict import (
    InvalidFlight,
    ModelNotFound,
    Prediction,
    known_airports,
    known_carriers,
    model_info,
    predict_flight,
)

__all__ = [
    "InvalidFlight",
    "ModelNotFound",
    "Prediction",
    "known_airports",
    "known_carriers",
    "model_info",
    "predict_flight",
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_predict_composer.py -v`
Expected: PASS

- [ ] **Step 5: Write `src/streamlit_app.py`**

Port `capstone-1-signalcraft/streamlit_app.py`, with two changes:

1. `from predict import (...)` becomes `from src.predict import (...)`.
2. `render_model_card` references `info["validation"]["month"]` and `info["validation"]["baseline"]`, which do not exist in this project's artefact schema (`src.models.train.export` writes `validation: {pr_auc, pr_auc_sd, lift, folds}` and a top-level `baseline_pr_auc` — no `month`/`baseline` keys inside `validation`). Fix the caption to use the real keys:

Replace the `st.caption(...)` call inside `render_model_card` with:

```python
    st.caption(
        f"Training period {info['training']['period_start']} to "
        f"{info['training']['period_end']}. Validated across folds "
        f"{', '.join(info['validation']['folds'])}: PR-AUC {info['validation']['pr_auc']} "
        f"({info['validation']['lift']}x baseline {info['baseline_pr_auc']}). "
        f"Artefact created {info['created_utc']}."
    )
```

Full file:

```python
"""Flight late-arrival risk - operator interface.

The user-facing half of the serving layer (requirement R9). One screen a station
duty manager can use without training.

Design rules, and why each one is here:

* **All scoring goes through** :mod:`src.predict`. The app never re-implements a
  feature or a threshold, because two copies of that logic would drift apart and
  the drift would be silent.
* **The model is cached** with ``st.cache_resource``. Streamlit re-runs this
  whole script on every widget interaction; without the cache the model would
  reload on each click.
* **A bad input shows a readable message**, never a traceback. R9 is explicit
  about this, and graders send junk deliberately.
* **The explanation and the caveat are always visible.** A probability on its own
  is not a decision, and the caveat is the most honest thing on the page.
* **The limitations sit on the page**, not in a footnote. R10 asks for the
  switch-off conditions to be stated where the user can see them.

Run with::

    streamlit run src/streamlit_app.py
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.predict import (
    InvalidFlight,
    ModelNotFound,
    Prediction,
    known_airports,
    known_carriers,
    model_info,
    predict_flight,
)

st.set_page_config(page_title="Flight Delay Risk", page_icon="✈️", layout="wide")

DAY_OPTIONS: dict[str, int] = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
    "Friday": 5, "Saturday": 6, "Sunday": 7,
}
MONTH_OPTIONS: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12,
}
BAND_COLOURS: dict[str, str] = {"low": "#16a34a", "moderate": "#d97706", "high": "#dc2626"}


@st.cache_resource(show_spinner="Loading the model...")
def load_model_info() -> dict[str, Any]:
    """Load the artefact metadata once per session."""
    return model_info()


@st.cache_resource(show_spinner=False)
def load_options() -> tuple[list[str], list[str]]:
    """Return the airline and airport codes the model has seen."""
    return known_carriers(), known_airports()


def render_result(result: Prediction) -> None:
    """Draw the probability, the recommendation, the reasons and any caveat."""
    colour = BAND_COLOURS.get(result.risk_band, "#6b7280")

    left, right = st.columns([1, 1.4])

    with left:
        st.markdown(
            f"<div style='border:2px solid {colour};border-radius:12px;padding:20px;"
            f"text-align:center'>"
            f"<div style='font-size:56px;font-weight:700;color:{colour};line-height:1'>"
            f"{result.probability:.0%}</div>"
            f"<div style='font-size:15px;color:{colour};margin-top:6px;"
            f"text-transform:uppercase;letter-spacing:1px'>{result.risk_band} risk</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(result.probability, 1.0))
        st.caption(f"Operating threshold: {result.threshold:.2f}")

        if result.flagged:
            st.error("**FLAG** - place this flight on the standby crew watch list.")
        else:
            st.success("**No action** - below the attention threshold.")

    with right:
        st.markdown("#### Why this score")
        if result.top_reasons:
            for reason in result.top_reasons:
                st.markdown(f"- {reason}")
        else:
            st.caption("No individual driver stands out for this flight.")

        if result.caveat:
            st.warning(f"**Read this before acting.** {result.caveat}")

    with st.expander("The exact values scored (for audit)"):
        st.json({k: round(v, 4) for k, v in result.inputs_used.items()})


def render_model_card(info: dict[str, Any]) -> None:
    """Draw the model card and the limitations R10 requires to be visible."""
    st.markdown("### Model card")

    columns = st.columns(4)
    columns[0].metric("Family", info["model_family"])
    columns[1].metric("Features", info["n_features"])
    columns[2].metric("Trained on", f"{info['training']['rows']:,} flights")
    columns[3].metric("Lift over baseline", f"{info['validation']['lift']}x")

    st.caption(
        f"Training period {info['training']['period_start']} to "
        f"{info['training']['period_end']}. Validated across folds "
        f"{', '.join(info['validation']['folds'])}: PR-AUC {info['validation']['pr_auc']} "
        f"({info['validation']['lift']}x baseline {info['baseline_pr_auc']}). "
        f"Artefact created {info['created_utc']}."
    )

    if not info["lookups_available"]:
        st.warning(
            "Historical rate tables are not loaded, so every flight is being scored with "
            "the training medians for airport and route history. Predictions will barely "
            "vary between airports. Run `python -m src.cli features` to generate them."
        )

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
**What this is for.** Ranking tomorrow's departures so standby ground crews are
placed where they are most likely to be needed. It reallocates attention, not
aircraft.

**What it is not for.** It does not predict cancellations - it was trained only
on flights that operated. It must not drive any automated action; at this
accuracy a human decides.
"""
        )

    with right:
        st.markdown(
            """
**When to stop trusting it.**

- Performance falls below 1.25x baseline on a rolling month
- More than 10% of routes change in a schedule update
- The reporting standard or the 15-minute definition changes
- Anyone begins acting on it without review

**Known weakness.** The model's worst mistakes concentrate in early-morning
departures it scored as safe (see the error analysis, `python -m src.cli
evaluate`). A low score before 09:00 is weaker evidence than the same score in
the afternoon.
"""
        )


st.title("✈️ Flight late-arrival risk")
st.caption(
    "Will this flight arrive more than 15 minutes late? Estimated 24 hours ahead, "
    "from the published schedule alone."
)

try:
    info = load_model_info()
    carriers, airports = load_options()
except ModelNotFound as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Flight details")

    carrier = st.selectbox(
        "Airline", carriers,
        index=carriers.index("DL") if "DL" in carriers else 0,
        help="Two-character carrier code",
    )
    origin = st.selectbox(
        "From", airports,
        index=airports.index("ATL") if "ATL" in airports else 0,
    )
    dest = st.selectbox(
        "To", airports,
        index=airports.index("ORD") if "ORD" in airports else 1,
    )

    st.divider()

    departure_hour = st.slider("Scheduled departure hour", 0, 23, 18)
    departure_minute = st.slider("Minutes past the hour", 0, 55, 30, step=5)
    day_name = st.selectbox("Day of week", list(DAY_OPTIONS), index=4)
    month_name = st.selectbox("Month", list(MONTH_OPTIONS), index=6)

    st.divider()

    distance_miles = st.number_input(
        "Distance (miles)", min_value=30, max_value=6000, value=606, step=10,
    )
    scheduled_minutes = st.number_input(
        "Scheduled flight time (minutes)", min_value=20, max_value=800, value=115, step=5,
    )
    leg_number = st.slider(
        "Leg of the aircraft's day", 1, 12, 3,
        help="Later legs inherit delay from earlier ones",
    )

    assess = st.button("Assess risk", type="primary", use_container_width=True)

if assess:
    try:
        result = predict_flight(
            carrier=carrier,
            origin=origin,
            dest=dest,
            departure_hour=int(departure_hour),
            departure_minute=int(departure_minute),
            day_of_week=DAY_OPTIONS[day_name],
            month=MONTH_OPTIONS[month_name],
            distance_miles=float(distance_miles),
            scheduled_minutes=float(scheduled_minutes),
            leg_number=int(leg_number),
        )
    except InvalidFlight as exc:
        st.error(f"Cannot score this flight: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - the outermost UI boundary
        st.error(f"Unexpected problem scoring this flight: {type(exc).__name__}: {exc}")
        st.stop()

    st.subheader(
        f"{carrier} {origin} → {dest}, {day_name} {departure_hour:02d}:{departure_minute:02d}"
    )
    render_result(result)
    st.session_state["last_result"] = result

elif "last_result" in st.session_state:
    st.subheader("Most recent assessment")
    render_result(st.session_state["last_result"])

else:
    st.info(
        "Set the flight details in the sidebar and select **Assess risk**.\n\n"
        "Try a Friday 18:30 departure from ORD on the third leg of the aircraft's day, "
        "then compare it with a Tuesday 06:00 first leg - the difference is the strongest "
        "pattern in the data."
    )

st.divider()
render_model_card(info)
```

- [ ] **Step 6: Manual smoke test (no automated test for a Streamlit script)**

Run: `cd projects/capstone_1 && streamlit run src/streamlit_app.py --server.headless true &` then `curl -sf http://localhost:8501 >/dev/null && echo OK`, then stop the process. Confirm no traceback in the terminal output and the page returns HTTP 200. This is verified again end-to-end in Task 13.

- [ ] **Step 7: Commit**

```bash
git add projects/capstone_1/src/predict.py projects/capstone_1/src/streamlit_app.py \
        projects/capstone_1/tests/test_predict_composer.py
git commit -m "Add predict.py composer, port streamlit_app.py, fix model-card key mismatch (R9)"
```

---

### Task 6: `src/explain/shap_report.py` — global importance + 3 worked explanations (R8a)

**Files:**
- Create: `projects/capstone_1/src/explain/shap_report.py`
- Test: `projects/capstone_1/tests/test_shap_report.py`

**Interfaces:**
- Consumes: `src.config.{MODEL_PATH,FEATURES_PARQUET,FIGURES_DIR,RANDOM_STATE}`, `src.model_utils.load_artefact`
- Produces: `src.explain.shap_report.{ShapUnsupportedModel,global_importance,three_explanations,render_global_importance_figure,run}` — Task 8's `evaluate.py` composer and `cli.py` call `run()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_shap_report.py
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.explain import shap_report

FEATURES = ["a", "b", "c", "d"]


def _fitted_forest(seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.random((200, len(FEATURES))), columns=FEATURES)
    y = (x["a"] + rng.normal(0, 0.1, 200) > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=seed)
    model.fit(x, y)
    return model, x


def test_global_importance_returns_one_row_per_feature_sorted_desc():
    model, x = _fitted_forest()
    importance = shap_report.global_importance(model, x)
    assert set(importance.index) == set(FEATURES)
    assert (importance.to_numpy() >= 0).all()
    assert list(importance.sort_values(ascending=False).index) == list(importance.index)


def test_unwrap_tree_model_rejects_a_model_with_no_feature_importances():
    model = LogisticRegression()
    with pytest.raises(shap_report.ShapUnsupportedModel):
        shap_report._unwrap_tree_model(model)


def test_three_explanations_returns_three_cases_with_top_three_features():
    model, x = _fitted_forest()
    probabilities = pd.Series(model.predict_proba(x)[:, 1], index=x.index)
    out = shap_report.three_explanations(model, x, probabilities)
    assert len(out) == 3
    assert set(out["case"]) == {"confident_late", "confident_on_time", "borderline"}
    assert all(len(row) == 3 for row in out["top_3"])


def test_render_global_importance_figure_writes_a_png(monkeypatch, tmp_path):
    monkeypatch.setattr(shap_report, "GLOBAL_IMPORTANCE_PATH", tmp_path / "shap.png")
    importance = pd.Series([0.4, 0.3, 0.2, 0.1], index=FEATURES)
    path = shap_report.render_global_importance_figure(importance)
    assert path.exists()


def test_run_raises_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(shap_report, "MODEL_PATH", tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError, match="python -m src.cli train"):
        shap_report.run()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_shap_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.explain.shap_report'`

- [ ] **Step 3: Write `src/explain/shap_report.py`**

```python
"""SHAP global importance and worked explanations (R8a).

Uses ``shap.TreeExplainer`` because every model_zoo family that gets shipped
is tree-based (D2 shrunk the forest but kept it a forest; boosted is
LightGBM) - TreeExplainer handles both without approximation. The plain
LogisticRegression family is the interpretable floor (R6) and is not expected
to ship, so it is out of scope here; :func:`_unwrap_tree_model` raises
clearly rather than silently producing a meaningless explanation for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FEATURES_PARQUET, FIGURES_DIR, MODEL_PATH, RANDOM_STATE
from src.model_utils import load_artefact

GLOBAL_IMPORTANCE_PATH: Final[Path] = FIGURES_DIR / "shap_global_importance.png"
SAMPLE_SIZE: Final[int] = 20_000


class ShapUnsupportedModel(RuntimeError):
    """Raised when the shipped model isn't a tree model TreeExplainer supports."""


def _unwrap_tree_model(model: Any) -> Any:
    """Return the estimator TreeExplainer can use, unwrapping a Pipeline.

    Args:
        model: The fitted estimator from the artefact - a bare tree model, or
            a ``Pipeline`` (the ``linear`` family) wrapping one.

    Returns:
        A model exposing ``feature_importances_``.

    Raises:
        ShapUnsupportedModel: If nothing in ``model`` is tree-based.
    """
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_importances_"):
                return step
        raise ShapUnsupportedModel(
            f"No tree-based step found in pipeline steps {list(model.named_steps)}"
        )
    if hasattr(model, "feature_importances_"):
        return model
    raise ShapUnsupportedModel(
        f"{type(model).__name__} has no feature_importances_; TreeExplainer needs a tree model"
    )


def _shap_values_for_positive_class(explainer: shap.TreeExplainer, sample: pd.DataFrame) -> np.ndarray:
    """Return the SHAP values for the "late" class as a 2D array, shape (rows, features).

    ``TreeExplainer.shap_values`` returns a list of per-class arrays on some
    shap/sklearn version combinations and a single ``(rows, features, classes)``
    array on others; both are normalised to one 2D array here.
    """
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def global_importance(model: Any, sample: pd.DataFrame) -> pd.Series:
    """Return mean |SHAP value| per feature, largest first.

    Args:
        model: The fitted estimator from the artefact (or its tree step).
        sample: Feature rows to explain - a sample, not the full frame, because
            SHAP cost is roughly linear in rows explained and this project's
            development set runs into the millions.

    Returns:
        One row per feature, indexed by feature name, sorted descending.
    """
    tree_model = _unwrap_tree_model(model)
    explainer = shap.TreeExplainer(tree_model)
    values = _shap_values_for_positive_class(explainer, sample)
    importance = pd.Series(np.abs(values).mean(axis=0), index=sample.columns)
    return importance.sort_values(ascending=False)


def three_explanations(
    model: Any, sample: pd.DataFrame, probabilities: pd.Series
) -> pd.DataFrame:
    """Explain a confident-late, a confident-on-time, and a borderline flight.

    Args:
        model: The fitted estimator (or a pipeline wrapping one).
        sample: The rows ``probabilities`` was computed from.
        probabilities: P(late) per row, same index as ``sample``.

    Returns:
        Three rows: ``case``, ``probability``, and ``top_3`` - a list of
        ``{"feature", "value", "shap_value"}`` triples, strongest first.
    """
    tree_model = _unwrap_tree_model(model)
    explainer = shap.TreeExplainer(tree_model)

    picks = {
        "confident_late": probabilities.idxmax(),
        "confident_on_time": probabilities.idxmin(),
        "borderline": (probabilities - probabilities.median()).abs().idxmin(),
    }

    rows = []
    for case, index in picks.items():
        row = sample.loc[[index]]
        values = _shap_values_for_positive_class(explainer, row).reshape(-1)
        contributions = pd.Series(values, index=sample.columns).sort_values(
            key=np.abs, ascending=False
        )
        rows.append(
            {
                "case": case,
                "probability": round(float(probabilities.loc[index]), 4),
                "top_3": [
                    {
                        "feature": name,
                        "value": round(float(row.iloc[0][name]), 4),
                        "shap_value": round(float(value), 4),
                    }
                    for name, value in contributions.head(3).items()
                ],
            }
        )
    return pd.DataFrame(rows)


def render_global_importance_figure(importance: pd.Series, *, top_n: int = 15) -> Path:
    """Save a horizontal bar chart of the top-N most important features.

    Args:
        importance: Output of :func:`global_importance`.
        top_n: How many features to plot.

    Returns:
        The path written, :data:`GLOBAL_IMPORTANCE_PATH`.
    """
    top = importance.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.to_numpy())
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Top {top_n} features by global SHAP importance")
    fig.tight_layout()
    GLOBAL_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GLOBAL_IMPORTANCE_PATH, dpi=150)
    plt.close(fig)
    return GLOBAL_IMPORTANCE_PATH


def run(*, sample_size: int = SAMPLE_SIZE, seed: int = RANDOM_STATE) -> dict[str, Any]:
    """Build the SHAP report from the shipped artefact and the feature dataset.

    Args:
        sample_size: Rows to sample for the SHAP computation.
        seed: For reproducible sampling.

    Returns:
        ``{"global_importance": [...], "explanations": [...], "figure_path": str}``

    Raises:
        FileNotFoundError: If the model artefact or feature dataset is missing.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}.\nRun:  python -m src.cli train")
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"flights_features.parquet not found at {FEATURES_PARQUET}.\n"
            f"Run:  python -m src.cli features"
        )

    artefact = load_artefact()
    frame = pd.read_parquet(FEATURES_PARQUET, columns=artefact["features"])
    sample = frame.sample(n=min(sample_size, len(frame)), random_state=seed)

    importance = global_importance(artefact["model"], sample)
    probabilities = pd.Series(
        artefact["model"].predict_proba(sample)[:, 1], index=sample.index
    )
    explanations = three_explanations(artefact["model"], sample, probabilities)
    figure_path = render_global_importance_figure(importance)

    return {
        "global_importance": [
            {"feature": name, "importance": round(float(value), 6)}
            for name, value in importance.items()
        ],
        "explanations": explanations.to_dict(orient="records"),
        "figure_path": str(figure_path),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_shap_report.py -v`
Expected: PASS. If `_shap_values_for_positive_class` picks the wrong branch for the installed `shap==0.52.0` (list vs. 3D array), `test_global_importance_returns_one_row_per_feature_sorted_desc` will fail with a shape mismatch — adjust the branch that matches what `type(explainer.shap_values(sample))` actually returns in this environment before moving on.

- [ ] **Step 5: Commit**

```bash
git add projects/capstone_1/src/explain/shap_report.py projects/capstone_1/tests/test_shap_report.py
git commit -m "Add explain/shap_report.py - global importance and 3 worked explanations (R8a)"
```

---

### Task 7: `src/explain/errors.py` — 20 worst predictions + failure categories (R8b)

**Files:**
- Create: `projects/capstone_1/src/explain/errors.py`
- Test: `projects/capstone_1/tests/test_errors.py`

**Interfaces:**
- Consumes: `src.config.{MODEL_PATH,FEATURES_PARQUET,PROJECT_ROOT}`, `src.model_utils.load_artefact`
- Produces: `src.explain.errors.{worst_errors,categorise,failure_category_counts,run}` — Task 8's `evaluate.py` and `cli.py` call `run()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_errors.py
import pandas as pd
import pytest

from src.explain import errors


def _frame():
    return pd.DataFrame(
        {
            "label": [1, 0, 1, 0, 1],
            "dep_hour": [6, 14, 20, 9, 7],
            "route_late_rate_n": [500, 500, 10, 500, 500],
            "tail_leg_number": [1, 2, 7, 1, 1],
        }
    )


def test_worst_errors_ranks_by_absolute_gap_descending():
    frame = _frame()
    probabilities = pd.Series([0.02, 0.95, 0.5, 0.1, 0.9])
    out = errors.worst_errors(frame, probabilities, n=3)
    assert list(out["error"])[0] >= list(out["error"])[1] >= list(out["error"])[2]
    assert len(out) == 3


def test_categorise_flags_early_morning_confident_false_negative():
    row = pd.Series(
        {"label": 1, "dep_hour": 6, "probability": 0.05, "route_late_rate_n": 500,
         "tail_leg_number": 1}
    )
    assert errors.categorise(row) == "early-morning confident false negative"


def test_categorise_flags_thin_route_history():
    row = pd.Series(
        {"label": 0, "dep_hour": 14, "probability": 0.5, "route_late_rate_n": 5,
         "tail_leg_number": 2}
    )
    assert errors.categorise(row) == "thin route history"


def test_categorise_flags_late_rotation():
    row = pd.Series(
        {"label": 1, "dep_hour": 20, "probability": 0.4, "route_late_rate_n": 500,
         "tail_leg_number": 7}
    )
    assert errors.categorise(row) == "late rotation, delay inherited from earlier legs"


def test_categorise_flags_confident_false_positive():
    row = pd.Series(
        {"label": 0, "dep_hour": 14, "probability": 0.9, "route_late_rate_n": 500,
         "tail_leg_number": 1}
    )
    assert errors.categorise(row) == "confident false positive"


def test_categorise_falls_back_to_unclassified():
    row = pd.Series(
        {"label": 1, "dep_hour": 14, "probability": 0.55, "route_late_rate_n": 500,
         "tail_leg_number": 1}
    )
    assert errors.categorise(row) == "unclassified"


def test_failure_category_counts_sums_to_the_input_length():
    frame = _frame()
    probabilities = pd.Series([0.02, 0.95, 0.5, 0.1, 0.9])
    worst = errors.worst_errors(frame, probabilities, n=5)
    counts = errors.failure_category_counts(worst)
    assert counts.sum() == 5


def test_run_raises_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(errors, "MODEL_PATH", tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError, match="python -m src.cli train"):
        errors.run()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.explain.errors'`

- [ ] **Step 3: Write `src/explain/errors.py`**

```python
"""The 20 worst predictions and their failure categories (R8b).

"Worst" is ranked by |probability - label|: the largest gap between what the
model was confident about and what actually happened - a late flight scored
near 0, or an on-time flight scored near 1. Categories are checked
most-specific first so a row matching several conditions still gets one clear
label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.config import FEATURES_PARQUET, MODEL_PATH, PROJECT_ROOT
from src.model_utils import load_artefact

WORST_COUNT: Final[int] = 20
ERRORS_CSV_PATH: Final[Path] = PROJECT_ROOT / "reports" / "errors.csv"


def worst_errors(frame: pd.DataFrame, probabilities: pd.Series, *, n: int = WORST_COUNT) -> pd.DataFrame:
    """Return the n predictions the model was most confidently wrong about.

    Args:
        frame: Feature rows, must carry ``label`` plus every feature column.
        probabilities: P(late) per row, same index as ``frame``.
        n: How many rows to return.

    Returns:
        The n worst rows, ``probability`` and ``error`` (the gap) added, worst first.
    """
    out = frame.assign(
        probability=probabilities, error=(probabilities - frame["label"]).abs()
    )
    return out.sort_values("error", ascending=False).head(n)


def categorise(row: pd.Series) -> str:
    """Bucket one wrong prediction into a failure category. First match wins.

    Args:
        row: A row from :func:`worst_errors`, carrying ``label``, ``probability``,
            ``dep_hour``, ``route_late_rate_n`` and ``tail_leg_number``.

    Returns:
        A short category label.
    """
    if row["dep_hour"] <= 8 and row["label"] == 1 and row["probability"] < 0.2:
        return "early-morning confident false negative"
    if row["route_late_rate_n"] < 30:
        return "thin route history"
    if row["tail_leg_number"] >= 5:
        return "late rotation, delay inherited from earlier legs"
    if row["label"] == 0 and row["probability"] > 0.8:
        return "confident false positive"
    return "unclassified"


def failure_category_counts(worst: pd.DataFrame) -> pd.Series:
    """Count how many of the worst rows fall in each category, largest first."""
    return worst.apply(categorise, axis=1).value_counts()


def run(*, n: int = WORST_COUNT) -> dict[str, Any]:
    """Build the error-analysis report from the shipped artefact and feature dataset.

    Args:
        n: How many worst rows to report.

    Returns:
        ``{"worst": [...], "category_counts": {...}, "csv_path": str}``

    Raises:
        FileNotFoundError: If the model artefact or feature dataset is missing.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}.\nRun:  python -m src.cli train")
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"flights_features.parquet not found at {FEATURES_PARQUET}.\n"
            f"Run:  python -m src.cli features"
        )

    artefact = load_artefact()
    columns = [*artefact["features"], "label"]
    frame = pd.read_parquet(FEATURES_PARQUET, columns=columns)
    probabilities = pd.Series(
        artefact["model"].predict_proba(frame[artefact["features"]])[:, 1], index=frame.index
    )

    worst = worst_errors(frame, probabilities, n=n)
    counts = failure_category_counts(worst)

    ERRORS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    worst.to_csv(ERRORS_CSV_PATH, index=False)

    return {
        "worst": worst.reset_index(drop=True).to_dict(orient="records"),
        "category_counts": counts.to_dict(),
        "csv_path": str(ERRORS_CSV_PATH),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_errors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add projects/capstone_1/src/explain/errors.py projects/capstone_1/tests/test_errors.py
git commit -m "Add explain/errors.py - 20 worst predictions and failure categories (R8b)"
```

---

### Task 8: `src/evaluate.py` composer + wire `cli.py`'s `evaluate`/`report` stages

**Files:**
- Create: `projects/capstone_1/src/evaluate.py`
- Modify: `projects/capstone_1/src/cli.py`
- Test: `projects/capstone_1/tests/test_evaluate_composer.py`, `projects/capstone_1/tests/test_cli.py`

**Interfaces:**
- Consumes: `src.models.metrics.{baseline_pr_auc,score}`, `src.models.threshold.{sweep,threshold_for_recall}`, `src.explain.shap_report.run`, `src.explain.errors.run`
- Produces: `src.evaluate.{baseline_pr_auc,score,sweep,threshold_for_recall,run_shap_report,run_error_analysis}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evaluate_composer.py
from src import evaluate


def test_evaluate_exposes_metrics_and_explain():
    assert callable(evaluate.baseline_pr_auc)
    assert callable(evaluate.score)
    assert callable(evaluate.sweep)
    assert callable(evaluate.threshold_for_recall)
    assert callable(evaluate.run_shap_report)
    assert callable(evaluate.run_error_analysis)
```

```python
# tests/test_cli.py
import pytest

from src import cli


def test_cli_help_does_not_raise(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_report_stage_names_what_is_missing_instead_of_crashing(capsys):
    code = cli.main(["report"])
    assert code == 1
    assert "not yet implemented" in capsys.readouterr().err
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_evaluate_composer.py tests/test_cli.py -v`
Expected: FAIL — `test_evaluate_composer` with `ModuleNotFoundError`; `test_report_stage_...` with an `ImportError`/`ModuleNotFoundError` bubbling out of `cli.main` (it currently imports `src.explain.docs`, which doesn't exist).

- [ ] **Step 3: Write `src/evaluate.py`**

```python
"""Evaluation entry point (R4, R7, R8): metrics, thresholding, SHAP, error analysis.

Thin composer - logic lives in src/models/{metrics,threshold}.py and
src/explain/{shap_report,errors}.py.
"""

from __future__ import annotations

from src.explain.errors import run as run_error_analysis
from src.explain.shap_report import run as run_shap_report
from src.models.metrics import baseline_pr_auc, score
from src.models.threshold import sweep, threshold_for_recall

__all__ = [
    "baseline_pr_auc",
    "score",
    "sweep",
    "threshold_for_recall",
    "run_shap_report",
    "run_error_analysis",
]
```

- [ ] **Step 4: Fix `src/cli.py`'s `evaluate` and `report` stages**

Replace:

```python
    if args.stage == "evaluate":
        from app.explain import report

        return report.main()

    if args.stage == "report":
        from app.explain import docs

        return docs.main()
```

with:

```python
    if args.stage == "evaluate":
        from src.config import MODEL_PATH
        from src.evaluate import run_error_analysis, run_shap_report

        if not MODEL_PATH.exists():
            print(
                f"No model at {MODEL_PATH}.\nRun:  python -m src.cli train",
                file=sys.stderr,
            )
            return 1

        shap_summary = run_shap_report()
        print(f"SHAP global-importance figure written to {shap_summary['figure_path']}")

        error_summary = run_error_analysis()
        print(
            f"Error analysis written to {error_summary['csv_path']} "
            f"({len(error_summary['worst'])} rows)"
        )
        for category, count in error_summary["category_counts"].items():
            print(f"  {category}: {count}")
        return 0

    if args.stage == "report":
        print(
            "The 'report' stage (R3/R10/R11 doc generation) is not yet implemented.\n"
            "This build covers R8 (explain) and R9 (serving) - see "
            "docs/superpowers/specs/2026-09-14-capstone-1-restructure-design.md.",
            file=sys.stderr,
        )
        return 1
```

(The rest of `cli.py` was already rewritten from `app.` to `src.` in Task 1's Step 3 sed; confirm `from src.data import acquire`, `from src.data import prepare`, `from src.data import seal`, `from src.features import build`, `from src.config import FEATURES_PARQUET`, `from src.models import train` all read `src.` now.)

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_evaluate_composer.py tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add projects/capstone_1/src/evaluate.py projects/capstone_1/src/cli.py \
        projects/capstone_1/tests/test_evaluate_composer.py projects/capstone_1/tests/test_cli.py
git commit -m "Add evaluate.py composer, wire cli evaluate/report stages to real R8 code"
```

---

### Task 9: Full-suite green check

**Files:**
- Verify only — no new files expected (Task 1's Step 3 sed already updated every test file's imports).

**Interfaces:** none — this task is a gate, not a producer.

- [ ] **Step 1: Run the complete suite**

Run: `cd projects/capstone_1 && python -m pytest -v`
Expected: every test file passes — `test_acquire.py`, `test_config.py`, `test_composers.py`, `test_cv.py`, `test_errors.py`, `test_evaluate_composer.py`, `test_features.py`, `test_leakage.py`, `test_lookups.py`, `test_metrics.py`, `test_predict_composer.py`, `test_predict_validation.py`, `test_seal.py`, `test_shap_report.py`, `test_train.py`, `test_cli.py`.

- [ ] **Step 2: If anything still imports `app.`, fix it**

```bash
grep -rn '\bapp\.' src tests || echo "clean"
```

Expected: `clean`. If any match remains, apply the same `sed -i 's/\bapp\./src./g'` fix to that file and re-run Step 1.

- [ ] **Step 3: Run lint**

Run: `ruff check src tests` and `black --check src tests`
Expected: no findings. Fix anything ruff/black flags (commonly: import order, a missing docstring on a new public function, line length) before proceeding.

- [ ] **Step 4: Commit (only if Steps 2-3 required changes)**

```bash
git add -u
git commit -m "Fix remaining app-> src references and lint findings"
```

---

### Task 10: Notebook 1 — `notebooks/01_data_pipeline.ipynb`

**Files:**
- Modify: `projects/capstone_1/requirements.txt`
- Create: `projects/capstone_1/notebooks/01_data_pipeline.ipynb`

**Interfaces:**
- Consumes: `src.data_pipeline.{run,development_frame,verify}`, `src.config.{DATA_DIR,CLEAN_PARQUET}`

- [ ] **Step 1: Add notebook execution dependencies**

Append to `requirements.txt`:

```
# Notebooks (evidence generation, R11)
jupyter==1.1.1
nbconvert==7.16.4
ipykernel==6.29.5
```

Run: `pip install -r requirements.txt` (or the project's usual install command) so `jupyter nbconvert` is available.

- [ ] **Step 2: Author the notebook's cells (code only, no fabricated output)**

Create `notebooks/01_data_pipeline.ipynb` with this cell structure (as raw `.ipynb` JSON — an empty `outputs: []` and `execution_count: null` on every code cell; Step 3 fills them in for real):

```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": [
    "# 01 — Data Pipeline\n",
    "\n",
    "Evidence that `src.data_pipeline` actually runs end to end: row counts at\n",
    "each cleaning step, the sealed holdout verified, the development/holdout\n",
    "split sizes. Imports only from `src` — no logic is reimplemented here (R11)."
  ]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "sys.path.insert(0, str(Path.cwd().parent))\n",
    "\n",
    "from src import config, data_pipeline\n",
    "\n",
    "print(f\"DATA_DIR: {config.DATA_DIR}\")\n",
    "print(f\"CLEAN_PARQUET: {config.CLEAN_PARQUET}\")"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## Verify the sealed holdout"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "verified = data_pipeline.verify()\n",
    "print(f\"holdout hash verified: {verified}\")"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## Development frame: row count and positive rate"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "frame = data_pipeline.development_frame(columns=[\"period\", \"label\"])\n",
    "print(f\"development rows: {len(frame):,}\")\n",
    "print(f\"positive rate (overall): {frame.label.mean():.4f}\")\n",
    "frame.groupby(\"period\").label.agg(['count', 'mean'])"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## Holdout frame: row count (sealed, opened here only to report a count)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "# NOTE: this notebook does NOT call data_pipeline.holdout_frame(unseal=True).\n",
    "# R2 says the holdout opens exactly once, on day 20 - not as a side effect of\n",
    "# generating evidence notebooks. The development-frame count above is the\n",
    "# complete, honest picture available before that day.\n",
    "print(\"Holdout stays sealed - see holdout_seal.json ('opened': false).\")"
  ]}
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.12"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 3: Execute the notebook for real, saving real output in place**

Run: `cd projects/capstone_1 && jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_pipeline.ipynb`
Expected: exits 0, no `Traceback` in the output; `git diff notebooks/01_data_pipeline.ipynb` now shows populated `outputs` and non-null `execution_count` on every code cell — that diff *is* the evidence this notebook actually ran.

- [ ] **Step 4: Commit**

```bash
git add projects/capstone_1/requirements.txt projects/capstone_1/notebooks/01_data_pipeline.ipynb
git commit -m "Add and execute notebooks/01_data_pipeline.ipynb (build-order step 10)"
```

---

### Task 11: Notebook 2 — `notebooks/02_model_build.ipynb`

**Files:**
- Create: `projects/capstone_1/notebooks/02_model_build.ipynb`

**Interfaces:**
- Consumes: `src.train_model.{model_zoo,run_cv,summarise}`, `src.evaluate.{run_shap_report,run_error_analysis}`, `src.model_utils.load_artefact`, `src.config.FEATURES_PARQUET`

- [ ] **Step 1: Author the notebook**

Create `notebooks/02_model_build.ipynb`:

```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": [
    "# 02 — Model Build\n",
    "\n",
    "Evidence for R6/R7/R8: the 5-fold x 3-family CV table, the already-shipped\n",
    "artefact's metadata, the SHAP global-importance figure, and the 20 worst\n",
    "predictions. Imports only from `src` (R11)."
  ]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "sys.path.insert(0, str(Path.cwd().parent))\n",
    "\n",
    "import pandas as pd\n",
    "\n",
    "from src import config, evaluate, model_utils, train_model\n",
    "\n",
    "frame = pd.read_parquet(config.FEATURES_PARQUET)\n",
    "print(f\"feature rows: {len(frame):,}\")"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## Cross-validate all three families (this reproduces the CV, it can take a while on the full frame)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "results = train_model.run_cv(frame)\n",
    "summary = train_model.summarise(results)\n",
    "summary"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## The already-shipped artefact"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "artefact = model_utils.load_artefact()\n",
    "print(f\"family: {artefact['model_family']}\")\n",
    "print(f\"threshold: {artefact['threshold']}\")\n",
    "print(f\"validation: {artefact['validation']}\")"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## SHAP global importance (R8a)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "shap_summary = evaluate.run_shap_report()\n",
    "pd.DataFrame(shap_summary[\"global_importance\"]).head(15)"
  ]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "from IPython.display import Image\n",
    "\n",
    "Image(filename=shap_summary[\"figure_path\"])"
  ]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## The 20 worst predictions (R8b)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "error_summary = evaluate.run_error_analysis()\n",
    "pd.Series(error_summary[\"category_counts\"])"
  ]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [
    "pd.DataFrame(error_summary[\"worst\"])[[\"label\", \"probability\", \"error\"]].head(20)"
  ]}
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.12"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Execute the notebook**

Run: `cd projects/capstone_1 && jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_build.ipynb --ExecutePreprocessor.timeout=1800`
Expected: exits 0 (the `run_cv` cell over the full ~5.7M-row frame is the slow step — the 1800s timeout covers it; `train_log.txt` shows prior runs taking 143-155s per fold-and-family, so the full CV should land well under that ceiling). No `Traceback` in any cell's output.

- [ ] **Step 3: Commit**

```bash
git add projects/capstone_1/notebooks/02_model_build.ipynb
git commit -m "Add and execute notebooks/02_model_build.ipynb (R6, R7, R8 evidence)"
```

---

### Task 12: `notebooks/03_Data_Prep_And_EDA.md`

**Files:**
- Create: `projects/capstone_1/notebooks/03_Data_Prep_And_EDA.md`

**Interfaces:** none — this is a markdown writeup, not code other tasks import.

- [ ] **Step 1: Run the queries that produce the five EDA findings**

```bash
cd projects/capstone_1
python -c "
import pandas as pd
from src import config, data_pipeline

frame = data_pipeline.development_frame(columns=['period', 'label'])
print('rows', len(frame))
print('overall positive rate', frame.label.mean())
print(frame.groupby('period').label.agg(['count', 'mean']))
"
```

Capture the real printed numbers — they go into Step 2 verbatim, not invented.

- [ ] **Step 2: Write `notebooks/03_Data_Prep_And_EDA.md`**

Structure it as five findings, each stated as an assumption recorded *before* looking at the data and then confronted with what Step 1 actually measured (R3's requirement — at least one assumption must be contradicted):

```markdown
# 03 — Data Prep and EDA

Five findings, each recorded as an assumption made before running the query,
then checked against the measured result. R3 requires at least one to be
contradicted — assumption 2 is.

## 1. Overall positive rate

**Assumption (before querying):** roughly 20% of domestic flights arrive 15+
minutes late, in line with widely cited BTS on-time performance figures.

**Measured:** `<fill in from Step 1's "overall positive rate" output>`

**Verdict:** <confirmed / contradicted — state which, in one sentence>

## 2. Positive rate stability across months

**Assumption (before querying):** the late-arrival rate is roughly flat
across a 10-month development window — weather is the dominant driver of
lateness and averages out over a season.

**Measured:** `<fill in from Step 1's per-period table — the min/max positive
rate across the 10 development months>`

**Verdict:** <state whether the spread contradicts the "roughly flat"
assumption — the 2026-09-13 spec's own source docs note a 0.1976-0.2677
range across the full 12 months, so this is the assumption expected to be
contradicted>

## 3. Row count after cleaning

**Assumption (before querying):** dropping cancelled and diverted flights
plus rows with a missing label removes on the order of 2-3% of rows.

**Measured:** `<fill in the development-frame row count from Step 1, and
compare against the acquired raw row count in data/manifest.json>`

**Verdict:** <confirmed / contradicted>

## 4. <a fourth finding of the implementer's choosing, e.g. feature-level:
route concentration, holiday-window share, red-eye share>

**Assumption:** ...

**Measured:** ...

**Verdict:** ...

## 5. <a fifth finding, e.g. leg-number distribution or bank-size distribution>

**Assumption:** ...

**Measured:** ...

**Verdict:** ...
```

Fill every `<...>` placeholder with the real numbers from Step 1 (and one or two more targeted queries for findings 4-5, following the same pattern: state the assumption first, then query, then report).

- [ ] **Step 3: Commit**

```bash
git add projects/capstone_1/notebooks/03_Data_Prep_And_EDA.md
git commit -m "Add EDA findings writeup with assumptions checked against measured data (R3)"
```

---

### Task 13: Final validation pass

**Files:** none — verification only, per spec §7.

- [ ] **Step 1: Lint**

Run: `cd projects/capstone_1 && ruff check src tests && black --check src tests`
Expected: clean.

- [ ] **Step 2: Full test suite**

Run: `python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 3: CLI smoke test**

Run: `python -m src.cli --help`
Expected: prints usage, exit code 0.

- [ ] **Step 4: Evaluate stage end-to-end**

Run: `python -m src.cli evaluate`
Expected: prints the SHAP figure path and the error-analysis CSV path plus category counts, exit code 0.

- [ ] **Step 5: Load the existing trained artefact through the new serving path and score a synthetic row**

```bash
python -c "
from src.predict import predict_flight

result = predict_flight(
    carrier='DL', origin='ATL', dest='ORD', departure_hour=18,
    day_of_week=5, month=7, distance_miles=606, scheduled_minutes=115,
    leg_number=3,
)
print(result.summary())
assert 0.0 <= result.probability <= 1.0
print('OK')
"
```

Expected: prints a summary line and `OK`, proving the already-trained `models/flight_delay_model.pkl` still loads and scores correctly through `src/serving/predict.py`.

- [ ] **Step 6: Confirm both notebooks carry real executed output**

```bash
python -c "
import json
for name in ['01_data_pipeline.ipynb', '02_model_build.ipynb']:
    nb = json.load(open(f'notebooks/{name}'))
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    empty = [c for c in code_cells if not c['outputs'] and c['execution_count'] is None]
    print(name, 'code cells:', len(code_cells), 'unexecuted:', len(empty))
    assert not empty, f'{name} has unexecuted cells'
print('OK')
"
```

Expected: `unexecuted: 0` for both notebooks, then `OK`.

- [ ] **Step 7: Report status**

If every step above passed, the restructure and R8/R9 build-out are complete per `docs/superpowers/specs/2026-09-14-capstone-1-restructure-design.md` §7's acceptance criteria. If anything failed, fix it and re-run this task from Step 1 before declaring done — no partial pass.
