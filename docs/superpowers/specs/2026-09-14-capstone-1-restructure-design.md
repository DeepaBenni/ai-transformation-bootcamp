# Capstone 1 — Flat Layout Restructure — Design

**Date:** 2026-09-14
**Branch:** `feature/capstone-1`
**Target:** `projects/capstone_1` (restructured in place)
**Supersedes:** §4 (Architecture, file layout only) of
`docs/superpowers/specs/2026-09-13-capstone-1-signalcraft-design.md`. Everything else in that
document — decisions D1–D6, requirement coverage R1–R11, the testing philosophy, the data flow,
the error handling, the build order's substance — stays authoritative. This document only changes
where the code lives and how it is named.

---

## 1. Purpose

The existing `projects/capstone_1/app/` package is a stage-based, deeply-modular implementation of
the signalcraft design: separate files per concern (`cv.py`, `zoo.py`, `threshold.py`, `train.py`,
`forbidden.py`, `history.py`, `build.py`, `seal.py`, `predict.py`, `lookups.py`), each independently
tested.

A flatter reference layout was supplied separately (`src/data/acquire.py`, `src/data_pipeline.py`,
`src/feature_engineering.py`, `src/train_model.py`, `src/evaluate.py`, `src/predict.py`,
`src/model_utils.py`, `src/streamlit_app.py`, root `config.json`) — the file surface a reader expects
to find. This restructure reconciles the two: adopt the flatter file *names* as the public surface,
without discarding the tested, isolated modules the graded requirements (R2 sealed holdout, R8
leakage guard) depend on.

## 2. Decisions taken before design

| # | Decision | Rationale |
|---|---|---|
| E1 | Rename `app/` → `src/`; existing submodules move unchanged | Preserves working, tested code; renaming is mechanical, not a rewrite |
| E2 | Add six thin entry-point files at `src/` root matching the flat layout | Each composes/re-exports the real logic from the submodule beneath it — satisfies the expected file list without merging tested units |
| E3 | Keep `zoo.py`'s current hyperparameters (100/10 forest, unregularized boosted) | A model is already trained against this config; switching requires a retrain and stale-doc cleanup with no measured benefit (D2's rationale still holds) |
| E4 | Add root `config.json` read by `src/config.py`, env vars still override | Satisfies the requested config-file pairing without breaking the many call sites that already import constants from `app.config` |
| E5 | Add `notebooks/` with 3 files, generated and executed last | This was already build-order step 10 in the 2026-09-13 spec, never completed. Notebooks import from `src/` — no duplicated logic (R11) |

## 3. Target layout

```
projects/capstone_1/
├── config.json                        NEW
├── src/
│   ├── __init__.py
│   ├── config.py                      moved from app/config.py; reads config.json, env overrides
│   ├── cli.py                         moved from app/cli.py, unchanged
│   ├── data_pipeline.py               NEW thin composer -> src/data/{prepare,seal}.py
│   ├── feature_engineering.py         NEW thin composer -> src/features/{forbidden,history,build}.py
│   ├── train_model.py                 NEW thin composer -> src/models/{cv,zoo,threshold,train}.py
│   ├── evaluate.py                    NEW thin composer -> src/models/metrics.py, src/explain/
│   ├── predict.py                     NEW thin composer -> src/serving/{predict,lookups}.py
│   ├── model_utils.py                 NEW — path/preprocessing/loader helpers consolidated
│   ├── streamlit_app.py               moved, imports updated to src.*
│   ├── data/
│   │   ├── __init__.py
│   │   ├── acquire.py                 moved, unchanged
│   │   ├── prepare.py                 moved, unchanged
│   │   └── seal.py                    moved, unchanged
│   ├── features/
│   │   ├── __init__.py
│   │   ├── forbidden.py               moved, unchanged (own module + test, per 2026-09-13 spec §4)
│   │   ├── history.py                 moved, unchanged
│   │   └── build.py                   moved, unchanged
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cv.py                      moved, unchanged
│   │   ├── zoo.py                     moved, unchanged (hyperparameters unchanged — E3)
│   │   ├── metrics.py                 moved, unchanged
│   │   ├── threshold.py               moved, unchanged
│   │   └── train.py                   moved, unchanged
│   ├── explain/
│   │   ├── __init__.py
│   │   ├── shap_report.py             moved, unchanged
│   │   └── errors.py                  moved, unchanged
│   └── serving/
│       ├── __init__.py
│       ├── predict.py                 moved, unchanged
│       └── lookups.py                 moved, unchanged
├── notebooks/                          NEW
│   ├── 01_data_pipeline.ipynb          generated + executed, imports src.data_pipeline
│   ├── 02_model_build.ipynb            generated + executed, imports src.train_model / src.evaluate
│   └── 03_Data_Prep_And_EDA.md         EDA writeup (feeds R3's five findings)
├── data/, models/, holdout_seal.json    untouched — reused as-is
├── tests/                               import paths updated app.→src., filenames unchanged
├── reports/figures/                     untouched
└── README.md, pyproject.toml, requirements.txt, .env.example, .gitignore
    paths and package refs updated from app. to src.
```

## 4. config.json

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

`src/config.py` loads this file for defaults, then applies the existing `.env` overrides on top
(`DATA_DIR`, `MODEL_DIR`, `RANDOM_STATE`, etc.), preserving D4's "point at the existing parquet
without a re-download" behavior. No call site that already does `from app.config import X` changes
its usage — only the import path (`app` → `src`) changes.

## 5. Entry-point composition pattern

Each new top-level file is intentionally thin — it imports from its submodule(s), re-exports the
public functions/classes under the expected name, and adds nothing else. Example shape for
`train_model.py`:

```python
"""Model training entry point (R6, R7). Thin composer — logic lives in src/models/."""
from src.models.cv import expanding_folds
from src.models.zoo import model_zoo
from src.models.threshold import threshold_for_recall
from src.models.train import run_cv, summarise, fit_final

__all__ = ["expanding_folds", "model_zoo", "threshold_for_recall", "run_cv", "summarise", "fit_final"]
```

`evaluate.py` additionally re-exports the SHAP/error-analysis entry points from `src/explain/` so
R8's outputs are reachable from the expected file, without moving that code out of its own tested
module.

## 6. Notebooks

Generated after the restructure is validated, each notebook imports from `src.*` — never
reimplements pipeline logic in a cell (2026-09-13 spec's R11 rule, unchanged):

- `01_data_pipeline.ipynb` — runs `src.data_pipeline`, shows row counts at each cleaning step,
  positive-rate check
- `02_model_build.ipynb` — runs `src.train_model` + `src.evaluate`, shows the 5-fold × 3-family
  results table, the threshold sweep, final metrics
- `03_Data_Prep_And_EDA.md` — the five EDA findings (R3), assumptions recorded before looking at
  data, at least one contradicted, per the existing spec's §5 "report" description

## 7. Testing and validation

Existing 8 test files keep their names and scope; only `from app.X import Y` becomes
`from src.X import Y`. New tests added for the six entry-point files verify they expose the right
names and that a thin call-through returns what the underlying module returns — not a re-test of
logic already covered.

**Validation sequence for this restructure:**

1. `ruff check` / `black --check` over `src/` and `tests/`
2. Full `pytest` suite green (all 8 existing files + new entry-point tests)
3. `python -m src.cli --help` — package imports cleanly end to end
4. Load the existing `models/flight_delay_model.pkl` through `src/predict.py` and score one
   synthetic row — proves the moved serving path still works against the already-trained artifact
5. Notebooks execute top to bottom with no errors, cells saved with real output (not committed
   empty, per the source-material defect this restructure is explicitly avoiding — see the
   2026-09-13 spec §3)

## 8. Out of scope

- No retraining (E3)
- No change to holdout sealing behavior, leakage guard logic, or any R1–R11 requirement's substance
- No change to `capstone-1-signalcraft/` (the read-only source material) or `projects/capstone_2`
