# SignalCraft — Flight Delay Risk Prediction

**Will this flight arrive more than 15 minutes late — knowable the afternoon
before it departs?**

SignalCraft is a classical machine-learning pipeline built for a station duty
manager's real decision: whether to pre-position standby ground crew against
tomorrow's departure bank. Every prediction is made at **T-24h**, using only
information a dispatcher would actually have a day out.

> *Capstone 1 of the AI Transformation Readiness bootcamp. Assesses data
> acquisition discipline, feature engineering without leakage, correct
> time-series validation, threshold selection, and honest serving.*

---

## Stack

pandas · scikit-learn · LightGBM · Streamlit · pytest · ruff · black

## Data

[BTS On-Time Reporting](https://transtats.bts.gov) — U.S. DOT public domain
flight performance data, no login wall. 12 months acquired
(`data/manifest.json` records the exact files, checksums and row counts):
10 months for development (5,696,742 rows), 2 months (May–June 2026) held
out and **sealed** — `holdout_seal.json` records `"opened": false` and the
seal is enforced in code, not by promise.

## Quick start

```bash
cd projects/capstone_1
python -m venv .venv && .venv\Scripts\activate      # or source .venv/bin/activate
pip install -r requirements.txt

python -m app.cli acquire --start 2025-07 --end 2026-06   # download + checksum
python -m app.cli prepare                                  # clean
python -m app.cli seal                                      # split + seal the holdout
python -m app.cli features                                  # build the 24 features
python -m app.cli train                                     # cross-validate, fit, export
python -m app.cli evaluate                                   # score on the sealed holdout
pytest                                                        # 96 passed

streamlit run streamlit_app.py                                # the operator UI
```

Each stage reads a file and writes a file, so a failure late in the pipeline
never costs an earlier, expensive step. `SIGNALCRAFT_DATA_DIR` (see
`.env.example`) lets the pipeline point at an already-prepared data folder
instead of re-downloading ~30 GB of zips.

---

## The pipeline

`app/cli.py` dispatches seven stages, each its own module:

| Stage | Module | What it does |
|---|---|---|
| `acquire` | `app/data/acquire.py` | Downloads and checksums each month's BTS zip; records source, licence and row counts to `manifest.json` |
| `prepare` | `app/data/prepare.py` | Cleans and standardises the raw schema |
| `seal` | `app/data/seal.py` | Splits development vs. holdout by month and hashes the holdout so a later change is detectable |
| `features` | `app/features/build.py` | Builds 24 features (18 derived), each checked against `app/features/forbidden.py`'s leakage guard |
| `train` | `app/models/train.py` | Expanding-window cross-validation across three model families, then fits the winner on the full development set |
| `evaluate` | — | Scores the sealed holdout exactly once |
| `report` | — | Writes the results tables |

## Modelling approach

- **Features:** 24 total, 18 derived — time-of-day, business-hours,
  holiday-adjacency, route history (expanding, never look-ahead), carrier
  and airport effects. Every feature answers one question: *would a
  dispatcher have known this yesterday?* Anything that fails that test
  lives in `app/features/forbidden.py` instead, and `assert_no_leakage()`
  checks for it at build time.
- **Validation:** expanding-window cross-validation (`app/models/cv.py`) —
  never random k-fold, because shuffling time-series rows would leak the
  future into training.
- **Three model families** (`app/models/zoo.py`), compared on PR-AUC rather
  than accuracy:

  | Model | PR-AUC | Recall | Precision | Fit time |
  |---|---|---|---|---|
  | Random Forest | 0.3304 | 0.2047 | 0.3885 | 143.0 s |
  | **LightGBM (shipped)** | 0.3282 | 0.2907 | 0.3629 | 155.3 s |
  | Logistic Regression | 0.3142 | 0.2407 | 0.3651 | 8.0 s |

  LightGBM is shipped for handling the categorical explosion (airports,
  carriers) natively and for the best recall at a comparable PR-AUC. The
  random forest deliberately runs at a shrunk configuration (100 trees,
  depth 10) — it's the slowest family and still loses on the metrics that
  matter, so it's present to be compared, not to be shipped, and that's
  reported rather than hidden.
- **Threshold:** chosen from a target recall (`app/models/threshold.py`),
  not the default 0.5 — the source analysis found that a naive 0.5 cut
  achieves only 2.9% recall on this class imbalance, which would be
  operationally useless.

## Serving

`app/serving/predict.py` + `streamlit_app.py` — a one-screen UI a duty
manager could use without training:
- Every prediction ships with a **plain-English explanation** (`_top_reasons`)
  ranked by the model's own feature importances — never a bare probability.
- A **caveat is always visible** — e.g. the model states plainly that
  early-morning departures are where it's least reliable.
- Bad input shows a readable message, never a traceback; the model is
  loaded once via `st.cache_resource` so it doesn't reload on every click.
- The Streamlit app and the CLI both call `app/serving/predict.py` — there
  is exactly one scoring implementation, so the UI and the pipeline can
  never silently drift apart.

---

## Requirement traceability

| Req | What it asks | Where |
|---|---|---|
| R1 | Reproducible acquisition | `app/data/acquire.py`, `data/manifest.json` |
| R2 | Sealed holdout | `app/data/seal.py`, `holdout_seal.json` — opens once, one-way |
| R3 | Five EDA findings | `notebooks/01_data_pipeline.ipynb` |
| R4 | Baseline before models | `app/models/metrics.py` — every result reported as lift over baseline |
| R5 | 12+ features, 4+ derived | `app/features/build.py` — 24 features, 18 derived |
| R6 | 3 model families, correct validation | `app/models/zoo.py`, `app/models/cv.py` |
| R7 | PR-AUC + recall at threshold | `app/models/threshold.py` |
| R8 | Error analysis | `notebooks/03_model_build.ipynb` |
| R9 | Serving layer | `app/serving/predict.py`, `streamlit_app.py` |
| R10 | Decision framing | this README, the app's own explanation and caveat |
| R11 | Engineering quality | `tests/` (96 passing), pinned `requirements.txt`, no notebooks in the execution path |

## Known limitations

- **Explainability is feature-importance-based, not SHAP.** The design
  originally scoped a SHAP-driven error analysis (`app/explain/`); the
  shipped explanation instead ranks the model's built-in feature
  importances into plain English, which is simpler but less precise
  per-prediction than a SHAP value.
- **PR-AUC (~0.33) is modest in absolute terms** — flight delays are a
  genuinely hard, high-entropy prediction problem (weather, ATC, downstream
  aircraft rotation), which is exactly why the app frames its output as a
  risk signal with a stated caveat, not a certainty.
- **The random forest is deliberately under-tuned** for a fair three-way
  comparison on a laptop-scale budget — a production deployment would
  re-tune whichever family is actually shipped.
- **Synthetic seed months:** the data is real BTS data, but the sealed
  holdout has not yet been opened as of this writing — the reported metrics
  are cross-validation estimates, not a final holdout score.

## Project structure

```
app/
├── data/       acquire · prepare · seal
├── features/   build · forbidden (leakage guard) · history
├── models/     zoo (3 families) · cv (expanding-window) · threshold · train · metrics
├── serving/    predict · lookups
├── explain/    (scaffolded, not yet populated)
└── cli.py      one entry point for every stage
streamlit_app.py   operator-facing UI
notebooks/         01 data pipeline · 02 feature engineering · 03 model build
data/manifest.json record of every acquired file: source, licence, checksum
holdout_seal.json  the sealed-holdout record (R2)
tests/             96 tests
```
