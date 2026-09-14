# Capstone 1 — SignalCraft (Track A) — Design

**Date:** 2026-09-13
**Branch:** `feature/capstone-1`
**Target:** `projects/capstone_1`
**Source material:** `capstone-1-signalcraft/` — read-only, not modified by this work

---

## 1. Purpose and track

Track A, flight punctuality. Predict whether a flight arrives more than 15 minutes late
(`ArrDel15`), at **T-24h** — the afternoon before departure.

Track justification against the brief's three criteria:

| Criterion | Answer |
|---|---|
| Genuinely public, no login wall | BTS On-Time Reporting, US federal public domain (17 U.S.C. 105), direct zip download |
| Supports a real operator decision | A station duty manager pre-positions standby ground crew against tomorrow's departure bank |
| Volume large enough for evaluation to mean something | 7,043,316 raw rows over 12 months; 6,896,283 after cleaning |

Track A was chosen on day one and never changed.

---

## 2. Decisions taken before design

| # | Decision | Rationale |
|---|---|---|
| D1 | Build the pipeline **and train a real model** | `predict.py` and the Streamlit app load an artefact that does not exist; without training, the serving layer is dead |
| D2 | Random forest runs **shrunk** — 100 trees, depth 10, against the specified 200/14 | It is the slowest family (772.78 s/fold measured) and loses to LightGBM on every metric; R6 needs three families, not three expensive ones. The reduced config is recorded in the results table so the comparison is not silently unfair |
| D3 | **Package + CLI**, notebooks generated afterwards | R11 forbids notebooks in the execution path; notebooks are produced last, executed, importing from `app/` so no logic is duplicated |
| D4 | `DATA_DIR` **config pointing at the existing parquet** | Avoids a 359 MB re-download; `acquire` is still ported so the one-command path exists for R1 |
| D5 | Holdout **stays sealed** | R2 says it opens exactly once, on day 20. Enforced in code, not by promise |
| D6 | Stretch goals: **rolling backtest only** | R6 already mandates expanding-window CV, which *is* a rolling backtest. Calibration, drift monitor and retrain CLI are out of scope |

### Departure from the source folder's stated philosophy

`06_Build_Plan.md` states: *"every notebook is self-contained. No notebook imports from a project
module."* That rule is correct for `capstone-1-signalcraft`, which is a learning folder. It is wrong
for `projects/capstone_1`, which is a product, and R11 says so directly. Both remain true in their
own place because the source folder is left untouched.

---

## 3. State of the source material

Verified by inspection on 2026-09-13, because `06_Build_Plan.md`'s status table overstates it.

**Real and portable:**

| Asset | State |
|---|---|
| `src/data/acquire.py` | 400 lines, CLI, SHA-256 manifest, banned-column guard — R1 complete |
| `data/manifest.json` | 12 months, licence, retrieval dates, per-file hashes |
| `holdout_seal.reference.json` | 2026-05/06, 1,199,541 rows, `opened: false` — R2 complete |
| `predict.py` | 13 functions, validation, risk bands, early-morning caveat |
| `streamlit_app.py` | Working UI |
| `tests/` | 15 tests over acquire, leakage, predict validation |
| `data/processed/flights_clean.parquet` | 141 MB, 6,896,283 rows — notebook 01 did run |

**Does not exist, despite being marked "Done, executed":**

- `notebooks/02_feature_engineering.ipynb` — 7 code cells, all empty, zero outputs
- `notebooks/03_model_build.ipynb` — 9 code cells, all empty, zero outputs
- `models/flight_delay_model.pkl` — absent
- All six `docs/*.md` — empty templates

So R5, R6, R7 and R8 have no executable code behind them. The planning documents specify the
design down to measured numbers, which is what makes rebuilding it tractable.

**Numbers quoted throughout this spec come from those planning documents.** They were measured by a
run that is not in the folder, with code and seeds unavailable. A re-run produces *our* numbers.
Close agreement is expected; exact reproduction is not guaranteed, and a large gap is itself a
finding to investigate rather than to hide.

---

## 4. Architecture

Stage-based CLI with artefacts between stages, rather than one end-to-end command: the expensive
step sits in the middle, and a failed SHAP run must not cost the CV.

```
projects/capstone_1/
├── app/
│   ├── config.py              DATA_DIR / MODEL_DIR / constants, from env
│   ├── cli.py                 acquire | prepare | seal | features | train | evaluate | report
│   ├── data/
│   │   ├── acquire.py         ported                                   R1
│   │   ├── prepare.py         notebook 01's logic: clean -> 6,896,283 rows
│   │   └── seal.py            create/verify holdout hash               R2
│   ├── features/
│   │   ├── forbidden.py       28 banned columns + assertion            leakage guard
│   │   ├── history.py         expanding means, alpha=20 toward 0.2242
│   │   └── build.py           the 24 features                          R5
│   ├── models/
│   │   ├── cv.py              expanding-window folds                   R6
│   │   ├── zoo.py             linear | forest | boosted                R6
│   │   ├── metrics.py         PR-AUC, lift, recall@threshold, baseline R4, R7
│   │   ├── threshold.py       sweep + threshold_for_recall             R7
│   │   └── train.py           final fit, artefact export
│   ├── explain/
│   │   ├── shap_report.py     global importance, 3 explanations        R8a
│   │   └── errors.py          20 worst, failure categories             R8b
│   └── serving/
│       ├── predict.py         ported                                   R9
│       └── lookups.py         history tables needed at score time
├── streamlit_app.py           ported                                   R9
├── notebooks/                 generated LAST, executed, importing app/
├── docs/                      EDA · BASELINE · METRICS · ERROR_ANALYSIS · DECISION · MODEL_CARD
├── reports/figures/
├── models/                    flight_delay_model.pkl (gitignored)
├── tests/
└── .env.example · pyproject.toml · requirements.txt · README.md
```

**`features/forbidden.py` is its own module.** It is the entire leakage defence, and the brief names
leakage as Track A's defining trap. It gets a module and a test, not a comment.

**`serving/lookups.py` is the missing join.** Six of the 24 features are historical rates. At score
time the app has a carrier and a route, not a rate, so `train` must export lookup tables alongside
the model or the serving layer cannot build a feature row. Nothing in the source produces this.

---

## 5. Data flow

```
  BTS monthly zips
         |  acquire      12 x ontime_YYYY_MM.parquet + manifest.json    R1
         v                7,043,316 raw rows · SHA-256 per file
  data/interim/
         |  prepare      drop cancelled 127,324 · diverted 19,708
         v                · missing label -> 147,033 removed (-2.09%)
  flights_clean.parquet   6,896,283 rows · positive rate 0.2242
         |
         +-- seal -----> verify holdout hash                            R2
         |                2026-05 + 2026-06 · 1,199,541 rows · SEALED
         |  features     24 columns, 18 derived                         R5
         v                history rates: expanding mean, alpha=20
  flights_features.parquet
         |  train        5 expanding folds x 3 families = 15 fits       R4 R6
         v                then final fit on all development months
  flight_delay_model.pkl + lookups.pkl
         |  evaluate     threshold sweep · SHAP · 20 worst errors       R7 R8
         v                figures -> reports/figures/
  metrics.json + errors.csv
         |  report       emit measured numbers for the six docs
         v
  docs/*.md                                                             R3 R10 R11
```

**What `report` does and does not do.** It writes `metrics.json` and renders the numeric tables —
baselines, per-fold lift, the threshold sweep, failure-category counts. It does **not** author the
prose. R3 asks for assumptions recorded *before* looking at the data and at least one of them
contradicted; that is a human judgement, not a generated artefact. The prose is written against the
assumptions already recorded in the source folder's `03_Data_Prep_And_EDA.md` (Cell 39), with the
numbers this pipeline actually measures substituted in. Where a measured number disagrees with the
source document, the measured one wins and the disagreement is noted.

### Holdout enforcement

`config.py` holds `HOLDOUT_MONTHS = ("2026-05", "2026-06")`. `data/seal.py` exposes
`development_frame()` as the **only** sanctioned loader; it filters those months before returning,
and `features`, `train` and `evaluate` all go through it. Opening requires `--unseal`, which refuses
unless the seal file still reads `opened: false`, then flips it and records a timestamp. R2's
"opened exactly once" becomes a property of the code.

A hash mismatch raises and refuses to continue. No `--force`.

### Folds

History features are expanding means over strictly prior months, so early months have little
history — 90.958% of rows have prior route history once burn-in passes. Validating on month one
would score the model on mostly-null features.

| Fold | Trains on | Validates on |
|---|---|---|
| 1 | 2025-07 → 2025-11 | 2025-12 |
| 2 | 2025-07 → 2025-12 | 2026-01 |
| 3 | 2025-07 → 2026-01 | 2026-02 |
| 4 | 2025-07 → 2026-02 | 2026-03 |
| 5 | 2025-07 → 2026-03 | 2026-04 |

The window expands and never wraps. No fold trains on a month later than the one it validates. The
five windows are also the rolling backtest (D6).

**The baseline is recomputed per fold.** The positive rate moves across months (0.1976–0.2677 in the
source docs), so every reported number is a lift multiple, never a bare score (R4).

### Artefacts

`flight_delay_model.pkl` carries the estimator, the 24 feature names **in order**, the threshold,
the baseline, validation metrics and training metadata. Order matters: a model handed its columns
in a different order returns confident nonsense with no error.

`lookups.pkl` carries the six history tables (route, carrier, origin, dest, carrier x origin,
origin x hour) plus UI defaults.

---

## 6. Error handling

### Pipeline stages

Each stage checks its input artefact and names the command that produces a missing one:

```
flights_features.parquet not found at <path>.
Run:  python -m app.cli features
```

`acquire` retries transient HTTP failures and verifies each zip's SHA-256 against the manifest — a
truncated download producing a short frame would silently corrupt everything downstream.

### Serving input (R9)

Ported from `predict.py`, which already does this correctly. Every message names field, rule and
value received:

```python
if not 0 <= departure_hour <= 23:
    raise InvalidFlight(f"departure_hour must be 0-23, got {departure_hour}")
```

`InvalidFlight` is caught in `streamlit_app.py` and rendered with `st.error()`. Missing artefact
raises `ModelNotFound` carrying the training command.

### Cold start — not specified in the source

| Situation | `route_late_rate` | `route_late_rate_n` | UI |
|---|---|---|---|
| Well-observed | smoothed rate | e.g. 762 | normal explanation |
| Thin | shrunk toward 0.2242 | e.g. 4 | "thin route — limited history" |
| Unseen | **the prior, 0.2242** | **0** | "no history for this route" |

This degrades honestly because `route_late_rate_n` is itself a trained feature — the model learned
to discount rates resting on thin evidence, which is why all three selection methods kept it.
Unknown carriers and airports follow the same rule against their own priors.

### The caveat from the error analysis

All twenty worst predictions were confident false negatives, eighteen early-morning departures
scored near 0.04:

```python
if departure_hour <= 8 and probability < 0.2:
    caveat = ("Early-morning departures are where this model is least reliable. "
              "Its confident misses concentrate here — overnight maintenance, crew "
              "hours, or an aircraft finishing the previous day out of position.")
```

The model telling the operator where not to trust it, derived from measured failure.

### Response budget

R9 wants sub-two-second responses. Artefacts load once, lazily, at first call (~1 s); each
subsequent score is a single-row predict. Both loads sit behind `@st.cache_resource` because
Streamlit re-runs the script on every interaction.

---

## 7. Testing

The 15 existing tests port unchanged. **All tests run on synthetic fixtures**, never the 141 MB
parquet or a trained model.

### Added tests

**History-feature leakage — the important one.** Tests the property, not a value: information
cannot flow backwards.

```python
def test_history_feature_ignores_future_months():
    frame = _tiny_frame()
    before = build_history(frame)
    frame.loc[frame.month == 4, "label"] = 1     # rewrite only the future
    after = build_history(frame)
    pd.testing.assert_series_equal(              # months 1-3 must be untouched
        before[before.month < 4].route_late_rate,
        after[after.month < 4].route_late_rate,
    )
```

A whole-dataset group mean fails this immediately.

**Fold ordering.** For every fold, `max(train_months) < validation_month`. R6's actual requirement,
made structural.

**Feature contract.** 24 long, no duplicates, no forbidden column, **stable order**.

**Metrics and threshold.** Lift against a known positive rate; `threshold_for_recall` achieving at
least the requested recall on a hand-checked array.

**Cold start.** Unseen route yields the prior and `n=0` rather than raising or producing `NaN`.

**Seal integrity.** A tampered row set changes the hash; the sanctioned loader never returns a
holdout month.

### Deliberately not tested

Model quality (belongs to evaluation; `pr_auc > 0.30` is a flaky test), the full training run
(`train` is exercised on a synthetic frame proving wiring, not science), Streamlit rendering (the
logic lives in `predict.py` and is tested there).

`ruff` and `black` config ports from the source `pyproject.toml`: line length 100, Google
docstrings, `ANN` enforced.

---

## 8. Requirement coverage

| Req | Where | Note |
|---|---|---|
| R1 Reproducible acquisition | `app/data/acquire.py`, `data/manifest.json` | Ported; source, licence, retrieval date, row count documented |
| R2 Sealed holdout | `app/data/seal.py`, `holdout_seal.reference.json` | Stays sealed; `--unseal` is one-way and recorded |
| R3 Five EDA findings | `docs/EDA.md` | Assumptions recorded first; at least one contradicted |
| R4 Baseline before models | `app/models/metrics.py`, `docs/BASELINE.md` | Per-fold baseline; everything reported as lift |
| R5 12+ features, 4+ derived | `app/features/build.py`, README table | 24 features, 18 derived, each with signal rationale and T-24h availability |
| R6 3 families, correct validation | `app/models/zoo.py`, `app/models/cv.py` | linear, forest, boosted; expanding windows, never random k-fold |
| R7 PR-AUC + recall at threshold | `app/models/threshold.py`, `docs/METRICS.md` | Threshold set from a target recall, not the 0.5 default |
| R8 SHAP + 20 worst | `app/explain/`, `docs/ERROR_ANALYSIS.md` | Highest-value section; failure categories and remedies |
| R9 Serving layer | `app/serving/predict.py`, `streamlit_app.py` | Validation, sub-2s, graceful on malformed input |
| R10 Decision framing | `docs/DECISION.md`, README, visible in the app | Consumer, decision changed, threshold cost reasoning, switch-off conditions |
| R11 Engineering quality | `tests/`, `README.md`, `requirements.txt`, `docs/MODEL_CARD.md` | Pinned requirements; no notebooks in the execution path; env-var config |

### Self-honesty clause

The holdout stays sealed in this work. When it is opened on day 20, if the score comes in materially
below validation, **the README's first paragraph says so**, with the explanation. That is a standing
requirement of the brief, recorded here so it is not forgotten at the point it becomes inconvenient.

---

## 9. Build order

1. Scaffold `projects/capstone_1`, `config.py`, `pyproject.toml`, pinned `requirements.txt`,
   `.env.example`, and a local `.gitignore` covering `data/`, `models/*.pkl` and
   `reports/figures/*.png` — no raw data and no artefacts committed (R1, R11)
2. Port `acquire.py`, `predict.py`, `streamlit_app.py`, the 15 tests
3. `data/prepare.py` + `data/seal.py`, seal verified against the reference
4. `features/` — forbidden list, history encodings, the 24 features, **leakage test first**
5. `models/` — metrics and baseline, CV folds, the zoo, threshold
6. **Run training.** 5 folds x 3 families, then the final fit
7. `explain/` — SHAP, 20 worst, failure categories
8. `serving/lookups.py`, wire `predict.py` to the real artefact, verify the app scores a flight
9. Fill the six docs with measured numbers; write the README
10. Generate and **execute** the notebooks as evidence, importing from `app/`

Step 10 is last by design. Notebooks committed with empty cells are worse than no notebooks, because
they look like evidence without being any — the defect this port exists to correct.
