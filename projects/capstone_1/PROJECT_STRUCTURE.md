# Project Structure — What Every File Is For

A map of `capstone_1/`: what each file does, and where it sits in the run flow described in `RUN_GUIDE.md`. Files are grouped by folder, in the order you'd actually touch them.

**Legend** — how each file participates in the flow:
- 🏃 **You run this directly**
- 📦 **Imported by other code** — you don't run it yourself; something else calls it
- 🗃️ **Generated automatically** — created by running the notebooks/CLI; don't hand-edit, don't worry if it's missing (something else will (re)build it)
- 📄 **Reference / read-only data** — provenance or config, not code
- ✅ **Test** — run automatically by `pytest`, not part of the demo flow
- 📘 **Documentation** — for you, not for the code

---

## Root

| File | Purpose | Flow |
|---|---|---|
| `streamlit_app.py` | The UI. Renders the sidebar form, calls `app.serving.predict.predict_flight`, and draws the result, the reasons, the caveat, and the model card. | 🏃 `streamlit run streamlit_app.py` — **step 4** of `RUN_GUIDE.md` |
| `RUN_GUIDE.md` | Step-by-step: verify the pickle → run the notebooks → sanity-check `predict.py` → run Streamlit, plus a suggested presentation order. | 📘 Read this first |
| `STREAMLIT_GUIDE.md` | What each part of the running app shows, good flights to demo, and a troubleshooting table for things going wrong live. | 📘 Read before presenting |
| `PROJECT_STRUCTURE.md` | This file. | 📘 |
| `requirements.txt` | Every pinned dependency (pandas, scikit-learn, LightGBM, Streamlit, Jupyter, pytest, ...). | 🏃 `pip install -r requirements.txt` — **step 0** |
| `pyproject.toml` | Lint/format/test config (`ruff`, `black`, `pytest`). Not something you run — tools read it automatically. | 📦 |
| `.env.example` | Template for an optional `.env` — lets you point `SIGNALCRAFT_DATA_DIR` somewhere else instead of the default data location. Copy to `.env` only if you need that. | 📄 |
| `.gitignore` | Tells git not to track data files, trained models, caches, etc. | 📄 |
| `holdout_seal.json` | The sealed final-2-months holdout: its row-count and a hash, plus `"opened": false`. `app/data/seal.py` reads this to guarantee nothing downstream ever trains or validates on those months. | 🗃️ Don't edit — and don't let anything flip `opened` to `true` before you're ready to report that result |

---

## `app/` — the package

Everything under here is Python code, imported by the notebooks, the CLI, and `streamlit_app.py`. Nothing in `app/` is meant to be run directly except `cli.py` and `serving/predict.py`.

| File | Purpose | Flow |
|---|---|---|
| `config.py` | Every path and constant in one place: `DATA_DIR`, `MODEL_PATH`, `RANDOM_STATE`, `HOLDOUT_MONTHS`, etc. | 📦 Imported by almost everything |
| `cli.py` | `python -m app.cli <stage>` — a single entry point for `acquire`, `prepare`, `seal`, `features`, `train` (the stages the notebooks now do interactively). | 🏃 Optional — the notebooks are the primary flow now; the CLI is the original automated path |
| `data/acquire.py` | Downloads the 12 months of BTS on-time data and writes `data/manifest.json`. | 📦 Called by `cli.py acquire`; not needed if `data/manifest.json` already exists |
| `data/prepare.py` | Drops cancelled/diverted flights, repairs one bad schedule field, writes `flights_clean.parquet`. | 📦 Called by `cli.py prepare` and by notebook 1 (indirectly, via `seal.py`) |
| `data/seal.py` | Carves out May/June 2026 as the holdout, hashes it, and is the **only** sanctioned loader (`development_frame()`) every other stage must use so nothing trains on the holdout by accident. | 📦 Used by notebook 1 (`seal.verify()`, `seal.development_frame()`) |
| `features/forbidden.py` | The 28-column leakage blocklist (things like `DepDelay`, `ArrTime`) plus the check that raises if any of them reach the feature matrix. Its own file on purpose — it's the entire leakage defence. | 📦 Used inside `features/build.py` |
| `features/history.py` | Backward-looking "late rate" averages (by route, carrier, airport...) computed as **expanding means over strictly prior months** — never a whole-dataset average, which would leak the label. | 📦 Used inside `features/build.py` |
| `features/build.py` | Builds all 24 model features and writes `data/processed/flights_features.parquet`. | 📦 Used by notebook 2 (reads the parquet it already wrote) |
| `models/cv.py` | Defines the 5 expanding-window folds (train on the past, validate on one future month — never random k-fold). | 📦 Used by notebook 2's cross-validation cell |
| `models/zoo.py` | Defines the 3 model families: `linear` (LogisticRegression), `forest` (shrunk RandomForest), `boosted` (LightGBM). | 📦 Used by notebook 2 |
| `models/metrics.py` | PR-AUC and lift-over-baseline scoring — every result is reported as a multiple of that fold's own baseline, since the late-arrival rate moves month to month. | 📦 Used by notebook 2 via `train.run_cv` |
| `models/threshold.py` | Picks the operating threshold from a target recall (60% by default), instead of blindly using 0.5. | 📦 Used by `train.export()` |
| `models/train.py` | Runs cross-validation (`run_cv`), fits the final model on all development rows (`fit_final`), and writes the artefact (`export`) — this is the actual training logic notebook 2 calls. | 📦 Used by notebook 2 — **this is what produces `models/flight_delay_model.pkl`** |
| `serving/lookups.py` | Builds the 7 historical-rate lookup tables (by route, carrier, airport, hour) a live prediction needs but a user can't type in — derived from the most recent month of feature data and cached to `models/lookup_tables.pkl`. | 📦 Used by `serving/predict.py` |
| `serving/predict.py` | The scoring function `predict_flight()`: validates input, assembles the 24-feature row, calls the model, and returns a probability plus plain-English reasons and a caveat. | 🏃 `python -m app.serving.predict` — **step 3** of `RUN_GUIDE.md` (also 📦 imported by `streamlit_app.py`) |
| `explain/__init__.py` | Empty placeholder package for future SHAP/error-analysis work (not built in this pass — out of scope for the presentation). | 📦 Not currently used |

---

## `data/`

| File | Purpose | Flow |
|---|---|---|
| `manifest.json` | Provenance record for the 12 acquired months: source URL, SHA-256, row/column counts. The raw data itself is **not** committed (see `.gitignore`) — this file is the evidence it was legitimately downloaded. | 📄 |
| `processed/flights_features.parquet` | The built feature table: ~5.7M development rows × 24 features, already excludes the sealed holdout. | 🗃️ Read by notebook 2 and by `serving/lookups.py`; rebuilt by `app.features.build.run()` if ever deleted |

---

## `models/`

| File | Purpose | Flow |
|---|---|---|
| `flight_delay_model.pkl` | **The trained model artefact.** A dict: the fitted estimator, the 24 feature names in order, the operating threshold, training/validation metadata. This is what `streamlit_app.py` and `predict.py` read. | 🗃️ Produced by notebook 2's export cell — **this is "the pickle file"** |
| `lookup_tables.pkl` | Cached output of `serving/lookups.py`, so it isn't rebuilt from the 5.7M-row parquet on every app restart. | 🗃️ Auto-created the first time `predict.py` or the app runs; safe to delete, it just rebuilds |
| `train_log.txt` | A saved console log from an earlier full 5-fold CLI training run — useful as a second data point next to notebook 2's own (faster, 2-fold) run. | 📄 Reference only |
| `.gitkeep` | Keeps the empty `models/` directory tracked in git even though the `.pkl` files inside it are gitignored. | 📄 |

---

## `notebooks/` — your three presentation files

| File | Purpose | Flow |
|---|---|---|
| `01_data_pipeline.ipynb` | Verifies the sealed holdout hash, shows development-row counts and the positive rate by month. Read-only checks — produces nothing. | 🏃 Run first — **step 2a** |
| `02_model_build.ipynb` | Shows the 3 model definitions, cross-validates them on 2 expanding folds, fits the winning family on the **full** 5.7M-row development set, and **exports `models/flight_delay_model.pkl`**. | 🏃 Run second — **step 2b, this is the notebook that builds the pickle** |
| `03_Data_Prep_And_EDA.md` | Five EDA findings, each an assumption stated before querying, then checked against the real measured number (finding 2 contradicts the "roughly flat" assumption). A written report, not a notebook to execute. | 📘 Read/present, nothing to run |

---

## `reports/figures/`

| File | Purpose | Flow |
|---|---|---|
| `.gitkeep` | Keeps the empty directory tracked; figures generated here later are gitignored. Currently empty — no figure-generating code has been built yet. | 📄 |

---

## `tests/`

Everything here runs under `pytest`, not as part of the demo — it's what proves the code behind the demo is correct. If someone asks "how do you know this works," this is the answer.

| File | Covers | Flow |
|---|---|---|
| `conftest.py` | Puts the project root on the import path so `import app` works from anywhere `pytest` is invoked. | 📦 |
| `test_acquire.py` | `data/acquire.py` — URL building, retry logic, manifest writing. | ✅ |
| `test_config.py` | `config.py` — path resolution. | ✅ |
| `test_cv.py` | `models/cv.py` — expanding folds never train on the future. | ✅ |
| `test_features.py` | `features/build.py` — all 24 features present, correct order, no leakage. | ✅ |
| `test_leakage.py` | `features/forbidden.py` — the leakage guard itself. | ✅ |
| `test_lookups.py` | `serving/lookups.py` — the history tables (new, this session). | ✅ |
| `test_metrics.py` | `models/metrics.py` + `threshold.py` — lift and threshold selection. | ✅ |
| `test_predict_validation.py` | `serving/predict.py` — every malformed input raises a readable error (new, this session). | ✅ |
| `test_seal.py` | `data/seal.py` — the sealed holdout never leaks. | ✅ |
| `test_train.py` | `models/train.py` — CV, summarisation, artefact export. | ✅ |

Run them all with:

```bash
python -m pytest -q
```

---

## The short version

**To present:** `RUN_GUIDE.md` → the 3 files in `notebooks/` → `streamlit_app.py`.
**Everything in `app/`** is the engine underneath those; you don't run most of it directly, but every notebook cell and every Streamlit interaction is calling into it.
**Everything in `data/` and `models/`** is generated output — if it's ever missing, re-running the notebooks in order rebuilds it.
**`tests/`** proves the engine is correct; it's not part of the live demo.
