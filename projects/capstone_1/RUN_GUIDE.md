# Run Guide — Flight Late-Arrival Risk (Capstone 1)

This is the order to run things in for your presentation: verify the pickle exists → run the notebooks → run the Streamlit app.

## 0. One-time setup

```bash
cd projects/capstone_1
pip install -r requirements.txt
```

This installs pandas, scikit-learn, LightGBM, Streamlit, Jupyter/nbconvert, and everything else the project needs. Only needs to happen once per machine.

---

## 1. Make sure the pickle exists

The Streamlit app reads a trained model from `models/flight_delay_model.pkl`. Check it's there:

```bash
ls -la models/flight_delay_model.pkl
```

- **File exists** → skip to step 2 if you just want to demo the app, or continue to step 2 if you want to show the notebooks producing it live.
- **File missing** → you must run notebook 2 (step 2 below) before the app will work.

You can also check what's actually in it from the command line:

```bash
python -c "
import joblib
a = joblib.load('models/flight_delay_model.pkl')
print('family:', a['model_family'])
print('threshold:', a['threshold'])
print('trained on:', a['training']['rows'], 'rows')
print('validation:', a['validation'])
"
```

If this prints without error, the pickle is valid and loadable.

---

## 2. Run the notebooks, in order

There are three files in `notebooks/`:

| File | What it does | Produces the pickle? |
|---|---|---|
| `01_data_pipeline.ipynb` | Verifies the sealed holdout, shows row counts and the positive rate by month | No — read-only checks |
| `02_model_build.ipynb` | Shows the 3 model families, cross-validates them, fits the winner on the full dataset, **exports the pickle** | **Yes — this is the notebook that builds `models/flight_delay_model.pkl`** |
| `03_Data_Prep_And_EDA.md` | Five EDA findings with real measured numbers (not a notebook to run — it's a written report) | No |

### Running a notebook

**Option A — Jupyter in the browser:**

```bash
jupyter notebook notebooks/
```

Open the file, then **Run → Run All Cells**. Watch for a green checkmark / no red error output on any cell.

**Option B — VS Code:**

Open the `.ipynb` file, click **Run All** at the top of the notebook.

**Option C — from the command line (no browser needed, good for a quick re-run before presenting):**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_build.ipynb --ExecutePreprocessor.timeout=1800
```

`02_model_build.ipynb` does real training (cross-validation plus a final fit on ~5.7 million rows), so it takes several minutes — the `--ExecutePreprocessor.timeout=1800` gives it up to 30 minutes before nbconvert would time it out. If you're running it live during a presentation, say so up front and either let it run in the background or have a pre-executed copy ready as a fallback (see step 3 below).

### Confirming a notebook ran successfully

- No cell shows a red error box / traceback.
- Every code cell has a number in `[ ]` next to it (not empty brackets) — that number is the execution order.
- In `02_model_build.ipynb`, the last cell should print something like `78% chance of arriving 15+ min late - FLAG for attention` — if you see that, the freshly-exported model just scored a real flight successfully.

If a cell fails, read the error message at the bottom of that cell first — it's almost always either a missing file (re-run the cell above it, or check `data/processed/flights_features.parquet` exists) or a stale kernel (Kernel → Restart & Run All).

---

## 3. Run `predict.py` directly — sanity check before launching the UI

Before opening the Streamlit app, run the serving module on its own from the command line:

```bash
python -m app.serving.predict
```

This is the exact same code the Streamlit app calls to score a flight — running it standalone first means if something's wrong (pickle missing, lookups missing, a bad import), you find out here as a plain, readable error, not as a blank or broken browser tab in front of your audience.

**Expected output:**

1. A `MODEL` section — the shipped family (`forest`, `boosted`, or `linear`), its threshold, its validation lift over baseline, and whether the lookup tables loaded.
2. Three worked examples, each with a real probability, a risk band, and 2-4 plain-English reasons.
3. Four deliberately invalid inputs, each producing a one-line readable error (e.g. `departure_hour must be an integer 0-23, got 25`) instead of a traceback — this proves the input validation works before you ever put it in front of someone typing garbage into the UI.

If it fails with `No model at ... Run: python -m app.cli train`, the pickle wasn't produced — go back to step 2 and confirm `02_model_build.ipynb` actually ran its export cell.

---

## 4. Run the Streamlit app

Once the pickle exists (step 1), the notebooks have run (step 2), and the standalone check above passed (step 3):

```bash
streamlit run streamlit_app.py
```

This prints a local URL — normally `http://localhost:8501`. It should open in your browser automatically; if not, copy the URL there.

**For a live presentation**, it's worth starting this a minute or two before you need it, since the first load takes about a second to read the model from disk (after that it's cached for the rest of the session).

See `STREAMLIT_GUIDE.md` for what to actually show on the page and how to handle things going wrong live.

---

## Presentation order that tells the whole story

1. Open `03_Data_Prep_And_EDA.md` — walk through 1-2 of the five findings (finding 2, the positive-rate spread across months, is the most interesting one to show).
2. Open `01_data_pipeline.ipynb` with its saved output — point at the verified holdout hash and the row counts.
3. Open `02_model_build.ipynb` with its saved output — point at the three model definitions, the cross-validation table, and the final line showing the exported model scoring a real flight. (You don't need to re-run it live unless you want to; the saved output already proves it ran.)
4. Run `python -m app.serving.predict` in a terminal (step 3 above) — this is prep, not usually something to dwell on in front of the audience, but it's your last checkpoint that everything downstream of the notebook actually works before you open the app.
5. Switch to the Streamlit app (step 4 above) — pick a flight in the sidebar, click **Assess risk**, and walk through the probability, the reasons, and the caveat.

That order goes data → model → live product, which is the easiest structure for an audience to follow.
