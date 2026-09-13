# Streamlit App Guide — Flight Late-Arrival Risk

Everything about running `streamlit_app.py` and handling it during a live demo.

## Starting it

```bash
cd projects/capstone_1
streamlit run streamlit_app.py
```

You'll see something like:

```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

It opens your default browser automatically. If it doesn't, click the Local URL.

## Stopping it

`Ctrl+C` in the terminal it's running in. Closing the browser tab does **not** stop the server — it keeps running until you `Ctrl+C` it (or close the terminal).

## If port 8501 is already in use

Something else is using the default port (maybe a previous run you forgot to stop). Either:

```bash
# find and stop whatever's on 8501, or just use a different port:
streamlit run streamlit_app.py --server.port 8502
```

then open `http://localhost:8502` instead.

## What the page actually shows

- **Sidebar (left):** every input the model needs — airline, origin/destination airport, scheduled departure time, day, month, distance, flight duration, and which leg of the aircraft's day this is. Click **Assess risk** to score it.
- **Result panel:** a big percentage (the probability of arriving 15+ minutes late), a colored risk band (green/amber/red), and a FLAG / No action recommendation based on the model's operating threshold.
- **"Why this score":** plain-English reasons, strongest first — no feature names or jargon, e.g. *"a 06:00 departure lowers the risk"*.
- **Caveat box (when it appears):** a warning specific to two known weak spots — early-morning departures (the model's confident mistakes cluster there) and routes with almost no history in the training data. Not every flight triggers this.
- **"The exact values scored" (expandable):** the full 24-feature row that was actually fed to the model — useful if someone asks "how do I know it used the right numbers."
- **Model card (bottom of page):** which family shipped (forest / boosted / linear), how many flights it trained on, its validation lift over baseline, and — importantly — the stated limitations and "when to stop trusting it" conditions. This section exists because a probability alone isn't a decision; showing this unprompted is part of the assignment's honesty requirement.

## Good flights to demo

- **Friday 18:30, ATL → ORD, leg 3 of the day** — a solidly "moderate/high risk" example with several reasons lit up.
- **Tuesday 06:00, PHX → LAS, leg 1** — triggers the early-morning caveat, good for showing the model's self-reported weakness.
- Try the same route/time with a different leg number (e.g. leg 1 vs leg 6) — rotation depth is one of the stronger features, and the reasons panel should visibly change.

## Handling things going wrong live

| What you see | What it means | What to do |
|---|---|---|
| Red error box: *"No model at ... Run notebook to train and export it."* | `models/flight_delay_model.pkl` doesn't exist or was deleted. | Stop the app (`Ctrl+C`), run `notebooks/02_model_build.ipynb` (see `RUN_GUIDE.md`), restart. |
| Red error box: *"Cannot score this flight: ..."* | You (or the audience) entered something the input validation caught — this is a **feature**, not a bug: it proves bad input never reaches the model as a crash. | Just adjust the input in the sidebar and click **Assess risk** again. |
| Yellow warning: *"Historical rate tables are not loaded..."* | `models/lookup_tables.pkl` is missing, so route/airport history features fall back to training medians — predictions will look flatter than they should. | Run `python -c "from app.serving.lookups import build_lookups; import joblib; from app import config; joblib.dump(build_lookups(), config.LOOKUP_PATH, compress=3)"` once, then reload the page. |
| Page loads but says "Loading the model..." and hangs | First load reads the pickle from disk — normally under 2 seconds. If it's stuck much longer, the model file may be corrupted (interrupted write mid-training). | Stop the app, re-run notebook 2 to regenerate the pickle, restart. |
| Browser shows nothing / connection refused | The Streamlit process isn't actually running, or you're pointed at the wrong port. | Check the terminal — is `streamlit run` still running? Confirm the port in the URL matches what the terminal printed. |
| You change code in `streamlit_app.py` or `app/serving/predict.py` mid-demo | Streamlit doesn't auto-reload by default in some configurations. | Click the "Rerun" option Streamlit shows in the top-right, or just refresh the browser tab. |

## Before you present: a 60-second sanity check

```bash
cd projects/capstone_1
python -c "
from app.serving.predict import predict_flight
r = predict_flight(carrier='DL', origin='ATL', dest='ORD', departure_hour=18,
                    day_of_week=5, month=7, distance_miles=606,
                    scheduled_minutes=115, leg_number=3)
print(r.summary())
"
```

If this prints a result with no traceback, the model, the lookups, and the whole prediction path all work — the Streamlit app is just a UI on top of exactly this function, so if this works, the app will too.
