"""Flight late-arrival risk - operator interface.

The user-facing half of the serving layer (requirement R9). One screen a station
duty manager can use without training.

Design rules, and why each one is here:

* **All scoring goes through** :mod:`app.serving.predict`. The app never
  re-implements a feature or a threshold, because two copies of that logic
  would drift apart and the drift would be silent.
* **The model is cached** with ``st.cache_resource``. Streamlit re-runs this
  whole script on every widget interaction; without the cache the model would
  reload on each click.
* **A bad input shows a readable message**, never a traceback.
* **The explanation and the caveat are always visible.** A probability on its
  own is not a decision, and the caveat is the most honest thing on the page.
* **The limitations sit on the page**, not in a footnote.

Run with::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.serving.predict import (
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
    """Draw the model card and the limitations section."""
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
            "vary between airports. Run `python -m app.cli features` to generate them."
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
departures it scored as safe. A low score before 09:00 is weaker evidence than
the same score in the afternoon.
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
