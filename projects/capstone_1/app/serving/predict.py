"""Score a single flight for late-arrival risk.

This is the serving layer (requirement R9). It loads the artefact produced by
``app.models.train.export`` and turns a plain description of one flight into a
probability, a recommendation and a plain-English explanation.

Design rules, and why each one is here:

* **The model is loaded once**, lazily, at first use. Loading takes about a
  second; doing it per call would blow the two-second response budget R9 sets.
* **Every input is validated** before it reaches the model. Bad values raise
  :class:`InvalidFlight` with a message naming the field, the rule and the
  value received - never a stack trace.
* **The public signature takes plain Python values.** No pandas object crosses
  the boundary; the DataFrame is assembled inside.
* **It runs standalone.** ``python -m app.serving.predict`` prints a worked
  example and three deliberately invalid inputs.

Usage::

    from app.serving.predict import predict_flight

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

import joblib
import numpy as np
import pandas as pd

from app.config import MODEL_PATH
from app.serving.lookups import load_lookups

HOLIDAYS: Final[pd.DatetimeIndex] = pd.to_datetime([
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-11-28", "2025-12-24",
    "2025-12-25", "2025-12-31", "2026-01-01", "2026-01-19", "2026-02-16",
    "2026-05-25",
])

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


class InvalidFlight(ValueError):
    """Raised when a caller's flight description cannot be scored."""


class ModelNotFound(FileNotFoundError):
    """Raised when the trained artefact is missing, with the command that fixes it."""


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


_artefact: dict[str, Any] | None = None


def _load_artefact() -> dict[str, Any]:
    """Load and cache the trained model artefact.

    Returns:
        The dictionary saved by ``app.models.train.export``: model, feature
        order, threshold, defaults and provenance.

    Raises:
        ModelNotFound: if the pickle does not exist, naming the training command.
    """
    global _artefact
    if _artefact is None:
        if not MODEL_PATH.exists():
            raise ModelNotFound(
                f"No model at {MODEL_PATH}.\nRun:  python -m app.cli train"
            )
        _artefact = joblib.load(MODEL_PATH)
    return _artefact


def model_info() -> dict[str, Any]:
    """Return the artefact's metadata, without the model object itself."""
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


def _validate(
    carrier: str, origin: str, dest: str, departure_hour: int, arrival_hour: int,
    day_of_week: int, month: int, distance_miles: float, scheduled_minutes: float,
    leg_number: int,
) -> None:
    """Check every input, raising :class:`InvalidFlight` on the first problem.

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


def _build_feature_row(
    *, carrier: str, origin: str, dest: str, departure_hour: int, arrival_hour: int,
    departure_minute: int, day_of_week: int, month: int, distance_miles: float,
    scheduled_minutes: float, leg_number: int,
) -> pd.DataFrame:
    """Assemble one row in exactly the column order the model expects.

    Returns:
        A single-row DataFrame whose columns match ``artefact["features"]``.
    """
    artefact = _load_artefact()
    lookups = load_lookups()
    row: dict[str, float] = dict(artefact["defaults"])

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


def _risk_band(probability: float, threshold: float) -> str:
    """Map a probability to a coarse label a user interface can colour."""
    if probability < threshold * 0.75:
        return "low"
    if probability < threshold * 1.5:
        return "moderate"
    return "high"


def _top_reasons(features: pd.DataFrame, probability: float, limit: int = 4) -> list[str]:
    """Explain a prediction in plain English using the model's own feature importances.

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
    """Return a warning when this flight falls where the model is known to fail."""
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
