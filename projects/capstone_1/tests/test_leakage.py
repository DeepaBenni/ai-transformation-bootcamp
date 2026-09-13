import pandas as pd
import pytest

from app.features import history
from app.features.forbidden import FORBIDDEN, LeakageError, assert_no_leakage


def _tiny_frame():
    """Four periods, one route, alternating labels - hand-checkable."""
    return pd.DataFrame(
        {
            "period": ["2026-01", "2026-02", "2026-03", "2026-04"],
            "route": ["ATL-ORD"] * 4,
            "label": [1, 1, 0, 0],
        }
    )


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
    assert_no_leakage(["Distance", "dep_hour", "route_late_rate"]) is None  # noqa: B015


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
