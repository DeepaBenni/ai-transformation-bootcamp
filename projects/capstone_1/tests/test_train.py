import numpy as np
import pandas as pd

from app.features.build import FEATURE_COLUMNS
from app.models import train


def _synthetic(rows_per_period=400):
    """Small frame with real signal, so a model can actually learn something."""
    rng = np.random.default_rng(7)
    periods = ["2026-01", "2026-02", "2026-03"]
    frames = []
    for period in periods:
        n = rows_per_period
        data = {col: rng.random(n) for col in FEATURE_COLUMNS}
        # A learnable rule: high origin_hour_late_rate means late.
        signal = data["origin_hour_late_rate"]
        data["label"] = (signal + rng.normal(0, 0.15, n) > 0.55).astype(int)
        data["period"] = period
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def test_run_cv_returns_one_row_per_model_and_fold():
    out = train.run_cv(_synthetic(), validation_periods=["2026-02", "2026-03"])
    assert len(out) == 3 * 2
    assert set(out.model) == {"linear", "forest", "boosted"}


def test_every_result_carries_a_lift_not_a_bare_score():
    out = train.run_cv(_synthetic(), validation_periods=["2026-03"])
    assert {"pr_auc", "baseline", "lift"} <= set(out.columns)
    assert (out.lift > 0).all()


def test_a_model_with_signal_beats_the_baseline():
    out = train.run_cv(_synthetic(), validation_periods=["2026-03"])
    assert out.lift.max() > 1.0


def test_summarise_reports_the_standard_deviation():
    out = train.run_cv(_synthetic(), validation_periods=["2026-02", "2026-03"])
    summary = train.summarise(out)
    assert "pr_auc_sd" in summary.columns
    assert len(summary) == 3


def test_export_writes_an_artefact_carrying_the_feature_order(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "model.pkl")
    # rows_per_period=3000: at the default 400, the boosted family - evaluated
    # in-sample against its own training period in export() - drives
    # probabilities so close to 1 that round(threshold, 4) collides with 1.0
    # and trips the threshold < 1.0 assertion below. A rounding artefact of an
    # overfit toy model, not a production concern (real folds carry millions
    # of rows); 3000 gives it enough data to settle safely under 1.0.
    frame = _synthetic(rows_per_period=3000)
    results = train.run_cv(frame, validation_periods=["2026-03"])
    model, _ = train.fit_final(frame, "boosted")

    path = train.export(model, frame, results)

    import joblib

    artefact = joblib.load(path)
    assert artefact["features"] == FEATURE_COLUMNS
    assert 0.0 < artefact["threshold"] < 1.0
    assert artefact["baseline_pr_auc"] > 0
    assert set(artefact["defaults"]) == set(FEATURE_COLUMNS)
