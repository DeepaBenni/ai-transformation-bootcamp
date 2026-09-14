import numpy as np
import pytest

from app.models import metrics, threshold


def test_baseline_is_the_positive_rate():
    y = np.array([0, 0, 0, 1])
    assert metrics.baseline_pr_auc(y) == pytest.approx(0.25)


def test_lift_is_pr_auc_over_baseline():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])  # perfect ranking
    out = metrics.score(y, p, threshold=0.5)
    assert out["baseline"] == pytest.approx(0.5)
    assert out["lift"] == pytest.approx(out["pr_auc"] / out["baseline"])
    assert out["lift"] > 1.0


def test_a_useless_ranker_lifts_about_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    p = rng.random(20_000)  # no signal at all
    out = metrics.score(y, p, threshold=0.5)
    assert out["lift"] == pytest.approx(1.0, abs=0.05)


def test_recall_is_measured_at_the_given_threshold():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.3, 0.2, 0.1])
    assert metrics.score(y, p, threshold=0.5)["recall"] == pytest.approx(0.5)
    assert metrics.score(y, p, threshold=0.25)["recall"] == pytest.approx(1.0)


def test_threshold_for_recall_achieves_at_least_the_target():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.7, 0.4, 0.2, 0.35, 0.25, 0.15, 0.05])
    cut = threshold.threshold_for_recall(y, p, 0.75)
    achieved = (p >= cut)[y == 1].mean()
    assert achieved >= 0.75


def test_lower_target_recall_gives_a_higher_threshold():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.7, 0.4, 0.2, 0.35, 0.25, 0.15, 0.05])
    assert threshold.threshold_for_recall(y, p, 0.5) >= threshold.threshold_for_recall(y, p, 1.0)


def test_sweep_reports_the_share_of_schedule_flagged():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    table = threshold.sweep(y, p, [0.5])
    assert table.flagged_share.iloc[0] == pytest.approx(0.5)
