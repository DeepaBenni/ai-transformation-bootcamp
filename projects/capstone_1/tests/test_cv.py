import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from app.models import cv, zoo

PERIODS = [
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
]
VALIDATION = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]


def test_there_are_five_folds():
    assert len(cv.expanding_folds(PERIODS, VALIDATION)) == 5


def test_no_fold_trains_on_a_month_later_than_it_validates():
    """Random k-fold on a time series is a defect, not a shortcut (R6)."""
    for fold in cv.expanding_folds(PERIODS, VALIDATION):
        assert max(fold.train_periods) < fold.validation_period


def test_the_training_window_expands():
    folds = cv.expanding_folds(PERIODS, VALIDATION)
    sizes = [len(f.train_periods) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert set(earlier.train_periods) <= set(later.train_periods)


def test_the_first_fold_trains_on_the_burn_in_months():
    first = cv.expanding_folds(PERIODS, VALIDATION)[0]
    assert first.validation_period == "2025-12"
    assert first.train_periods == ("2025-07", "2025-08", "2025-09", "2025-10", "2025-11")


def test_validation_periods_never_overlap_training():
    for fold in cv.expanding_folds(PERIODS, VALIDATION):
        assert fold.validation_period not in fold.train_periods


def test_a_validation_period_absent_from_the_data_is_rejected():
    with pytest.raises(ValueError, match="2027-01"):
        cv.expanding_folds(PERIODS, ["2027-01"])


def test_a_validation_period_with_nothing_before_it_is_rejected():
    with pytest.raises(ValueError, match="nothing precedes"):
        cv.expanding_folds(PERIODS, [PERIODS[0]])


def test_zoo_offers_exactly_three_families():
    models = zoo.model_zoo()
    assert set(models) == {"linear", "forest", "boosted"}


def test_zoo_returns_unfitted_estimators():
    for name, model in zoo.model_zoo().items():
        assert hasattr(model, "fit"), name
        assert hasattr(model, "predict_proba"), name
        with pytest.raises(NotFittedError):
            check_is_fitted(model)
