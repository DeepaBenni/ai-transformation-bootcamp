"""Cross-validate the three families, fit the winner, export the artefact."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import FOLD_VALIDATION_MONTHS, MODEL_PATH, PREDICTION_TIME
from app.features.build import FEATURE_COLUMNS
from app.models.cv import expanding_folds
from app.models.metrics import baseline_pr_auc, score
from app.models.threshold import threshold_for_recall
from app.models.zoo import FOREST_CONFIG, model_zoo


def run_cv(
    frame: pd.DataFrame,
    *,
    validation_periods: list[str] | None = None,
    threshold: float = 0.30,
) -> pd.DataFrame:
    """Fit every family on every expanding fold.

    Args:
        frame: Features plus ``period`` and ``label``.
        validation_periods: Defaults to the five configured months.
        threshold: The operating point recall is measured at during CV.

    Returns:
        One row per model and fold, carrying pr_auc, baseline, lift and fit time.
    """
    validation_periods = list(validation_periods or FOLD_VALIDATION_MONTHS)
    folds = expanding_folds(frame.period.unique(), validation_periods)
    rows: list[dict[str, Any]] = []

    for fold in folds:
        train_mask = frame.period.isin(fold.train_periods)
        valid_mask = frame.period == fold.validation_period

        x_train = frame.loc[train_mask, FEATURE_COLUMNS]
        y_train = frame.loc[train_mask, "label"].to_numpy()
        x_valid = frame.loc[valid_mask, FEATURE_COLUMNS]
        y_valid = frame.loc[valid_mask, "label"].to_numpy()

        for name, model in model_zoo().items():
            started = time.perf_counter()
            model.fit(x_train, y_train)
            fit_s = time.perf_counter() - started

            probabilities = model.predict_proba(x_valid)[:, 1]
            result = score(y_valid, probabilities, threshold)
            rows.append(
                {
                    "model": name,
                    "period": fold.validation_period,
                    "train_months": len(fold.train_periods),
                    "fit_s": round(fit_s, 1),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Average each family across folds, keeping the spread.

    The standard deviation is not decoration: if one family's spread overlaps
    another's mean, the ranking between them is not established.

    Args:
        results: Output of :func:`run_cv`.

    Returns:
        One row per family.
    """
    grouped = results.groupby("model")
    summary = grouped.agg(
        pr_auc=("pr_auc", "mean"),
        pr_auc_sd=("pr_auc", "std"),
        lift=("lift", "mean"),
        recall=("recall", "mean"),
        precision=("precision", "mean"),
        roc_auc=("roc_auc", "mean"),
        brier=("brier", "mean"),
        fit_s=("fit_s", "mean"),
    )
    return summary.sort_values("lift", ascending=False).round(4)


def fit_final(frame: pd.DataFrame, family: str) -> tuple[Any, dict[str, Any]]:
    """Fit one family on every development row.

    Args:
        frame: Features plus ``period`` and ``label``.
        family: A key of :func:`app.models.zoo.model_zoo`.

    Returns:
        The fitted model and its training metadata.
    """
    model = model_zoo()[family]
    started = time.perf_counter()
    model.fit(frame[FEATURE_COLUMNS], frame.label.to_numpy())

    meta = {
        "rows": int(len(frame)),
        "period_start": str(frame.period.min()),
        "period_end": str(frame.period.max()),
        "fit_s": round(time.perf_counter() - started, 1),
    }
    return model, meta


def export(
    model: object,
    frame: pd.DataFrame,
    cv_results: pd.DataFrame,
    *,
    family: str = "boosted",
    target_recall: float = 0.60,
) -> Path:
    """Write the model plus everything needed to use it six weeks from now.

    A model alone is a trap: nobody remembers which columns it wants, in what
    order, or which threshold was chosen.

    Args:
        model: The fitted estimator.
        frame: The development frame it was fitted on.
        cv_results: Output of :func:`run_cv`.
        family: Which family ``model`` belongs to.
        target_recall: The recall the operating point is chosen for.

    Returns:
        The artefact path.
    """
    last_period = frame.period.max()
    holdback = frame[frame.period == last_period]
    probabilities = model.predict_proba(holdback[FEATURE_COLUMNS])[:, 1]
    y_true = holdback.label.to_numpy()

    chosen = threshold_for_recall(y_true, probabilities, target_recall)
    family_rows = cv_results[cv_results.model == family]

    artefact = {
        "model": model,
        "model_family": family,
        "features": list(FEATURE_COLUMNS),
        "threshold": round(float(chosen), 4),
        "baseline_pr_auc": round(baseline_pr_auc(frame.label.to_numpy()), 4),
        "forest_config": FOREST_CONFIG,
        "validation": {
            "pr_auc": round(float(family_rows.pr_auc.mean()), 4),
            "pr_auc_sd": round(float(family_rows.pr_auc.std()), 4),
            "lift": round(float(family_rows.lift.mean()), 4),
            "folds": family_rows.period.tolist(),
        },
        "training": {
            "rows": int(len(frame)),
            "period_start": str(frame.period.min()),
            "period_end": str(last_period),
        },
        "defaults": {col: float(np.nanmedian(frame[col])) for col in FEATURE_COLUMNS},
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "notes": f"Predicts P(arrival 15+ min late) at {PREDICTION_TIME}.",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, MODEL_PATH, compress=3)
    return MODEL_PATH
