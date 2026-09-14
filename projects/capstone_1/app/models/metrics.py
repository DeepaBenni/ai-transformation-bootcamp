"""Scoring, always relative to a baseline that moves.

R4 forbids reporting a bare score. The positive rate shifts month to month, so
the same PR-AUC means different things in different folds; only the multiple
over that fold's own baseline compares.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def baseline_pr_auc(y: np.ndarray) -> float:
    """Return the PR-AUC a no-skill ranker achieves: the positive rate.

    Args:
        y: Binary labels.

    Returns:
        The proportion of positives.
    """
    return float(np.mean(y))


def score(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Score predictions against this fold's own baseline.

    Args:
        y_true: Binary labels.
        probabilities: Predicted P(late).
        threshold: The operating point recall and precision are measured at.

    Returns:
        A mapping with pr_auc, baseline, lift, recall, precision, roc_auc, brier.
    """
    flagged = (probabilities >= threshold).astype(int)
    base = baseline_pr_auc(y_true)
    pr_auc = float(average_precision_score(y_true, probabilities))

    return {
        "pr_auc": pr_auc,
        "baseline": base,
        "lift": pr_auc / base if base else float("nan"),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }
