"""Choosing the operating point.

The operator states the recall the operation needs; the threshold follows. Doing
it the other way round - taking 0.5 because it is the default, then reporting
whatever recall falls out - produced 2.9% recall in the source analysis.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def sweep(y_true: np.ndarray, probabilities: np.ndarray, grid: Iterable[float]) -> pd.DataFrame:
    """Tabulate recall, precision and how much of the schedule each cut flags.

    Args:
        y_true: Binary labels.
        probabilities: Predicted P(late).
        grid: Thresholds to evaluate.

    Returns:
        One row per threshold.
    """
    rows = []
    positives = y_true.sum()
    for cut in grid:
        flagged = probabilities >= cut
        hits = int((flagged & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": float(cut),
                "recall": hits / positives if positives else 0.0,
                "precision": hits / flagged.sum() if flagged.sum() else 0.0,
                "flagged_share": float(flagged.mean()),
            }
        )
    return pd.DataFrame(rows)


def threshold_for_recall(
    y_true: np.ndarray, probabilities: np.ndarray, target_recall: float
) -> float:
    """Return the highest threshold still achieving ``target_recall``.

    Args:
        y_true: Binary labels.
        probabilities: Predicted P(late).
        target_recall: The recall the operation requires, 0-1.

    Returns:
        The threshold. Falls to the minimum probability if the target is
        unreachable, which flags everything and is reported as such.
    """
    positive_scores = np.sort(probabilities[y_true == 1])[::-1]
    if positive_scores.size == 0:
        return float(probabilities.min())

    index = int(np.ceil(target_recall * positive_scores.size)) - 1
    index = min(max(index, 0), positive_scores.size - 1)
    return float(positive_scores[index])
