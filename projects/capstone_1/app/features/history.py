"""Backward-looking rate features, computed without look-ahead.

The naive version - a group mean over the whole dataset - leaks the label into
its own predictor. These are expanding means over strictly prior periods, so a
row's feature depends only on months that had already happened when a dispatcher
would have needed it.

Thin groups are smoothed toward the global prior: a route with three flights
should not claim a 0.0 late rate.
"""

from __future__ import annotations

import pandas as pd

from app.config import ALPHA, PRIOR


def expanding_rate(
    frame: pd.DataFrame,
    keys: list[str],
    name: str,
    *,
    alpha: int = ALPHA,
    prior: float = PRIOR,
) -> pd.DataFrame:
    """Add a smoothed late rate over strictly prior periods, and its count.

    Args:
        frame: Must carry ``period`` and ``label``, plus every column in ``keys``.
        keys: The grouping columns, e.g. ``["route"]``.
        name: Output column name; the count lands in ``f"{name}_n"``.
        alpha: Smoothing weight toward ``prior``.
        prior: The global positive rate a thin group is shrunk toward.

    Returns:
        ``frame`` with two columns added.
    """
    periods = sorted(frame.period.unique())
    rate_col = pd.Series(prior, index=frame.index, dtype="float64")
    count_col = pd.Series(0.0, index=frame.index, dtype="float64")

    running_sum: dict[tuple, float] = {}
    running_count: dict[tuple, float] = {}

    for period in periods:
        mask = frame.period == period

        # Score this period from what was known BEFORE it. Nothing from
        # `period` itself has been folded in yet.
        if running_count:
            group_keys = list(frame.loc[mask, keys].itertuples(index=False, name=None))
            rates, counts = [], []
            for key in group_keys:
                n = running_count.get(key, 0.0)
                s = running_sum.get(key, 0.0)
                rates.append((s + alpha * prior) / (n + alpha))
                counts.append(n)
            rate_col.loc[mask] = rates
            count_col.loc[mask] = counts

        # Only now does this period join the history.
        grouped = frame.loc[mask].groupby(keys, observed=True).label.agg(["sum", "count"])
        for key, row in grouped.iterrows():
            key_tuple = key if isinstance(key, tuple) else (key,)
            running_sum[key_tuple] = running_sum.get(key_tuple, 0.0) + float(row["sum"])
            running_count[key_tuple] = running_count.get(key_tuple, 0.0) + float(row["count"])

    frame[name] = rate_col
    frame[f"{name}_n"] = count_col
    return frame
