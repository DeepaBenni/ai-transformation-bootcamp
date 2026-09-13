"""Expanding-window cross-validation.

A random fold would train on June to predict January. Worse here than usual:
the history features are expanding means over prior months, so shuffled folds
would let a model see rates computed from its own validation rows. Every fold
trains strictly on the past.

The five windows also serve as the rolling backtest - stability across them is
reported rather than averaged away.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """One expanding window: everything before, validated on one month."""

    train_periods: tuple[str, ...]
    validation_period: str


def expanding_folds(periods: Iterable[str], validation_periods: Iterable[str]) -> list[Fold]:
    """Build folds whose training window grows and never wraps.

    Args:
        periods: Every period present, ``YYYY-MM``.
        validation_periods: The periods to validate on, in order.

    Returns:
        One fold per validation period.

    Raises:
        ValueError: If a validation period is absent, or nothing precedes it.
    """
    ordered = sorted(set(periods))
    folds: list[Fold] = []

    for period in sorted(validation_periods):
        if period not in ordered:
            raise ValueError(f"validation period {period} is not in the data")
        train = tuple(p for p in ordered if p < period)
        if not train:
            raise ValueError(f"nothing precedes {period}; it cannot be validated")
        folds.append(Fold(train_periods=train, validation_period=period))
    return folds
