"""Three model families, so complexity has to justify itself (R6).

The random forest runs deliberately smaller than a tuned configuration would.
It is the slowest family and loses to boosting on every metric that matters, so
it is present to be compared, not to be shipped - and its reduced settings are
reported alongside its score rather than hidden.
"""

from __future__ import annotations

from typing import Final

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import RANDOM_STATE

# Reported in the results table so the comparison is not silently unfair.
FOREST_CONFIG: Final[dict[str, int]] = {"n_estimators": 100, "max_depth": 10}


def model_zoo() -> dict[str, object]:
    """Return one unfitted estimator per family.

    Returns:
        ``linear`` - the interpretable floor; if boosting cannot beat it,
        complexity is not earning its keep.
        ``forest`` - bagging, captures interactions without boosting.
        ``boosted`` - LightGBM, which handles the categorical explosion natively.
    """
    return {
        "linear": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "forest": RandomForestClassifier(
            n_estimators=FOREST_CONFIG["n_estimators"],
            max_depth=FOREST_CONFIG["max_depth"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "boosted": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }
