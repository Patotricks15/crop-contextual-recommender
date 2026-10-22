"""Individual feature transforms.

Every transform is fitted strictly on ``split.train_indices``: the rounds
LinUCB replays for online learning. Validation and test rounds never
influence a mean, std, popularity count, or feasible range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data.loader import CONTEXT_COLUMNS, BanditLog
from recommender.data.splitting import Split

#: Expand each crop's observed min/max by this fraction of its own range
#: before treating a context as infeasible. Without slack, a test-time round
#: that is marginally outside the training extremes for its true crop would
#: be rejected outright, which is too strict for a dataset this small.
FEASIBILITY_TOLERANCE = 0.08


def context_zscore(log: BanditLog, split: Split) -> pd.Series:
    """Per-feature (mean, std) fitted on the training rounds.

    Returns:
        A Series indexed by ``f"{feature}_mean"`` / ``f"{feature}_std"``.
    """
    train_context = log.context[split.train_indices]
    mean = train_context.mean(axis=0)
    std = train_context.std(axis=0)
    std[std == 0] = 1.0  # a constant feature would otherwise divide by zero

    index = [f"{col}_mean" for col in CONTEXT_COLUMNS] + [f"{col}_std" for col in CONTEXT_COLUMNS]
    values = np.concatenate([mean, std])
    return pd.Series(values, index=index)


def arm_feasible_range(log: BanditLog, split: Split) -> pd.DataFrame:
    """Per-arm (min, max) per raw feature, expanded by a tolerance margin.

    Returns:
        A frame indexed by ``arm_index`` with ``f"{feature}_min"`` /
        ``f"{feature}_max"`` columns.
    """
    train_frame = log.frame.iloc[split.train_indices]
    rows = {}
    for arm in range(log.n_arms):
        arm_rows = train_frame[train_frame["arm_index"] == arm]
        record = {}
        for col in CONTEXT_COLUMNS:
            if arm_rows.empty:
                record[f"{col}_min"], record[f"{col}_max"] = -np.inf, np.inf
                continue
            lo, hi = arm_rows[col].min(), arm_rows[col].max()
            span = hi - lo
            slack = span * FEASIBILITY_TOLERANCE if span > 0 else abs(lo) * FEASIBILITY_TOLERANCE
            record[f"{col}_min"] = lo - slack
            record[f"{col}_max"] = hi + slack
        rows[arm] = record
    return pd.DataFrame.from_dict(rows, orient="index")


def arm_popularity(log: BanditLog, split: Split) -> pd.Series:
    """Training-time frequency count per arm."""
    train_frame = log.frame.iloc[split.train_indices]
    counts = train_frame["arm_index"].value_counts()
    return counts.reindex(range(log.n_arms), fill_value=0).astype("float64")


def arm_context_centroid(log: BanditLog, split: Split) -> pd.DataFrame:
    """Mean raw soil/climate vector per arm, for descriptive reporting."""
    train_frame = log.frame.iloc[split.train_indices]
    return train_frame.groupby("arm_index")[CONTEXT_COLUMNS].mean().reindex(range(log.n_arms))
