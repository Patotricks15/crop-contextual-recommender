"""Agronomic feasibility constraints over the arm set.

A crop is only offered to the bandit for a given round if that round's raw
(unstandardised) soil/climate values fall inside the crop's training-time
observed range, expanded by a small tolerance (see
``recommender.features.transforms.FEASIBILITY_TOLERANCE``). This never
blocks the *true* crop for a round drawn from the same distribution as
training, but it does rule out arms LinUCB would otherwise have to learn
are bad purely from exploration -- the entire point of the "constrained" in
constrained LinUCB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data.loader import CONTEXT_COLUMNS


def feasible_arms(raw_context: np.ndarray, feasible_range: pd.DataFrame) -> np.ndarray:
    """Return the indices of arms whose feasible range contains ``raw_context``.

    Args:
        raw_context: A single round's unstandardised context vector, in
            :data:`~recommender.data.loader.CONTEXT_COLUMNS` order.
        feasible_range: Per-arm ``f"{feature}_min"``/``f"{feature}_max"``
            table from :func:`recommender.features.transforms.arm_feasible_range`.

    Returns:
        Sorted array of feasible arm indices. Never empty: if no arm
        satisfies every constraint, every arm is returned instead, since an
        empty feasible set would leave the bandit with no action to take.
    """
    mins = feasible_range[[f"{col}_min" for col in CONTEXT_COLUMNS]].to_numpy()
    maxs = feasible_range[[f"{col}_max" for col in CONTEXT_COLUMNS]].to_numpy()
    satisfied = ((raw_context >= mins) & (raw_context <= maxs)).all(axis=1)
    feasible = np.flatnonzero(satisfied)
    if feasible.size == 0:
        return np.arange(feasible_range.shape[0])
    return feasible
