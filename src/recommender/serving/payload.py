"""Serialize a fitted LinUCB policy into a small JSON artifact for serving.

LinUCB's state is just per-arm ``A`` (``d x d``) and ``b`` (``d``) arrays
plus the context scaler and the agronomic constraint table -- small enough
to ship as one JSON blob read once at Lambda cold start, with no need to
replay any training data at request time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from recommender.data.loader import CONTEXT_COLUMNS
from recommender.features.pipeline import ContextScaler
from recommender.models.constraints import feasible_arms
from recommender.models.linucb import LinUCB


def serialize_policy(
    policy: LinUCB,
    scaler: ContextScaler,
    feasible_range: pd.DataFrame,
    arm_names: np.ndarray,
) -> dict[str, Any]:
    """Pack a fitted policy into a JSON-serialisable dict.

    Args:
        policy: A fitted (or partially fitted) :class:`LinUCB`.
        scaler: The training-fitted context standardisation.
        feasible_range: The per-arm agronomic constraint table.
        arm_names: Crop names, positionally aligned to arm index.

    Returns:
        A dict safe to ``json.dumps`` directly.
    """
    return {
        "alpha": policy.alpha,
        "arm_names": arm_names.tolist(),
        "scaler_mean": scaler.mean.tolist(),
        "scaler_std": scaler.std.tolist(),
        "A": [a.tolist() for a in policy.A],
        "b": [b.tolist() for b in policy.b],
        "feasible_range": feasible_range.to_dict(orient="index"),
    }


def recommend_from_payload(
    payload: dict[str, Any], raw_context: dict[str, float]
) -> dict[str, Any]:
    """Score every feasible arm for one raw context, using a serialized policy.

    Args:
        payload: Output of :func:`serialize_policy`, round-tripped through JSON.
        raw_context: Mapping of :data:`~recommender.data.loader.CONTEXT_COLUMNS`
            to their unstandardised values for this request.

    Returns:
        ``{"crop": ..., "score": ..., "candidate_pool_size": ...}``.

    Raises:
        KeyError: If ``raw_context`` is missing a required feature.
    """
    raw = np.array([raw_context[col] for col in CONTEXT_COLUMNS], dtype=np.float64)
    mean = np.array(payload["scaler_mean"])
    std = np.array(payload["scaler_std"])
    context = (raw - mean) / std

    arm_names = payload["arm_names"]
    feasible_range = pd.DataFrame.from_dict(payload["feasible_range"], orient="index")
    feasible_range.index = feasible_range.index.astype(int)
    candidates = feasible_arms(raw, feasible_range)

    alpha = payload["alpha"]
    best_arm, best_score = None, -np.inf
    for arm in candidates:
        a = np.array(payload["A"][arm])
        b = np.array(payload["b"][arm])
        a_inv = np.linalg.inv(a)
        theta = a_inv @ b
        score = float(context @ theta) + alpha * float(np.sqrt(context @ a_inv @ context))
        if score > best_score:
            best_arm, best_score = int(arm), score

    return {
        "crop": arm_names[best_arm],
        "score": round(best_score, 4),
        "candidate_pool_size": int(candidates.shape[0]),
    }
