"""Bandit replay evaluation: online learning curve and frozen-policy scoring.

Two distinct protocols are implemented:

``replay_train``
    Streams the training rounds *in order* through a policy, updating it
    after every round (if the policy supports ``update``). Returns the
    reward obtained round by round, which is the online learning curve.

``evaluate_frozen``
    Streams validation/test rounds through an already-trained policy with
    no further updates, measuring generalisation: reward (match) rate,
    regret against an oracle that always recommends the logged crop, and
    the agronomic constraint violation rate of the arm actually recommended
    by the *unconstrained* relevance score (diagnostic only; the policy
    itself is always restricted to the feasible set when constraints are
    enabled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from recommender.data.loader import BanditLog
from recommender.features.pipeline import ContextScaler


class BanditPolicy(Protocol):
    def select_arm(self, context: np.ndarray, candidate_arms: np.ndarray) -> int: ...
    def update(self, arm: int, context: np.ndarray, reward: float) -> None: ...


def _candidates_for(
    raw_context: np.ndarray, feasible_range: pd.DataFrame | None, n_arms: int
) -> np.ndarray:
    if feasible_range is None:
        return np.arange(n_arms)
    from recommender.models.constraints import feasible_arms

    return feasible_arms(raw_context, feasible_range)


def replay_train(
    policy: BanditPolicy,
    log: BanditLog,
    indices: np.ndarray,
    scaler: ContextScaler,
    feasible_range: pd.DataFrame | None = None,
) -> np.ndarray:
    """Stream rounds through ``policy``, updating it after each one.

    Args:
        policy: A bandit exposing ``select_arm`` and ``update``.
        log: The full replay log.
        indices: Round indices to replay, in the order they are replayed.
        scaler: Context standardisation fitted on the training rounds.
        feasible_range: Agronomic constraint table, or ``None`` to allow
            every arm every round (plain, unconstrained LinUCB).

    Returns:
        ``(len(indices),)`` binary reward obtained at each round.
    """
    rewards = np.empty(indices.shape[0])
    for i, round_idx in enumerate(indices):
        raw_context = log.context[round_idx]
        context = scaler.transform(raw_context)
        candidates = _candidates_for(raw_context, feasible_range, log.n_arms)
        chosen = policy.select_arm(context, candidates)
        reward = float(chosen == log.arm_index[round_idx])
        policy.update(chosen, context, reward)
        rewards[i] = reward
    return rewards


@dataclass(frozen=True)
class FrozenEvaluationReport:
    """Generalisation metrics for a policy replayed with learning frozen.

    Attributes:
        model: Name of the evaluated policy.
        match_rate: Fraction of rounds where the chosen arm equals the
            logged crop (the simulated reward rate).
        regret: Mean per-round regret against an oracle that always matches
            (``1 - match_rate``, reported directly since the oracle's
            reward is always 1).
        mean_candidate_pool_size: Mean number of arms offered per round
            (diagnostic on how restrictive the agronomic constraints are).
        n_rounds: Number of rounds evaluated.
    """

    model: str
    match_rate: float
    regret: float
    mean_candidate_pool_size: float
    n_rounds: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "match_rate": round(self.match_rate, 4),
            "regret": round(self.regret, 4),
            "mean_candidate_pool_size": round(self.mean_candidate_pool_size, 2),
            "n_rounds": self.n_rounds,
        }


def evaluate_frozen(
    policy: BanditPolicy,
    log: BanditLog,
    indices: np.ndarray,
    scaler: ContextScaler,
    feasible_range: pd.DataFrame | None = None,
    model_name: str = "model",
) -> FrozenEvaluationReport:
    """Replay held-out rounds through ``policy`` without updating it.

    Args:
        policy: A fitted bandit exposing ``select_arm``.
        log: The full replay log.
        indices: Held-out round indices (validation or test).
        scaler: Context standardisation fitted on the training rounds.
        feasible_range: Agronomic constraint table, or ``None``.
        model_name: Label attached to the report.

    Returns:
        A :class:`FrozenEvaluationReport`.

    Raises:
        ValueError: If ``indices`` is empty.
    """
    if indices.size == 0:
        raise ValueError("indices must not be empty.")

    matches = np.empty(indices.shape[0])
    pool_sizes = np.empty(indices.shape[0])
    for i, round_idx in enumerate(indices):
        raw_context = log.context[round_idx]
        context = scaler.transform(raw_context)
        candidates = _candidates_for(raw_context, feasible_range, log.n_arms)
        pool_sizes[i] = candidates.shape[0]
        chosen = policy.select_arm(context, candidates)
        matches[i] = float(chosen == log.arm_index[round_idx])

    match_rate = float(matches.mean())
    return FrozenEvaluationReport(
        model=model_name,
        match_rate=match_rate,
        regret=1.0 - match_rate,
        mean_candidate_pool_size=float(pool_sizes.mean()),
        n_rounds=indices.shape[0],
    )
