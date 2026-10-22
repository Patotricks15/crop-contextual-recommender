"""LinUCB: a linear, disjoint-model contextual bandit (Li et al., 2010).

Each arm ``a`` keeps a ridge-regression posterior over its own reward model:

    A_a = I_d + sum_{t: chosen a} x_t x_t^T
    b_a = sum_{t: chosen a} r_t x_t
    theta_a = A_a^{-1} b_a

The upper-confidence-bound score for context ``x`` under arm ``a`` is

    p_a(x) = x^T theta_a + alpha * sqrt(x^T A_a^{-1} x),

where the first term is the point estimate of expected reward and the
second is an exploration bonus proportional to the model's uncertainty
about ``x`` under arm ``a``'s current posterior. This is closed-form linear
algebra updated incrementally after every round; there is no gradient
descent and no neural network anywhere in this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinUCB:
    """Disjoint LinUCB over ``n_arms`` arms and ``n_features`` context dims.

    Attributes:
        n_arms: Number of arms (crops).
        n_features: Context dimensionality.
        alpha: Exploration strength; larger values widen the confidence
            bonus and favor exploring less-visited arms.
    """

    n_arms: int
    n_features: int
    alpha: float = 1.0

    A: list[np.ndarray] = field(init=False)
    b: list[np.ndarray] = field(init=False)
    pulls: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.A = [np.eye(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.pulls = np.zeros(self.n_arms, dtype=np.int64)

    def scores(self, context: np.ndarray, candidate_arms: np.ndarray) -> np.ndarray:
        """UCB score for every arm in ``candidate_arms``, same order."""
        out = np.empty(candidate_arms.shape[0])
        for i, arm in enumerate(candidate_arms):
            a_inv = np.linalg.inv(self.A[arm])
            theta = a_inv @ self.b[arm]
            mean = float(context @ theta)
            bonus = self.alpha * float(np.sqrt(context @ a_inv @ context))
            out[i] = mean + bonus
        return out

    def select_arm(self, context: np.ndarray, candidate_arms: np.ndarray) -> int:
        """Pick the arm in ``candidate_arms`` with the highest UCB score.

        Args:
            context: Standardised context vector for the current round.
            candidate_arms: Feasible arm indices for this round (output of
                :func:`recommender.models.constraints.feasible_arms`).

        Returns:
            The chosen arm index.

        Raises:
            ValueError: If ``candidate_arms`` is empty.
        """
        if candidate_arms.size == 0:
            raise ValueError("candidate_arms must not be empty.")
        scores = self.scores(context, candidate_arms)
        return int(candidate_arms[np.argmax(scores)])

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Incorporate one round's observed reward into arm ``arm``'s posterior."""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.pulls[arm] += 1
