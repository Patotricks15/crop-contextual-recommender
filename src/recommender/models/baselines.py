"""Non-contextual bandit baselines: Random and frequency-weighted greedy.

Both ignore the soil/climate context entirely, which is exactly the
behaviour LinUCB's contextual, constrained policy should beat: a crop that
is popular overall is not necessarily feasible or correct for any given
round's conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RandomPolicy:
    """Picks uniformly among the feasible arms, ignoring context."""

    random_state: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)

    def select_arm(self, context: np.ndarray, candidate_arms: np.ndarray) -> int:  # noqa: ARG002
        return int(self._rng.choice(candidate_arms))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:  # noqa: ARG002
        return


@dataclass
class EpsilonGreedyPopularity:
    """With probability epsilon explores uniformly; otherwise picks the
    feasible arm with the highest observed reward rate so far.
    """

    n_arms: int
    epsilon: float = 0.1
    random_state: int = 0

    successes: np.ndarray = field(init=False)
    pulls: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.successes = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self._rng = np.random.default_rng(self.random_state)

    def select_arm(self, context: np.ndarray, candidate_arms: np.ndarray) -> int:  # noqa: ARG002
        if self._rng.random() < self.epsilon:
            return int(self._rng.choice(candidate_arms))
        rates = np.where(
            self.pulls[candidate_arms] > 0,
            self.successes[candidate_arms] / np.maximum(self.pulls[candidate_arms], 1),
            0.0,
        )
        return int(candidate_arms[np.argmax(rates)])

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:  # noqa: ARG002
        self.pulls[arm] += 1
        self.successes[arm] += reward
