"""Row-level partitioning of the bandit replay log.

There is no user/session concept here, so the split is a simple random
holdout over rounds rather than the user-aware splits used elsewhere in
this portfolio. The **training** partition doubles as the online-learning
replay stream: LinUCB updates its per-arm parameters round by round while
replaying it, in a fixed shuffled order. The **validation**/**test**
partitions are replayed afterwards with the policy frozen (no further
updates), which is what isolates "does the learned policy generalise" from
"how fast does it learn."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.loader import BanditLog


@dataclass(frozen=True)
class Split:
    """Train (learning stream) / validation / test round indices."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    n_arms: int

    def __post_init__(self) -> None:
        if self.train_indices.size == 0:
            raise ValueError("The training partition is empty.")


def random_holdout_split(
    log: BanditLog,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> Split:
    """Shuffle rounds once and cut them into train/validation/test.

    Args:
        log: The full replay log.
        validation_fraction: Fraction of rounds held out for validation.
        test_fraction: Fraction of rounds held out for test.
        random_state: Seed controlling both the shuffle and, downstream,
            the replay order used for online learning.

    Returns:
        A :class:`Split`.

    Raises:
        ValueError: If the fractions are not in ``(0, 1)`` and summing to
            less than 1.
    """
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("validation_fraction and test_fraction must be in (0, 1).")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1.")

    rng = np.random.default_rng(random_state)
    order = rng.permutation(log.n_rounds)

    n_val = int(log.n_rounds * validation_fraction)
    n_test = int(log.n_rounds * test_fraction)

    validation_indices = order[:n_val]
    test_indices = order[n_val : n_val + n_test]
    train_indices = order[n_val + n_test :]

    return Split(
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        n_arms=log.n_arms,
    )
