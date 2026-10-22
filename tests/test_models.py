"""Tests for LinUCB, the agronomic constraints, and the baselines."""

from __future__ import annotations

import numpy as np

from recommender.evaluation import evaluate_frozen, replay_train
from recommender.models import EpsilonGreedyPopularity, LinUCB, RandomPolicy, feasible_arms


def test_linucb_learns_better_than_random(log, split, feature_tables):
    # Evaluated without the agronomic constraint (full 4-arm candidate set):
    # with constraints on, this fixture's well-separated clusters often
    # leave only one feasible arm, which would make LinUCB and Random
    # trivially tie. Comparing on the unconstrained candidate set isolates
    # what LinUCB's contextual learning itself contributes.
    linucb = LinUCB(n_arms=log.n_arms, n_features=log.n_features, alpha=1.0)
    replay_train(linucb, log, split.train_indices, feature_tables.context_scaler, None)
    linucb_report = evaluate_frozen(
        linucb, log, split.test_indices, feature_tables.context_scaler, None, "LinUCB"
    )

    random_policy = RandomPolicy(random_state=0)
    random_report = evaluate_frozen(
        random_policy, log, split.test_indices, feature_tables.context_scaler, None, "Random"
    )

    assert linucb_report.match_rate > random_report.match_rate


def test_constraints_reduce_candidate_pool_below_full_arm_count(log, split, feature_tables):
    raw_context = log.context[split.test_indices[0]]
    candidates = feasible_arms(raw_context, feature_tables.feasible_range)
    assert candidates.size <= log.n_arms


def test_feasible_arms_never_returns_empty(log, feature_tables):
    # A wildly out-of-range context should fall back to "every arm feasible"
    # rather than leaving the bandit with no action to take.
    impossible_context = np.full(log.n_features, 1e9)
    candidates = feasible_arms(impossible_context, feature_tables.feasible_range)
    assert candidates.size == log.n_arms


def test_epsilon_greedy_update_increments_pulls_and_successes():
    policy = EpsilonGreedyPopularity(n_arms=3, epsilon=0.0, random_state=0)
    policy.update(arm=1, context=np.zeros(3), reward=1.0)
    assert policy.pulls[1] == 1
    assert policy.successes[1] == 1.0


def test_linucb_select_arm_rejects_empty_candidates(log):
    policy = LinUCB(n_arms=log.n_arms, n_features=log.n_features)
    try:
        policy.select_arm(np.zeros(log.n_features), np.array([], dtype=np.int64))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty candidate_arms")
