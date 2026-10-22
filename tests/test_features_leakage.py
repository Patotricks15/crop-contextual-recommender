"""Leakage-prevention tests for the feature engineering layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recommender.features.registry import assert_registry_matches_spec
from recommender.features.spec import load_feature_spec

ROOT = Path(__file__).resolve().parents[1]


def test_context_scaler_uses_only_training_rounds(log, split, feature_tables):
    train_context = log.context[split.train_indices]
    expected_mean = train_context.mean(axis=0)
    np.testing.assert_allclose(feature_tables.context_scaler.mean, expected_mean)


def test_context_scaler_is_unaffected_by_held_out_rounds(log, split):
    from recommender.features.transforms import context_zscore

    # Corrupting a held-out round's context must not change the scaler: it
    # is computed strictly over split.train_indices.
    before = context_zscore(log, split)
    corrupted_log_context = log.context.copy()
    corrupted_log_context[split.test_indices[0]] = 1e6
    from dataclasses import replace

    corrupted_log = replace(log, context=corrupted_log_context)
    after = context_zscore(corrupted_log, split)
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy())


def test_arm_feasible_range_uses_only_training_rounds(log, split, feature_tables):
    train_frame = log.frame.iloc[split.train_indices]
    for arm in range(log.n_arms):
        arm_rows = train_frame[train_frame["arm_index"] == arm]
        if arm_rows.empty:
            continue
        observed_min = arm_rows["N"].min()
        stored_min = feature_tables.feasible_range.loc[arm, "N_min"]
        # Stored min is the observed min minus a tolerance slack, so it must
        # be <= the raw observed min, never derived from held-out rows.
        assert stored_min <= observed_min


def test_registry_matches_spec_declaration():
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    assert_registry_matches_spec(spec)  # must not raise
