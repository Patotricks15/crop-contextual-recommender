"""Schema-contract tests for the feature spec and the raw data loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recommender.data.loader import build_bandit_log, load_observations
from recommender.features.spec import load_feature_spec

ROOT = Path(__file__).resolve().parents[1]


def test_feature_spec_declares_known_families_only():
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    assert spec.names() == {
        "context_zscore",
        "arm_feasible_range",
        "arm_popularity",
        "arm_context_centroid",
    }


def test_load_observations_rejects_missing_columns(tmp_path):
    bad = tmp_path / "observations.csv"
    pd.DataFrame({"N": [1], "label": ["rice"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_observations(bad)


def test_load_observations_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_observations(tmp_path / "missing.csv")


def test_build_bandit_log_assigns_contiguous_arm_indices():
    observations = pd.DataFrame(
        {
            "N": [1, 2, 3], "P": [1, 2, 3], "K": [1, 2, 3], "temperature": [1, 2, 3],
            "humidity": [1, 2, 3], "ph": [6, 6, 6], "rainfall": [100, 100, 100],
            "label": ["rice", "maize", "rice"],
        }
    )
    log = build_bandit_log(observations)
    assert log.n_arms == 2
    assert set(log.arm_index.tolist()) == {0, 1}
