"""Shared fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import build_bandit_log, random_holdout_split  # noqa: E402
from recommender.data.loader import BanditLog  # noqa: E402
from recommender.features import build_feature_tables, load_feature_spec  # noqa: E402

CROPS = {
    "rice": ((90, 40, 40, 24, 82, 6.5, 230), (8, 5, 4, 1, 3, 0.2, 10)),
    "maize": ((75, 50, 20, 23, 60, 6.3, 85), (8, 5, 4, 1, 3, 0.2, 10)),
    "chickpea": ((40, 65, 80, 19, 17, 7.3, 80), (8, 5, 4, 1, 3, 0.2, 10)),
    "cotton": ((120, 45, 20, 24, 75, 6.9, 90), (8, 5, 4, 1, 3, 0.2, 10)),
}


@pytest.fixture
def observations() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    names = list(CROPS.keys())
    rows = []
    for _ in range(200):
        crop = rng.choice(names)
        mean, std = CROPS[crop]
        values = rng.normal(mean, std)
        rows.append(
            {
                "N": values[0], "P": values[1], "K": values[2], "temperature": values[3],
                "humidity": values[4], "ph": values[5], "rainfall": values[6], "label": crop,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def log(observations: pd.DataFrame) -> BanditLog:
    return build_bandit_log(observations)


@pytest.fixture
def split(log: BanditLog):
    return random_holdout_split(log, random_state=0)


@pytest.fixture
def feature_tables(log: BanditLog, split):
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    return build_feature_tables(log, split, spec)
