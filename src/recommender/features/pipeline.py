"""Build the context scaler and the per-arm feature tables.

Unlike the user/item portfolio projects, this pipeline has no per-user
table: the context scaler standardises every round's soil/climate vector,
and the arm table holds the agronomic feasible range, training popularity,
and descriptive centroid for each crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.loader import CONTEXT_COLUMNS, BanditLog
from recommender.data.splitting import Split
from recommender.features.registry import TRANSFORMS, assert_registry_matches_spec
from recommender.features.spec import FeatureSpec


@dataclass(frozen=True)
class ContextScaler:
    """Per-feature standardisation, fitted on training rounds only."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, context: np.ndarray) -> np.ndarray:
        return (context - self.mean) / self.std


@dataclass(frozen=True)
class FeatureTables:
    """Materialized feature tables."""

    context_scaler: ContextScaler
    feasible_range: pd.DataFrame
    popularity: pd.Series
    centroid: pd.DataFrame


def build_feature_tables(log: BanditLog, split: Split, spec: FeatureSpec) -> FeatureTables:
    """Materialize every declared feature, strictly from training rounds.

    Args:
        log: The full replay log.
        split: Leakage-safe partition; only ``split.train_indices`` feeds
            every transform.
        spec: The loaded declarative feature specification.

    Returns:
        The context scaler and the per-arm feature tables.

    Raises:
        ValueError: If the registry and the spec disagree.
    """
    assert_registry_matches_spec(spec)

    outputs = {feature.name: TRANSFORMS[feature.transform](log, split) for feature in spec.features}

    zscore = outputs["context_zscore"]
    mean = zscore[[f"{col}_mean" for col in CONTEXT_COLUMNS]].to_numpy()
    std = zscore[[f"{col}_std" for col in CONTEXT_COLUMNS]].to_numpy()

    return FeatureTables(
        context_scaler=ContextScaler(mean=mean, std=std),
        feasible_range=outputs["arm_feasible_range"],
        popularity=outputs["arm_popularity"],
        centroid=outputs["arm_context_centroid"],
    )
