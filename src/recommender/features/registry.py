"""Registry mapping declared feature names to their transform implementations.

``assert_registry_matches_spec`` is the two-way check referenced by
``conf/features.yaml``: every declared feature must have a registered
transform, and every registered transform must be declared.
"""

from __future__ import annotations

from collections.abc import Callable

from recommender.features.spec import FeatureSpec
from recommender.features.transforms import (
    arm_context_centroid,
    arm_feasible_range,
    arm_popularity,
    context_zscore,
)

TRANSFORMS: dict[str, Callable] = {
    "context_zscore": context_zscore,
    "arm_feasible_range": arm_feasible_range,
    "arm_popularity": arm_popularity,
    "arm_context_centroid": arm_context_centroid,
}


def assert_registry_matches_spec(spec: FeatureSpec) -> None:
    """Fail loudly if the spec and the registry disagree in either direction.

    Args:
        spec: The loaded declarative specification.

    Raises:
        ValueError: If a declared transform is unimplemented, or an
            implemented transform is undeclared.
    """
    declared = {f.transform for f in spec.features}
    implemented = set(TRANSFORMS)

    undeclared = implemented - declared
    unimplemented = declared - implemented
    if undeclared or unimplemented:
        raise ValueError(
            "Feature registry and conf/features.yaml disagree: "
            f"implemented-but-undeclared={sorted(undeclared)}, "
            f"declared-but-unimplemented={sorted(unimplemented)}."
        )
