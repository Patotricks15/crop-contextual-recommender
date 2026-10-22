"""Parse and validate the declarative feature specification.

See ``conf/features.yaml`` for the specification this module loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_FAMILIES = {"user", "item", "interaction", "context", "text"}
VALID_FIT_SCOPES = {"train_only", "catalogue"}


@dataclass(frozen=True)
class FeatureDef:
    """One declared feature."""

    name: str
    family: str
    dtype: str
    transform: str
    fit_on: str
    fill: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class FeatureSpec:
    """The full declarative specification."""

    version: int
    dataset: str
    features: tuple[FeatureDef, ...]

    def names(self) -> set[str]:
        return {f.name for f in self.features}


def load_feature_spec(path: str | Path) -> FeatureSpec:
    """Load and validate ``conf/features.yaml``.

    Args:
        path: Path to the YAML specification.

    Returns:
        A validated :class:`FeatureSpec`.

    Raises:
        FileNotFoundError: If the spec file does not exist.
        ValueError: If a feature declares an unknown family, fit scope, or a
            name is duplicated.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    raw = yaml.safe_load(path.read_text())
    features = []
    seen = set()
    for item in raw["features"]:
        if item["family"] not in VALID_FAMILIES:
            raise ValueError(f"Unknown family {item['family']!r} for feature {item['name']!r}.")
        if item["fit_on"] not in VALID_FIT_SCOPES:
            raise ValueError(f"Unknown fit_on {item['fit_on']!r} for feature {item['name']!r}.")
        if item["name"] in seen:
            raise ValueError(f"Duplicate feature name {item['name']!r}.")
        seen.add(item["name"])
        features.append(
            FeatureDef(
                name=item["name"],
                family=item["family"],
                dtype=item["dtype"],
                transform=item["transform"],
                fit_on=item["fit_on"],
                fill=item.get("fill", {}),
                rationale=item["rationale"].strip(),
            )
        )

    return FeatureSpec(version=raw["version"], dataset=raw["dataset"], features=tuple(features))
