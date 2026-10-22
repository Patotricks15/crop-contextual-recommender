"""Build and persist the context scaler and per-arm feature tables.

    python scripts/build_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import build_bandit_log, load_observations, random_holdout_split  # noqa: E402
from recommender.features import build_feature_tables, load_feature_spec  # noqa: E402


def main() -> None:
    observations = load_observations(ROOT / "data" / "observations.csv")
    log = build_bandit_log(observations)
    split = random_holdout_split(log)
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")

    tables = build_feature_tables(log, split, spec)

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    tables.feasible_range.to_parquet(out_dir / "arm_feasible_range.parquet")
    tables.centroid.to_parquet(out_dir / "arm_context_centroid.parquet")
    (out_dir / "context_scaler.json").write_text(
        json.dumps(
            {"mean": tables.context_scaler.mean.tolist(), "std": tables.context_scaler.std.tolist()}
        )
    )
    print(f"wrote context scaler and arm tables for {log.n_arms} crops to {out_dir}")


if __name__ == "__main__":
    main()
