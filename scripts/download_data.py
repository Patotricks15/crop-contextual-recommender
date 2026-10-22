"""Fetch the Crop Recommendation dataset, or generate a synthetic fixture.

Real data (requires ``KAGGLE_USERNAME``/``KAGGLE_KEY``):

    python scripts/download_data.py --kaggle

Synthetic fixture with the same schema, no Kaggle account needed:

    python scripts/download_data.py --synthetic
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KAGGLE_REF = "atharvaingle/crop-recommendation-dataset"

# (N, P, K, temperature, humidity, pH, rainfall) cluster centers and spreads
# per synthetic crop -- loosely inspired by real agronomic ranges, enough to
# give each crop a genuinely distinct, partially overlapping feasible
# region so the constraint layer and LinUCB both have real signal to learn.
CROPS = {
    "rice": ((90, 40, 40, 24, 82, 6.5, 230), (15, 10, 8, 2, 5, 0.4, 25)),
    "maize": ((75, 50, 20, 23, 60, 6.3, 85), (12, 10, 6, 3, 8, 0.5, 20)),
    "chickpea": ((40, 65, 80, 19, 17, 7.3, 80), (8, 10, 8, 2, 4, 0.5, 15)),
    "cotton": ((120, 45, 20, 24, 75, 6.9, 90), (15, 10, 6, 3, 6, 0.4, 20)),
    "coffee": ((100, 20, 30, 25, 58, 6.8, 160), (12, 8, 6, 2, 6, 0.4, 20)),
    "banana": ((100, 80, 50, 27, 78, 6.0, 110), (15, 10, 8, 2, 5, 0.4, 20)),
    "mango": ((20, 25, 30, 32, 50, 5.8, 95), (10, 8, 6, 3, 8, 0.5, 20)),
    "watermelon": ((100, 15, 50, 25, 85, 6.5, 45), (12, 6, 8, 2, 5, 0.4, 10)),
}


def generate_synthetic(n_rounds: int = 900, seed: int = 42) -> pd.DataFrame:
    """Sample soil/climate rounds from per-crop Gaussian clusters."""
    rng = np.random.default_rng(seed)
    names = list(CROPS.keys())
    rows = []
    for _ in range(n_rounds):
        crop = rng.choice(names)
        mean, std = CROPS[crop]
        values = rng.normal(mean, std)
        values = np.clip(values, a_min=[0, 0, 0, 5, 10, 3.5, 0], a_max=None)
        rows.append(
            {
                "N": round(values[0], 1),
                "P": round(values[1], 1),
                "K": round(values[2], 1),
                "temperature": round(values[3], 2),
                "humidity": round(values[4], 2),
                "ph": round(values[5], 2),
                "rainfall": round(values[6], 2),
                "label": crop,
            }
        )
    return pd.DataFrame(rows)


def download_kaggle() -> None:
    """Download the real dataset via the Kaggle CLI (requires credentials)."""
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_REF, "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    raw = DATA_DIR / "Crop_recommendation.csv"
    if not raw.exists():
        raise FileNotFoundError("Expected Crop_recommendation.csv after download.")
    pd.read_csv(raw).to_csv(DATA_DIR / "observations.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true")
    group.add_argument("--kaggle", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if args.synthetic:
        observations = generate_synthetic()
        observations.to_csv(DATA_DIR / "observations.csv", index=False)
        print(f"wrote {len(observations)} rounds across {len(CROPS)} crops to {DATA_DIR}")
    else:
        download_kaggle()
        print(f"downloaded and normalized the real dataset into {DATA_DIR}")


if __name__ == "__main__":
    sys.exit(main())
