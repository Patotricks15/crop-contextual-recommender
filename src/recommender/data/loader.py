"""Load the Crop Recommendation dataset as a contextual-bandit log.

Each row is a soil/climate observation (``N``, ``P``, ``K``, temperature,
humidity, pH, rainfall) paired with the crop an agronomist would recommend
for it. Framed as a bandit: the context is the soil/climate vector, the arm
is the crop, and the simulated reward for choosing arm ``a`` on round ``t``
is ``1`` iff ``a`` equals the logged crop for that round. This is the
standard way to turn a labelled classification dataset into an offline
contextual-bandit replay log (Li et al., 2011).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CONTEXT_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
REQUIRED_COLUMNS = {*CONTEXT_COLUMNS, "label"}


@dataclass(frozen=True)
class BanditLog:
    """A contextual-bandit replay log with contiguous arm indices.

    Attributes:
        frame: Raw rows plus an ``arm_index`` column.
        context: ``(n_rounds, n_features)`` context matrix, column order
            fixed to :data:`CONTEXT_COLUMNS`.
        arm_index: ``(n_rounds,)`` logged (ground-truth) arm per round.
        arm_names: Crop names, positionally aligned to arm index.
    """

    frame: pd.DataFrame
    context: np.ndarray
    arm_index: np.ndarray
    arm_names: np.ndarray

    @property
    def n_rounds(self) -> int:
        return int(self.context.shape[0])

    @property
    def n_arms(self) -> int:
        return int(self.arm_names.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.context.shape[1])


def load_observations(path: str | pd.io.common.PathLike) -> pd.DataFrame:
    """Read the raw crop CSV and validate its contract.

    Args:
        path: Path to a CSV with :data:`CONTEXT_COLUMNS` plus ``label``.

    Returns:
        The raw frame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py --synthetic"
        )
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows.")
    return frame


def build_bandit_log(observations: pd.DataFrame) -> BanditLog:
    """Reindex crop labels to contiguous arm indices and extract the context.

    Args:
        observations: Raw frame from :func:`load_observations`.

    Returns:
        A :class:`BanditLog`.
    """
    arm_names = np.sort(observations["label"].unique())
    arm_map = {name: idx for idx, name in enumerate(arm_names)}

    frame = observations.copy()
    frame["arm_index"] = frame["label"].map(arm_map).astype(np.int64)

    context = frame[CONTEXT_COLUMNS].to_numpy(dtype=np.float64)
    arm_index = frame["arm_index"].to_numpy(dtype=np.int64)
    return BanditLog(frame=frame, context=context, arm_index=arm_index, arm_names=arm_names)
