"""Train/test split for next-track prediction.

For each playlist with at least `min_len` tracks, hold out a random subset of
its tracks as the target set, keep the rest as the seed (history). This mirrors
the MPD challenge format (predict missing tracks given partial playlist).

Note: a strict time-based split would be more realistic, but MPD doesn't
expose per-track add timestamps consistently. The held-out-track formulation
is what the official RecSys 2018 challenge uses.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def holdout_split(
    matrix: sp.csr_matrix,
    holdout_frac: float = 0.2,
    min_len: int = 10,
    seed: int = 0,
) -> tuple[sp.csr_matrix, dict[int, np.ndarray]]:
    """Split each long-enough playlist into seed tracks (train) and held-out tracks (test).

    Returns:
        train: csr_matrix with held-out tracks zeroed out
        test:  {playlist_idx: np.array of held-out track indices}
    """
    rng = np.random.default_rng(seed)
    matrix = matrix.tocsr()
    train = matrix.copy().tolil()
    test: dict[int, np.ndarray] = {}

    for pid in range(matrix.shape[0]):
        track_indices = matrix.indices[matrix.indptr[pid]:matrix.indptr[pid + 1]]
        if len(track_indices) < min_len:
            continue
        n_hold = max(1, int(len(track_indices) * holdout_frac))
        held = rng.choice(track_indices, size=n_hold, replace=False)
        for t in held:
            train[pid, t] = 0
        test[pid] = held

    return train.tocsr(), test
