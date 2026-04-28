"""Non-personalized baselines.

The popularity baseline is the most important sanity check in recsys: if
ALS doesn't beat "always recommend the most popular tracks the user hasn't
seen", the model is just memorizing popularity rather than learning
playlist-level structure.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class PopularityRecommender:
    """Recommend the globally most-frequent tracks not already in the playlist."""

    def __init__(self):
        self.track_popularity: np.ndarray | None = None
        self.ranked_track_ids: np.ndarray | None = None

    def fit(self, train_matrix: sp.csr_matrix) -> "PopularityRecommender":
        counts = np.asarray(train_matrix.sum(axis=0)).ravel()
        self.track_popularity = counts
        self.ranked_track_ids = np.argsort(-counts)
        return self

    def recommend(self, train_matrix: sp.csr_matrix, playlist_idx: int, k: int = 10):
        assert self.ranked_track_ids is not None
        in_playlist = set(train_matrix[playlist_idx].indices.tolist())
        out: list[int] = []
        for tid in self.ranked_track_ids:
            if int(tid) in in_playlist:
                continue
            out.append(int(tid))
            if len(out) == k:
                break
        return np.array(out, dtype=np.int64)
