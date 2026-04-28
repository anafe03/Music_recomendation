"""FAISS-based item retrieval over learned item factors.

Builds an inner-product index over track embeddings so we can do fast
nearest-neighbor lookup ("tracks similar to this track") and ad-hoc
playlist scoring (sum of seed track vectors -> query the index).
"""
from __future__ import annotations

import numpy as np

import faiss


class TrackIndex:
    def __init__(self, item_factors: np.ndarray):
        self.item_factors = np.ascontiguousarray(item_factors.astype(np.float32))
        self.dim = self.item_factors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.item_factors)

    def neighbors(self, track_idx: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = self.item_factors[track_idx:track_idx + 1]
        scores, ids = self.index.search(q, k + 1)
        return ids[0][1:], scores[0][1:]

    def score_playlist(self, seed_track_indices: list[int], k: int = 10, exclude_seeds: bool = True):
        """Pool seed track vectors (mean) and retrieve top-k tracks."""
        if not seed_track_indices:
            raise ValueError("Need at least one seed track")
        q = self.item_factors[seed_track_indices].mean(axis=0, keepdims=True)
        scores, ids = self.index.search(q.astype(np.float32), k + len(seed_track_indices))
        ids, scores = ids[0], scores[0]
        if exclude_seeds:
            mask = np.array([i not in set(seed_track_indices) for i in ids])
            ids, scores = ids[mask], scores[mask]
        return ids[:k], scores[:k]
