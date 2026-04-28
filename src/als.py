"""ALS baseline using the `implicit` library.

ALS (Alternating Least Squares) for implicit feedback factorizes the
user-item matrix into low-rank user and item factors. For playlists,
"user" = playlist and "item" = track. Trained factors give us a dense
embedding space we can query with FAISS.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import implicit


def train_als(
    train_matrix: sp.csr_matrix,
    factors: int = 64,
    regularization: float = 0.05,
    iterations: int = 15,
    alpha: float = 40.0,
) -> implicit.als.AlternatingLeastSquares:
    """Train an ALS model.

    `alpha` scales the implicit confidence: in the Hu/Koren formulation,
    confidence c_ui = 1 + alpha * r_ui. Higher alpha means stronger
    confidence in observed (playlist, track) pairs.
    """
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        use_gpu=False,
    )
    model.fit((train_matrix * alpha).astype(np.float32))
    return model


def recommend_for_playlist(
    model: implicit.als.AlternatingLeastSquares,
    train_matrix: sp.csr_matrix,
    playlist_idx: int,
    n: int = 10,
    filter_already_in_playlist: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-n (track_indices, scores) for a playlist."""
    user_items = train_matrix[playlist_idx]
    ids, scores = model.recommend(
        playlist_idx,
        user_items,
        N=n,
        filter_already_liked_items=filter_already_in_playlist,
    )
    return ids, scores
