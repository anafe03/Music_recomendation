"""End-to-end: load -> split -> train ALS -> evaluate -> persist artifacts."""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import scipy.sparse as sp

from .als import train_als
from .data import load_or_synthetic
from .eval import evaluate
from .split import holdout_split

ARTIFACT_DIR = "artifacts"


def run(
    mpd_dir: str = "data/mpd",
    factors: int = 64,
    iterations: int = 15,
    holdout_frac: float = 0.2,
    eval_sample: int = 500,
    seed: int = 0,
) -> dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("[1/4] Loading dataset...")
    ds = load_or_synthetic(mpd_dir=mpd_dir, seed=seed)
    print(f"      playlists={ds.n_playlists}  tracks={ds.n_tracks}  nnz={ds.matrix.nnz}")

    print("[2/4] Holdout split...")
    train_matrix, test = holdout_split(ds.matrix, holdout_frac=holdout_frac, seed=seed)
    print(f"      test playlists={len(test)}")

    print("[3/4] Training ALS...")
    t0 = time.time()
    model = train_als(train_matrix, factors=factors, iterations=iterations)
    print(f"      trained in {time.time() - t0:.1f}s")

    print("[4/4] Evaluating...")
    metrics = evaluate(model, train_matrix, test, k=10, sample=eval_sample, seed=seed)
    metrics_500 = evaluate(model, train_matrix, test, k=500, sample=eval_sample, seed=seed)
    metrics.update(metrics_500)
    print(f"      {metrics}")

    print("Persisting artifacts...")
    np.savez(
        os.path.join(ARTIFACT_DIR, "factors.npz"),
        item_factors=model.item_factors,
        user_factors=model.user_factors,
    )
    sp.save_npz(os.path.join(ARTIFACT_DIR, "train_matrix.npz"), train_matrix)
    with open(os.path.join(ARTIFACT_DIR, "catalog.pkl"), "wb") as f:
        pickle.dump(
            {
                "track_ids": ds.track_ids,
                "track_names": ds.track_names,
                "track_to_idx": ds.track_to_idx,
                "playlist_names": ds.playlist_names,
            },
            f,
        )
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    run()
