"""End-to-end: load -> split -> train ALS + popularity -> evaluate both -> persist."""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import scipy.sparse as sp

from .als import train_als
from .audio import build_audio_embeddings
from .baselines import PopularityRecommender
from .data import load_or_synthetic
from .eval import evaluate_als, evaluate_popularity
from .split import holdout_split

ARTIFACT_DIR = "artifacts"


def run(
    mpd_dir: str = "data/mpd",
    factors: int = 64,
    iterations: int = 15,
    holdout_frac: float = 0.2,
    eval_sample: int = 500,
    seed: int = 0,
    skip_audio: bool = False,
) -> dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("[1/5] Loading dataset...")
    ds = load_or_synthetic(mpd_dir=mpd_dir, seed=seed)
    print(f"      playlists={ds.n_playlists}  tracks={ds.n_tracks}  nnz={ds.matrix.nnz}")

    print("[2/5] Holdout split...")
    train_matrix, test = holdout_split(ds.matrix, holdout_frac=holdout_frac, seed=seed)
    print(f"      test playlists={len(test)}")

    print("[3/5] Training ALS...")
    t0 = time.time()
    als = train_als(train_matrix, factors=factors, iterations=iterations)
    als_train_seconds = time.time() - t0
    print(f"      trained in {als_train_seconds:.2f}s")

    print("[4/5] Training popularity baseline...")
    t0 = time.time()
    pop = PopularityRecommender().fit(train_matrix)
    pop_train_seconds = time.time() - t0
    print(f"      trained in {pop_train_seconds:.4f}s")

    print("[5/5] Evaluating both models...")
    big_k = min(500, max(50, ds.n_tracks // 2))
    metrics = {
        "als": {
            **evaluate_als(als, train_matrix, test, k=10, sample=eval_sample, seed=seed),
            **evaluate_als(als, train_matrix, test, k=big_k, sample=eval_sample, seed=seed),
            "train_seconds": als_train_seconds,
        },
        "popularity": {
            **evaluate_popularity(pop, train_matrix, test, k=10, sample=eval_sample, seed=seed),
            **evaluate_popularity(pop, train_matrix, test, k=big_k, sample=eval_sample, seed=seed),
            "train_seconds": pop_train_seconds,
        },
        "dataset": {
            "n_playlists": int(ds.n_playlists),
            "n_tracks": int(ds.n_tracks),
            "nnz": int(ds.matrix.nnz),
            "source": (
                "mpd" if os.path.isdir(mpd_dir) and any(f.endswith(".json") for f in os.listdir(mpd_dir) if os.path.isfile(os.path.join(mpd_dir, f)))
                else ("real-seeded" if any(m.get("preview_url") for m in (ds.track_meta or [])) else "synthetic")
            ),
        },
        "model": {"factors": factors, "iterations": iterations},
    }
    print(json.dumps(metrics, indent=2))

    if not skip_audio:
        print("[audio] Building content-based audio embeddings...")
        audio_emb, audio_valid = build_audio_embeddings(ds.track_meta)
    else:
        audio_emb = np.zeros((ds.n_tracks, 1), dtype=np.float32)
        audio_valid = np.zeros(ds.n_tracks, dtype=bool)

    print("Persisting artifacts...")
    np.savez(
        os.path.join(ARTIFACT_DIR, "factors.npz"),
        item_factors=als.item_factors,
        user_factors=als.user_factors,
        track_popularity=pop.track_popularity,
        audio_embeddings=audio_emb,
        audio_valid=audio_valid,
    )
    sp.save_npz(os.path.join(ARTIFACT_DIR, "train_matrix.npz"), train_matrix)
    with open(os.path.join(ARTIFACT_DIR, "catalog.pkl"), "wb") as f:
        pickle.dump(
            {
                "track_ids": ds.track_ids,
                "track_names": ds.track_names,
                "track_to_idx": ds.track_to_idx,
                "playlist_names": ds.playlist_names,
                "track_meta": ds.track_meta,
            },
            f,
        )
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    run()
