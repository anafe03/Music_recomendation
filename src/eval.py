"""Offline evaluation: Recall@K and NDCG@K.

Both treat held-out tracks as ground truth. NDCG rewards correct items
appearing higher in the ranked list; Recall just measures hit rate within K.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

import implicit

from .baselines import PopularityRecommender


def _dcg(relevances: np.ndarray) -> float:
    if len(relevances) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances * discounts))


def _evaluate_with(
    recommend_fn: Callable[[int, int], np.ndarray],
    test: dict[int, np.ndarray],
    k: int,
    sample: int | None,
    seed: int,
    desc: str,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pids = list(test.keys())
    if sample and sample < len(pids):
        pids = list(rng.choice(pids, size=sample, replace=False))

    recalls: list[float] = []
    ndcgs: list[float] = []
    for pid in tqdm(pids, desc=desc):
        held = set(int(x) for x in test[pid])
        if not held:
            continue
        ids = recommend_fn(pid, k)
        hits = np.array([1.0 if int(t) in held else 0.0 for t in ids])
        recalls.append(hits.sum() / len(held))
        ideal = np.ones(min(len(held), k))
        ndcgs.append(_dcg(hits) / _dcg(ideal) if _dcg(ideal) > 0 else 0.0)

    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_evaluated": len(recalls),
    }


def evaluate_als(
    model: implicit.als.AlternatingLeastSquares,
    train_matrix: sp.csr_matrix,
    test: dict[int, np.ndarray],
    k: int = 10,
    sample: int | None = 500,
    seed: int = 0,
) -> dict[str, float]:
    def fn(pid: int, kk: int) -> np.ndarray:
        ids, _ = model.recommend(pid, train_matrix[pid], N=kk, filter_already_liked_items=True)
        return ids
    return _evaluate_with(fn, test, k, sample, seed, desc=f"als@{k}")


def evaluate_popularity(
    model: PopularityRecommender,
    train_matrix: sp.csr_matrix,
    test: dict[int, np.ndarray],
    k: int = 10,
    sample: int | None = 500,
    seed: int = 0,
) -> dict[str, float]:
    def fn(pid: int, kk: int) -> np.ndarray:
        return model.recommend(train_matrix, pid, k=kk)
    return _evaluate_with(fn, test, k, sample, seed, desc=f"pop@{k}")
