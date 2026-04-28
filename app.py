"""Streamlit demo: pick a playlist, see recommended tracks and *why*.

Run:
    streamlit run app.py

Loads artifacts produced by `python -m src.pipeline`.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import scipy.sparse as sp
import streamlit as st

from src.retrieval import TrackIndex

ARTIFACT_DIR = "artifacts"


@st.cache_resource
def load_artifacts():
    factors = np.load(os.path.join(ARTIFACT_DIR, "factors.npz"))
    train_matrix = sp.load_npz(os.path.join(ARTIFACT_DIR, "train_matrix.npz"))
    with open(os.path.join(ARTIFACT_DIR, "catalog.pkl"), "rb") as f:
        catalog = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    item_factors = factors["item_factors"]
    user_factors = factors["user_factors"]
    index = TrackIndex(item_factors)
    return {
        "item_factors": item_factors,
        "user_factors": user_factors,
        "train_matrix": train_matrix,
        "catalog": catalog,
        "metrics": metrics,
        "index": index,
    }


def explain(seed_track_indices, candidate_idx, item_factors, track_names, top_n=3):
    """Return the seed tracks whose vectors most align with the candidate.

    Inner product between candidate and each seed -> top contributors.
    """
    cand = item_factors[candidate_idx]
    seed_vecs = item_factors[seed_track_indices]
    contribs = seed_vecs @ cand
    order = np.argsort(-contribs)[:top_n]
    return [(track_names[seed_track_indices[i]], float(contribs[i])) for i in order]


def main():
    st.set_page_config(page_title="Playlist Recommender", layout="wide")
    st.title("Playlist Track Recommender")
    st.caption("ALS collaborative filtering + FAISS retrieval. Built as a recsys learning project.")

    art = load_artifacts()
    catalog = art["catalog"]
    train_matrix = art["train_matrix"]
    item_factors = art["item_factors"]

    with st.sidebar:
        st.header("Model")
        st.write(f"**Tracks:** {len(catalog['track_ids']):,}")
        st.write(f"**Playlists:** {train_matrix.shape[0]:,}")
        st.write(f"**Factor dim:** {item_factors.shape[1]}")
        st.subheader("Offline metrics")
        st.json(art["metrics"])

    tab_playlist, tab_track = st.tabs(["Recommend from a playlist", "Find similar tracks"])

    with tab_playlist:
        st.subheader("Pick a playlist")
        n_playlists = train_matrix.shape[0]
        sample_pids = np.random.RandomState(0).choice(n_playlists, size=min(200, n_playlists), replace=False)
        options = {f"[{pid}] {catalog['playlist_names'][pid]}": pid for pid in sample_pids}
        choice = st.selectbox("Playlist", list(options.keys()))
        pid = options[choice]

        seed_indices = train_matrix[pid].indices.tolist()
        st.write(f"**Seed tracks ({len(seed_indices)}):**")
        st.write(", ".join(catalog["track_names"][i] for i in seed_indices[:30])
                 + (" ..." if len(seed_indices) > 30 else ""))

        k = st.slider("How many recommendations?", 5, 30, 10)
        if st.button("Recommend"):
            ids, scores = art["index"].score_playlist(seed_indices, k=k)
            st.subheader("Recommended tracks")
            for rank, (tid, score) in enumerate(zip(ids, scores), 1):
                with st.expander(f"{rank}. {catalog['track_names'][tid]} (score {score:.3f})"):
                    contribs = explain(seed_indices, int(tid), item_factors, catalog["track_names"])
                    st.write("**Most similar seed tracks** (why this was recommended):")
                    for name, c in contribs:
                        st.write(f"- {name}  ·  contribution {c:.3f}")

    with tab_track:
        st.subheader("Find tracks similar to a single track")
        track_options = {catalog["track_names"][i]: i for i in range(min(2000, len(catalog["track_names"])))}
        chosen = st.selectbox("Track", list(track_options.keys()))
        tid = track_options[chosen]
        k = st.slider("Neighbors", 5, 30, 10, key="nbrs")
        if st.button("Find similar", key="similar_btn"):
            ids, scores = art["index"].neighbors(tid, k=k)
            for rank, (nid, score) in enumerate(zip(ids, scores), 1):
                st.write(f"{rank}. {catalog['track_names'][nid]} (similarity {score:.3f})")


if __name__ == "__main__":
    main()
