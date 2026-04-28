"""Streamlit demo: visual walkthrough of how the recommender works.

Run:  streamlit run app.py

Sections:
  1. Hero — dataset stats and headline metrics
  2. How it works — pipeline diagram + prose
  3. Inside the model — embedding-space visualization (PCA of item factors)
  4. ALS vs Popularity — side-by-side metric comparison
  5. Try it — interactive recommender with per-track explanations
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp
import streamlit as st
from sklearn.decomposition import PCA

from src.baselines import PopularityRecommender
from src.retrieval import TrackIndex

ARTIFACT_DIR = "artifacts"


# ---------- artifact loading ----------

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
    track_popularity = factors["track_popularity"]

    pop = PopularityRecommender()
    pop.track_popularity = track_popularity
    pop.ranked_track_ids = np.argsort(-track_popularity)

    return {
        "item_factors": item_factors,
        "user_factors": user_factors,
        "track_popularity": track_popularity,
        "train_matrix": train_matrix,
        "catalog": catalog,
        "metrics": metrics,
        "index": TrackIndex(item_factors),
        "popularity": pop,
    }


@st.cache_data
def pca_projection(item_factors: np.ndarray, max_points: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = item_factors.shape[0]
    sample = rng.choice(n, size=min(max_points, n), replace=False)
    coords = PCA(n_components=2, random_state=seed).fit_transform(item_factors[sample])
    return sample, coords


# ---------- helpers ----------

def explain(seed_indices, candidate_idx, item_factors, track_names, top_n=3):
    """Inner-product attribution: which seed tracks pulled this candidate up?"""
    cand = item_factors[candidate_idx]
    contribs = item_factors[seed_indices] @ cand
    order = np.argsort(-contribs)[:top_n]
    return [(track_names[seed_indices[i]], float(contribs[i])) for i in order]


def topic_from_name(name: str) -> str:
    if name.startswith("Topic"):
        return name.split(" - ", 1)[0]
    return "real"


# ---------- page ----------

st.set_page_config(page_title="Playlist Recommender", page_icon="🎵", layout="wide")

# Custom CSS for a cleaner, more polished look
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; }
      .metric-card {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 1.2rem 1.4rem; border-radius: 14px; color: white;
        box-shadow: 0 4px 14px rgba(29,185,84,0.25);
      }
      .metric-card.alt {
        background: linear-gradient(135deg, #2a2a2a 0%, #404040 100%);
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
      }
      .metric-label { font-size: 0.78rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.06em; }
      .metric-value { font-size: 1.9rem; font-weight: 700; line-height: 1.1; margin-top: 0.2rem; }
      .metric-sub   { font-size: 0.8rem; opacity: 0.85; margin-top: 0.15rem; }
      .step-card {
        background: #f6f8fa; border-left: 4px solid #1DB954;
        padding: 0.9rem 1.1rem; border-radius: 8px; margin-bottom: 0.6rem;
      }
      .step-num { color: #1DB954; font-weight: 700; margin-right: 0.5rem; }
      .badge { display:inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
               font-size: 0.72rem; background:#e8f5ee; color:#0d7a3a; margin-right: 0.3rem; }
      .seed-pill { display:inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
                   font-size: 0.78rem; background:#eef1f5; color:#333; margin: 0.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, sub: str = "", alt: bool = False):
    cls = "metric-card alt" if alt else "metric-card"
    st.markdown(
        f"""<div class="{cls}">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


art = load_artifacts()
catalog = art["catalog"]
m = art["metrics"]
src_label = m["dataset"]["source"].upper()

# ---------- HERO ----------

st.title("🎵 Playlist Track Recommender")
st.markdown(
    "**ALS collaborative filtering + FAISS retrieval** for next-track prediction. "
    f"Trained on **{src_label}** data."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Playlists", f"{m['dataset']['n_playlists']:,}", "training set")
with c2:
    metric_card("Tracks", f"{m['dataset']['n_tracks']:,}", "vocabulary")
with c3:
    metric_card("Recall@10", f"{m['als']['recall@10']:.3f}",
                f"vs {m['popularity']['recall@10']:.3f} popularity",
                alt=True)
with c4:
    metric_card("NDCG@10", f"{m['als']['ndcg@10']:.3f}",
                f"vs {m['popularity']['ndcg@10']:.3f} popularity",
                alt=True)

st.divider()

# ---------- HOW IT WORKS ----------

st.header("How it works")
st.markdown(
    "The model learns a low-dimensional **embedding** for every playlist and every track from "
    "co-occurrence patterns alone — no audio, no lyrics, no metadata. Tracks that show up in similar "
    "playlists end up close together in the embedding space. To recommend, we pool the embeddings of "
    "your seed tracks and look up the closest unseen tracks via FAISS."
)

steps = [
    ("Load", "Parse playlists into a sparse `(playlist × track)` matrix of 1.0s. "
             "Most entries are zero — a 2k×3k matrix has 99% empty cells."),
    ("Split", "For each playlist with ≥10 tracks, hide 20% as targets. The other 80% are seeds the "
              "model sees during training. This mirrors the RecSys 2018 challenge."),
    ("Train ALS", "Alternating Least Squares factorizes the matrix into a playlist-factor matrix "
                  f"and a track-factor matrix, both of dim {m['model']['factors']}. "
                  "The dot product approximates whether a track belongs in a playlist."),
    ("Index", "Load the learned track factors into a FAISS inner-product index for millisecond "
              "nearest-neighbor lookup."),
    ("Recommend", "Average the seed track embeddings → use as a query vector → FAISS returns the "
                  "top-K closest unseen tracks."),
]
for i, (title, body) in enumerate(steps, 1):
    st.markdown(
        f'<div class="step-card"><span class="step-num">{i}.</span>'
        f"<b>{title}</b> — {body}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------- EMBEDDING SPACE ----------

st.header("Inside the model: the embedding space")
st.markdown(
    "Every track lives at a point in a "
    f"{art['item_factors'].shape[1]}-dimensional space. PCA projects that down to 2D so we can see "
    "it. Tracks the model considers similar are close together. **The clusters below were learned "
    "from co-occurrence alone — the model was never told topics existed.**"
)

sample, coords = pca_projection(art["item_factors"])
df = pd.DataFrame(
    {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "track": [catalog["track_names"][i] for i in sample],
        "popularity": art["track_popularity"][sample],
        "topic": [topic_from_name(catalog["track_names"][i]) for i in sample],
    }
)

color_by = st.radio(
    "Color points by",
    ["Topic (synthetic ground-truth)" if df["topic"].nunique() > 1 else "Popularity",
     "Popularity"],
    horizontal=True,
)
if color_by.startswith("Topic"):
    fig = px.scatter(df, x="x", y="y", color="topic", hover_data=["track", "popularity"],
                     opacity=0.75, height=520,
                     title="Item factors after PCA — colored by topic")
else:
    fig = px.scatter(df, x="x", y="y", color="popularity", hover_data=["track", "topic"],
                     color_continuous_scale="Viridis", opacity=0.8, height=520,
                     title="Item factors after PCA — colored by track popularity")
fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "On synthetic data, you should see clean topic clusters — that's the model recovering the "
    "latent genre structure of the generator. On real Spotify data, clusters correspond to "
    "loose genre/mood groupings emerging from listener behavior."
)

st.divider()

# ---------- BASELINE COMPARISON ----------

st.header("ALS vs Popularity baseline")
st.markdown(
    "**The most important sanity check in recsys.** If a model can't beat 'recommend the most-played "
    "tracks the user hasn't seen', it's not learning anything personal. The bars below should show "
    "ALS clearly ahead — if they didn't, the model would be broken."
)

cmp_df = pd.DataFrame(
    {
        "Model": ["ALS", "Popularity", "ALS", "Popularity"],
        "Metric": ["Recall@10", "Recall@10", "NDCG@10", "NDCG@10"],
        "Score": [m["als"]["recall@10"], m["popularity"]["recall@10"],
                  m["als"]["ndcg@10"], m["popularity"]["ndcg@10"]],
    }
)
fig2 = px.bar(cmp_df, x="Metric", y="Score", color="Model", barmode="group",
              color_discrete_map={"ALS": "#1DB954", "Popularity": "#888888"},
              height=380, title="ALS vs Popularity baseline")
fig2.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

lift_recall = m["als"]["recall@10"] / max(m["popularity"]["recall@10"], 1e-9)
lift_ndcg = m["als"]["ndcg@10"] / max(m["popularity"]["ndcg@10"], 1e-9)
st.markdown(
    f"**ALS lift over popularity:** Recall@10 ×{lift_recall:.1f}, NDCG@10 ×{lift_ndcg:.1f}. "
    f"Trained in {m['als']['train_seconds']:.2f}s."
)

st.divider()

# ---------- INTERACTIVE ----------

st.header("Try it")
tab_playlist, tab_track = st.tabs(["📋 Recommend from a playlist", "🎯 Find similar tracks"])

with tab_playlist:
    n_playlists = art["train_matrix"].shape[0]
    sample_pids = np.random.RandomState(0).choice(n_playlists, size=min(200, n_playlists), replace=False)
    options = {f"[{pid}] {catalog['playlist_names'][pid]}": pid for pid in sample_pids}
    choice = st.selectbox("Pick a playlist", list(options.keys()))
    pid = options[choice]
    seed_indices = art["train_matrix"][pid].indices.tolist()

    st.markdown(f"**Seed tracks ({len(seed_indices)}):**")
    pills = " ".join(f"<span class='seed-pill'>{catalog['track_names'][i]}</span>"
                     for i in seed_indices[:25])
    if len(seed_indices) > 25:
        pills += f" <span class='seed-pill'>+{len(seed_indices)-25} more</span>"
    st.markdown(pills, unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        k = st.slider("How many recommendations?", 5, 30, 10)
    with col_b:
        show_pop = st.checkbox("Also show popularity baseline", value=True)

    if st.button("🚀 Recommend", type="primary"):
        ids, scores = art["index"].score_playlist(seed_indices, k=k)

        st.subheader("ALS recommendations")
        for rank, (tid, score) in enumerate(zip(ids, scores), 1):
            with st.expander(f"**{rank}. {catalog['track_names'][tid]}**  ·  score {score:.3f}"):
                contribs = explain(seed_indices, int(tid), art["item_factors"],
                                   catalog["track_names"], top_n=5)
                cdf = pd.DataFrame(contribs, columns=["Seed track", "Contribution"])
                bar = px.bar(cdf, x="Contribution", y="Seed track", orientation="h",
                             color="Contribution", color_continuous_scale="Greens",
                             height=240)
                bar.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
                st.plotly_chart(bar, use_container_width=True)
                st.caption("Bars show inner product between each seed track's vector and the "
                           "candidate's vector — i.e. which seed tracks 'pulled' this candidate up.")

        if show_pop:
            st.subheader("Popularity baseline (for comparison)")
            pop_ids = art["popularity"].recommend(art["train_matrix"], pid, k=k)
            for rank, tid in enumerate(pop_ids, 1):
                pop_count = int(art["track_popularity"][tid])
                st.write(f"{rank}. {catalog['track_names'][tid]}  "
                         f"·  <span class='badge'>in {pop_count} playlists</span>",
                         unsafe_allow_html=True)

with tab_track:
    track_options = {catalog["track_names"][i]: i
                     for i in range(min(2000, len(catalog["track_names"])))}
    chosen = st.selectbox("Pick a track", list(track_options.keys()))
    tid = track_options[chosen]
    k2 = st.slider("How many neighbors?", 5, 30, 10, key="nbrs")
    if st.button("🔎 Find similar", type="primary", key="similar_btn"):
        ids, scores = art["index"].neighbors(tid, k=k2)
        rows = pd.DataFrame(
            {
                "Rank": list(range(1, len(ids) + 1)),
                "Track": [catalog["track_names"][i] for i in ids],
                "Cosine-ish similarity": [round(float(s), 3) for s in scores],
            }
        )
        st.dataframe(rows, hide_index=True, use_container_width=True)

# ---------- FOOTER ----------

st.divider()
with st.expander("ℹ️ About this project"):
    st.markdown(
        f"""
**Data source:** {src_label}
{"This is the synthetic generator with 20 latent topics — clean enough to validate the pipeline, not real music. Drop Spotify MPD slices into `data/mpd/` and rerun `python -m src.pipeline` to swap in real data." if src_label == "SYNTHETIC" else "Spotify Million Playlist Dataset."}

**Model:** ALS with {m["model"]["factors"]} factors, {m["model"]["iterations"]} iterations, trained in {m['als']['train_seconds']:.2f}s.

**What's missing (honest list):**
- No two-tower neural model yet (planned next).
- No content features (audio, artist, album).
- Held-out split is random per playlist, not stratified by length — short playlists over-represent test failures.
- Cold-start tracks (zero co-occurrences) get a zero vector and never surface.
"""
    )
