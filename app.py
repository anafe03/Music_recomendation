"""Streamlit demo: visual walkthrough of the playlist recommender.

Run:  streamlit run app.py

Sections:
  1. Hero — dataset stats and headline metrics
  2. Plain-English: what is ALS, what is FAISS
  3. How it works — pipeline diagram
  4. Embedding-space visualization (PCA, colored by genre)
  5. ALS vs Popularity comparison
  6. Try it — pick a playlist, see recommendations as cards with art + audio
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
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

    audio_emb = factors["audio_embeddings"] if "audio_embeddings" in factors.files else None
    audio_valid = factors["audio_valid"] if "audio_valid" in factors.files else None

    pop = PopularityRecommender()
    pop.track_popularity = track_popularity
    pop.ranked_track_ids = np.argsort(-track_popularity)

    audio_index = TrackIndex(audio_emb) if audio_emb is not None and audio_emb.shape[1] > 1 else None

    return {
        "item_factors": item_factors,
        "user_factors": user_factors,
        "track_popularity": track_popularity,
        "audio_embeddings": audio_emb,
        "audio_valid": audio_valid,
        "audio_index": audio_index,
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

GENRE_COLORS = {
    "Pop": "#ff6b9d", "Hip-Hop": "#fbb13c", "R&B": "#9d4edd",
    "Rock": "#e63946", "Indie": "#06d6a0", "Country": "#a0522d",
    "Electronic": "#00b4d8", "Jazz": "#ffd166", "Classical": "#7b8cde",
    "Latin": "#ef476f", "Metal": "#444444", "Folk": "#588157",
}


def genre_color(g: str | None) -> str:
    return GENRE_COLORS.get(g or "", "#888")


def explain(seed_indices, candidate_idx, item_factors, track_names, top_n=3):
    cand = item_factors[candidate_idx]
    contribs = item_factors[seed_indices] @ cand
    order = np.argsort(-contribs)[:top_n]
    return [(track_names[seed_indices[i]], float(contribs[i])) for i in order]


# ---------- page setup ----------

st.set_page_config(page_title="Playlist Recommender", page_icon="🎵", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
      .metric-card {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 1.2rem 1.4rem; border-radius: 14px; color: white;
        box-shadow: 0 4px 14px rgba(29,185,84,0.25);
      }
      .metric-card.alt {
        background: linear-gradient(135deg, #2a2a2a 0%, #404040 100%);
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
      }
      .metric-label { font-size: 0.78rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.06em; color: white;}
      .metric-value { font-size: 1.9rem; font-weight: 700; line-height: 1.1; margin-top: 0.2rem; color: white;}
      .metric-sub   { font-size: 0.8rem; opacity: 0.85; margin-top: 0.15rem; color: white;}
      .step-card {
        background: #f6f8fa; border-left: 4px solid #1DB954;
        padding: 0.9rem 1.1rem; border-radius: 8px; margin-bottom: 0.6rem;
      }
      .step-num { color: #1DB954; font-weight: 700; margin-right: 0.5rem; }
      .genre-badge {
        display:inline-block; padding: 0.18rem 0.6rem; border-radius: 999px;
        font-size: 0.72rem; color: white; font-weight: 600; letter-spacing: 0.02em;
      }
      .seed-pill {
        display:inline-flex; align-items:center; gap:6px;
        padding: 0.25rem 0.7rem; border-radius: 999px;
        font-size: 0.78rem; background:#eef1f5; color:#222; margin: 0.15rem;
      }
      .seed-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
      .track-row {
        display:flex; align-items:center; gap:14px;
        padding: 10px 14px; background:#fafbfc; border:1px solid #eaeef2;
        border-radius: 12px; margin-bottom: 8px;
      }
      .track-art { width:60px; height:60px; border-radius:8px; object-fit:cover;
                   box-shadow:0 2px 6px rgba(0,0,0,0.12); }
      .track-title { font-weight: 600; font-size: 0.98rem; }
      .track-artist { color: #555; font-size: 0.85rem; }
      .rank-num { font-size: 1.2rem; font-weight: 700; color:#1DB954; min-width: 26px; }
      .glossary {
        background:#fff8e1; border-left:4px solid #ffb300;
        padding: 1rem 1.2rem; border-radius: 8px;
      }
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


def genre_badge(g: str) -> str:
    return f'<span class="genre-badge" style="background:{genre_color(g)}">{g}</span>'


def render_track_card(rank: int, meta: dict, score: float | None = None, sub_label: str = ""):
    art = meta.get("artwork_url") or "https://via.placeholder.com/60?text=♪"
    artist = meta.get("artist") or "Unknown"
    title = meta.get("title") or ""
    g = meta.get("genre")
    score_html = f"<span style='color:#888;font-size:0.85rem;'>score {score:.3f}</span>" if score is not None else ""
    sub_html = f"<span style='color:#888;font-size:0.85rem;'>{sub_label}</span>" if sub_label else ""
    st.markdown(
        f"""
        <div class="track-row">
          <div class="rank-num">{rank}</div>
          <img class="track-art" src="{art}" />
          <div style="flex:1; min-width:0;">
            <div class="track-title">{title}</div>
            <div class="track-artist">{artist}</div>
            <div style="margin-top:4px;">{genre_badge(g) if g else ""} {score_html} {sub_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if meta.get("preview_url"):
        st.audio(meta["preview_url"])


# ---------- data ----------

art = load_artifacts()
catalog = art["catalog"]
m = art["metrics"]
src_label = m["dataset"]["source"].upper()
meta_list = catalog.get("track_meta", [])

# ---------- HERO ----------

st.title("🎵 Playlist Track Recommender")
st.markdown(
    "Pick a playlist of real songs — the model recommends tracks you'd likely add next, "
    "with **30-second previews** and album artwork. Built as a hands-on recsys learning project."
)

c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("Tracks", f"{m['dataset']['n_tracks']:,}", "in catalog")
with c2: metric_card("Playlists", f"{m['dataset']['n_playlists']:,}", "training set")
with c3: metric_card("ALS Recall@10", f"{m['als']['recall@10']:.3f}",
                     f"vs {m['popularity']['recall@10']:.3f} popularity", alt=True)
with c4: metric_card("ALS NDCG@10", f"{m['als']['ndcg@10']:.3f}",
                     f"vs {m['popularity']['ndcg@10']:.3f} popularity", alt=True)

st.divider()

# ---------- WTF IS THIS ----------

st.header("Wait — what is ALS and FAISS?")

col_als, col_faiss = st.columns(2)
with col_als:
    st.markdown(
        """
        <div class="glossary">
        <h4 style="margin:0 0 0.3rem 0;">🧮 ALS — Alternating Least Squares</h4>
        <p style="margin:0;">A way to <b>learn what tracks have in common from the playlists they appear in.</b>
        Imagine a giant spreadsheet: rows are playlists, columns are tracks, cells are 1 if the
        track is in the playlist. Most cells are empty.</p>
        <p>ALS rips that sparse spreadsheet into two smaller dense ones — a "playlist taste" sheet
        and a "track flavor" sheet. Multiplying any playlist row by any track column gives you a
        score for whether they belong together. The model learns those flavors by watching which
        tracks co-occur. <b>No audio, no lyrics, no genre tags needed</b> — just co-occurrence.</p>
        <p style="margin:0;">"Alternating" because it solves one sheet at a time, locking the other
        — an old-school 2008 algorithm that still beats most modern neural recommenders on
        playlist data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_faiss:
    st.markdown(
        """
        <div class="glossary">
        <h4 style="margin:0 0 0.3rem 0;">⚡ FAISS — Facebook AI Similarity Search</h4>
        <p style="margin:0;">After ALS, every track is a point in a 64-dimensional space. To
        recommend, we need to find which tracks are <b>nearest</b> to a query point.</p>
        <p>Doing that by checking every track is fine for 88 tracks, but Spotify has ~100M.
        FAISS is a library that builds an <b>index</b> over those points so nearest-neighbor
        lookup is sub-millisecond even at billion-scale.</p>
        <p style="margin:0;">For this demo it's overkill, but the same code works at production
        scale — that's the point of including it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------- PIPELINE ----------

st.header("The pipeline, step by step")
steps = [
    ("Load", f"Pull {len(meta_list)} real tracks from iTunes (artist, title, album art, 30s preview, genre tag). "
             f"Build a sparse {m['dataset']['n_playlists']:,}×{m['dataset']['n_tracks']} playlist×track matrix."),
    ("Split", "For each playlist, hide 20% of its tracks as test targets. The model only sees the other 80%."),
    ("Train", f"Run ALS for {m['model']['iterations']} iterations to learn {m['model']['factors']}-dim "
              f"embeddings for every playlist and every track. Took {m['als']['train_seconds']:.2f}s."),
    ("Index", "Load the track embeddings into a FAISS inner-product index for fast neighbor lookup."),
    ("Recommend", "Average a playlist's seed track embeddings → query FAISS → return the top-K closest "
                  "tracks the playlist hasn't already heard."),
]
for i, (title, body) in enumerate(steps, 1):
    st.markdown(
        f'<div class="step-card"><span class="step-num">{i}.</span><b>{title}</b> — {body}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ---------- EMBEDDING SPACE ----------

st.header("Inside the model: the embedding space")
st.markdown(
    f"Every track lives at a point in {art['item_factors'].shape[1]}-dimensional space — "
    "PCA flattens it to 2D so we can look at it. **Tracks that show up in similar playlists end up "
    "near each other.** Watch the genres separate."
)

sample, coords = pca_projection(art["item_factors"])
df = pd.DataFrame({
    "x": coords[:, 0], "y": coords[:, 1],
    "track": [catalog["track_names"][i] for i in sample],
    "genre": [meta_list[i].get("genre") if i < len(meta_list) else "?" for i in sample],
    "popularity": art["track_popularity"][sample],
})
fig = px.scatter(
    df, x="x", y="y", color="genre",
    color_discrete_map=GENRE_COLORS, hover_data=["track", "popularity"],
    opacity=0.85, height=520,
    title="Track embeddings (PCA 2D), colored by genre",
)
fig.update_traces(marker=dict(size=11, line=dict(width=0.5, color="white")))
fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="Genre")
st.plotly_chart(fig, use_container_width=True)
st.caption("Each dot is one song. **The model learned these clusters from co-occurrence alone — it was never told what genre anything is.**")

st.divider()

# ---------- ALS vs AUDIO ----------

st.header("Two ways to recommend music")

col_co, col_au = st.columns(2)
with col_co:
    st.markdown(
        """
        <div class="glossary" style="background:#e7f5ff; border-left-color:#1c7ed6;">
        <h4 style="margin:0 0 0.3rem 0;">👥 ALS — collaborative filtering</h4>
        <p style="margin:0;"><b>Learns from human behavior.</b> "If lots of playlists with track A
        also have track B, then A and B are similar."</p>
        <ul>
          <li>✅ Captures <b>cultural</b> similarity — genres, scenes, eras, moods that listeners group together</li>
          <li>✅ Picks up "people who like X also like Y" patterns the audio doesn't reveal</li>
          <li>❌ <b>Cold-start fail:</b> a brand-new song has no playlists, so no recommendation</li>
          <li>❌ Can over-recommend popular tracks that ride along with everything</li>
        </ul>
        <p style="margin:0;"><i>Has never heard the song. Has read every playlist.</i></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_au:
    st.markdown(
        """
        <div class="glossary" style="background:#fff0f6; border-left-color:#d6336c;">
        <h4 style="margin:0 0 0.3rem 0;">🎧 Audio — content-based</h4>
        <p style="margin:0;"><b>Learns from the actual sound.</b> Each 30s preview is turned into a
        53-dim feature vector capturing timbre, harmony, rhythm, and dynamics
        (MFCCs, chroma, spectral contrast, tempo, RMS).</p>
        <ul>
          <li>✅ <b>No cold-start problem</b> — a brand-new song has audio from day one</li>
          <li>✅ Catches <b>sonic</b> similarity (texture, BPM, key) that co-occurrence misses</li>
          <li>❌ Ignores cultural context — a punk cover of a folk song sounds different but belongs together</li>
          <li>❌ Confuses "sounds the same" with "fans of one like the other" — not always true</li>
        </ul>
        <p style="margin:0;"><i>Has heard every song. Has read no playlists.</i></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="margin-top:1rem; padding:1rem 1.2rem; background:#f3f0ff;
                border-left:4px solid #7048e8; border-radius:8px;">
    <b>The fix is to blend them.</b> Real systems (Spotify, YouTube Music, Apple Music)
    use a <b>hybrid</b>: collaborative filtering for the cultural signal + audio embeddings
    for the sonic signal + content metadata + sequential context. Below you can see each
    method on its own and blend them yourself.
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Side-by-side neighbors for a chosen track ---
if art["audio_index"] is not None:
    st.subheader("Same track, two definitions of similar")
    track_options = {f"{meta_list[i].get('artist','?')} — {meta_list[i].get('title','?')}": i
                     for i in range(len(meta_list))}
    pick_label = st.selectbox(
        "Pick any track",
        list(track_options.keys()),
        index=min(5, len(track_options) - 1),
    )
    pick_idx = track_options[pick_label]
    chosen_md = meta_list[pick_idx]

    art_url = chosen_md.get("artwork_url") or "https://via.placeholder.com/120?text=♪"
    chosen_col, _ = st.columns([1, 3])
    with chosen_col:
        st.markdown(
            f"""<div style="text-align:center;">
                <img src="{art_url}" style="width:140px; border-radius:8px;
                     box-shadow:0 2px 8px rgba(0,0,0,0.15);" />
                <div style="margin-top:6px; font-weight:600;">{chosen_md.get('title','')}</div>
                <div style="color:#555; font-size:0.85rem;">{chosen_md.get('artist','')}</div>
                <div style="margin-top:4px;">{genre_badge(chosen_md.get('genre',''))}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        if chosen_md.get("preview_url"):
            st.audio(chosen_md["preview_url"])

    n_show = 5
    als_ids, als_scores = art["index"].neighbors(pick_idx, k=n_show)
    aud_ids, aud_scores = art["audio_index"].neighbors(pick_idx, k=n_show)

    nb_col_a, nb_col_b = st.columns(2)
    with nb_col_a:
        st.markdown("**👥 ALS neighbors** _(by playlist co-occurrence)_")
        for r, (tid, sc) in enumerate(zip(als_ids, als_scores), 1):
            md = meta_list[int(tid)]
            st.markdown(
                f"<div class='track-row'><div class='rank-num'>{r}</div>"
                f"<img class='track-art' src='{md.get('artwork_url','')}' style='width:46px;height:46px;'/>"
                f"<div style='flex:1;'><div class='track-title' style='font-size:0.9rem;'>{md.get('title','')}</div>"
                f"<div class='track-artist'>{md.get('artist','')}</div>"
                f"<div>{genre_badge(md.get('genre',''))}</div></div></div>",
                unsafe_allow_html=True,
            )
    with nb_col_b:
        st.markdown("**🎧 Audio neighbors** _(by waveform features)_")
        for r, (tid, sc) in enumerate(zip(aud_ids, aud_scores), 1):
            md = meta_list[int(tid)]
            st.markdown(
                f"<div class='track-row'><div class='rank-num'>{r}</div>"
                f"<img class='track-art' src='{md.get('artwork_url','')}' style='width:46px;height:46px;'/>"
                f"<div style='flex:1;'><div class='track-title' style='font-size:0.9rem;'>{md.get('title','')}</div>"
                f"<div class='track-artist'>{md.get('artist','')}</div>"
                f"<div>{genre_badge(md.get('genre',''))}</div></div></div>",
                unsafe_allow_html=True,
            )

    overlap = set(int(x) for x in als_ids) & set(int(x) for x in aud_ids)
    st.caption(
        f"**{len(overlap)}/{n_show} overlap.** When the lists agree, both signals confirm the match. "
        "When they disagree, that's where the magic of a hybrid system is — each side catches something the other misses."
    )

st.divider()

# ---------- BASELINE COMPARISON ----------

st.header("ALS vs Popularity")
st.markdown(
    "**The most important sanity check in recsys.** A model that can't beat 'just recommend the "
    "globally most popular tracks' isn't actually learning your taste."
)

cmp_df = pd.DataFrame({
    "Model": ["ALS", "Popularity", "ALS", "Popularity"],
    "Metric": ["Recall@10", "Recall@10", "NDCG@10", "NDCG@10"],
    "Score": [m["als"]["recall@10"], m["popularity"]["recall@10"],
              m["als"]["ndcg@10"], m["popularity"]["ndcg@10"]],
})
fig2 = px.bar(cmp_df, x="Metric", y="Score", color="Model", barmode="group",
              color_discrete_map={"ALS": "#1DB954", "Popularity": "#888888"},
              height=380, title="ALS vs Popularity baseline")
fig2.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

lift_recall = m["als"]["recall@10"] / max(m["popularity"]["recall@10"], 1e-9)
st.markdown(f"**ALS lift:** ×{lift_recall:.1f} on Recall@10. Trained in {m['als']['train_seconds']:.2f}s.")

st.divider()

# ---------- INTERACTIVE ----------

st.header("Try it: build a playlist, get recommendations")

# Track index -> "Artist - Title" label (for the multiselect)
all_genres = sorted({(md.get("genre") or "?") for md in meta_list})
track_label_to_idx: dict[str, int] = {}
for i, md in enumerate(meta_list):
    label = f"{md.get('artist','?')} — {md.get('title','?')}  ·  {md.get('genre','?')}"
    track_label_to_idx[label] = i

# Order labels by global popularity descending, so recognizable songs are at the top
pop_order = np.argsort(-art["track_popularity"])
ordered_labels = []
seen_idx: set[int] = set()
for ti in pop_order:
    label = next((lbl for lbl, idx in track_label_to_idx.items() if idx == int(ti)), None)
    if label and int(ti) not in seen_idx:
        ordered_labels.append(label)
        seen_idx.add(int(ti))

# Preset starter playlists — one click fills the seed list
PRESETS: dict[str, list[tuple[str, str]]] = {
    "🎤 Pop hits": [("Taylor Swift", "Anti-Hero"), ("Dua Lipa", "Levitating"),
                   ("Harry Styles", "As It Was"), ("The Weeknd", "Blinding Lights")],
    "🎤 Hip-Hop": [("Kendrick Lamar", "HUMBLE."), ("Drake", "God's Plan"),
                  ("Travis Scott", "SICKO MODE"), ("J. Cole", "No Role Modelz")],
    "🎸 Classic Rock": [("Queen", "Bohemian Rhapsody"), ("Led Zeppelin", "Stairway to Heaven"),
                       ("The Beatles", "Hey Jude"), ("Pink Floyd", "Money")],
    "🎧 Indie chill": [("Tame Impala", "The Less I Know The Better"), ("Arctic Monkeys", "505"),
                      ("Beach House", "Space Song"), ("Mac DeMarco", "Chamber of Reflection")],
    "💃 R&B vibes": [("Frank Ocean", "Thinkin Bout You"), ("SZA", "Good Days"),
                    ("The Weeknd", "Starboy"), ("Daniel Caesar", "Best Part")],
}


def _find_label_for(artist: str, title: str) -> str | None:
    for lbl, idx in track_label_to_idx.items():
        md = meta_list[idx]
        if (md.get("artist") or "").lower() == artist.lower() and \
           (md.get("title") or "").lower().startswith(title.lower()[:10]):
            return lbl
    return None


# Initialize seeds in session state
if "user_seeds" not in st.session_state:
    st.session_state.user_seeds = []

# --- Preset row ---
st.markdown("**Quick start with a preset:**")
preset_cols = st.columns(len(PRESETS))
for col, (name, tracks) in zip(preset_cols, PRESETS.items()):
    if col.button(name, use_container_width=True):
        labels = [_find_label_for(a, t) for a, t in tracks]
        st.session_state.user_seeds = [lbl for lbl in labels if lbl]

# --- Filter + multiselect ---
genre_filter = st.multiselect("Filter the picker by genre (optional)", all_genres)
if genre_filter:
    pool = [lbl for lbl in ordered_labels
            if (meta_list[track_label_to_idx[lbl]].get("genre") in genre_filter)]
else:
    pool = ordered_labels

# Make sure current seeds are always available in the pool, even with a filter on
for s in st.session_state.user_seeds:
    if s not in pool:
        pool = [s] + pool

selected = st.multiselect(
    "🎵 Pick tracks for your playlist (search by artist or song)",
    pool,
    default=st.session_state.user_seeds,
    help="Type to search. Mix genres on purpose to see how the model blends them.",
)
st.session_state.user_seeds = selected
seed_indices = [track_label_to_idx[lbl] for lbl in selected]

# --- Show selected seeds as cards ---
if seed_indices:
    st.markdown(f"**Your playlist · {len(seed_indices)} track(s)**")
    seed_cols = st.columns(min(4, len(seed_indices)))
    for slot, ti in enumerate(seed_indices):
        with seed_cols[slot % len(seed_cols)]:
            md = meta_list[ti]
            art_url = md.get("artwork_url") or "https://via.placeholder.com/120?text=♪"
            st.markdown(
                f"""
                <div style="text-align:center; padding:6px;">
                  <img src="{art_url}" style="width:100%; max-width:140px; border-radius:8px;
                       box-shadow:0 2px 8px rgba(0,0,0,0.15);" />
                  <div style="font-size:0.85rem; font-weight:600; margin-top:6px;">{md.get('title','')}</div>
                  <div style="font-size:0.78rem; color:#555;">{md.get('artist','')}</div>
                  <div style="margin-top:4px;">{genre_badge(md.get('genre',''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if md.get("preview_url"):
                st.audio(md["preview_url"])
else:
    st.info("Pick a few tracks above (or click a preset) to get recommendations.")

# --- Controls ---
col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    k = st.slider("How many recommendations?", 5, 20, 8)
with col_b:
    show_pop = st.checkbox("Also show popularity baseline", value=False)
with col_c:
    if st.button("🗑 Clear playlist"):
        st.session_state.user_seeds = []
        st.rerun()

# Hybrid blend: 0 = pure audio, 1 = pure ALS
if art["audio_index"] is not None:
    blend = st.slider(
        "🎚 Blend the two signals — 0 = pure audio (sonic), 1 = pure ALS (cultural)",
        0.0, 1.0, 0.7, step=0.05,
        help="At 0.7, recommendations are 70% co-occurrence + 30% audio similarity. "
             "Slide left to favor sonically similar tracks, right to favor culturally similar ones."
    )
else:
    blend = 1.0


def blended_recommend(seed_indices: list[int], k: int, blend: float):
    """Score every track as blend*ALS + (1-blend)*audio similarity, return top-k unseen."""
    n = len(meta_list)
    seen = set(seed_indices)

    # ALS score: dot product between mean seed factor and each item factor
    seed_als = art["item_factors"][seed_indices].mean(axis=0)
    als_scores = art["item_factors"] @ seed_als
    als_scores = (als_scores - als_scores.min()) / (als_scores.ptp() + 1e-9)

    if art["audio_index"] is not None and blend < 1.0:
        seed_aud = art["audio_embeddings"][seed_indices].mean(axis=0)
        aud_scores = art["audio_embeddings"] @ seed_aud
        aud_scores = (aud_scores - aud_scores.min()) / (aud_scores.ptp() + 1e-9)
    else:
        aud_scores = np.zeros(n, dtype=np.float32)

    final = blend * als_scores + (1.0 - blend) * aud_scores
    for s in seen:
        final[s] = -1e9
    top = np.argsort(-final)[:k]
    return top, final[top], als_scores[top], aud_scores[top]


# --- Recommend ---
if st.button("🚀 Recommend", type="primary", disabled=not seed_indices):
    ids, scores, als_part, aud_part = blended_recommend(seed_indices, k, blend)

    if blend == 1.0:
        header = "🎯 ALS recommendations (pure cultural)"
    elif blend == 0.0:
        header = "🎯 Audio recommendations (pure sonic)"
    else:
        header = f"🎯 Hybrid recommendations ({int(blend*100)}% ALS + {int((1-blend)*100)}% audio)"
    st.subheader(header)

    for rank, (tid, score, a, u) in enumerate(zip(ids, scores, als_part, aud_part), 1):
        meta = meta_list[int(tid)] if int(tid) < len(meta_list) else {"title": catalog["track_names"][tid]}
        sub = (f"ALS {a:.2f} · Audio {u:.2f}"
               if art["audio_index"] is not None and 0.0 < blend < 1.0
               else "")
        render_track_card(rank, meta, score=float(score), sub_label=sub)
        contribs = explain(seed_indices, int(tid), art["item_factors"], catalog["track_names"], top_n=4)
        with st.expander("Why was this recommended?"):
            cdf = pd.DataFrame(contribs, columns=["Seed track", "Contribution"])
            bar = px.bar(cdf, x="Contribution", y="Seed track", orientation="h",
                         color="Contribution", color_continuous_scale="Greens", height=220)
            bar.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(bar, use_container_width=True)
            st.caption("ALS contribution: which seed tracks the model thinks this candidate co-occurs with the most.")

    if show_pop:
        st.subheader("📊 Popularity baseline (for comparison)")
        seen_set = set(seed_indices)
        pop_ranked = [int(t) for t in art["popularity"].ranked_track_ids if int(t) not in seen_set][:k]
        for rank, tid in enumerate(pop_ranked, 1):
            meta = meta_list[int(tid)] if int(tid) < len(meta_list) else {"title": catalog["track_names"][tid]}
            n_in = int(art["track_popularity"][tid])
            render_track_card(rank, meta, sub_label=f"in {n_in} playlists")

st.divider()
with st.expander("ℹ️ About this project"):
    st.markdown(
        f"""
**Data source:** {src_label} · {len(meta_list)} real tracks fetched from the iTunes Search API,
plus a synthetic playlist generator that clusters them by genre.

**Why iTunes data instead of Spotify MPD right now?** MPD is gated behind an AIcrowd application
process. iTunes is open and gives us album art + 30s previews for free, so we can build the
demo today. The same code drops MPD in unchanged when you have it.

**Model:** ALS, {m["model"]["factors"]}-dim factors, {m["model"]["iterations"]} iterations.
Trained in {m['als']['train_seconds']:.2f}s.

**Honest limits:**
- The catalog is small (~88 tracks across 9 genres) — this is a teaching demo, not a benchmark.
- Playlists are synthetic (genre-clustered) rather than real listener behavior.
- Cold-start tracks have no co-occurrences and never surface — needs content features to fix.
- No two-tower neural model yet (planned next).
"""
    )
