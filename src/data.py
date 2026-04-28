"""Load playlists into a sparse user-item matrix.

Three data sources, in priority order:
1. Spotify Million Playlist Dataset (MPD) JSON slices in data/mpd/
2. Real-track seeded synthetic dataset (recommended demo path) — uses the
   curated seed list and iTunes Search API to get real tracks with album
   artwork and 30s preview URLs, then builds genre-clustered playlists.
3. Pure synthetic — fully fake "Topic12" tracks. Last-resort fallback when
   the network is unavailable.
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class Dataset:
    matrix: sp.csr_matrix          # shape (n_playlists, n_tracks), implicit 1.0s
    track_ids: list[str]           # index -> track id
    track_names: list[str]         # index -> "artist - title"
    track_to_idx: dict[str, int]
    playlist_names: list[str]      # index -> playlist name
    track_meta: list[dict] = field(default_factory=list)  # per-track: genre, preview_url, artwork_url, ...

    @property
    def n_playlists(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_tracks(self) -> int:
        return self.matrix.shape[1]


# ---------- MPD ----------

def load_mpd(mpd_dir: str, max_slices: int | None = None) -> Dataset:
    paths = sorted(glob.glob(os.path.join(mpd_dir, "mpd.slice.*.json")))
    if not paths:
        raise FileNotFoundError(
            f"No MPD slices found in {mpd_dir}. "
            "Get the dataset from https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge"
        )
    if max_slices:
        paths = paths[:max_slices]

    track_to_idx: dict[str, int] = {}
    track_names: list[str] = []
    track_meta: list[dict] = []
    playlist_names: list[str] = []
    rows: list[int] = []
    cols: list[int] = []

    pid = 0
    for path in paths:
        with open(path) as f:
            blob = json.load(f)
        for p in blob["playlists"]:
            playlist_names.append(p.get("name") or f"playlist_{pid}")
            for t in p["tracks"]:
                uri = t["track_uri"]
                idx = track_to_idx.get(uri)
                if idx is None:
                    idx = len(track_to_idx)
                    track_to_idx[uri] = idx
                    track_names.append(f"{t.get('artist_name','?')} - {t.get('track_name','?')}")
                    track_meta.append({
                        "artist": t.get("artist_name"),
                        "title": t.get("track_name"),
                        "album": t.get("album_name"),
                        "genre": None,
                        "artwork_url": None,
                        "preview_url": None,
                    })
                rows.append(pid)
                cols.append(idx)
            pid += 1

    data = np.ones(len(rows), dtype=np.float32)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(pid, len(track_to_idx)), dtype=np.float32)
    return Dataset(matrix, list(track_to_idx.keys()), track_names, track_to_idx, playlist_names, track_meta)


# ---------- Pure synthetic (last resort) ----------

def make_synthetic(
    n_playlists: int = 2000,
    n_tracks: int = 3000,
    n_topics: int = 20,
    avg_playlist_len: int = 25,
    seed: int = 0,
) -> Dataset:
    """Fully fake tracks. Use only if the network is unavailable."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    track_topic = np_rng.integers(0, n_topics, size=n_tracks)
    topic_to_tracks: dict[int, list[int]] = {t: [] for t in range(n_topics)}
    for i, t in enumerate(track_topic):
        topic_to_tracks[int(t)].append(i)

    rows: list[int] = []
    cols: list[int] = []
    playlist_names: list[str] = []
    for pid in range(n_playlists):
        primary = rng.randrange(n_topics)
        playlist_names.append(f"synthetic_topic{primary}_{pid}")
        length = max(5, int(np_rng.normal(avg_playlist_len, 5)))
        seen: set[int] = set()
        while len(seen) < length:
            tid = rng.choice(topic_to_tracks[primary]) if rng.random() < 0.8 else rng.randrange(n_tracks)
            seen.add(tid)
        for tid in seen:
            rows.append(pid); cols.append(tid)

    data = np.ones(len(rows), dtype=np.float32)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_playlists, n_tracks), dtype=np.float32)
    track_ids = [f"synthetic:track:{i}" for i in range(n_tracks)]
    track_names = [f"Topic{int(track_topic[i])} - Track{i}" for i in range(n_tracks)]
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    track_meta = [
        {"artist": f"Topic{int(track_topic[i])}", "title": f"Track{i}",
         "genre": f"Topic{int(track_topic[i])}", "artwork_url": None, "preview_url": None}
        for i in range(n_tracks)
    ]
    return Dataset(matrix, track_ids, track_names, track_to_idx, playlist_names, track_meta)


# ---------- Real-seeded synthetic (default demo path) ----------

def make_real_seeded(
    n_playlists: int = 1500,
    avg_playlist_len: int = 18,
    primary_genre_share: float = 0.8,
    seed: int = 0,
) -> Dataset:
    """Build playlists from real tracks fetched via iTunes Search API.

    Each playlist picks a primary genre, draws ~80% of its tracks from that
    genre, the rest from any other genre. Track names, album artwork, and
    30-second preview URLs come from iTunes (cached on disk).
    """
    from .catalog_seed import SEED_TRACKS
    from .itunes import fetch_metadata

    print("  Fetching iTunes metadata for seed tracks...")
    meta_by_key = fetch_metadata(SEED_TRACKS)

    track_meta: list[dict] = []
    track_ids: list[str] = []
    track_names: list[str] = []
    for artist, title, genre in SEED_TRACKS:
        key = f"{artist}|||{title}"
        md = meta_by_key.get(key)
        if md is None:
            continue
        track_ids.append(key)
        display_artist = md.get("artist") or artist
        display_title = md.get("title") or title
        track_names.append(f"{display_artist} - {display_title}")
        track_meta.append({
            "artist": display_artist,
            "title": display_title,
            "genre": genre,
            "itunes_genre": md.get("itunes_genre"),
            "artwork_url": md.get("artwork_url"),
            "preview_url": md.get("preview_url"),
            "track_view_url": md.get("track_view_url"),
        })

    if len(track_ids) < 30:
        print("  Too few iTunes lookups succeeded — falling back to pure synthetic.")
        return make_synthetic(seed=seed)

    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    genre_to_indices: dict[str, list[int]] = {}
    for i, m in enumerate(track_meta):
        genre_to_indices.setdefault(m["genre"], []).append(i)
    genres = list(genre_to_indices.keys())

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    rows: list[int] = []
    cols: list[int] = []
    playlist_names: list[str] = []

    for pid in range(n_playlists):
        primary = rng.choice(genres)
        playlist_names.append(f"{primary} mix #{pid}")
        length = max(8, int(np_rng.normal(avg_playlist_len, 4)))
        length = min(length, len(track_ids))
        seen: set[int] = set()
        attempts = 0
        while len(seen) < length and attempts < length * 5:
            attempts += 1
            if rng.random() < primary_genre_share:
                tid = rng.choice(genre_to_indices[primary])
            else:
                tid = rng.randrange(len(track_ids))
            seen.add(tid)
        for tid in seen:
            rows.append(pid); cols.append(tid)

    data = np.ones(len(rows), dtype=np.float32)
    matrix = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_playlists, len(track_ids)), dtype=np.float32
    )
    return Dataset(matrix, track_ids, track_names, track_to_idx, playlist_names, track_meta)


def load_or_synthetic(mpd_dir: str = "data/mpd", **synth_kwargs) -> Dataset:
    """Try MPD -> real-seeded -> pure synthetic, in that order."""
    if os.path.isdir(mpd_dir) and glob.glob(os.path.join(mpd_dir, "mpd.slice.*.json")):
        print("  Source: Spotify MPD")
        return load_mpd(mpd_dir)
    print("  Source: real-seeded synthetic (iTunes API)")
    try:
        return make_real_seeded(**{k: v for k, v in synth_kwargs.items() if k in ("seed",)})
    except Exception as e:
        print(f"  Real-seeded build failed ({e}) — using pure synthetic.")
        return make_synthetic(**synth_kwargs)
