"""Load playlists into a sparse user-item matrix.

Supports two sources:
1. Spotify Million Playlist Dataset (MPD) JSON slices placed in data/mpd/
2. A synthetic generator so the pipeline runs end-to-end before MPD is downloaded

The MPD ships as files like `mpd.slice.0-999.json`, each containing 1000 playlists.
Each playlist has a list of tracks with `track_uri` (Spotify URI) and `track_name`.
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Dataset:
    matrix: sp.csr_matrix          # shape (n_playlists, n_tracks), implicit 1.0s
    track_ids: list[str]           # index -> track_uri
    track_names: list[str]         # index -> human-readable "artist - title"
    track_to_idx: dict[str, int]
    playlist_names: list[str]      # index -> playlist name (or "playlist_<i>")

    @property
    def n_playlists(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_tracks(self) -> int:
        return self.matrix.shape[1]


def load_mpd(mpd_dir: str, max_slices: int | None = None) -> Dataset:
    """Load MPD JSON slices from a directory."""
    paths = sorted(glob.glob(os.path.join(mpd_dir, "mpd.slice.*.json")))
    if not paths:
        raise FileNotFoundError(
            f"No MPD slices found in {mpd_dir}. "
            "Download the dataset from https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge "
            "and unzip the JSON slices into data/mpd/."
        )
    if max_slices:
        paths = paths[:max_slices]

    track_to_idx: dict[str, int] = {}
    track_names: list[str] = []
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
                rows.append(pid)
                cols.append(idx)
            pid += 1

    data = np.ones(len(rows), dtype=np.float32)
    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(pid, len(track_to_idx)),
        dtype=np.float32,
    )
    track_ids = list(track_to_idx.keys())
    return Dataset(matrix, track_ids, track_names, track_to_idx, playlist_names)


def make_synthetic(
    n_playlists: int = 2000,
    n_tracks: int = 3000,
    n_topics: int = 20,
    avg_playlist_len: int = 25,
    seed: int = 0,
) -> Dataset:
    """Generate fake playlists with latent 'genre' structure.

    Each track belongs to one topic. Each playlist samples a primary topic
    and draws ~80% of its tracks from that topic, the rest uniformly. This
    produces a co-occurrence signal strong enough to validate the recsys
    pipeline end-to-end before real data is available.
    """
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
            if rng.random() < 0.8:
                tid = rng.choice(topic_to_tracks[primary])
            else:
                tid = rng.randrange(n_tracks)
            seen.add(tid)
        for tid in seen:
            rows.append(pid)
            cols.append(tid)

    data = np.ones(len(rows), dtype=np.float32)
    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_playlists, n_tracks),
        dtype=np.float32,
    )
    track_ids = [f"synthetic:track:{i}" for i in range(n_tracks)]
    track_names = [f"Topic{int(track_topic[i])} - Track{i}" for i in range(n_tracks)]
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    return Dataset(matrix, track_ids, track_names, track_to_idx, playlist_names)


def load_or_synthetic(mpd_dir: str = "data/mpd", **synth_kwargs) -> Dataset:
    """Try MPD first; fall back to synthetic so the pipeline always runs."""
    if os.path.isdir(mpd_dir) and glob.glob(os.path.join(mpd_dir, "mpd.slice.*.json")):
        return load_mpd(mpd_dir)
    return make_synthetic(**synth_kwargs)
