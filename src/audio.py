"""Content-based audio embeddings.

Pipeline:
  1. Download each track's 30s iTunes preview (cached on disk as .m4a).
  2. Load the waveform with librosa.
  3. Extract a hand-built feature vector:
       - 13 MFCCs (mean + std)            -> 26 dims  (timbre)
       - 12 chroma (mean)                  -> 12 dims  (key/harmony)
       - 7 spectral contrast (mean)        -> 7 dims   (tonal vs noisy)
       - spectral centroid (mean + std)    -> 2 dims   (brightness)
       - spectral rolloff (mean)           -> 1 dim
       - zero-crossing rate (mean + std)   -> 2 dims
       - RMS energy (mean + std)           -> 2 dims   (loudness dynamics)
       - tempo                             -> 1 dim    (BPM)
                                              ------
                                              53 dims
  4. L2-normalize so cosine = inner product.
  5. Save as a (n_tracks, 53) float32 array aligned with the catalog.

These are classical audio features (the kind ISMIR papers used pre-deep-learning).
They capture timbre, harmony, rhythm, and dynamics — *the actual sound* of a
track — independent of which playlists it's been in. That's the whole point:
a separate similarity channel from collaborative filtering.

Drop-in upgrade path: replace `extract_features` with a call to a deep model
(CLAP, PANNs, OpenL3) and the rest of the pipeline doesn't change.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Iterable

import numpy as np


PREVIEW_DIR = "data/previews"
USER_AGENT = "music-recsys-demo/0.1"


def download_preview(url: str, out_path: str, timeout: int = 15) -> bool:
    """Download a single iTunes preview if not already cached. Returns success."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1024:
            return False
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    download failed for {out_path}: {e}")
        return False


def download_all_previews(track_meta: list[dict], out_dir: str = PREVIEW_DIR) -> list[str | None]:
    """Download each track's preview. Returns list of local paths (or None if missing)."""
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str | None] = []
    for i, md in enumerate(track_meta):
        url = md.get("preview_url")
        if not url:
            paths.append(None)
            continue
        out = os.path.join(out_dir, f"{i:04d}.m4a")
        ok = download_preview(url, out)
        paths.append(out if ok else None)
    n_ok = sum(1 for p in paths if p)
    print(f"  Audio previews: {n_ok}/{len(track_meta)} cached at {out_dir}/")
    return paths


def extract_features(audio_path: str, sr: int = 22050, duration: float = 28.0) -> np.ndarray | None:
    """Extract a 53-dim feature vector from a single audio file."""
    import librosa
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    except Exception as e:
        print(f"    load failed for {audio_path}: {e}")
        return None
    if len(y) < sr:
        return None

    feats: list[float] = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feats.extend(mfcc.mean(axis=1).tolist())
    feats.extend(mfcc.std(axis=1).tolist())

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(chroma.mean(axis=1).tolist())

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats.extend(contrast.mean(axis=1).tolist())

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats.append(float(centroid.mean()))
    feats.append(float(centroid.std()))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats.append(float(rolloff.mean()))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats.append(float(zcr.mean()))
    feats.append(float(zcr.std()))

    rms = librosa.feature.rms(y=y)[0]
    feats.append(float(rms.mean()))
    feats.append(float(rms.std()))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats.append(float(np.asarray(tempo).flatten()[0]))

    vec = np.array(feats, dtype=np.float32)
    if not np.all(np.isfinite(vec)):
        return None
    return vec


def build_audio_embeddings(track_meta: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Download previews + extract features for the whole catalog.

    Returns:
        embeddings: (n_tracks, dim) float32, L2-normalized. Tracks without
                    audio get a zero vector.
        valid_mask: (n_tracks,) bool, True where a real embedding exists.
    """
    paths = download_all_previews(track_meta)
    n = len(track_meta)
    feats: list[np.ndarray | None] = [None] * n
    print("  Extracting audio features (this takes ~30s for 88 tracks)...")
    for i, p in enumerate(paths):
        if p is None:
            continue
        feats[i] = extract_features(p)

    valid = [v is not None for v in feats]
    valid_mask = np.array(valid, dtype=bool)
    if not valid_mask.any():
        return np.zeros((n, 53), dtype=np.float32), valid_mask

    dim = next(v for v in feats if v is not None).shape[0]
    raw = np.zeros((n, dim), dtype=np.float32)
    for i, v in enumerate(feats):
        if v is not None:
            raw[i] = v

    # Per-feature standardization on the valid rows, then L2-normalize.
    mu = raw[valid_mask].mean(axis=0)
    sd = raw[valid_mask].std(axis=0) + 1e-8
    raw[valid_mask] = (raw[valid_mask] - mu) / sd

    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = (raw / norms).astype(np.float32)

    print(f"  Audio embeddings: {valid_mask.sum()}/{n} tracks, dim={dim}")
    return embeddings, valid_mask
