"""iTunes Search API client with disk cache.

The iTunes Search API is free and requires no authentication. For each
(artist, title) we get a 30-second preview URL, album artwork URL, and
iTunes' own primary genre tag. The disk cache means we hit the network once
across runs.

Endpoint:
    https://itunes.apple.com/search?term=<artist+title>&entity=song&limit=1
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Iterable

CACHE_PATH = "data/itunes_cache.json"
USER_AGENT = "music-recsys-demo/0.1 (educational project)"


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _query(artist: str, title: str) -> dict | None:
    term = urllib.parse.quote(f"{artist} {title}")
    url = f"https://itunes.apple.com/search?term={term}&entity=song&limit=1"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  iTunes lookup failed for '{artist} - {title}': {e}")
        return None
    results = data.get("results") or []
    if not results:
        return None
    r = results[0]
    return {
        "artist": r.get("artistName"),
        "title": r.get("trackName"),
        "preview_url": r.get("previewUrl"),
        "artwork_url": (r.get("artworkUrl100") or "").replace("100x100", "300x300"),
        "itunes_genre": r.get("primaryGenreName"),
        "track_view_url": r.get("trackViewUrl"),
    }


def fetch_metadata(seed_tracks: Iterable[tuple[str, str, str]], rate_limit_seconds: float = 0.5) -> dict[str, dict]:
    """For each (artist, title, genre) tuple, return enriched metadata.

    Returns a dict keyed by f"{artist}|||{title}". Skips entries without
    matches. Persists to a JSON cache between runs.
    """
    cache = _load_cache()
    out: dict[str, dict] = {}
    new_lookups = 0
    for artist, title, genre in seed_tracks:
        key = f"{artist}|||{title}"
        if key in cache and cache[key]:
            md = dict(cache[key])
            md["genre"] = genre
            out[key] = md
            continue
        md = _query(artist, title)
        new_lookups += 1
        time.sleep(rate_limit_seconds)
        if md is None:
            # Don't cache failures — retry on next run.
            continue
        md["genre"] = genre
        cache[key] = {k: v for k, v in md.items() if k != "genre"}
        out[key] = md

    if new_lookups:
        _save_cache(cache)
        print(f"  iTunes: {new_lookups} new lookups, {len(out)} total tracks with metadata")
    else:
        print(f"  iTunes: served {len(out)} tracks from cache")
    return out
