# Music Recommendation

A playlist-aware music recommender. Given a partial playlist, predict tracks
the user is likely to add next. Built as a hands-on recsys project targeting
the Spotify Million Playlist Dataset (MPD).

## Status

**v0.1 — ALS baseline + FAISS retrieval + Streamlit demo.** Runs end-to-end on
a synthetic dataset out of the box so you can poke at the pipeline before
downloading MPD. Drop MPD slices into `data/mpd/` and the same code trains on
real data with no changes.

### Stack
- **Data:** Spotify MPD (1M playlists) — synthetic fallback for fast iteration
- **Baseline:** ALS implicit-feedback matrix factorization (`implicit` library)
- **Retrieval:** FAISS inner-product index over learned item factors
- **Eval:** Recall@K and NDCG@K with held-out tracks per playlist (mirrors
  the RecSys 2018 challenge formulation)
- **Demo:** Streamlit app — pick a playlist, get recommendations, see *which
  seed tracks contributed most* to each recommendation

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Trains on synthetic data if data/mpd/ is empty
python -m src.pipeline

# Launch the demo
streamlit run app.py
```

Artifacts (item factors, train matrix, catalog, metrics) land in `artifacts/`.

## Using real data (MPD)

```bash
python scripts/download_mpd.py   # prints setup instructions
# Drop mpd.slice.*.json files into data/mpd/
python -m src.pipeline
```

## Repo layout

```
src/
  data.py        # MPD loader + synthetic generator
  split.py       # holdout split for next-track prediction
  als.py         # ALS training wrapper
  eval.py        # Recall@K / NDCG@K
  retrieval.py   # FAISS index over item factors
  pipeline.py    # end-to-end driver
app.py           # Streamlit demo
scripts/
  download_mpd.py
artifacts/       # generated: factors, train matrix, metrics
data/mpd/        # MPD JSON slices go here (gitignored)
```

## Roadmap

- [ ] **Two-tower neural model** (PyTorch). Item tower over track features
      (artist, album, audio features if available); playlist tower pools
      seed track embeddings. Trained with in-batch negatives + sampled softmax.
- [ ] **Cold-playlist evaluation slice.** Hypothesis: ALS will beat the neural
      model on very short seeds (<5 tracks) because regularized factorization
      is more conservative than a learned nonlinear pooler. Worth measuring.
- [ ] **Sequential model** (SASRec / GRU4Rec) for ordered playlists.
- [ ] **Re-ranking stage.** Today this is retrieval-only. A real system would
      retrieve ~1k candidates from FAISS then re-rank with a heavier model
      using richer features (recency, popularity, audio similarity).
- [ ] **Deploy demo.** HuggingFace Spaces or Railway.

## Notes on choices

- **Why ALS first.** It's the right baseline for implicit feedback and trains
  in seconds on 100k playlists. If a neural model can't beat it, the neural
  model is wrong. Many published models don't.
- **Why held-out tracks instead of time-based split.** MPD doesn't ship reliable
  per-track add timestamps. The held-out-tracks formulation is what the official
  RecSys 2018 challenge uses, so results are comparable to published baselines.
- **Why FAISS even for the baseline.** Item factors are a dense embedding space
  whether they came from ALS or a neural net. Putting FAISS in the v0.1 stack
  means the retrieval scaffolding is already in place when we swap models.
- **Synthetic generator.** Fakes 20 latent "genres" with 80% in-genre / 20%
  out-of-genre playlists. Strong enough to validate the pipeline (Recall@500
  goes from ~0.001 random to ~0.82 with ALS), too clean to draw real
  conclusions from. It's scaffolding, not a benchmark.

## What doesn't work yet (honest)

- Held-out split is random per playlist, not stratified by playlist length.
  Short playlists are over-represented in test failure cases.
- No popularity baseline. Need to add "always recommend top-N most-frequent
  tracks" to confirm ALS is doing more than memorizing popular music.
- No cold-start handling. New tracks (no playlist co-occurrences) get a zero
  vector and are never recommended. The two-tower work fixes this via content
  features.
