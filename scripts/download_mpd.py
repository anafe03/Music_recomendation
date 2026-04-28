"""Helper printout for getting the Spotify Million Playlist Dataset.

The MPD is gated behind an AIcrowd account — there is no public direct download.
Run this script for instructions, then drop the unzipped slices into data/mpd/.
"""

INSTRUCTIONS = """
Spotify Million Playlist Dataset (MPD) setup
============================================

1. Create a free account at https://www.aicrowd.com/
2. Visit the challenge page:
   https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
3. Accept the dataset license, then download `spotify_million_playlist_dataset.zip`
4. Unzip it. You will get a `data/` folder containing 1000 files named
   `mpd.slice.0-999.json`, `mpd.slice.1000-1999.json`, etc.
5. Copy (or symlink) those JSON files into this repo at:
       data/mpd/

Once the slices are in place, `python -m src.pipeline` will detect them
and use real data instead of the synthetic generator.

For a fast iteration loop, start with just a few slices:
   mkdir -p data/mpd
   cp /path/to/spotify_million_playlist_dataset/data/mpd.slice.0-999.json data/mpd/
   cp /path/to/spotify_million_playlist_dataset/data/mpd.slice.1000-1999.json data/mpd/

That's 2k playlists — enough to get meaningful metrics without long training.
"""

if __name__ == "__main__":
    print(INSTRUCTIONS)
