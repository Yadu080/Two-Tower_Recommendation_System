"""
Fetch movie poster URLs from The Movie Database (TMDb).

Run ONCE after getting a free API key from https://www.themoviedb.org/signup

Usage:
  1. Add TMDB_API_KEY=your_key to .env in the project root
  2. source venv/bin/activate
  3. PYTHONUNBUFFERED=1 python ml/scripts/fetch_posters.py

Output: ml/data/poster_map.json  →  { "movieId": "https://image.tmdb.org/t/p/w500/..." }
Runtime: ~8-10 minutes for 10K movies (respects TMDb rate limits)
"""

import os, sys, json, time
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TMDB_KEY  = os.environ.get("TMDB_API_KEY", "")
API_BASE  = "https://api.themoviedb.org/3"
IMG_BASE  = "https://image.tmdb.org/t/p/w500"

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "ml" / "data"
EMB_DIR  = ROOT / "ml" / "embeddings"


def main():
    if not TMDB_KEY:
        print("✗ TMDB_API_KEY not found.")
        print("  Create a .env file in the project root:")
        print("      TMDB_API_KEY=your_key_here")
        sys.exit(1)

    # ── Only fetch for movies we actually serve ────────────────────────────
    meta_df  = pd.read_csv(EMB_DIR / "item_meta.csv")
    links_df = pd.read_csv(ROOT / "data" / "link.csv")

    # movieId (int) → tmdbId (int)
    tmdb_map: dict[int, int] = {}
    for _, row in links_df.iterrows():
        if pd.notna(row.get("tmdbId")):
            tmdb_map[int(row["movieId"])] = int(row["tmdbId"])

    catalogue_ids = meta_df["movieId"].astype(int).tolist()
    print(f"Catalogue size : {len(catalogue_ids):,} movies", flush=True)
    print(f"TMDb ID coverage: {sum(1 for m in catalogue_ids if m in tmdb_map):,} have a tmdbId", flush=True)
    print("Starting fetch … (this takes ~8-10 min)\n", flush=True)

    poster_map: dict[str, str] = {}
    session = requests.Session()
    errors  = 0

    for i, movie_id in enumerate(catalogue_ids):
        if i % 500 == 0 and i > 0:
            pct = len(poster_map) / i * 100
            print(f"  {i:,}/{len(catalogue_ids):,}  posters so far: {len(poster_map):,}  ({pct:.0f}% hit rate)", flush=True)

        tmdb_id = tmdb_map.get(movie_id)
        if not tmdb_id:
            continue

        try:
            resp = session.get(
                f"{API_BASE}/movie/{tmdb_id}",
                params={"api_key": TMDB_KEY, "language": "en-US"},
                timeout=8,
            )
            if resp.status_code == 200:
                poster_path = resp.json().get("poster_path")
                if poster_path:
                    poster_map[str(movie_id)] = f"{IMG_BASE}{poster_path}"
            elif resp.status_code == 429:
                # rate-limited — back off and retry once
                print("  ⚠ Rate limited, backing off 15s …", flush=True)
                time.sleep(15)
                resp = session.get(
                    f"{API_BASE}/movie/{tmdb_id}",
                    params={"api_key": TMDB_KEY},
                    timeout=8,
                )
                if resp.status_code == 200:
                    poster_path = resp.json().get("poster_path")
                    if poster_path:
                        poster_map[str(movie_id)] = f"{IMG_BASE}{poster_path}"
            elif resp.status_code == 404:
                pass  # movie not on TMDb — skip silently
            else:
                errors += 1

        except requests.RequestException:
            errors += 1

        # Respect TMDb rate limit: ~20 req/s comfortably under the 40/10s cap
        time.sleep(0.05)

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = DATA_DIR / "poster_map.json"
    with open(out_path, "w") as f:
        json.dump(poster_map, f)

    coverage = len(poster_map) / len(catalogue_ids) * 100
    print(f"\n✓ Done!", flush=True)
    print(f"  Saved  : {out_path}", flush=True)
    print(f"  Posters: {len(poster_map):,} / {len(catalogue_ids):,}  ({coverage:.1f}% coverage)", flush=True)
    if errors:
        print(f"  Errors : {errors} (network timeouts — re-run to fill gaps)", flush=True)


if __name__ == "__main__":
    main()
