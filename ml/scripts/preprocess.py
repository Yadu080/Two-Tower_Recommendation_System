"""
Data preprocessing pipeline for MovieLens 25M.

Steps:
  1. Load movies + ratings
  2. Filter to active users (>=20 ratings) and popular movies (>=50 ratings)
  3. Build user features  (genre preferences, activity stats)
  4. Build item features  (genre vector, popularity stats)
  5. Build interaction table (train/val split)
  6. Save everything to ml/data/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(OUT_DIR, exist_ok=True)

# ── tuneable knobs ────────────────────────────────────────────────────────────
MIN_USER_RATINGS  = 20     # drop users who rated fewer than this
MIN_MOVIE_RATINGS = 50     # drop movies that received fewer ratings than this
MAX_INTERACTIONS  = 2_000_000  # cap total rows to keep training fast locally
RANDOM_SEED       = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_movies(path: str) -> pd.DataFrame:
    print("Loading movies …")
    movies = pd.read_csv(path)
    # genres column: "Action|Comedy|Drama" → list ["Action", "Comedy", "Drama"]
    movies["genre_list"] = movies["genres"].apply(
        lambda g: g.split("|") if g != "(no genres listed)" else []
    )
    print(f"  {len(movies):,} movies loaded")
    return movies


def load_ratings(path: str) -> pd.DataFrame:
    print("Loading ratings (this may take ~30 s for 25 M rows) …")
    ratings = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"  {len(ratings):,} ratings loaded")
    return ratings


def filter_interactions(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    DSA Note: defaultdict(int) is a hash map — O(1) insert and lookup.
    Filtering with .isin(set) is O(1) per element vs O(N) list scan.
    """
    print("Filtering active users and popular movies …")

    # count ratings per user / movie using hash maps
    user_counts  = defaultdict(int)
    movie_counts = defaultdict(int)
    for uid, mid in zip(ratings["userId"], ratings["movieId"]):
        user_counts[uid]  += 1
        movie_counts[mid] += 1

    active_users  = {u for u, c in user_counts.items()  if c >= MIN_USER_RATINGS}
    popular_movies = {m for m, c in movie_counts.items() if c >= MIN_MOVIE_RATINGS}

    print(f"  Active users : {len(active_users):,}")
    print(f"  Popular movies: {len(popular_movies):,}")

    # .isin() uses a set internally → O(1) per lookup
    filtered = ratings[
        ratings["userId"].isin(active_users) &
        ratings["movieId"].isin(popular_movies)
    ].copy()

    print(f"  Interactions after filter: {len(filtered):,}")

    # cap to MAX_INTERACTIONS to keep local training fast
    if len(filtered) > MAX_INTERACTIONS:
        filtered = filtered.sample(MAX_INTERACTIONS, random_state=RANDOM_SEED)
        print(f"  Sampled down to: {len(filtered):,}")

    return filtered.reset_index(drop=True)


def build_genre_vocab(movies: pd.DataFrame) -> dict:
    """
    Collect all unique genres into a sorted list → index map.
    Using a set for deduplication: O(1) insert, O(G) total.
    """
    all_genres: set = set()
    for gl in movies["genre_list"]:
        all_genres.update(gl)
    all_genres.discard("")
    vocab = {g: i for i, g in enumerate(sorted(all_genres))}
    print(f"  Genre vocab size: {len(vocab)}")
    return vocab


def genre_vector(genre_list: list, vocab: dict) -> list:
    """Multi-hot encode genres. Length = len(vocab)."""
    vec = [0] * len(vocab)
    for g in genre_list:
        if g in vocab:
            vec[vocab[g]] = 1
    return vec


def build_item_features(movies: pd.DataFrame,
                        ratings: pd.DataFrame,
                        genre_vocab: dict) -> pd.DataFrame:
    """
    Per-movie features:
      - genre vector (multi-hot)
      - avg_rating, num_ratings, rating_std  (popularity signals)
    """
    print("Building item features …")

    # aggregate rating stats per movie — hash map of lists
    movie_ratings_map: dict = defaultdict(list)
    for mid, r in zip(ratings["movieId"], ratings["rating"]):
        movie_ratings_map[mid].append(r)

    rows = []
    for _, row in movies.iterrows():
        mid = row["movieId"]
        rlist = movie_ratings_map.get(mid, [])

        if not rlist:
            continue  # skip movies with no filtered interactions

        rows.append({
            "movieId"   : mid,
            "title"     : row["title"],
            "avg_rating": round(float(np.mean(rlist)), 4),
            "num_ratings": len(rlist),
            "rating_std": round(float(np.std(rlist)), 4),
            "genres"    : "|".join(row["genre_list"]),
            "genre_vec" : genre_vector(row["genre_list"], genre_vocab),
        })

    df = pd.DataFrame(rows)
    # normalise popularity to [0, 1]
    df["popularity"] = (df["num_ratings"] - df["num_ratings"].min()) / \
                       (df["num_ratings"].max() - df["num_ratings"].min() + 1e-9)
    print(f"  Item features built for {len(df):,} movies")
    return df


def build_user_features(ratings: pd.DataFrame,
                        item_features: pd.DataFrame,
                        genre_vocab: dict) -> pd.DataFrame:
    """
    Per-user features:
      - avg_rating, num_ratings  (activity level)
      - genre preference vector  (weighted avg of genre vecs of rated movies)
    """
    print("Building user features …")

    # build movieId → genre_vec lookup (hash map for O(1) access)
    movie_genre_map: dict = {
        row["movieId"]: row["genre_vec"]
        for _, row in item_features.iterrows()
    }

    G = len(genre_vocab)

    user_data: dict = defaultdict(lambda: {
        "ratings": [],
        "genre_acc": np.zeros(G, dtype=np.float32),
    })

    for uid, mid, r in zip(ratings["userId"], ratings["movieId"], ratings["rating"]):
        user_data[uid]["ratings"].append(r)
        gv = movie_genre_map.get(mid)
        if gv is not None:
            # accumulate genre signal weighted by rating
            user_data[uid]["genre_acc"] += np.array(gv, dtype=np.float32) * r

    rows = []
    for uid, d in user_data.items():
        rlist = d["ratings"]
        # normalise genre preference vector
        genre_pref = d["genre_acc"] / (len(rlist) + 1e-9)
        rows.append({
            "userId"    : uid,
            "avg_rating": round(float(np.mean(rlist)), 4),
            "num_ratings": len(rlist),
            "genre_pref": genre_pref.tolist(),
        })

    df = pd.DataFrame(rows)
    print(f"  User features built for {len(df):,} users")
    return df


def train_val_split(interactions: pd.DataFrame,
                    val_ratio: float = 0.1) -> tuple:
    """
    Temporal split: for each user, their most recent 10% of ratings → val.
    This avoids data leakage (we never train on future interactions).
    DSA: sort once O(N log N), then slice — better than repeated filtering.
    """
    print("Splitting train / val (temporal) …")
    interactions = interactions.sort_values(["userId", "timestamp"])

    val_mask = interactions.groupby("userId")["timestamp"].transform(
        lambda t: t >= t.quantile(1 - val_ratio)
    )
    train = interactions[~val_mask].reset_index(drop=True)
    val   = interactions[val_mask].reset_index(drop=True)
    print(f"  Train: {len(train):,}  |  Val: {len(val):,}")
    return train, val


def remap_ids(interactions: pd.DataFrame,
              item_features: pd.DataFrame,
              user_features: pd.DataFrame):
    """
    Two-Tower needs contiguous 0-based integer IDs for embedding tables.
    We build two dicts (hash maps): original_id → contiguous_index.
    """
    unique_users  = sorted(interactions["userId"].unique())
    unique_movies = sorted(interactions["movieId"].unique())

    user_id_map  = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}

    interactions["user_idx"]  = interactions["userId"].map(user_id_map)
    interactions["movie_idx"] = interactions["movieId"].map(movie_id_map)

    item_features = item_features[
        item_features["movieId"].isin(movie_id_map)
    ].copy()
    item_features["movie_idx"] = item_features["movieId"].map(movie_id_map)

    user_features = user_features[
        user_features["userId"].isin(user_id_map)
    ].copy()
    user_features["user_idx"] = user_features["userId"].map(user_id_map)

    return interactions, item_features, user_features, user_id_map, movie_id_map


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    movies  = load_movies(os.path.join(DATA_DIR, "movie.csv"))
    ratings = load_ratings(os.path.join(DATA_DIR, "rating.csv"))

    # ── Filter ────────────────────────────────────────────────────────────────
    interactions = filter_interactions(ratings)
    del ratings  # free ~2 GB RAM

    # keep only movies present in filtered interactions
    active_movie_ids = set(interactions["movieId"].unique())
    movies = movies[movies["movieId"].isin(active_movie_ids)].reset_index(drop=True)

    # ── Build vocab + features ────────────────────────────────────────────────
    genre_vocab  = build_genre_vocab(movies)
    item_features = build_item_features(movies, interactions, genre_vocab)
    user_features = build_user_features(interactions, item_features, genre_vocab)

    # ── ID remapping ──────────────────────────────────────────────────────────
    interactions, item_features, user_features, user_id_map, movie_id_map = \
        remap_ids(interactions, item_features, user_features)

    # ── Train / val split ────────────────────────────────────────────────────
    train, val = train_val_split(interactions)

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\nSaving processed files …")

    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    item_features.to_csv(os.path.join(OUT_DIR, "item_features.csv"), index=False)
    user_features.to_csv(os.path.join(OUT_DIR, "user_features.csv"), index=False)

    # save genre vector column separately as numpy (faster to load later)
    import ast
    genre_vecs = np.array(item_features["genre_vec"].apply(
        lambda x: x if isinstance(x, list) else ast.literal_eval(x)
    ).tolist(), dtype=np.float32)
    np.save(os.path.join(OUT_DIR, "item_genre_vecs.npy"), genre_vecs)

    # save id maps as JSON (needed by API to translate back to real IDs)
    with open(os.path.join(OUT_DIR, "user_id_map.json"), "w") as f:
        json.dump({str(k): v for k, v in user_id_map.items()}, f)
    with open(os.path.join(OUT_DIR, "movie_id_map.json"), "w") as f:
        json.dump({str(k): v for k, v in movie_id_map.items()}, f)
    with open(os.path.join(OUT_DIR, "genre_vocab.json"), "w") as f:
        json.dump(genre_vocab, f)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n✓ Preprocessing complete")
    print(f"  Users  : {len(user_id_map):,}")
    print(f"  Movies : {len(movie_id_map):,}")
    print(f"  Train  : {len(train):,} interactions")
    print(f"  Val    : {len(val):,} interactions")
    print(f"  Genres : {len(genre_vocab)}")
    print(f"\n  Files saved to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
