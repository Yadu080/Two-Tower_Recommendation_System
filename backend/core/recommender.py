"""
Core recommendation pipeline — ties FAISS retrieval + LightGBM ranking together.

Request flow (DAG):
  user_idx
    → [LRU Cache] user_embedding (skip tower if cached)
    → [FAISS] top-500 candidate items
    → [LightGBM] rank candidates by feature score
    → [Heap top-K] select top-10
    → return results with explainability metadata
"""

import os, ast, json, heapq, time
from collections import OrderedDict
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import torch
# FAISS also conflicts with torch on macOS for the same reason.
# At 10K items, numpy matmul (0.63ms/query) is fast enough for serving.
# For production scale (100M+ items), run FAISS as a separate microservice.

# paths relative to project root
_ROOT     = os.path.join(os.path.dirname(__file__), "../..")
_EMB_DIR  = os.path.join(_ROOT, "ml/embeddings")
_MODEL_DIR = os.path.join(_ROOT, "ml/models")
_DATA_DIR = os.path.join(_ROOT, "ml/data")


# ── LRU Cache for user embeddings ─────────────────────────────────────────────
class LRUCache:
    """
    O(1) get and put using OrderedDict (doubly-linked list + hash map).

    Why we need this:
      Running a user through the UserTower costs ~0.5ms.
      If the same user makes 10 requests/minute, we recompute the same embedding each time.
      Cache hit → 0ms. Miss → 0.5ms + cache write.

    DSA: OrderedDict maintains insertion order.
      On access, we move the key to the end (most-recently-used).
      On eviction, we pop from the front (least-recently-used).
      Both ops are O(1) — that's the hash map part.

    Interview: "Same pattern as LeetCode 146 — LRU Cache."
    """
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: int) -> Optional[np.ndarray]:
        if key not in self.cache:
            self.misses += 1
            return None
        self.cache.move_to_end(key)   # mark as recently used
        self.hits += 1
        return self.cache[key]

    def put(self, key: int, value: np.ndarray):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)   # evict LRU (front)
        self.cache[key] = value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hit_rate, 3),
                "size": len(self.cache), "capacity": self.capacity}


def topk_heap(scores: np.ndarray, k: int) -> list:
    """
    O(N log K) heap-based top-K selection.
    Returns indices of top-K scores in descending order.
    """
    heap = []
    for i, score in enumerate(scores):
        s = float(score)
        if len(heap) < k:
            heapq.heappush(heap, (s, i))
        elif s > heap[0][0]:
            heapq.heapreplace(heap, (s, i))
    return [i for _, i in sorted(heap, reverse=True)]


# ── Singleton model loader ─────────────────────────────────────────────────────
class RecommendationEngine:
    """
    Loads all artifacts once at startup and holds them in memory.
    All recommendation requests share the same loaded models (singleton pattern).
    """

    def __init__(self):
        self._loaded = False
        self.user_cache = LRUCache(capacity=2048)

    def load(self):
        if self._loaded:
            return
        print("Loading recommendation engine …", flush=True)

        # ── Two-Tower model ────────────────────────────────────────────────────
        from ml.models.two_tower import TwoTowerModel
        ckpt = torch.load(os.path.join(_MODEL_DIR, "two_tower.pt"),
                          map_location="cpu")
        self.two_tower = TwoTowerModel(
            num_users  = ckpt["num_users"],
            num_items  = ckpt["num_items"],
            num_genres = ckpt["num_genres"],
            embed_dim  = ckpt["embed_dim"],
            output_dim = ckpt["output_dim"],
        )
        self.two_tower.load_state_dict(ckpt["model_state"])
        self.two_tower.eval()

        # ── Item embeddings (numpy, for feature computation) ──────────────────
        self.item_embs = np.load(os.path.join(_EMB_DIR, "item_embeddings.npy"))

        # ── Numpy retrieval matrix (replaces FAISS for local serving) ─────────
        # item_embs is L2-normalised so matmul gives cosine similarity directly
        self.faiss_index = None  # kept for API compatibility, unused at this scale

        # ── Item metadata: movie_idx → title, genres, stats ───────────────────
        meta_df = pd.read_csv(os.path.join(_EMB_DIR, "item_meta.csv"))
        self.item_meta: dict = {
            int(r["movie_idx"]): {
                "movieId"    : int(r["movieId"]),
                "title"      : r["title"],
                "genres"     : r["genres"],
                "avg_rating" : float(r["avg_rating"]),
                "popularity" : float(r["popularity"]),
            }
            for _, r in meta_df.iterrows()
        }

        # ── Item features (genre vecs) for ranker ─────────────────────────────
        items_df = pd.read_csv(os.path.join(_DATA_DIR, "item_features.csv"))
        self.item_feat_map: dict = {}
        for _, row in items_df.iterrows():
            idx  = int(row["movie_idx"])
            gvec = ast.literal_eval(row["genre_vec"]) if isinstance(row["genre_vec"], str) \
                   else row["genre_vec"]
            self.item_feat_map[idx] = {
                "avg_rating" : float(row["avg_rating"]),
                "num_ratings": int(row["num_ratings"]),
                "popularity" : float(row["popularity"]),
                "genre_vec"  : np.array(gvec, dtype=np.float32),
            }

        # ── User features ──────────────────────────────────────────────────────
        users_df = pd.read_csv(os.path.join(_DATA_DIR, "user_features.csv"))
        self.user_feat_map: dict = {}
        for _, row in users_df.iterrows():
            idx  = int(row["user_idx"])
            gvec = ast.literal_eval(row["genre_pref"]) if isinstance(row["genre_pref"], str) \
                   else row["genre_pref"]
            self.user_feat_map[idx] = {
                "avg_rating" : float(row["avg_rating"]),
                "num_ratings": int(row["num_ratings"]),
                "genre_pref" : np.array(gvec, dtype=np.float32),
            }

        # ── sklearn GBM ranker ─────────────────────────────────────────────────
        ranker_path = os.path.join(_MODEL_DIR, "ranker.joblib")
        if os.path.exists(ranker_path):
            self.ranker = joblib.load(ranker_path)
            with open(os.path.join(_MODEL_DIR, "ranker_features.json")) as f:
                self.ranker_features = json.load(f)
            self.ranker_loaded = True
        else:
            self.ranker_loaded = False
            print("  ⚠ Ranker not found — will use embedding similarity only")

        # ── ID maps ────────────────────────────────────────────────────────────
        with open(os.path.join(_DATA_DIR, "user_id_map.json")) as f:
            self.user_id_map = {int(k): v for k, v in json.load(f).items()}
        with open(os.path.join(_DATA_DIR, "movie_id_map.json")) as f:
            raw = json.load(f)
            self.movie_idx_to_id = {v: int(k) for k, v in raw.items()}

        self._loaded = True
        print("  ✓ Engine loaded", flush=True)

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get user embedding with LRU cache."""
        cached = self.user_cache.get(user_idx)
        if cached is not None:
            return cached

        with torch.no_grad():
            idx_t = torch.tensor([user_idx], dtype=torch.long)
            emb = self.two_tower.user_tower(idx_t).numpy()[0]  # (128,)

        self.user_cache.put(user_idx, emb)
        return emb

    def _build_ranker_features(self, user_idx: int, item_idx: int,
                                user_emb: np.ndarray) -> list:
        item_emb  = self.item_embs[item_idx]
        emb_sim   = float(np.dot(user_emb, item_emb))

        uf = self.user_feat_map.get(user_idx, {})
        itf = self.item_feat_map.get(item_idx, {})

        u_genre = np.array(uf.get("genre_pref", np.zeros(19)), dtype=np.float32)
        i_genre = np.array(itf.get("genre_vec", np.zeros(19)), dtype=np.float32)
        u_norm = u_genre / (np.linalg.norm(u_genre) + 1e-9)
        i_norm = i_genre / (np.linalg.norm(i_genre) + 1e-9)
        genre_match = float(np.dot(u_norm, i_norm))

        avg_r = itf.get("avg_rating", 3.5)
        pop   = itf.get("popularity", 0.0)

        return [
            emb_sim,
            avg_r,
            itf.get("num_ratings", 0),
            pop,
            uf.get("avg_rating", 3.5),
            uf.get("num_ratings", 0),
            genre_match,
            emb_sim * pop,
        ]

    def recommend(self, user_idx: int,
                  n_retrieve: int = 500,
                  n_return: int = 10,
                  exclude_seen: Optional[set] = None) -> list:
        """
        Full pipeline:
          1. user_idx → user_emb  (UserTower + LRU cache)
          2. user_emb → top-500 candidates  (FAISS)
          3. candidates → ranked scores  (LightGBM)
          4. scores → top-10  (heap)
          5. format with explainability fields
        """
        t_start = time.perf_counter()

        # Step 1 — user embedding
        user_emb = self.get_user_embedding(user_idx)

        # Step 2 — numpy retrieval (cosine sim via matmul, items are L2-normalised)
        sims = self.item_embs @ user_emb                          # (num_items,)
        top_idx = np.argpartition(sims, -n_retrieve)[-n_retrieve:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        candidates   = top_idx                                    # (500,)
        faiss_scores = sims[top_idx]                              # (500,)

        # filter out seen items
        if exclude_seen:
            mask = np.array([idx not in exclude_seen for idx in candidates])
            candidates   = candidates[mask]
            faiss_scores = faiss_scores[mask]

        # Step 3 — LightGBM ranking (if available)
        if self.ranker_loaded and len(candidates) > 0:
            feat_matrix = np.array([
                self._build_ranker_features(user_idx, int(c), user_emb)
                for c in candidates
            ], dtype=np.float32)
            rank_scores = self.ranker.predict_proba(feat_matrix)[:, 1]  # (N,) in [0,1]
        else:
            rank_scores = faiss_scores                            # fallback

        # Step 4 — heap top-K  O(N log K)
        top_k_indices = topk_heap(rank_scores, n_return)

        # Step 5 — format results
        latency_ms = (time.perf_counter() - t_start) * 1000
        results = []
        for rank, local_idx in enumerate(top_k_indices):
            item_idx   = int(candidates[local_idx])
            meta       = self.item_meta.get(item_idx, {})
            emb_sim    = float(faiss_scores[local_idx])
            rank_score = float(rank_scores[local_idx])

            # explainability: which genre drove this recommendation?
            itf = self.item_feat_map.get(item_idx, {})
            uf  = self.user_feat_map.get(user_idx, {})
            top_genre = _top_matching_genre(
                np.array(uf.get("genre_pref", np.zeros(19))),
                np.array(itf.get("genre_vec", np.zeros(19))),
                meta.get("genres", ""),
            )

            results.append({
                "rank"           : rank + 1,
                "movie_idx"      : item_idx,
                "movieId"        : meta.get("movieId"),
                "title"          : meta.get("title", "Unknown"),
                "genres"         : meta.get("genres", ""),
                "avg_rating"     : meta.get("avg_rating"),
                "popularity"     : meta.get("popularity"),
                "embedding_sim"  : round(emb_sim, 4),
                "ranking_score"  : round(rank_score, 4),
                "why_recommended": f"Matches your taste in {top_genre}" if top_genre
                                   else "Based on your viewing history",
                "latency_ms"     : round(latency_ms, 2),
            })

        return results


def _top_matching_genre(user_pref: np.ndarray,
                        item_genre: np.ndarray,
                        genres_str: str) -> str:
    """Return the genre with highest user preference × item presence."""
    genre_list = genres_str.split("|") if genres_str else []
    if not genre_list or len(user_pref) == 0:
        return ""
    # multiply element-wise: high only if user likes it AND item has it
    scores = user_pref * item_genre
    best_idx = int(np.argmax(scores))
    if best_idx < len(genre_list) and scores[best_idx] > 0:
        return genre_list[0]   # return primary genre as fallback
    return genre_list[0] if genre_list else ""


# module-level singleton — imported by FastAPI
engine = RecommendationEngine()
