"""
Phase 9 — Offline evaluation metrics.

Metrics:
  Recall@K   : fraction of held-out positives retrieved in top-K
  NDCG@K     : normalised discounted cumulative gain — rewards ranking the
               positive higher within top-K (position-sensitive recall)

Both are computed over the val set using numpy retrieval.

Note on FAISS: FAISS and PyTorch share conflicting OpenMP runtimes on macOS,
causing a segfault when both are loaded in the same process. At 10K items,
numpy matrix multiply (0.63ms/query) is fast enough for eval and serving.
For production scale (100M+ items), run FAISS in a separate microservice.

DSA: computing NDCG requires knowing the rank of a hit.
  - Brute-force: scan top-K list for the item → O(K) per user
  - Better: build a set of top-K indices → O(1) lookup per item → O(K) build once
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ml.models.two_tower import TwoTowerModel

DATA_DIR  = os.path.join(os.path.dirname(__file__), "../data")
EMB_DIR   = os.path.join(os.path.dirname(__file__), "../embeddings")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

K_VALUES    = [10, 50, 100]
N_EVAL_USERS = 2_000   # sample for speed; full eval takes ~10 min
POS_THRESHOLD = 3.5


def dcg_at_k(relevances: list, k: int) -> float:
    """
    Discounted Cumulative Gain @K.
    relevances[i] = 1 if item at rank i+1 is relevant, else 0.
    DCG = sum( rel_i / log2(i+2) )  for i in [0, k)
    """
    return sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    IDCG = ideal DCG (all relevant items at top positions).
    """
    if not relevant:
        return 0.0
    relevances = [1 if item in relevant else 0 for item in retrieved[:k]]
    actual_dcg  = dcg_at_k(relevances, k)
    # ideal: min(|relevant|, k) hits at the top
    ideal_rels  = [1] * min(len(relevant), k)
    ideal_dcg   = dcg_at_k(ideal_rels, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / min(len(relevant), k)


def main():
    print("Loading artifacts …", flush=True)

    # ── Load model + item embeddings ─────────────────────────────────────────
    ckpt = torch.load(os.path.join(MODEL_DIR, "two_tower.pt"), map_location="cpu")
    model = TwoTowerModel(
        num_users  = ckpt["num_users"],
        num_items  = ckpt["num_items"],
        num_genres = ckpt["num_genres"],
        embed_dim  = ckpt["embed_dim"],
        output_dim = ckpt["output_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    item_embs = np.load(os.path.join(EMB_DIR, "item_embeddings.npy"))
    item_embs = np.ascontiguousarray(item_embs, dtype=np.float32)

    # ── Precompute item embedding matrix for numpy retrieval ─────────────────
    # item_embs is already L2-normalised → dot product = cosine similarity
    # shape: (num_items, 128) — O(N×D) = O(10523×128) per query ≈ 0.6ms, fine for eval
    index = item_embs  # just use numpy directly

    # ── Load val set, build user → positive items map ─────────────────────────
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    pos_val = val_df[val_df["rating"] >= POS_THRESHOLD]

    user_positives: dict = {}
    for uid, mid in zip(pos_val["user_idx"], pos_val["movie_idx"]):
        user_positives.setdefault(int(uid), set()).add(int(mid))

    eval_users = list(user_positives.keys())
    if len(eval_users) > N_EVAL_USERS:
        rng = np.random.default_rng(42)
        eval_users = rng.choice(eval_users, N_EVAL_USERS, replace=False).tolist()

    print(f"Evaluating {len(eval_users):,} users …", flush=True)

    # ── Per-user metrics ──────────────────────────────────────────────────────
    metrics = {k: {"recall": [], "ndcg": []} for k in K_VALUES}
    max_k = max(K_VALUES)

    with torch.no_grad():
        for i, user_idx in enumerate(eval_users):
            if i % 500 == 0:
                print(f"  {i:,}/{len(eval_users):,} …", flush=True)

            # user embedding from tower
            idx_t = torch.tensor([user_idx], dtype=torch.long)
            user_emb = model.user_tower(idx_t).numpy()          # (1, 128)

            # numpy retrieval: dot product then argpartition O(N) + sort O(K log K)
            sims = (index @ user_emb[0])                         # (num_items,)
            retrieved = np.argpartition(sims, -(max_k))[-max_k:] # top-max_k unsorted
            retrieved = retrieved[np.argsort(sims[retrieved])[::-1]].tolist()

            relevant = user_positives[user_idx]

            for k in K_VALUES:
                metrics[k]["recall"].append(recall_at_k(retrieved, relevant, k))
                metrics[k]["ndcg"].append(ndcg_at_k(retrieved, relevant, k))

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print(f"{'Metric':<20} {'@10':>8} {'@50':>8} {'@100':>8}")
    print("-" * 45)
    for metric in ["recall", "ndcg"]:
        row = f"{metric.upper() + '@K':<20}"
        for k in K_VALUES:
            val = np.mean(metrics[k][metric])
            row += f" {val:>8.4f}"
        print(row)
    print("=" * 45)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        metric: {str(k): float(np.mean(metrics[k][metric])) for k in K_VALUES}
        for metric in ["recall", "ndcg"]
    }
    out_path = os.path.join(MODEL_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved eval results → {out_path}", flush=True)

    # ── Interview-ready interpretation ───────────────────────────────────────
    r10 = results["recall"]["10"]
    n10 = results["ndcg"]["10"]
    print(f"\nInterpretation:")
    print(f"  Recall@10 = {r10:.4f} → the model retrieves the relevant item in")
    print(f"    its top-10 {r10*100:.1f}% of the time")
    print(f"  NDCG@10   = {n10:.4f} → position-weighted; 1.0 = perfect ranking")


if __name__ == "__main__":
    main()
