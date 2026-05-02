"""
Generate and save item embeddings from a trained Two-Tower model.

Runs a single forward pass through the item tower for every item,
then saves the (num_items, 128) matrix as a .npy file for FAISS.

This is done ONCE offline — not at query time.
"""

import os, sys, ast, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ml.models.two_tower import TwoTowerModel

DATA_DIR  = os.path.join(os.path.dirname(__file__), "../data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
EMB_DIR   = os.path.join(os.path.dirname(__file__), "../embeddings")
os.makedirs(EMB_DIR, exist_ok=True)


def main():
    device = torch.device("cpu")  # embedding gen doesn't need GPU

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(os.path.join(MODEL_DIR, "two_tower.pt"), map_location=device)
    model = TwoTowerModel(
        num_users  = ckpt["num_users"],
        num_items  = ckpt["num_items"],
        num_genres = ckpt["num_genres"],
        embed_dim  = ckpt["embed_dim"],
        output_dim = ckpt["output_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    # ── load item features ────────────────────────────────────────────────────
    items_df  = pd.read_csv(os.path.join(DATA_DIR, "item_features.csv"))
    NUM_ITEMS = ckpt["num_items"]
    NUM_G     = ckpt["num_genres"]

    feat = torch.zeros(NUM_ITEMS, NUM_G + 2, dtype=torch.float32)
    for _, row in items_df.iterrows():
        idx = int(row["movie_idx"])
        if idx >= NUM_ITEMS:
            continue
        gvec = ast.literal_eval(row["genre_vec"]) if isinstance(row["genre_vec"], str) \
               else row["genre_vec"]
        feat[idx, :NUM_G]    = torch.tensor(gvec, dtype=torch.float32)
        feat[idx, NUM_G]     = float(row["avg_rating"]) / 5.0
        feat[idx, NUM_G + 1] = float(row["popularity"])

    # ── generate embeddings in batches ────────────────────────────────────────
    print(f"Generating embeddings for {NUM_ITEMS:,} items …")
    BATCH = 512
    all_embs = []

    with torch.no_grad():
        for start in range(0, NUM_ITEMS, BATCH):
            end      = min(start + BATCH, NUM_ITEMS)
            item_idx = torch.arange(start, end, dtype=torch.long)
            item_feat = feat[start:end]
            emb = model.item_tower(item_idx, item_feat)   # (B, 128)
            all_embs.append(emb.numpy())

    embeddings = np.vstack(all_embs).astype(np.float32)  # (num_items, 128)
    print(f"Embeddings shape: {embeddings.shape}")

    # ── save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(EMB_DIR, "item_embeddings.npy"), embeddings)

    # also save item metadata for the API (movie_idx → title, genres)
    meta = items_df[["movie_idx", "movieId", "title", "genres",
                      "avg_rating", "popularity"]].copy()
    meta.to_csv(os.path.join(EMB_DIR, "item_meta.csv"), index=False)

    print(f"✓ Saved item_embeddings.npy  →  {os.path.join(EMB_DIR, 'item_embeddings.npy')}")
    print(f"✓ Saved item_meta.csv        →  {os.path.join(EMB_DIR, 'item_meta.csv')}")

    # ── quick sanity check: nearest neighbours for Toy Story ─────────────────
    toy_story = items_df[items_df["title"].str.contains("Toy Story", na=False)]
    if len(toy_story) > 0:
        ts_idx = int(toy_story.iloc[0]["movie_idx"])
        ts_emb = embeddings[ts_idx]
        sims   = embeddings @ ts_emb                    # cosine (all L2-normed)
        top5   = np.argsort(sims)[::-1][1:6]           # skip self (rank 0)

        idx_to_title = dict(zip(items_df["movie_idx"], items_df["title"]))
        print(f"\nNearest neighbours of '{toy_story.iloc[0]['title']}':")
        for rank, idx in enumerate(top5, 1):
            print(f"  {rank}. {idx_to_title.get(idx, '?')}  (sim={sims[idx]:.4f})")


if __name__ == "__main__":
    main()
