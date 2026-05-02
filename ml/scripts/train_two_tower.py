"""
Training script for the Two-Tower retrieval model.

Loss: InfoNCE (aka NT-Xent)
  - For a batch of B (user, pos_item) pairs:
    * Similarity matrix S[i,j] = dot(user_i, item_j) / temperature
    * S[i,i] is the positive; every S[i,j≠i] is an in-batch negative
    * Loss = cross_entropy(S, labels=arange(B))
  - We get B² training signal from B samples — very sample-efficient.

DSA insight: the similarity matrix is O(B²) in memory; keep batch size ≤ 1024.
"""

import os, sys, json, ast, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ml.models.two_tower import TwoTowerModel

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), "../data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
EMB_DIR   = os.path.join(os.path.dirname(__file__), "../embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

# ── hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 512
EPOCHS       = 10
LR           = 3e-4
EMBED_DIM    = 64
OUTPUT_DIM   = 128
NUM_WORKERS  = 0      # 0 = main process (safe on macOS with MPS)
POS_THRESHOLD = 3.5   # ratings >= this are treated as positive interactions
# ─────────────────────────────────────────────────────────────────────────────


class InteractionDataset(Dataset):
    """
    Loads positive (user, item) pairs.
    Item features are pre-loaded as numpy arrays indexed by movie_idx
    for O(1) lookup during training (vs. searching a DataFrame each time).
    """
    def __init__(self, interactions_df: pd.DataFrame,
                 item_feat_tensor: torch.Tensor):
        pos = interactions_df[interactions_df["rating"] >= POS_THRESHOLD]
        self.user_idxs  = torch.tensor(pos["user_idx"].values,  dtype=torch.long)
        self.item_idxs  = torch.tensor(pos["movie_idx"].values, dtype=torch.long)
        self.item_feats = item_feat_tensor   # (num_items, 21) pre-built

    def __len__(self):
        return len(self.user_idxs)

    def __getitem__(self, idx):
        uid   = self.user_idxs[idx]
        iid   = self.item_idxs[idx]
        feat  = self.item_feats[iid]         # O(1) tensor index
        return uid, iid, feat


def build_item_feature_tensor(items_df: pd.DataFrame,
                               num_items: int,
                               num_genres: int = 19) -> torch.Tensor:
    """
    Pre-build a (num_items, 21) float tensor so __getitem__ is a
    single tensor index (O(1)) rather than a DataFrame lookup (O(N)).

    Layout: [genre_vec(19) | avg_rating | popularity]
    Uses vectorised numpy ops instead of iterrows() — ~100x faster.
    """
    feat = np.zeros((num_items, num_genres + 2), dtype=np.float32)

    idxs   = items_df["movie_idx"].values.astype(int)
    gvecs  = np.array(
        items_df["genre_vec"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        ).tolist(), dtype=np.float32
    )
    ratings    = items_df["avg_rating"].values.astype(np.float32) / 5.0
    popularity = items_df["popularity"].values.astype(np.float32)

    valid = idxs < num_items
    feat[idxs[valid], :num_genres]    = gvecs[valid]
    feat[idxs[valid], num_genres]     = ratings[valid]
    feat[idxs[valid], num_genres + 1] = popularity[valid]

    return torch.from_numpy(feat)


def infonce_loss(user_emb: torch.Tensor,
                 item_emb: torch.Tensor,
                 temperature: torch.Tensor) -> torch.Tensor:
    """
    InfoNCE / in-batch negative loss.

    user_emb : (B, D)  L2-normalised
    item_emb : (B, D)  L2-normalised

    Similarity matrix: (B, B) where S[i,i] = positive pair.
    Rows = users, columns = items.
    Cross-entropy treats each row as a B-class classification problem
    where the correct class is the diagonal (label = i).

    This is equivalent to contrastive learning (SimCLR, CLIP style).
    """
    B = user_emb.size(0)
    sim = (user_emb @ item_emb.T) / temperature   # (B, B)
    labels = torch.arange(B, device=user_emb.device)
    # symmetric: users-as-queries + items-as-queries
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss


def in_batch_recall(user_emb: torch.Tensor,
                    item_emb: torch.Tensor, k: int = 10) -> float:
    """
    Quick sanity metric: for each user in the batch, is the positive item
    in the top-K retrieved items? Recall@K within the batch.
    """
    sim = user_emb @ item_emb.T        # (B, B)
    topk_idx = sim.topk(k, dim=-1).indices  # (B, K)
    labels = torch.arange(user_emb.size(0), device=user_emb.device)
    hits = (topk_idx == labels.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def train():
    device = (
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cuda")  if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}", flush=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading data …", flush=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    items_df = pd.read_csv(os.path.join(DATA_DIR, "item_features.csv"))

    with open(os.path.join(DATA_DIR, "user_id_map.json")) as f:
        user_id_map = json.load(f)
    with open(os.path.join(DATA_DIR, "movie_id_map.json")) as f:
        movie_id_map = json.load(f)

    NUM_USERS  = max(user_id_map.values()) + 1
    NUM_ITEMS  = max(movie_id_map.values()) + 1
    NUM_GENRES = 19

    print(f"  Users: {NUM_USERS:,}  Items: {NUM_ITEMS:,}", flush=True)

    # ── build item feature tensor (O(1) lookup in Dataset) ───────────────────
    # Keep on CPU — DataLoader workers can't collate GPU/MPS tensors.
    # We move each batch to device inside the training loop instead.
    print("Building item feature tensor …", flush=True)
    item_feat_tensor = build_item_feature_tensor(items_df, NUM_ITEMS, NUM_GENRES)

    # ── datasets & loaders ────────────────────────────────────────────────────
    train_ds = InteractionDataset(train_df, item_feat_tensor)
    val_ds   = InteractionDataset(val_df,   item_feat_tensor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    print(f"  Train positives: {len(train_ds):,}", flush=True)
    print(f"  Val   positives: {len(val_ds):,}",  flush=True)

    # ── model & optimiser ─────────────────────────────────────────────────────
    model = TwoTowerModel(NUM_USERS, NUM_ITEMS, NUM_GENRES, EMBED_DIM, OUTPUT_DIM).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}", flush=True)

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_recall, steps = 0.0, 0.0, 0

        t0 = time.time()
        for user_idx, item_idx, item_feat in train_loader:
            user_idx  = user_idx.to(device)
            item_idx  = item_idx.to(device)
            item_feat = item_feat.to(device)

            user_emb, item_emb = model(user_idx, item_idx, item_feat)
            loss = infonce_loss(user_emb, item_emb, model.temperature)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            train_loss   += loss.item()
            train_recall += in_batch_recall(user_emb.detach(), item_emb.detach(), k=10)
            steps += 1

        scheduler.step()

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_recall, val_steps = 0.0, 0.0, 0
        with torch.no_grad():
            for user_idx, item_idx, item_feat in val_loader:
                user_idx  = user_idx.to(device)
                item_idx  = item_idx.to(device)
                item_feat = item_feat.to(device)
                user_emb, item_emb = model(user_idx, item_idx, item_feat)
                val_loss   += infonce_loss(user_emb, item_emb, model.temperature).item()
                val_recall += in_batch_recall(user_emb, item_emb, k=10)
                val_steps  += 1

        avg_train = train_loss / steps
        avg_val   = val_loss   / val_steps
        avg_tr_rec = train_recall / steps
        avg_vl_rec = val_recall   / val_steps
        elapsed    = time.time() - t0

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
              f"train_recall@10={avg_tr_rec:.3f}  val_recall@10={avg_vl_rec:.3f}  "
              f"temp={model.temperature.item():.3f}  "
              f"[{elapsed:.0f}s]", flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "num_users": NUM_USERS,
                "num_items": NUM_ITEMS,
                "num_genres": NUM_GENRES,
                "embed_dim": EMBED_DIM,
                "output_dim": OUTPUT_DIM,
            }, os.path.join(MODEL_DIR, "two_tower.pt"))
            print(f"  ✓ checkpoint saved (val_loss={avg_val:.4f})", flush=True)

    print(f"\n✓ Training complete — best val_loss: {best_val_loss:.4f}", flush=True)


if __name__ == "__main__":
    train()
