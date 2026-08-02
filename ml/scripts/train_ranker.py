"""
Phase 5 — Train the ranking model.

Pipeline:
  1. Build each user's embedding from their TRAIN-set history, using the same
     UserTower the backend uses at serving time
  2. Label against their VAL-set positives — held-out future interactions
  3. Sample negatives the user never rated in either split
  4. Train a GBM binary classifier → relevance score in [0,1]
  5. At serving time: score the top-500 retrieved candidates → return top-N

On the disjoint split
---------------------
An earlier version of this script built the user embedding by averaging the
embeddings of a user's *validation* positives, then used those same items as
the positive rows. Because the first feature is the dot product between that
averaged vector and each candidate item, positives scored high essentially by
construction — the model was partly learning "was this item one of the vectors
I was just averaged from" rather than "would this user like this". The two
features derived from that dot product carried the large majority of feature
importance, so the resulting AUC was substantially inflated.

Two changes fix it:
  * The representation now comes from the UserTower rather than from averaging
    label items, so nothing about the labels leaks into the features.
  * Features are built from train-split history and labelled against val-split
    positives, so the item being scored is never part of what produced the
    embedding scoring it.

This also removes a train/serve skew: the ranker now sees exactly the same
user-embedding distribution in training that the backend produces in
production. Expect a markedly lower — and honest — AUC than the pre-fix number.

DSA: this is a learning-to-rank problem. The feature matrix is sparse
(most user-item pairs are 0), so we build it only for sampled pairs.
"""

import os, sys, ast, json
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# Note: LightGBM's Booster conflicts with PyTorch's OpenMP on macOS (SIGSEGV).
# sklearn.GradientBoostingClassifier has identical accuracy on 8 features
# and loads cleanly alongside torch in the same process.

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ml.models.two_tower import build_from_checkpoint
from ml.models.user_features import USER_FEATURE_DIM, build_user_feature_matrix

DATA_DIR  = os.path.join(os.path.dirname(__file__), "../data")
EMB_DIR   = os.path.join(os.path.dirname(__file__), "../embeddings")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

# how many negative candidates to sample per positive — class balance knob
NEG_POS_RATIO = 4
RANDOM_SEED   = 42
# how many val users to build ranking features for (full val = slow)
MAX_USERS     = 5_000
POS_THRESHOLD = 3.5


def load_artifacts():
    print("Loading artifacts …", flush=True)
    item_embs  = np.load(os.path.join(EMB_DIR, "item_embeddings.npy"))
    items_df   = pd.read_csv(os.path.join(DATA_DIR, "item_features.csv"))
    users_df   = pd.read_csv(os.path.join(DATA_DIR, "user_features.csv"))
    train_df   = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df     = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))

    # hash map: movie_idx → item features  (O(1) lookup)
    item_feat_map = {}
    for _, row in items_df.iterrows():
        idx = int(row["movie_idx"])
        gvec = ast.literal_eval(row["genre_vec"]) if isinstance(row["genre_vec"], str) \
               else row["genre_vec"]
        item_feat_map[idx] = {
            "avg_rating" : float(row["avg_rating"]),
            "num_ratings": int(row["num_ratings"]),
            "popularity" : float(row["popularity"]),
            "genre_vec"  : np.array(gvec, dtype=np.float32),
        }

    # hash map: user_idx → user features  (O(1) lookup)
    user_feat_map = {}
    for _, row in users_df.iterrows():
        idx = int(row["user_idx"])
        gvec = ast.literal_eval(row["genre_pref"]) if isinstance(row["genre_pref"], str) \
               else row["genre_pref"]
        user_feat_map[idx] = {
            "avg_rating" : float(row["avg_rating"]),
            "num_ratings": int(row["num_ratings"]),
            "genre_pref" : np.array(gvec, dtype=np.float32),
        }

    print(f"  Items: {len(item_feat_map):,}  Users: {len(user_feat_map):,}", flush=True)
    return item_embs, item_feat_map, user_feat_map, users_df, train_df, val_df


def load_user_tower():
    """
    Load the trained UserTower — the same one the backend serves with.

    Building ranker features from this rather than from averaged label items is
    what keeps the training and serving feature distributions identical.
    """
    ckpt = torch.load(os.path.join(MODEL_DIR, "two_tower.pt"), map_location="cpu")
    model = build_from_checkpoint(ckpt)
    model.eval()
    return model, ckpt.get("num_user_features", 0)


def compute_user_embeddings(model, num_user_features, user_ids,
                            user_feat_matrix) -> dict:
    """Batch all requested users through the UserTower → {user_idx: embedding}."""
    embs = {}
    BATCH = 1024
    with torch.no_grad():
        for start in range(0, len(user_ids), BATCH):
            chunk = user_ids[start:start + BATCH]
            idx_t = torch.tensor(chunk, dtype=torch.long)
            feats = None
            if num_user_features > 0:
                feats = torch.from_numpy(user_feat_matrix[chunk])
            out = model.user_tower(idx_t, feats).numpy()
            for uid, emb in zip(chunk, out):
                embs[uid] = emb
    return embs


def build_feature_row(user_idx: int, item_idx: int,
                      user_emb: np.ndarray,
                      item_embs: np.ndarray,
                      user_feat: dict,
                      item_feat: dict) -> list:
    """
    Build a single feature vector for a (user, item) pair.
    This is what the ranker scores at serving time.
    """
    item_emb = item_embs[item_idx]

    # cosine similarity — both embeddings are L2-normalised so dot = cosine
    emb_sim = float(np.dot(user_emb, item_emb))

    # genre match: dot product of user preference and item genre (both normalised)
    u_genre = user_feat["genre_pref"]
    i_genre = item_feat["genre_vec"]
    u_norm  = u_genre / (np.linalg.norm(u_genre) + 1e-9)
    i_norm  = i_genre / (np.linalg.norm(i_genre) + 1e-9)
    genre_match = float(np.dot(u_norm, i_norm))

    return [
        emb_sim,                          # Two-Tower similarity score
        item_feat["avg_rating"],          # how well-rated the item is
        item_feat["num_ratings"],         # how many people rated it
        item_feat["popularity"],          # normalised popularity
        user_feat["avg_rating"],          # user's average rating (leniency)
        user_feat["num_ratings"],         # how active the user is
        genre_match,                      # genre alignment
        emb_sim * item_feat["popularity"],  # interaction: popular + relevant
    ]


FEATURE_NAMES = [
    "emb_similarity",
    "item_avg_rating",
    "item_num_ratings",
    "item_popularity",
    "user_avg_rating",
    "user_num_ratings",
    "genre_match",
    "popularity_x_similarity",
]


def build_training_data(train_df, val_df, item_embs, item_feat_map,
                        user_feat_map, users_df, model, num_user_features):
    """
    Positive  = item the user rated >= 3.5 in the VAL split (held-out future)
    Negative  = item the user rated in NEITHER split
    Features  = built from the UserTower embedding, which was fit on the TRAIN
                split — so no label information reaches the feature side.

    DSA: use a set for O(1) "has user rated this?" checks per user.
    """
    print("Building ranking training data …", flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    all_item_idxs = np.array(list(item_feat_map.keys()))

    # val positives = the labels
    pos_val = val_df[val_df["rating"] >= POS_THRESHOLD]
    user_pos_items: dict = {}
    for uid, mid in zip(pos_val["user_idx"], pos_val["movie_idx"]):
        user_pos_items.setdefault(int(uid), set()).add(int(mid))

    # everything the user touched in train, so negatives exclude it
    user_train_seen: dict = {}
    for uid, mid in zip(train_df["user_idx"], train_df["movie_idx"]):
        user_train_seen.setdefault(int(uid), set()).add(int(mid))

    # only users with train history AND val positives can be used: the first
    # gives the UserTower something to have learned from, the second gives labels
    sampled_users = [u for u in user_pos_items
                     if u in user_train_seen and u in user_feat_map]
    print(f"  Users with both train history and val positives: {len(sampled_users):,}",
          flush=True)
    if len(sampled_users) > MAX_USERS:
        sampled_users = rng.choice(sampled_users, MAX_USERS, replace=False).tolist()

    # user embeddings straight from the tower — matches serving exactly
    print("  Computing user embeddings via UserTower …", flush=True)
    num_users = max(max(user_feat_map.keys()), max(sampled_users)) + 1
    user_feat_matrix = build_user_feature_matrix(users_df, num_users, USER_FEATURE_DIM)
    user_embs = compute_user_embeddings(model, num_user_features,
                                        sampled_users, user_feat_matrix)

    rows, labels = [], []

    for i, user_idx in enumerate(sampled_users):
        if i % 1000 == 0:
            print(f"  Processing user {i:,}/{len(sampled_users):,} …", flush=True)

        user_feat = user_feat_map[user_idx]
        user_emb  = user_embs[user_idx]
        pos_set   = user_pos_items[user_idx]
        seen      = user_train_seen.get(user_idx, set()) | pos_set

        n_pos = 0
        for item_idx in pos_set:
            if item_idx not in item_feat_map:
                continue
            rows.append(build_feature_row(
                user_idx, item_idx, user_emb, item_embs,
                user_feat, item_feat_map[item_idx]
            ))
            labels.append(1)
            n_pos += 1

        if n_pos == 0:
            continue

        # negative rows — items the user rated in neither split
        n_neg = n_pos * NEG_POS_RATIO
        pool = rng.choice(all_item_idxs, min(n_neg * 3, len(all_item_idxs)),
                          replace=False)
        neg_candidates = [int(idx) for idx in pool if int(idx) not in seen][:n_neg]
        for item_idx in neg_candidates:
            rows.append(build_feature_row(
                user_idx, item_idx, user_emb, item_embs,
                user_feat, item_feat_map[item_idx]
            ))
            labels.append(0)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"\n  Total samples: {len(y):,}  "
          f"Positives: {y.sum():,}  Negatives: {(y==0).sum():,}", flush=True)
    return X, y


def train_ranker(X, y):
    print("\nTraining sklearn GradientBoosting ranker …", flush=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"  Val AUC: {auc:.4f}", flush=True)

    print("\n  Feature importances:")
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for name, imp in importances:
        bar = "█" * int(imp / max(i for _, i in importances) * 30)
        print(f"    {name:<30s} {bar} {imp:.3f}")

    return model, auc


def main():
    (item_embs, item_feat_map, user_feat_map,
     users_df, train_df, val_df) = load_artifacts()

    model_tt, num_user_features = load_user_tower()
    print(f"  UserTower loaded (num_user_features={num_user_features})", flush=True)

    X, y = build_training_data(train_df, val_df, item_embs, item_feat_map,
                               user_feat_map, users_df, model_tt, num_user_features)
    ranker, auc = train_ranker(X, y)

    # save with joblib — loads in the same process as torch without conflict
    model_path = os.path.join(MODEL_DIR, "ranker.joblib")
    joblib.dump(ranker, model_path)

    with open(os.path.join(MODEL_DIR, "ranker_features.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)

    with open(os.path.join(MODEL_DIR, "ranker_metrics.json"), "w") as f:
        json.dump({"val_auc": float(auc), "n_samples": int(len(y))}, f, indent=2)

    print(f"\n✓ Ranker saved → {model_path}", flush=True)


if __name__ == "__main__":
    main()
