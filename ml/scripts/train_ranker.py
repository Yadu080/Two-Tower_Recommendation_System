"""
Phase 5 — Train LightGBM ranking model.

Pipeline:
  1. For each user in val set, retrieve top-500 items using item embeddings
     (simulates what FAISS will do at serving time)
  2. Build feature vectors for each (user, candidate_item) pair
  3. Label = 1 if user actually rated item >= 3.5 in val, else 0
  4. Train LightGBM binary classifier → outputs a relevance score in [0,1]
  5. At serving time: score top-500 FAISS candidates → return top-10

DSA: this is a learning-to-rank problem. The feature matrix is sparse
(most user-item pairs are 0), so we build it only for sampled pairs.
"""

import os, sys, ast, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# Note: LightGBM's Booster conflicts with PyTorch's OpenMP on macOS (SIGSEGV).
# sklearn.GradientBoostingClassifier has identical accuracy on 8 features
# and loads cleanly alongside torch in the same process.

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR  = os.path.join(os.path.dirname(__file__), "../data")
EMB_DIR   = os.path.join(os.path.dirname(__file__), "../embeddings")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

# how many negative candidates to sample per positive — class balance knob
NEG_POS_RATIO = 4
RANDOM_SEED   = 42
# how many val users to build ranking features for (full val = slow)
MAX_USERS     = 5_000


def load_artifacts():
    print("Loading artifacts …", flush=True)
    item_embs  = np.load(os.path.join(EMB_DIR, "item_embeddings.npy"))
    items_df   = pd.read_csv(os.path.join(DATA_DIR, "item_features.csv"))
    users_df   = pd.read_csv(os.path.join(DATA_DIR, "user_features.csv"))
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
    return item_embs, item_feat_map, user_feat_map, val_df


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


def build_training_data(val_df, item_embs, item_feat_map, user_feat_map):
    """
    For each user in val, build (user, pos_item) and (user, neg_items) pairs.

    Positive  = item the user actually rated >= 3.5 in val
    Negative  = random items NOT in the user's rated set (sampled from all items)

    DSA: use a set for O(1) "has user rated this?" checks per user.
    """
    print("Building ranking training data …", flush=True)

    pos_val = val_df[val_df["rating"] >= 3.5].copy()
    all_item_idxs = list(item_feat_map.keys())
    rng = np.random.default_rng(RANDOM_SEED)

    # group val positives by user for O(1) lookup
    user_pos_items: dict = {}
    for uid, mid in zip(pos_val["user_idx"], pos_val["movie_idx"]):
        user_pos_items.setdefault(int(uid), set()).add(int(mid))

    sampled_users = list(user_pos_items.keys())
    if len(sampled_users) > MAX_USERS:
        sampled_users = rng.choice(sampled_users, MAX_USERS, replace=False).tolist()

    rows, labels = [], []

    for i, user_idx in enumerate(sampled_users):
        if i % 1000 == 0:
            print(f"  Processing user {i:,}/{len(sampled_users):,} …", flush=True)

        if user_idx not in user_feat_map:
            continue
        user_feat = user_feat_map[user_idx]
        user_emb  = item_embs[0]  # placeholder — replaced with actual user emb at serve time
        # NOTE: at training time we approximate user emb with mean of their rated items
        pos_set   = user_pos_items[user_idx]
        pos_embs  = [item_embs[idx] for idx in pos_set if idx < len(item_embs)]
        if not pos_embs:
            continue
        user_emb = np.mean(pos_embs, axis=0)
        user_emb /= (np.linalg.norm(user_emb) + 1e-9)

        # positive rows
        for item_idx in pos_set:
            if item_idx not in item_feat_map:
                continue
            rows.append(build_feature_row(
                user_idx, item_idx, user_emb, item_embs,
                user_feat, item_feat_map[item_idx]
            ))
            labels.append(1)

        # negative rows — sample items the user hasn't seen
        n_neg = len(pos_set) * NEG_POS_RATIO
        neg_candidates = [idx for idx in rng.choice(all_item_idxs, n_neg * 3, replace=False)
                         if idx not in pos_set and idx in item_feat_map][:n_neg]
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

    return model


def main():
    item_embs, item_feat_map, user_feat_map, val_df = load_artifacts()
    X, y = build_training_data(val_df, item_embs, item_feat_map, user_feat_map)
    model = train_ranker(X, y)

    # save with joblib — loads in the same process as torch without conflict
    model_path = os.path.join(MODEL_DIR, "ranker.joblib")
    joblib.dump(model, model_path)

    with open(os.path.join(MODEL_DIR, "ranker_features.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)

    print(f"\n✓ Ranker saved → {model_path}", flush=True)


if __name__ == "__main__":
    main()
