"""
User content features for the UserTower.

Kept in one place deliberately: the ranker previously built its user
representation differently at training time than the backend did at serving
time, and the resulting feature-distribution mismatch silently miscalibrated
the model. Everything that needs user features should call into here so the
two paths cannot drift apart again.

Layout: [genre_pref(19, L2-normalised)]
"""

import ast
import numpy as np

USER_FEATURE_DIM = 19   # genre preference vector


def parse_genre_pref(raw, num_genres: int = USER_FEATURE_DIM) -> np.ndarray:
    """
    Parse a genre_pref cell from user_features.csv into a fixed-width vector.

    Stored as a stringified list; raw counts are unnormalised (a heavy user can
    have values well above 1), so we L2-normalise to keep the scale comparable
    across users regardless of how much they've watched.
    """
    if raw is None:
        return np.zeros(num_genres, dtype=np.float32)
    vec = ast.literal_eval(raw) if isinstance(raw, str) else raw
    vec = np.asarray(vec, dtype=np.float32)

    if vec.shape[0] < num_genres:
        vec = np.pad(vec, (0, num_genres - vec.shape[0]))
    elif vec.shape[0] > num_genres:
        vec = vec[:num_genres]

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def build_user_feature_matrix(users_df, num_users: int,
                              num_genres: int = USER_FEATURE_DIM) -> np.ndarray:
    """
    Build a (num_users, USER_FEATURE_DIM) matrix indexed by user_idx.

    Users absent from the dataframe keep an all-zero row, which the tower reads
    as "no known preference" rather than a wrong one.
    """
    feat = np.zeros((num_users, num_genres), dtype=np.float32)
    for idx, raw in zip(users_df["user_idx"].values, users_df["genre_pref"].values):
        idx = int(idx)
        if 0 <= idx < num_users:
            feat[idx] = parse_genre_pref(raw, num_genres)
    return feat


def genre_names_to_vector(genre_names: list, genre_vocab: dict,
                          num_genres: int = USER_FEATURE_DIM) -> np.ndarray:
    """
    Build the same feature vector for a cold-start user who has picked genres
    but has no rating history — a normalised multi-hot over their choices.
    """
    vec = np.zeros(num_genres, dtype=np.float32)
    for g in genre_names:
        gi = genre_vocab.get(g)
        if gi is not None and 0 <= gi < num_genres:
            vec[gi] = 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)
