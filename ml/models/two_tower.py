"""
Two-Tower retrieval model.

UserTower : user_idx (+ optional content features) → 128-dim L2-normalised embedding
ItemTower : item_idx + content features           → 128-dim L2-normalised embedding

Both towers share the same output space so dot product gives cosine similarity.

Note on user content features:
  Originally the user tower was a bare ID embedding, which meant it could only
  memorise per-user patterns — it had no way to generalise across users with
  similar taste. It now optionally concatenates a normalised genre-preference
  vector, mirroring what the item tower already does with its content features.

  `num_user_features=0` reproduces the original architecture exactly. The value
  is written into the checkpoint, and checkpoints saved before this change
  simply lack the key — so loaders default it to 0 and old weights keep working.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users: int, embed_dim: int = 64, output_dim: int = 128,
                 num_user_features: int = 0):
        super().__init__()
        self.num_user_features = num_user_features
        self.embedding = nn.Embedding(num_users, embed_dim)
        feature_dim = embed_dim + num_user_features
        self.net = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, user_idx: torch.Tensor,
                user_features: torch.Tensor = None) -> torch.Tensor:
        """
        user_idx      : (B,)                    long
        user_features : (B, num_user_features)  float, required when the tower
                                                was built with features enabled
        """
        x = self.embedding(user_idx)               # (B, embed_dim)
        if self.num_user_features > 0:
            if user_features is None:
                raise ValueError(
                    f"UserTower was built with num_user_features="
                    f"{self.num_user_features} but forward() got no features."
                )
            x = torch.cat([x, user_features], dim=-1)
        x = self.net(x)                            # (B, 128)
        return F.normalize(x, dim=-1)              # unit-sphere → dot == cosine


class ItemTower(nn.Module):
    """
    Combines learnable ID embedding with content features:
      - genre multi-hot vector (19-dim)
      - avg_rating and popularity (2-dim)
    Total input to FC: 64 + 19 + 2 = 85
    """
    def __init__(self, num_items: int, num_genres: int = 19,
                 embed_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embed_dim)
        feature_dim = embed_dim + num_genres + 2   # 85
        self.net = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, item_idx: torch.Tensor,
                item_features: torch.Tensor) -> torch.Tensor:
        """
        item_idx      : (B,)       long
        item_features : (B, 21)    float  [genre_vec(19) | avg_rating | popularity]
        """
        x = self.embedding(item_idx)               # (B, 64)
        x = torch.cat([x, item_features], dim=-1)  # (B, 85)
        x = self.net(x)                             # (B, 128)
        return F.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int,
                 num_genres: int = 19, embed_dim: int = 64, output_dim: int = 128,
                 num_user_features: int = 0):
        super().__init__()
        self.num_user_features = num_user_features
        self.user_tower = UserTower(num_users, embed_dim, output_dim, num_user_features)
        self.item_tower = ItemTower(num_items, num_genres, embed_dim, output_dim)
        # learnable temperature — model decides how sharp the distribution should be
        self.log_temp = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        # clamp prevents temperature from collapsing to 0 or exploding
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def forward(self, user_idx, item_idx, item_features, user_features=None):
        user_emb = self.user_tower(user_idx, user_features)
        item_emb = self.item_tower(item_idx, item_features)
        return user_emb, item_emb


def build_from_checkpoint(ckpt: dict) -> TwoTowerModel:
    """
    Rebuild a model matching a saved checkpoint's architecture.

    Checkpoints written before user content features existed have no
    "num_user_features" key; defaulting to 0 keeps them loadable.
    """
    model = TwoTowerModel(
        num_users         = ckpt["num_users"],
        num_items         = ckpt["num_items"],
        num_genres        = ckpt["num_genres"],
        embed_dim         = ckpt["embed_dim"],
        output_dim        = ckpt["output_dim"],
        num_user_features = ckpt.get("num_user_features", 0),
    )
    model.load_state_dict(ckpt["model_state"])
    return model
