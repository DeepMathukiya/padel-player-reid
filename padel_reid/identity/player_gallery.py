"""The per-player appearance memory that new tracklets are matched against.

For each of the 4 players the gallery keeps:
  * a near-side and a far-side prototype (quality-weighted mean embedding), since a
    player looks very different large and sharp near the camera vs. small and
    blurry on the far side; and
  * a FIFO memory bank of recent high-quality crop embeddings, which catches
    looks the prototype averages away (pose, lighting).

score(player) = 0.5 * cos(query, prototype[side]) + 0.5 * max cos(query, bank)
then fused with the colour descriptor similarity.
"""

from __future__ import annotations

from collections import deque

import numpy as np

NUM_PLAYERS = 4


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


class PlayerGallery:
    def __init__(self, embed_dim: int, color_dim: int, bank_size: int, color_weight: float):
        self.color_weight = color_weight
        self.proto_sum = np.zeros((NUM_PLAYERS, 2, embed_dim), np.float64)    # [player, side]
        self.proto_weight = np.zeros((NUM_PLAYERS, 2), np.float64)
        self.color_sum = np.zeros((NUM_PLAYERS, color_dim), np.float64)
        self.color_weight_sum = np.zeros(NUM_PLAYERS, np.float64)
        self.banks = [deque(maxlen=bank_size) for _ in range(NUM_PLAYERS)]

    def add(self, player: int, side: int, crop_embeddings: np.ndarray, crop_quality: np.ndarray,
            color: np.ndarray, min_bank_quality: float = 0.3) -> None:
        """Add a confidently identified tracklet's evidence to a player's memory."""
        w = crop_quality[:, None]
        self.proto_sum[player, side] += (crop_embeddings * w).sum(0)
        self.proto_weight[player, side] += crop_quality.sum()
        self.color_sum[player] += color * crop_quality.sum()
        self.color_weight_sum[player] += crop_quality.sum()
        for emb, q in zip(crop_embeddings, crop_quality):
            if q >= min_bank_quality:
                self.banks[player].append(emb)

    def prototype(self, player: int, side: int) -> np.ndarray:
        """Side prototype, falling back to the other side if this side has no data yet."""
        if self.proto_weight[player, side] > 0:
            return _normalize(self.proto_sum[player, side])
        return _normalize(self.proto_sum[player].sum(0))

    def has_player(self, player: int) -> bool:
        return self.proto_weight[player].sum() > 0

    def scores(self, query: np.ndarray, query_color: np.ndarray, side: int) -> np.ndarray:
        """Similarity of one tracklet query to each player, roughly in [-1, 1]."""
        out = np.full(NUM_PLAYERS, -1.0)
        for p in range(NUM_PLAYERS):
            if not self.has_player(p):
                continue
            proto = float(query @ self.prototype(p, side))
            bank = float(np.max(np.stack(self.banks[p]) @ query)) if self.banks[p] else proto
            deep = 0.5 * proto + 0.5 * bank
            color = float(query_color @ _normalize(self.color_sum[p])) if self.color_weight_sum[p] > 0 else 0.0
            out[p] = (1 - self.color_weight) * deep + self.color_weight * color
        return out


def scores_to_log_prob(scores: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """Softmax over players of similarity / temperature, as log-probabilities."""
    z = scores / temperature
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
