"""Aggregate per-crop embeddings into one descriptor per tracklet.

A tracklet is never matched by a single crop: its query is the quality-weighted
mean of its crop embeddings, which cancels blur and odd poses in single frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrackletDescriptors:
    tracklet_ids: np.ndarray        # (T,)
    embedding: np.ndarray           # (T, 512) L2-normalised quality-weighted mean
    color: np.ndarray               # (T, D) L2-normalised quality-weighted mean
    evidence: np.ndarray            # (T,) summed crop quality (how much to trust it)
    crop_rows: dict                 # tracklet_id -> (rows, embeddings, quality) for gallery banks


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def side_means(features: np.ndarray, side: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """(2, D) quality-weighted mean of the L2-normalised crop descriptors per half (far = 0,
    near = 1); zero for a half with fewer than 2 crops, which then stays uncentred."""
    features = _normalize(features)
    means = np.zeros((2, features.shape[1]), np.float32)
    for value in (0, 1):
        mask = side == value
        if mask.sum() >= 2:
            means[value] = (features[mask] * weight[mask, None]).sum(0) / weight[mask].sum()
    return means


def center_by_side(features: np.ndarray, side: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Subtract each court half's mean crop descriptor (side_means), then re-normalise.

    Near crops are large and sharp, far crops small and blurry, so raw descriptors
    mostly encode the half a crop came from: any two near players look more alike
    than one player seen near and then far. After a changeover that makes the
    newcomers on a half match whoever stood there before. Removing the per-half mean
    leaves what tells the people apart, so a player matches themselves across halves.
    """
    return _normalize(_normalize(features) - means[side.astype(int)])


def build_tracklet_descriptors(table: pd.DataFrame, embeddings_npz, num_tracklets: int,
                               side_centering: bool = False,
                               means: tuple[np.ndarray, np.ndarray] | None = None) -> TrackletDescriptors:
    """Quality-weighted mean descriptor over all embedded crops of each tracklet.

    means: (embedding, colour) side_means to centre with; None = computed from these crops.
    Live mode passes its session-wide means so queries and gallery are centred alike.
    """
    rows = embeddings_npz["rows"]
    deep = embeddings_npz["embeddings"].astype(np.float32)
    color = embeddings_npz["colors"].astype(np.float32)
    quality = embeddings_npz["quality"].astype(np.float32) + 1e-3
    if side_centering and len(rows):
        side = table["side"].to_numpy()[rows]
        deep_means, color_means = means if means is not None else (side_means(deep, side, quality),
                                                                   side_means(color, side, quality))
        deep = center_by_side(deep, side, deep_means)
        color = center_by_side(color, side, color_means)
    tracklet_of_row = table["tracklet_id"].to_numpy()[rows]

    order = np.lexsort((rows, tracklet_of_row))
    groups = np.split(order, np.flatnonzero(np.diff(tracklet_of_row[order])) + 1) if len(order) else []

    embedding = np.zeros((num_tracklets, deep.shape[1]), np.float32)
    color_mean = np.zeros((num_tracklets, color.shape[1] if len(color) else 1), np.float32)
    evidence = np.zeros(num_tracklets, np.float32)
    crop_rows = {}
    for group in groups:
        t = int(tracklet_of_row[group[0]])
        w = quality[group]
        embedding[t] = (deep[group] * w[:, None]).sum(0) / w.sum()
        color_mean[t] = (color[group] * w[:, None]).sum(0) / w.sum()
        evidence[t] = w.sum()
        crop_rows[t] = (rows[group], deep[group], w)
    return TrackletDescriptors(np.arange(num_tracklets), _normalize(embedding), _normalize(color_mean),
                               evidence, crop_rows)
