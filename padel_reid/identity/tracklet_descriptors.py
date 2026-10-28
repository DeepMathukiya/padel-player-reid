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


def build_tracklet_descriptors(table: pd.DataFrame, embeddings_npz, num_tracklets: int) -> TrackletDescriptors:
    """Quality-weighted mean descriptor over all embedded crops of each tracklet."""
    rows = embeddings_npz["rows"]
    deep = embeddings_npz["embeddings"].astype(np.float32)
    color = embeddings_npz["colors"].astype(np.float32)
    quality = embeddings_npz["quality"].astype(np.float32) + 1e-3
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
