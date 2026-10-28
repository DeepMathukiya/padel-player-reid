"""Find interaction events: two players overlapping in the image or standing close.

During an interaction the tracker's motion model cannot be trusted to keep the
right id on the right person, so tracklets are cut at interaction boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - intersection
    return intersection / np.maximum(union, 1e-6)


def overlap_fraction(boxes: np.ndarray) -> np.ndarray:
    """Per box: the largest fraction of its own area covered by another box."""
    if len(boxes) < 2:
        return np.zeros(len(boxes))
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    np.fill_diagonal(intersection, 0)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return (intersection / np.maximum(area[:, None], 1e-6)).max(axis=1)


def detect_interactions(detections: pd.DataFrame, iou_threshold: float,
                        foot_distance_m: float) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (row in interaction flag, row overlap fraction, interaction pairs table)."""
    in_interaction = np.zeros(len(detections), dtype=bool)
    overlap = np.zeros(len(detections))
    pairs = []
    boxes_all = detections[["x1", "y1", "x2", "y2"]].to_numpy()
    court_all = detections[["court_x", "court_y"]].to_numpy()
    track_all = detections["raw_track_id"].to_numpy()
    for frame, index in detections.groupby("frame").indices.items():
        if len(index) < 2:
            continue
        boxes = boxes_all[index]
        overlap[index] = overlap_fraction(boxes)
        iou = pairwise_iou(boxes)
        distance = np.linalg.norm(court_all[index, None, :] - court_all[None, index, :], axis=2)
        close = (iou > iou_threshold) | (distance < foot_distance_m)
        np.fill_diagonal(close, False)
        for a, b in zip(*np.where(np.triu(close))):
            in_interaction[index[a]] = in_interaction[index[b]] = True
            pairs.append((frame, track_all[index[a]], track_all[index[b]], iou[a, b], distance[a, b]))
    pair_table = pd.DataFrame(pairs, columns=["frame", "track_a", "track_b", "iou", "foot_distance_m"])
    return in_interaction, overlap, pair_table


def group_interaction_events(pair_table: pd.DataFrame, max_gap_frames: int = 5) -> pd.DataFrame:
    """Collapse per-frame interacting pairs into events (start, end, players involved)."""
    events = []
    if pair_table.empty:
        return pd.DataFrame(columns=["start_frame", "end_frame", "track_a", "track_b"])
    for (a, b), group in pair_table.groupby(["track_a", "track_b"]):
        frames = np.sort(group["frame"].unique())
        start = prev = frames[0]
        for f in frames[1:]:
            if f - prev > max_gap_frames:
                events.append((start, prev, a, b))
                start = f
            prev = f
        events.append((start, prev, a, b))
    return pd.DataFrame(events, columns=["start_frame", "end_frame", "track_a", "track_b"]).sort_values("start_frame")
