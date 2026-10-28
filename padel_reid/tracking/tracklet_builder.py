"""Cut raw tracker tracks into short, high-purity tracklets.

A raw track is cut whenever it enters or leaves an interaction and whenever the
tracker bridged a long gap. Each piece is then very likely to contain one person
only; *who* that person is gets decided later by Re-ID + global assignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from padel_reid.tracking.interaction_detector import detect_interactions

ENDPOINT_FRAMES = 5


def _pad_flags(frames: np.ndarray, flags: np.ndarray, pad: int) -> np.ndarray:
    """Grow True runs by `pad` frames on both sides (within one track)."""
    if pad <= 0 or not flags.any():
        return flags
    flagged_frames = frames[flags]
    distance = np.abs(frames[:, None] - flagged_frames[None, :]).min(axis=1)
    return distance <= pad


def build_tracklets(detections: pd.DataFrame, tracking_cfg, max_gap_frames: int = 10) -> pd.DataFrame:
    """Add `tracklet_id`, `in_interaction` and `overlap` columns to the detections."""
    table = detections.sort_values(["raw_track_id", "frame"]).reset_index(drop=True)
    in_interaction, overlap, _ = detect_interactions(
        table, tracking_cfg.interaction_iou, tracking_cfg.interaction_foot_distance_m)
    table["overlap"] = overlap
    table["in_interaction"] = in_interaction

    tracklet_ids = np.empty(len(table), dtype=np.int64)
    next_id = 0
    for _, index in table.groupby("raw_track_id").indices.items():
        frames = table["frame"].to_numpy()[index]
        flags = _pad_flags(frames, table["in_interaction"].to_numpy()[index],
                           tracking_cfg.interaction_pad_frames)
        table.loc[index, "in_interaction"] = flags
        cut = np.zeros(len(index), dtype=bool)
        cut[0] = True
        cut[1:] |= flags[1:] != flags[:-1]
        cut[1:] |= np.diff(frames) > max_gap_frames
        segment = np.cumsum(cut) - 1
        tracklet_ids[index] = next_id + segment
        next_id += segment[-1] + 1
    table["tracklet_id"] = tracklet_ids

    sizes = table.groupby("tracklet_id")["frame"].transform("size")
    table = table[sizes >= tracking_cfg.min_tracklet_frames].copy()
    # Re-number densely so ids can index arrays.
    table["tracklet_id"] = pd.factorize(table["tracklet_id"], sort=True)[0]
    return table.sort_values(["frame", "tracklet_id"]).reset_index(drop=True)


def summarize_tracklets(table: pd.DataFrame, frame_size: tuple[int, int], border_fraction: float) -> pd.DataFrame:
    """One row per tracklet: time span, position, side, quality and border flags."""
    width, height = frame_size
    margin_x, margin_y = border_fraction * width, border_fraction * height

    def touches_border(row) -> bool:
        return (row.x1 <= margin_x or row.x2 >= width - margin_x
                or row.y2 >= height - margin_y or row.y1 <= margin_y)

    records = []
    for tracklet_id, group in table.groupby("tracklet_id", sort=True):
        group = group.sort_values("frame")
        first, last = group.iloc[0], group.iloc[-1]
        # Endpoints as the median of a few frames: single-frame foot positions jump
        # when a player bends or is cut off by the frame.
        head, tail = group.iloc[:ENDPOINT_FRAMES], group.iloc[-ENDPOINT_FRAMES:]
        records.append({
            "tracklet_id": tracklet_id,
            "raw_track_id": int(first.raw_track_id),
            "start_frame": int(first.frame),
            "end_frame": int(last.frame),
            "frames": len(group),
            "near_fraction": float(group["side"].mean()),
            "court_x": float(group["court_x"].median()),
            "court_y": float(group["court_y"].median()),
            "start_court_x": float(head["court_x"].median()), "start_court_y": float(head["court_y"].median()),
            "end_court_x": float(tail["court_x"].median()), "end_court_y": float(tail["court_y"].median()),
            "start_truncated": bool(head["truncated_bottom"].any()),
            "end_truncated": bool(tail["truncated_bottom"].any()),
            "box_height": float((group["y2"] - group["y1"]).median()),
            "in_interaction": bool(group["in_interaction"].mean() > 0.5),
            "truncated_fraction": float(group["truncated_bottom"].mean()),
            "starts_at_border": touches_border(first),
            "ends_at_border": touches_border(last),
        })
    return pd.DataFrame(records)
