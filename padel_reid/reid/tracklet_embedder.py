"""Embed a subsample of every tracklet's detections (deep + colour + quality).

The video is decoded once, sequentially; frames with nothing to embed are only
grabbed, not decoded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from padel_reid.common.cache_paths import CachePaths
from padel_reid.common.video_io import iterate_frames
from padel_reid.reid.color_descriptor import color_descriptor
from padel_reid.reid.crop_quality import crop_box, crop_quality
from padel_reid.reid.embedding_model import ReidEmbedder


def select_rows_to_embed(table: pd.DataFrame, stride: int) -> np.ndarray:
    """Every `stride`-th detection of each tracklet, always including the first and last."""
    position = table.groupby("tracklet_id").cumcount().to_numpy()
    size = table.groupby("tracklet_id")["frame"].transform("size").to_numpy()
    keep = (position % stride == 0) | (position == size - 1)
    return np.where(keep)[0]


def embed_tracklet_detections(cfg, table: pd.DataFrame, paths: CachePaths) -> int:
    """Write embeddings.npz keyed by row index into the tracklet table; returns frames decoded."""
    embedder = ReidEmbedder(cfg.reid)
    rows = select_rows_to_embed(table, cfg.reid.embed_stride)
    rows_by_frame = pd.Series(rows).groupby(table["frame"].to_numpy()[rows]).apply(list).to_dict()

    out_rows, out_embed, out_color, out_quality = [], [], [], []
    pending_crops, pending_rows = [], []

    def flush() -> None:
        if not pending_crops:
            return
        out_embed.append(embedder.embed(pending_crops).astype(np.float16))
        out_rows.extend(pending_rows)
        pending_crops.clear()
        pending_rows.clear()

    columns = table[["x1", "y1", "x2", "y2", "confidence", "overlap", "truncated_bottom"]].to_numpy()
    frames = iterate_frames(cfg.video_path, start=min(rows_by_frame), wanted=set(rows_by_frame))
    decoded = 0
    for frame_index, frame in tqdm(frames, total=len(rows_by_frame), desc="embed", unit="frame"):
        decoded += 1
        for row in rows_by_frame[frame_index]:
            x1, y1, x2, y2, conf, overlap, truncated = columns[row]
            crop = crop_box(frame, x1, y1, x2, y2)
            if crop is None:
                continue
            out_color.append(color_descriptor(crop).astype(np.float16))
            out_quality.append(crop_quality(crop, y2 - y1, conf, overlap, bool(truncated)))
            pending_crops.append(crop)
            pending_rows.append(row)
            if len(pending_crops) >= cfg.reid.batch_size:
                flush()
    flush()

    np.savez_compressed(
        paths.embeddings,
        rows=np.asarray(out_rows, dtype=np.int64),
        embeddings=np.concatenate(out_embed) if out_embed else np.zeros((0, 512), np.float16),
        colors=np.asarray(out_color, dtype=np.float16),
        quality=np.asarray(out_quality, dtype=np.float32),
    )
    return decoded
