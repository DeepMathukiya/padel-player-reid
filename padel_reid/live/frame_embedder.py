"""Re-ID evidence for live rows, cut from frames that are still in memory.

Same selection as the batch embedder (every `embed_stride`-th row of a tracklet,
plus its last row) and the same deep + colour + quality features, but no second
pass over the video. Crops are cut, and colour and quality computed, the moment a
row is final; the OSNet pass is batched (`live.embed_batch` crops, or after
`live.embed_max_wait_frames`) because each call costs ~20 ms of fixed overhead,
which at one frame per call would not fit the 33 ms real-time budget.
"""

from __future__ import annotations

import numpy as np

from padel_reid.live.online_tracklet_builder import TRACKLET_COLUMNS
from padel_reid.reid.color_descriptor import color_descriptor
from padel_reid.reid.crop_quality import crop_box, crop_quality
from padel_reid.reid.embedding_model import ReidEmbedder

_COL = {name: i for i, name in enumerate(TRACKLET_COLUMNS)}


class FrameEmbedder:
    def __init__(self, reid_cfg, live_cfg):
        self.embedder = ReidEmbedder(reid_cfg)
        self.stride = reid_cfg.embed_stride
        self.batch = live_cfg.embed_batch
        self.max_wait = live_cfg.embed_max_wait_frames
        self.pending: list[tuple[int, int, np.ndarray, np.ndarray, float]] = []   # row, frame, crop, colour, quality

    def wants(self, position: int) -> bool:
        return position % self.stride == 0

    def add(self, row_index: int, row: tuple, frame: np.ndarray) -> None:
        """Cut the crop now (the frame will be released); embed later in a batch."""
        x1, y1, x2, y2 = (row[_COL[c]] for c in ("x1", "y1", "x2", "y2"))
        crop = crop_box(frame, x1, y1, x2, y2)
        if crop is None:
            return
        quality = crop_quality(crop, y2 - y1, row[_COL["confidence"]], row[_COL["overlap"]],
                               bool(row[_COL["truncated_bottom"]]))
        self.pending.append((row_index, row[_COL["frame"]], crop.copy(), color_descriptor(crop).astype(np.float16),
                             quality))

    def oldest_pending_frame(self) -> int | None:
        """Frame of the oldest crop not yet embedded (the identity frontier must stay before it)."""
        return self.pending[0][1] if self.pending else None

    def flush(self, newest_frame: int, force: bool = False) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
        """(row index, embedding, colour, quality) for the pending crops, once a batch is due."""
        if not self.pending:
            return []
        if not force and len(self.pending) < self.batch and newest_frame - self.pending[0][1] < self.max_wait:
            return []
        pending, self.pending = self.pending, []
        embeddings = self.embedder.embed([p[2] for p in pending]).astype(np.float16)
        return [(p[0], e, p[3], p[4]) for p, e in zip(pending, embeddings)]
