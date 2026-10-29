"""Incremental version of tracking/tracklet_builder.py for live mode.

Detections arrive one frame at a time. A raw track is cut into a new tracklet when
its (padded) interaction flag flips or when it jumps more than `max_gap_frames`,
exactly as the batch builder does. The padding needs `interaction_pad_frames` of
lookahead, so a frame's rows are final that many frames after it arrived.
Tracklets shorter than `min_tracklet_frames` are dropped; the others get dense,
increasing ids at the moment they reach that length.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from padel_reid.court.court_side import court_side
from padel_reid.detection.player_detector import DETECTION_COLUMNS
from padel_reid.tracking.interaction_detector import overlap_fraction, pairwise_iou

TRACKLET_COLUMNS = DETECTION_COLUMNS + ["side", "overlap", "in_interaction", "tracklet_id"]
_RAW_TRACK, _COURT_Y = DETECTION_COLUMNS.index("raw_track_id"), DETECTION_COLUMNS.index("court_y")
_BOX = slice(DETECTION_COLUMNS.index("x1"), DETECTION_COLUMNS.index("y2") + 1)
_COURT = slice(DETECTION_COLUMNS.index("court_x"), _COURT_Y + 1)


@dataclass
class _Segment:
    flag: bool
    last_frame: int
    tracklet_id: int | None = None
    size: int = 0
    buffered: list = field(default_factory=list)     # (partial row, position) until the segment is long enough
    last_row: tuple | None = None


@dataclass
class BuilderOutput:
    rows: list = field(default_factory=list)          # (row in TRACKLET_COLUMNS order, position in tracklet)
    closed: list = field(default_factory=list)        # (tracklet_id, last position, last row)
    pairs: list = field(default_factory=list)         # (frame, track_a, track_b, iou, foot_distance_m)

    def extend(self, other: "BuilderOutput") -> None:
        self.rows += other.rows
        self.closed += other.closed
        self.pairs += other.pairs


class OnlineTrackletBuilder:
    def __init__(self, tracking_cfg, max_gap_frames: int = 10):
        self.iou_threshold = tracking_cfg.interaction_iou
        self.foot_distance_m = tracking_cfg.interaction_foot_distance_m
        self.pad = tracking_cfg.interaction_pad_frames
        self.min_frames = tracking_cfg.min_tracklet_frames
        self.max_gap = max_gap_frames
        self.pending: deque = deque()                   # (frame, [(row, raw_flag, overlap)])
        self.flags: dict[int, deque] = {}               # raw track -> recent (frame, raw_flag)
        self.segments: dict[int, _Segment] = {}         # raw track -> open segment
        self.next_id = 0

    def _interactions(self, frame_index: int, rows: list[tuple], out: BuilderOutput) -> tuple[np.ndarray, np.ndarray]:
        """Per-row raw interaction flag and overlap fraction, as detect_interactions computes them."""
        flags, overlap = np.zeros(len(rows), bool), np.zeros(len(rows))
        if len(rows) < 2:
            return flags, overlap
        boxes = np.array([r[_BOX] for r in rows], float)
        court = np.array([r[_COURT] for r in rows], float)
        overlap = overlap_fraction(boxes)
        iou = pairwise_iou(boxes)
        distance = np.linalg.norm(court[:, None, :] - court[None, :, :], axis=2)
        close = (iou > self.iou_threshold) | (distance < self.foot_distance_m)
        np.fill_diagonal(close, False)
        for a, b in zip(*np.where(np.triu(close))):
            flags[a] = flags[b] = True
            out.pairs.append((frame_index, rows[a][_RAW_TRACK], rows[b][_RAW_TRACK], iou[a, b], distance[a, b]))
        return flags, overlap

    def push(self, frame_index: int, rows: list[tuple]) -> BuilderOutput:
        """rows: this frame's tracked detections in DETECTION_COLUMNS order. Returns what became final."""
        out = BuilderOutput()
        flags, overlap = self._interactions(frame_index, rows, out)
        self.pending.append((frame_index, [(r, bool(f), float(o)) for r, f, o in zip(rows, flags, overlap)]))
        for r, f in zip(rows, flags):
            self.flags.setdefault(r[_RAW_TRACK], deque()).append((frame_index, bool(f)))
        while self.pending and self.pending[0][0] <= frame_index - self.pad:
            self._finalize(*self.pending.popleft(), out)
        return out

    def flush(self) -> BuilderOutput:
        """End of stream: finalize everything and close all open tracklets."""
        out = BuilderOutput()
        while self.pending:
            self._finalize(*self.pending.popleft(), out)
        for raw_track in list(self.segments):
            self._close(raw_track, out)
        return out

    def _finalize(self, frame_index: int, entries: list, out: BuilderOutput) -> None:
        for raw_track in [k for k, s in self.segments.items() if frame_index - s.last_frame > self.max_gap]:
            self._close(raw_track, out)
        for row, _, overlap in entries:
            raw_track = row[_RAW_TRACK]
            padded = any(f and abs(fr - frame_index) <= self.pad for fr, f in self.flags[raw_track])
            segment = self.segments.get(raw_track)
            if segment is None or segment.flag != padded:
                if segment is not None:
                    self._close(raw_track, out)
                segment = self.segments[raw_track] = _Segment(padded, frame_index)
            segment.last_frame = frame_index
            self._append(segment, (*row, int(court_side(row[_COURT_Y])), overlap, padded), out)
        for raw_track, recent in list(self.flags.items()):
            while recent and recent[0][0] < frame_index + 1 - self.pad:
                recent.popleft()
            if not recent and raw_track not in self.segments:
                del self.flags[raw_track]

    def _append(self, segment: _Segment, partial: tuple, out: BuilderOutput) -> None:
        position = segment.size
        segment.size += 1
        if segment.tracklet_id is None:
            segment.buffered.append((partial, position))
            if segment.size < self.min_frames:
                return
            segment.tracklet_id, self.next_id = self.next_id, self.next_id + 1
            for buffered, buffered_position in segment.buffered:
                out.rows.append(((*buffered, segment.tracklet_id), buffered_position))
            segment.buffered.clear()
        else:
            out.rows.append(((*partial, segment.tracklet_id), position))
        segment.last_row = out.rows[-1][0]

    def _close(self, raw_track: int, out: BuilderOutput) -> None:
        segment = self.segments.pop(raw_track)
        if segment.tracklet_id is not None:              # shorter segments are dropped, as in batch
            out.closed.append((segment.tracklet_id, segment.size - 1, segment.last_row))
