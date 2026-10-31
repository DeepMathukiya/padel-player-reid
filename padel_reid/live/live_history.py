"""Everything the live run has seen so far, shared by perception, identity and emitter.

Perception appends tracklet rows and crop evidence; the identity thread takes
`snapshot`s of a recent time window in the same table/summary/descriptor formats
the offline modules use (with tracklet ids renumbered densely for the solver); the
emitter commits the labels it puts on screen. At the end `flush` writes the
offline cache files, so `refine`, `heldout` and `baselines` run on a live session
without decoding the video again.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from padel_reid.common.cache_paths import CachePaths
from padel_reid.court.court_side import court_side
from padel_reid.detection.player_detector import DETECTION_COLUMNS
from padel_reid.identity.tracklet_descriptors import (TrackletDescriptors, build_tracklet_descriptors,
                                                     center_by_side)
from padel_reid.live.online_tracklet_builder import TRACKLET_COLUMNS
from padel_reid.tracking.interaction_detector import group_interaction_events
from padel_reid.tracking.tracklet_builder import summarize_tracklets

_TRACKLET, _FRAME, _SIDE = (TRACKLET_COLUMNS.index(c) for c in ("tracklet_id", "frame", "side"))
PAIR_COLUMNS = ["frame", "track_a", "track_b", "iou", "foot_distance_m"]


@dataclass
class Snapshot:
    """A time window of the history with dense local tracklet ids 0..T-1."""
    table: pd.DataFrame
    summary: pd.DataFrame                  # indexed by local id
    descriptors: TrackletDescriptors
    evidence_seconds: np.ndarray
    global_ids: np.ndarray                 # local -> global tracklet id
    local_ids: dict[int, int]              # global -> local
    frontier: int                          # last frame whose rows were final when taken

    def to_local(self, labels: dict[int, int]) -> dict[int, int]:
        return {self.local_ids[t]: k for t, k in labels.items() if t in self.local_ids}

    def to_global(self, labels: dict[int, int]) -> dict[int, int]:
        return {int(self.global_ids[t]): k for t, k in labels.items()}


@dataclass
class IdentityState:
    """Decisions shared between the identity thread (latest_*) and the emitter (committed_*)."""
    latest_labels: dict[int, int] = field(default_factory=dict)
    latest_confident: set[int] = field(default_factory=set)
    latest_sides: dict[int, bool] = field(default_factory=dict)
    solved_through: int = -1               # frontier of the snapshot behind latest_*
    reference: dict[int, int] = field(default_factory=dict)      # symmetry-breaking fixed labels
    seeds: dict[int, int] = field(default_factory=dict)
    committed: dict[int, int] = field(default_factory=dict)      # shown on screen: never changes again
    committed_confident: set[int] = field(default_factory=set)
    committed_sides: dict[int, bool] = field(default_factory=dict)
    seeding: list[dict] = field(default_factory=list)


class LiveHistory:
    def __init__(self, fps: float, frame_size: tuple[int, int], border_fraction: float, embed_stride: int,
                 side_centering: bool = False):
        self.lock = threading.RLock()
        self.side_centering = side_centering
        # Session-wide quality-weighted sums of normalised crop descriptors per half (far 0, near 1):
        # snapshot queries and the gallery are centred with the same means (tracklet_descriptors).
        self.side_sums: list[np.ndarray] | None = None       # [embedding (2, D), colour (2, C)]
        self.side_weight = np.zeros(2)
        self.side_crops = np.zeros(2, np.int64)
        self.fps, self.frame_size = fps, frame_size
        self.border_fraction, self.embed_stride = border_fraction, embed_stride
        self.rows: list[tuple] = []
        self.rows_by_frame: dict[int, list[int]] = {}
        self.tracklet_rows: dict[int, list[int]] = {}
        self.near_rows: dict[int, int] = {}
        self.closed_summary: dict[int, dict] = {}
        self.crops: dict[int, list[tuple]] = {}          # tracklet -> [(row index, embedding, colour, quality)]
        self.detections: list[tuple] = []
        self.pairs: list[tuple] = []
        self.frontier = -1
        self.state = IdentityState()

    # ---- written by perception (main thread) ----

    def add_detections(self, rows: list[tuple], pairs: list[tuple]) -> None:
        with self.lock:
            self.detections += rows
            self.pairs += pairs

    def add_rows(self, rows: list[tuple[tuple, int]]) -> list[tuple[int, tuple, int]]:
        """Builder rows -> [(row index, row, position in tracklet)]."""
        out = []
        with self.lock:
            for row, position in rows:
                index = len(self.rows)
                self.rows.append(row)
                self.rows_by_frame.setdefault(row[_FRAME], []).append(index)
                t = row[_TRACKLET]
                self.tracklet_rows.setdefault(t, []).append(index)
                self.near_rows[t] = self.near_rows.get(t, 0) + int(row[_SIDE])
                out.append((index, row, position))
        return out

    def last_row_index(self, tracklet: int) -> int:
        with self.lock:
            return self.tracklet_rows[tracklet][-1]

    def add_crops(self, crops: list[tuple[int, np.ndarray, np.ndarray, float]]) -> None:
        with self.lock:
            for row_index, embedding, color, quality in crops:
                self.crops.setdefault(self.rows[row_index][_TRACKLET], []).append((row_index, embedding, color, quality))
                self._add_side_sums(int(self.rows[row_index][_SIDE]), embedding, color, quality)

    def _add_side_sums(self, side: int, embedding: np.ndarray, color: np.ndarray, quality: float) -> None:
        if self.side_sums is None:
            self.side_sums = [np.zeros((2, len(embedding))), np.zeros((2, len(color)))]
        w = float(quality) + 1e-3
        for sums, x in zip(self.side_sums, (embedding, color)):
            x = x.astype(np.float64)
            sums[side] += w * x / max(np.linalg.norm(x), 1e-6)
        self.side_weight[side] += w
        self.side_crops[side] += 1

    def side_means(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(embedding, colour) per-half means of every crop so far; None when centring is off."""
        with self.lock:
            if not self.side_centering or self.side_sums is None:
                return None
            ok = (self.side_crops >= 2)[:, None]
            return tuple(np.where(ok, sums / np.maximum(self.side_weight[:, None], 1e-6), 0.0).astype(np.float32)
                         for sums in self.side_sums)

    def centered_crops(self, crops: list[tuple], means: tuple[np.ndarray, np.ndarray] | None
                       ) -> tuple[np.ndarray, np.ndarray]:
        """(embeddings, colours) of a tracklet's crops, centred by each crop's half when means is given."""
        emb = np.stack([c[1] for c in crops]).astype(np.float32)
        color = np.stack([c[2] for c in crops]).astype(np.float32)
        if means is None:
            return emb, color
        with self.lock:
            side = np.asarray([self.rows[c[0]][_SIDE] for c in crops])
        return center_by_side(emb, side, means[0]), center_by_side(color, side, means[1])

    def close(self, tracklet: int) -> None:
        with self.lock:
            rows = [self.rows[i] for i in self.tracklet_rows[tracklet]]
        summary = summarize_tracklets(pd.DataFrame(rows, columns=TRACKLET_COLUMNS), self.frame_size, self.border_fraction)
        with self.lock:
            self.closed_summary[tracklet] = summary.iloc[0].to_dict()

    # ---- read by the identity thread ----

    def snapshot(self, window_frames: int) -> Snapshot | None:
        """Every tracklet with a row in the last `window_frames` frames (its rows up to the frontier).

        Rows after the frontier are left out: their crops may still wait to be embedded,
        and a solve that sees a tracklet's court side but not its appearance labels it
        from the side alone - wrong exactly when the teams change ends.
        """
        with self.lock:
            frontier = self.frontier
            tracklets = sorted(t for t, rows in self.tracklet_rows.items()
                               if self.rows[rows[0]][_FRAME] <= frontier
                               and self.rows[rows[-1]][_FRAME] >= frontier - window_frames)
            if not tracklets:
                return None
            row_index = [i for t in tracklets for i in self.tracklet_rows[t] if self.rows[i][_FRAME] <= frontier]
            rows = [self.rows[i] for i in row_index]
            crops = [c for t in tracklets for c in self.crops.get(t, []) if self.rows[c[0]][_FRAME] <= frontier]
            closed = {t: dict(self.closed_summary[t]) for t in tracklets
                      if t in self.closed_summary and self.closed_summary[t]["end_frame"] <= frontier}

        local_ids = {t: i for i, t in enumerate(tracklets)}
        table = pd.DataFrame(rows, columns=TRACKLET_COLUMNS)
        table["row_index"] = row_index
        table["tracklet_id"] = table["tracklet_id"].map(local_ids)
        table = table.sort_values(["frame", "tracklet_id"]).reset_index(drop=True)

        open_local = [local_ids[t] for t in tracklets if t not in closed]
        records = [{**s, "tracklet_id": local_ids[t]} for t, s in closed.items()]
        if open_local:
            records += summarize_tracklets(table[table["tracklet_id"].isin(open_local)], self.frame_size,
                                           self.border_fraction).to_dict("records")
        summary = pd.DataFrame(records).sort_values("tracklet_id").reset_index(drop=True)
        summary.index = summary["tracklet_id"].to_numpy()

        position = pd.Series(np.arange(len(table)), index=table["row_index"].to_numpy())
        embeddings = {
            "rows": position.loc[[c[0] for c in crops]].to_numpy() if crops else np.zeros(0, np.int64),
            "embeddings": np.stack([c[1] for c in crops]) if crops else np.zeros((0, 512), np.float16),
            "colors": np.stack([c[2] for c in crops]) if crops else np.zeros((0, 1), np.float16),
            "quality": np.asarray([c[3] for c in crops], np.float32),
        }
        means = self.side_means()
        descriptors = build_tracklet_descriptors(table, embeddings, len(tracklets), means is not None, means)
        evidence_seconds = descriptors.evidence * self.embed_stride / self.fps
        return Snapshot(table, summary, descriptors, evidence_seconds, np.asarray(tracklets), local_ids, frontier)

    def gallery_evidence(self, tracklets) -> dict[int, tuple[int, int, list]]:
        """tracklet -> (start frame, near side?, crops) for building a gallery from any tracklets."""
        out = {}
        with self.lock:
            for t in tracklets:
                rows = self.tracklet_rows.get(t)
                if rows and self.crops.get(t):
                    near = self.near_rows[t] / len(rows) > 0.5
                    out[t] = (self.rows[rows[0]][_FRAME], int(near), list(self.crops[t]))
        return out

    # ---- end of the session ----

    def flush(self, paths: CachePaths) -> None:
        """Write the offline cache files (same names and formats as the batch stages)."""
        with self.lock:
            detections = pd.DataFrame(self.detections, columns=DETECTION_COLUMNS)
            detections["side"] = court_side(detections["court_y"].to_numpy())
            detections.to_parquet(paths.detections)

            table = pd.DataFrame(self.rows, columns=TRACKLET_COLUMNS)
            table["row_index"] = np.arange(len(table))
            table = table.sort_values(["frame", "tracklet_id"]).reset_index(drop=True)
            new_position = np.empty(len(table), np.int64)
            new_position[table["row_index"].to_numpy()] = np.arange(len(table))
            table = table.drop(columns="row_index")
            table.to_parquet(paths.tracklets)
            summarize_tracklets(table, self.frame_size, self.border_fraction).to_parquet(paths.tracklet_summary)
            group_interaction_events(pd.DataFrame(self.pairs, columns=PAIR_COLUMNS)).to_parquet(paths.interactions)

            crops = [c for t in sorted(self.crops) for c in self.crops[t]]
            np.savez_compressed(
                paths.embeddings,
                rows=new_position[[c[0] for c in crops]] if crops else np.zeros(0, np.int64),
                embeddings=np.stack([c[1] for c in crops]) if crops else np.zeros((0, 512), np.float16),
                colors=np.stack([c[2] for c in crops]) if crops else np.zeros((0, 1), np.float16),
                quality=np.asarray([c[3] for c in crops], np.float32),
            )

    def write_events(self, path: Path, extra: dict) -> None:
        path.write_text(json.dumps({**extra, "seeding": self.state.seeding}, indent=2, default=int), encoding="utf-8")
