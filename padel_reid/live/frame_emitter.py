"""Show frames `live.lag_seconds` late, with the identities decided by then.

Frames wait in a ring buffer until they are `lag` frames old and the identity
thread has solved a window that reaches past them. When a frame leaves:
  * every labelled tracklet in it is *committed*: its player is fixed for good,
    so an ID never changes after it has been shown;
  * short gaps are filled and boxes smoothed over the buffered slice, which still
    holds the frames after it (the lag makes the offline post-processing causal);
  * boxes, mini-map and status panel are drawn and the frame goes to the display
    window and/or the output video (written on its own thread).

The post-processing mirrors identity/gap_interpolation.py in numpy: it runs for
almost every frame in real time, where the pandas version would be too slow.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque, namedtuple
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from padel_reid.common.video_io import VideoWriter
from padel_reid.live.live_history import LiveHistory
from padel_reid.live.online_tracklet_builder import TRACKLET_COLUMNS
from padel_reid.rendering.frame_overlay import PLAYER_COLORS, draw_badge, draw_overlay

ASSIGNED_COLUMNS = ["frame", "player_id", "x1", "y1", "x2", "y2", "court_x", "court_y",
                    "tracklet_id", "confidence", "interpolated"]
Box = namedtuple("Box", ASSIGNED_COLUMNS)
_COLUMNS = [TRACKLET_COLUMNS.index(c) for c in ("frame", "tracklet_id", "confidence",
                                                 "x1", "y1", "x2", "y2", "court_x", "court_y")]
_TRACKLET = TRACKLET_COLUMNS.index("tracklet_id")
WINDOW_TITLE = "padel_reid live (q to quit)"


def post_process_slice(rows: list[tuple], labels: dict[int, int], exits: set[tuple[int, int]], first: int,
                       last: int, max_gap: int, smoothing_window: int) -> dict[int, list[Box]]:
    """Boxes to draw for frames first..last: one per player, short gaps filled, boxes smoothed.

    rows: TRACKLET_COLUMNS tuples around [first, last] (with frames after `last` for the
    look-ahead). exits: (player, frame) where a player left the view: no gap filling after it.
    Unlabelled rows come back with player_id 0.
    """
    out: dict[int, list[Box]] = {}
    if not rows:
        return out
    data = np.array([[r[c] for c in _COLUMNS] for r in rows], np.float64)
    frames, tracklets, conf = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2]
    players = np.array([labels.get(t, -1) + 1 for t in tracklets])
    in_range = (frames >= first) & (frames <= last)
    for i in np.flatnonzero(in_range & (players == 0)):
        out.setdefault(frames[i], []).append(Box(frames[i], 0, *data[i, 3:9], tracklets[i], conf[i], False))

    for player in range(1, len(PLAYER_COLORS) + 1):
        sel = np.flatnonzero(players == player)
        if not len(sel):
            continue
        sel = sel[np.lexsort((-conf[sel], frames[sel]))]
        sel = sel[np.r_[True, np.diff(frames[sel]) > 0]]            # most confident box per frame
        f, values = frames[sel], data[sel, 3:9]                     # x1, y1, x2, y2, court_x, court_y
        t, c, interpolated = tracklets[sel], conf[sel], np.zeros(len(sel), bool)
        gaps = np.flatnonzero((np.diff(f) > 1) & (np.diff(f) <= max_gap + 1))
        gaps = [i for i in gaps if (player, int(f[i])) not in exits]
        if gaps:
            new_f = np.concatenate([np.arange(f[i] + 1, f[i + 1]) for i in gaps])
            new_v = np.concatenate([values[i] + (np.arange(1, f[i + 1] - f[i])[:, None] / (f[i + 1] - f[i]))
                                    * (values[i + 1] - values[i]) for i in gaps])
            order = np.argsort(np.concatenate([f, new_f]), kind="stable")
            f = np.concatenate([f, new_f])[order]
            values = np.concatenate([values, new_v])[order]
            t = np.concatenate([t, np.full(len(new_f), -1)])[order]
            c = np.concatenate([c, np.zeros(len(new_f))])[order]
            interpolated = np.concatenate([interpolated, np.ones(len(new_f), bool)])[order]
        if smoothing_window >= 5:
            for run in np.split(np.arange(len(f)), np.flatnonzero(np.diff(f) > 1) + 1):
                if len(run) >= smoothing_window:
                    values[run, :4] = savgol_filter(values[run, :4], smoothing_window, 2, axis=0)
        for i in np.flatnonzero((f >= first) & (f <= last)):
            out.setdefault(int(f[i]), []).append(Box(int(f[i]), player, *values[i], int(t[i]), c[i],
                                                     bool(interpolated[i])))
    return out


class _VideoSink(threading.Thread):
    """Encodes the output video off the perception thread (~7 ms per 1080p frame)."""

    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]):
        super().__init__(daemon=True)
        self.writer = VideoWriter(path, fps, frame_size)
        self.queue: queue.Queue = queue.Queue(maxsize=64)

    def run(self) -> None:
        while (frame := self.queue.get()) is not None:
            self.writer.write(frame)
        self.writer.close()

    def close(self) -> None:
        self.queue.put(None)
        self.join()


class FrameEmitter:
    def __init__(self, cfg, history: LiveHistory, fps: float, frame_size: tuple[int, int],
                 output_video: Path | None, display: bool):
        self.cfg, self.history, self.fps = cfg, history, fps
        self.lag = int(round(cfg.live.lag_seconds * fps))
        self.lookahead = max(self.lag // 3, 1)       # a solve must have seen this far past a frame
        self.interp = cfg.identity.interpolate_max_gap_frames
        self.side_window_frames = int(cfg.identity.window_seconds * fps)
        self.frames: dict[int, tuple[np.ndarray, float]] = {}
        self.sink = _VideoSink(output_video, fps, frame_size) if output_video else None
        if self.sink:
            self.sink.start()
        self.display = display
        self.display_width = cfg.live.display_width
        self.last_tracklet: dict[int, int] = {}       # player -> last tracklet shown
        self.assigned: list[Box] = []
        self.delays: list[float] = []
        self.stopped = False                          # user pressed q / Esc
        self.shown_at: deque = deque(maxlen=int(fps) + 1)   # wall times of the last second of shown frames

    def add_frame(self, frame_index: int, frame: np.ndarray, arrival: float) -> None:
        self.frames[frame_index] = (frame, arrival)

    def frame(self, frame_index: int) -> np.ndarray | None:
        entry = self.frames.get(frame_index)
        return entry[0] if entry else None

    def ready(self, newest: int, final: bool = False) -> list[int]:
        """Buffered frames that may be shown now."""
        if final:
            return sorted(self.frames)
        with self.history.lock:
            solved = self.history.state.solved_through
        return sorted(f for f in self.frames if f <= newest - self.lag and f + self.lookahead <= solved)

    def emit(self, indices: list[int]) -> None:
        if not indices:
            return
        first, last = indices[0], indices[-1]
        history, state = self.history, self.history.state
        with history.lock:
            for f in indices:                                         # commit what is about to be shown
                for i in history.rows_by_frame.get(f, []):
                    t = history.rows[i][_TRACKLET]
                    if t not in state.committed and t in state.latest_labels:
                        state.committed[t] = state.latest_labels[t]
                        if t in state.latest_confident:
                            state.committed_confident.add(t)
            for w, a_near in state.latest_sides.items():              # sides of fully shown windows
                if (w + 1) * self.side_window_frames - 1 <= last and w not in state.committed_sides:
                    state.committed_sides[w] = a_near
            labels = {**state.latest_labels, **state.committed}
            start = first - self.interp - 1
            rows = [history.rows[i] for f in range(start, max(history.frontier, last) + 1)
                    for i in history.rows_by_frame.get(f, [])]
            exits = {(labels[t] + 1, int(s["end_frame"])) for t, s in history.closed_summary.items()
                     if t in labels and s["ends_at_border"] and s["end_frame"] >= start}
            last_ended = {p: history.closed_summary.get(t) for p, t in self.last_tracklet.items()}
            note = None if state.reference else (
                "identifying players: waiting for a player in clear view" if self.cfg.live.region_numbering
                else "identifying players: waiting for all 4 in clear view")

        boxes = post_process_slice(rows, labels, exits, first, last, self.interp,
                                   self.cfg.identity.smoothing_window)
        for f in indices:
            frame, arrival = self.frames.pop(f)
            frame_boxes = boxes.get(f, [])
            status = {p: "out of frame" if (s := last_ended.get(p)) and s["ends_at_border"] else "not detected"
                      for p in PLAYER_COLORS}
            draw_overlay(frame, frame_boxes, status, f, self.fps, self.cfg.rendering.draw_minimap, note)
            now = time.perf_counter()
            self.shown_at.append(now)
            if len(self.shown_at) > 1:
                shown_fps = (len(self.shown_at) - 1) / max(self.shown_at[-1] - self.shown_at[0], 1e-6)
                draw_badge(frame, f"{shown_fps:5.1f} FPS  |  delay {now - arrival:.1f} s")
            for box in frame_boxes:
                if box.player_id:
                    self.assigned.append(box)
                    if box.tracklet_id >= 0:
                        self.last_tracklet[box.player_id] = box.tracklet_id
            self._output(frame)
            self.delays.append(time.perf_counter() - arrival)

    def _output(self, frame: np.ndarray) -> None:
        if self.sink is not None:
            self.sink.queue.put(frame)
        if self.display and not self.stopped:
            height, width = frame.shape[:2]
            scale = min(1.0, self.display_width / width)
            view = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1 else frame
            cv2.imshow(WINDOW_TITLE, view)
            self.poll_keys()

    def poll_keys(self) -> None:
        """q / Esc stops the session and closes the window at once (also call while no frame is shown)."""
        if not self.display or self.stopped:
            return
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            self.stopped = True
            cv2.destroyAllWindows()
            print("stopped by user: finishing up and saving the session to the cache...")

    def assigned_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.assigned, columns=ASSIGNED_COLUMNS)

    def close(self) -> None:
        if self.sink is not None:
            self.sink.close()
        if self.display:
            cv2.destroyAllWindows()
