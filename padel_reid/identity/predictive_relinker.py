"""Constant-velocity gap re-linking with a gate calibrated on this video.

Adapted from Baek & Yun (2026), "Bridging the gap: doubles badminton analysis with
singles-trained models": when a tracklet ends, extrapolate
    S(t_s + t) = S(t_s) + [S(t_s) - S(t_s - 1)] * t
and accept a new tracklet that starts within a radius of the prediction.

Differences from the paper: the radius is in court metres (not 200 px), depends on
the gap length, and is calibrated by deleting frames from clean tracklets and
measuring the prediction error (95th percentile). Motion is only a *gate* here;
appearance decides between the candidates inside it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VELOCITY_FRAMES = 5     # velocity = mean displacement over the last few frames (less jitter)


def _velocity(xy: np.ndarray) -> np.ndarray:
    n = min(VELOCITY_FRAMES, len(xy) - 1)
    return (xy[-1] - xy[-1 - n]) / n if n > 0 else np.zeros(2)


@dataclass
class GapGate:
    gap_frames: np.ndarray          # calibrated gap lengths
    radius_m: np.ndarray            # 95th-percentile prediction error per gap length
    fps: float
    max_speed_mps: float

    def radius(self, gap_frames: int) -> float:
        if gap_frames <= self.gap_frames[-1]:
            return float(np.interp(gap_frames, self.gap_frames, self.radius_m))
        # Beyond the calibrated range grow at the physical speed limit.
        extra = (gap_frames - self.gap_frames[-1]) / self.fps * self.max_speed_mps
        return float(self.radius_m[-1] + extra)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({"gap_frames": self.gap_frames.tolist(), "radius_m": self.radius_m.tolist(),
                                    "fps": self.fps, "max_speed_mps": self.max_speed_mps}, indent=2))


def default_gate(fps: float, max_speed_mps: float) -> GapGate:
    gaps = np.arange(1, 61)
    return GapGate(gaps, 0.5 + gaps / fps * max_speed_mps, fps, max_speed_mps)


def simulate_gap_errors(table: pd.DataFrame, max_gap: int = 60, min_length: int = 90,
                        samples_per_tracklet: int = 4, seed: int = 0) -> pd.DataFrame:
    """Delete `gap` frames inside clean tracklets and measure the extrapolation error."""
    rng = np.random.default_rng(seed)
    records = []
    clean = table[~table["in_interaction"] & ~table["truncated_bottom"]]
    for _, group in clean.groupby("tracklet_id"):
        if len(group) < min_length:
            continue
        frames = group["frame"].to_numpy()
        xy = group[["court_x", "court_y"]].to_numpy()
        for _ in range(samples_per_tracklet):
            cut = int(rng.integers(VELOCITY_FRAMES + 1, len(group) - max_gap - 1))
            v = _velocity(xy[:cut + 1])
            for gap in range(1, max_gap + 1):
                target = cut + gap
                dt = frames[target] - frames[cut]
                error = np.linalg.norm(xy[cut] + v * dt - xy[target])
                records.append((gap, error))
    return pd.DataFrame(records, columns=["gap_frames", "error_m"])


def calibrate_gate(table: pd.DataFrame, fps: float, max_speed_mps: float, percentile: float = 95) -> tuple[GapGate, pd.DataFrame]:
    errors = simulate_gap_errors(table)
    if errors.empty:
        return default_gate(fps, max_speed_mps), errors
    per_gap = errors.groupby("gap_frames")["error_m"].quantile(percentile / 100).sort_index()
    radius = np.maximum.accumulate(per_gap.to_numpy()) + 0.3   # monotone, plus box-jitter slack
    return GapGate(per_gap.index.to_numpy(), radius, fps, max_speed_mps), errors


@dataclass
class RelinkCandidate:
    earlier: int        # tracklet that ended
    later: int          # tracklet that started
    gap_frames: int
    distance_m: float   # between the prediction and the later tracklet's start
    within_gate: bool


def find_relink_candidates(table: pd.DataFrame, summary: pd.DataFrame, gate: GapGate,
                           max_gap_frames: int) -> list[RelinkCandidate]:
    """All (ended, started) tracklet pairs within max_gap_frames, scored by the gate."""
    tails = {t: g[["frame", "court_x", "court_y"]].to_numpy()[-(VELOCITY_FRAMES + 1):]
             for t, g in table.groupby("tracklet_id")}
    starts = summary.sort_values("start_frame")
    start_frames = starts["start_frame"].to_numpy()
    candidates = []
    for earlier in summary.itertuples():
        lo = np.searchsorted(start_frames, earlier.end_frame + 1)
        hi = np.searchsorted(start_frames, earlier.end_frame + max_gap_frames, side="right")
        tail = tails[earlier.tracklet_id]
        v = _velocity(tail[:, 1:])
        for later in starts.iloc[lo:hi].itertuples():
            gap = int(later.start_frame - earlier.end_frame)
            predicted = tail[-1, 1:] + v * gap
            distance = float(np.linalg.norm(predicted - np.array([later.start_court_x, later.start_court_y])))
            candidates.append(RelinkCandidate(int(earlier.tracklet_id), int(later.tracklet_id), gap, distance,
                                              distance <= gate.radius(gap)))
    return candidates
