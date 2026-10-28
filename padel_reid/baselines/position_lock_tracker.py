"""Re-implementation of the Baek & Yun (2026) doubles tracker as a baseline.

Logic from their repo (utils_stgcn_transformer_64indim.py, detect_court_and_players):
  * the first time exactly 4 tracks are present, ids are locked by position:
    sort by box-centre y (2 smallest = back), then by x -> 1 back-left, 2 back-right,
    3 front-left, 4 front-right; the lock is never revised
  * a lost display id keeps a constant-velocity predicted position for <= 20 frames
  * a new tracker id takes the display id whose prediction is nearest within 200 px,
    else the lowest free display id
Identity therefore means "court position at lock time": every crossing the tracker
gets wrong and every changeover becomes a permanent swap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_PREDICT_FRAMES = 20
RADIUS_PX = 200.0


def run_position_lock(detections: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[int, int] = {}                 # raw track id -> display id 1..4
    last_pos: dict[int, np.ndarray] = {}         # display id -> last centre
    velocity: dict[int, np.ndarray] = {}
    lost_for: dict[int, int] = {}
    locked = False
    out = []
    for frame, group in detections.sort_values("frame").groupby("frame"):
        centres = {int(r.raw_track_id): np.array([(r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2]) for r in group.itertuples()}
        if not locked and len(centres) == 4:
            by_y = sorted(centres.items(), key=lambda kv: kv[1][1])
            back = sorted(by_y[:2], key=lambda kv: kv[1][0])
            front = sorted(by_y[2:], key=lambda kv: kv[1][0])
            for display, (raw, _) in enumerate(back + front, start=1):
                mapping[raw] = display
            locked = True
        if not locked:
            continue
        seen = set()
        for raw, centre in centres.items():
            if raw not in mapping:
                active = {mapping[r] for r in centres if r in mapping}
                best, best_dist = None, RADIUS_PX
                for display, pos in last_pos.items():
                    if display in active or lost_for.get(display, 0) > MAX_PREDICT_FRAMES:
                        continue
                    predicted = pos + velocity.get(display, 0) * lost_for.get(display, 0)
                    dist = float(np.linalg.norm(predicted - centre))
                    if dist < best_dist:
                        best, best_dist = display, dist
                if best is None:
                    free = [d for d in range(1, 5) if d not in active]
                    if not free:
                        continue
                    best = free[0]
                mapping = {r: d for r, d in mapping.items() if d != best or r in centres}
                mapping[raw] = best
            display = mapping[raw]
            if display in seen:
                continue
            seen.add(display)
            if display in last_pos and lost_for.get(display, 0) == 0:
                velocity[display] = centre - last_pos[display]
            last_pos[display] = centre
            lost_for[display] = 0
            row = group[group["raw_track_id"] == raw].iloc[0]
            out.append((frame, display, row.x1, row.y1, row.x2, row.y2, row.court_x, row.court_y))
        for display in last_pos:
            if display not in seen:
                lost_for[display] = lost_for.get(display, 0) + 1
    return pd.DataFrame(out, columns=["frame", "player_id", "x1", "y1", "x2", "y2", "court_x", "court_y"])
