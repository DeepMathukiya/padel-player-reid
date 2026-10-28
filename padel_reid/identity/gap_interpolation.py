"""Fill short detection gaps per player and smooth boxes for rendering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

BOX_COLUMNS = ["x1", "y1", "x2", "y2"]


def fill_short_gaps(table: pd.DataFrame, max_gap_frames: int, exit_frames: set[tuple[int, int]]) -> pd.DataFrame:
    """Linearly interpolate boxes over gaps <= max_gap_frames for each player.

    Gaps that are known exit->re-entry absences are left empty (the player was
    really out of view). exit_frames holds (player, exit_frame) pairs.
    """
    filled = []
    for player, group in table.groupby("player_id"):
        group = group.sort_values("frame")
        frames = group["frame"].to_numpy()
        for i in np.where((np.diff(frames) > 1) & (np.diff(frames) <= max_gap_frames + 1))[0]:
            if (player, int(frames[i])) in exit_frames:
                continue
            a, b = group.iloc[i], group.iloc[i + 1]
            for f in range(int(frames[i]) + 1, int(frames[i + 1])):
                alpha = (f - frames[i]) / (frames[i + 1] - frames[i])
                row = {c: (1 - alpha) * a[c] + alpha * b[c] for c in BOX_COLUMNS + ["court_x", "court_y"]}
                row.update(frame=f, player_id=player, tracklet_id=-1, confidence=0.0, interpolated=True)
                filled.append(row)
    out = table.assign(interpolated=False)
    if filled:
        out = pd.concat([out, pd.DataFrame(filled)], ignore_index=True)
    return out.sort_values(["frame", "player_id"]).reset_index(drop=True)


def smooth_boxes(table: pd.DataFrame, window: int) -> pd.DataFrame:
    """Savitzky-Golay smoothing of box coordinates over each contiguous run."""
    if window < 5:
        return table
    table = table.copy()
    for _, group in table.groupby("player_id"):
        group = group.sort_values("frame")
        frames = group["frame"].to_numpy()
        runs = np.split(group.index.to_numpy(), np.where(np.diff(frames) > 1)[0] + 1)
        for run in runs:
            if len(run) >= window:
                table.loc[run, BOX_COLUMNS] = savgol_filter(table.loc[run, BOX_COLUMNS].to_numpy(), window, 2, axis=0)
    return table
