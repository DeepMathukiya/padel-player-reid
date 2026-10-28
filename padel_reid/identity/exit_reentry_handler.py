"""Players leaving the camera view and coming back (ball fetch, break, changeover walk).

Handled as its own Re-ID event, separate from occlusion:
  * EXIT  - a tracklet ends touching the frame border, outside any interaction.
  * ENTER - a tracklet starts touching the frame border.
While a player is out, motion gating means nothing, so exit->enter pairs are exempt
from the speed constraint, and the re-entering tracklet's identity comes from
appearance restricted to the players currently absent (enforced in the global
assignment through the per-frame exclusivity constraint).
"""

from __future__ import annotations

import pandas as pd


def is_exit(summary_row) -> bool:
    return bool(summary_row.ends_at_border) and not bool(summary_row.in_interaction)


def is_entry(summary_row) -> bool:
    return bool(summary_row.starts_at_border) and not bool(summary_row.in_interaction)


def exit_entry_flags(summary: pd.DataFrame) -> tuple[set[int], set[int]]:
    exits = {int(r.tracklet_id) for r in summary.itertuples() if is_exit(r)}
    entries = {int(r.tracklet_id) for r in summary.itertuples() if is_entry(r)}
    return exits, entries


def find_exit_reentry_events(summary: pd.DataFrame, labels: dict[int, int], fps: float,
                             min_absence_frames: int = 15) -> list[dict]:
    """After assignment: one event per (player left at border, came back at border)."""
    labelled = summary[summary["tracklet_id"].isin(labels)].copy()
    labelled["player"] = labelled["tracklet_id"].map(labels)
    events = []
    for player, group in labelled.sort_values("start_frame").groupby("player"):
        rows = list(group.itertuples())
        for before, after in zip(rows, rows[1:]):
            gap = after.start_frame - before.end_frame
            if gap >= min_absence_frames and before.ends_at_border and after.starts_at_border:
                events.append({
                    "player": int(player) + 1,
                    "exit_frame": int(before.end_frame),
                    "enter_frame": int(after.start_frame),
                    "absent_seconds": round(gap / fps, 2),
                    "exit_court_xy": [round(before.end_court_x, 2), round(before.end_court_y, 2)],
                    "enter_court_xy": [round(after.start_court_x, 2), round(after.start_court_y, 2)],
                    "exit_tracklet": int(before.tracklet_id),
                    "enter_tracklet": int(after.tracklet_id),
                })
    return events
