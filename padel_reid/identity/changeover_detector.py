"""Changeovers (teams switching ends), read from the solver's per-window side state."""

from __future__ import annotations


def changeovers_from_sides(team_a_near: dict[int, bool], window_frames: int, fps: float,
                           teams: list[list[int]]) -> list[dict]:
    events = []
    windows = sorted(team_a_near)
    for prev, cur in zip(windows, windows[1:]):
        if team_a_near[prev] != team_a_near[cur]:
            frame = cur * window_frames
            near_team = teams[0] if team_a_near[cur] else teams[1]
            events.append({
                "frame": int(frame),
                "time_s": round(frame / fps, 1),
                "near_side_players_after": [p + 1 for p in near_team],
            })
    return events
