"""Offline renderer: draw persistent Player 1-4 boxes onto the whole recorded video."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from padel_reid.common.cache_paths import CachePaths
from padel_reid.common.video_io import VideoWriter, iterate_frames, read_video_info, resolve_frame_range
from padel_reid.rendering.frame_overlay import PLAYER_COLORS, draw_overlay


def _absence_lookup(events_path: Path) -> list[tuple[int, int, int]]:
    if not events_path.exists():
        return []
    events = json.loads(events_path.read_text(encoding="utf-8"))
    return [(e["player"], e["exit_frame"], e["enter_frame"]) for e in events.get("exit_reentry", [])]


def render_video(cfg, table: pd.DataFrame, output: Path, start: int | None = None, end: int | None = None) -> None:
    info = read_video_info(cfg.video_path)
    start, end = resolve_frame_range(info, start if start is not None else int(table["frame"].min()),
                                     end if end is not None else int(table["frame"].max()) + 1)
    absences = _absence_lookup(CachePaths.from_config(cfg).events)
    by_frame = {f: g for f, g in table[(table["frame"] >= start) & (table["frame"] < end)].groupby("frame")}

    with VideoWriter(output, info.fps, (info.width, info.height)) as writer:
        for frame_index, frame in tqdm(iterate_frames(cfg.video_path, start, end), total=end - start,
                                       desc="render", unit="frame"):
            rows = by_frame.get(frame_index)
            status = {p: "not detected" for p in PLAYER_COLORS}
            for p, exit_frame, enter_frame in absences:
                if exit_frame < frame_index < enter_frame:
                    status[p] = "out of frame"
            draw_overlay(frame, rows.itertuples() if rows is not None else [], status, frame_index, info.fps,
                         cfg.rendering.draw_minimap)
            writer.write(frame)
