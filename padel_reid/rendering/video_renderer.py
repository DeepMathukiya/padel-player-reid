"""Draw persistent Player 1-4 boxes, a status panel and a court mini-map."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from padel_reid.common.cache_paths import CachePaths
from padel_reid.common.video_io import VideoWriter, iterate_frames, read_video_info, resolve_frame_range

# BGR; partners share a colour family (1 & 2 warm, 3 & 4 cool), none is court-blue.
PLAYER_COLORS = {1: (0, 215, 255), 2: (0, 120, 255), 3: (200, 60, 255), 4: (80, 220, 60)}
MINIMAP_ORIGIN = (20, 20)
MINIMAP_SCALE = 9          # pixels per metre
COURT_W, COURT_L = 10, 20


def _draw_label(frame, text, x, y, color) -> None:
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y = max(y, h + 6)
    cv2.rectangle(frame, (x, y - h - 8), (x + w + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)


def _draw_minimap(frame, positions: dict[int, tuple[float, float]]) -> None:
    x0, y0 = MINIMAP_ORIGIN
    w, h = COURT_W * MINIMAP_SCALE, COURT_L * MINIMAP_SCALE
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 8), (x0 + w + 8, y0 + h + 8), (40, 40, 40), -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (170, 90, 30), -1)
    frame[:] = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    cv2.line(frame, (x0, y0 + h // 2), (x0 + w, y0 + h // 2), (255, 255, 255), 2)          # net
    for y in (3.05, 16.95):
        yy = y0 + int(y * MINIMAP_SCALE)
        cv2.line(frame, (x0, yy), (x0 + w, yy), (230, 230, 230), 1)
    cv2.line(frame, (x0 + w // 2, y0 + int(3.05 * MINIMAP_SCALE)),
             (x0 + w // 2, y0 + int(16.95 * MINIMAP_SCALE)), (230, 230, 230), 1)
    for player, (cx, cy) in positions.items():
        px = int(np.clip(x0 + cx * MINIMAP_SCALE, x0 - 6, x0 + w + 6))
        py = int(np.clip(y0 + cy * MINIMAP_SCALE, y0 - 6, y0 + h + 6))
        cv2.circle(frame, (px, py), 6, PLAYER_COLORS[player], -1)
        cv2.putText(frame, str(player), (px + 7, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def _draw_status_panel(frame, status: dict[int, str], frame_index: int, fps: float) -> None:
    x0, y0 = MINIMAP_ORIGIN[0] + COURT_W * MINIMAP_SCALE + 25, 20
    cv2.rectangle(frame, (x0 - 8, y0 - 8), (x0 + 280, y0 + 130), (40, 40, 40), -1)
    minutes, seconds = divmod(frame_index / fps, 60)
    cv2.putText(frame, f"frame {frame_index}  {int(minutes):02d}:{seconds:05.2f}", (x0, y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    for i, player in enumerate(sorted(PLAYER_COLORS)):
        y = y0 + 35 + 24 * i
        cv2.circle(frame, (x0 + 8, y - 5), 7, PLAYER_COLORS[player], -1)
        cv2.putText(frame, f"Player {player}: {status[player]}", (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (240, 240, 240), 1, cv2.LINE_AA)


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
            positions, status = {}, {p: "not detected" for p in PLAYER_COLORS}
            for p, exit_frame, enter_frame in absences:
                if exit_frame < frame_index < enter_frame:
                    status[p] = "out of frame"
            if rows is not None:
                for r in rows.itertuples():
                    player = int(r.player_id)
                    color = PLAYER_COLORS[player]
                    thickness = 1 if getattr(r, "interpolated", False) else 3
                    cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, thickness)
                    _draw_label(frame, f"Player {player}", int(r.x1), int(r.y1) - 2, color)
                    positions[player] = (r.court_x, r.court_y)
                    status[player] = "visible"
            if cfg.rendering.draw_minimap:
                _draw_minimap(frame, positions)
            _draw_status_panel(frame, status, frame_index, info.fps)
            writer.write(frame)
