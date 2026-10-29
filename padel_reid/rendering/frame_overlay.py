"""Draw Player 1-4 boxes, a status panel and a court mini-map onto one frame.

Shared by the offline renderer and the live emitter. Rows with player_id 0 are
tracklets without an identity yet (live mode before seeding, or left unknown by
the solver); they get a grey "?" box and stay off the mini-map.
"""

from __future__ import annotations

import cv2
import numpy as np

# BGR; partners share a colour family (1 & 2 warm, 3 & 4 cool), none is court-blue.
PLAYER_COLORS = {1: (0, 215, 255), 2: (0, 120, 255), 3: (200, 60, 255), 4: (80, 220, 60)}
UNKNOWN_COLOR = (160, 160, 160)
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
    # Blend only the panel region (blending the whole frame cost ~4 ms per 1080p frame).
    panel = frame[y0 - 8:y0 + h + 9, x0 - 8:x0 + w + 9]
    overlay = np.empty_like(panel)
    overlay[:] = (40, 40, 40)
    cv2.rectangle(overlay, (8, 8), (8 + w, 8 + h), (170, 90, 30), -1)
    panel[:] = cv2.addWeighted(overlay, 0.75, panel, 0.25, 0)
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


def _draw_status_panel(frame, status: dict[int, str], frame_index: int, fps: float, note: str | None) -> None:
    x0, y0 = MINIMAP_ORIGIN[0] + COURT_W * MINIMAP_SCALE + 25, 20
    cv2.rectangle(frame, (x0 - 8, y0 - 8), (x0 + 280, y0 + (150 if note else 130)), (40, 40, 40), -1)
    minutes, seconds = divmod(frame_index / fps, 60)
    cv2.putText(frame, f"frame {frame_index}  {int(minutes):02d}:{seconds:05.2f}", (x0, y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    for i, player in enumerate(sorted(PLAYER_COLORS)):
        y = y0 + 35 + 24 * i
        cv2.circle(frame, (x0 + 8, y - 5), 7, PLAYER_COLORS[player], -1)
        cv2.putText(frame, f"Player {player}: {status[player]}", (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (240, 240, 240), 1, cv2.LINE_AA)
    if note:
        cv2.putText(frame, note, (x0, y0 + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 200, 255), 1, cv2.LINE_AA)


def draw_badge(frame: np.ndarray, text: str) -> None:
    """Small text badge in the top-right corner (live mode: display FPS and delay)."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    x0, y0 = frame.shape[1] - w - 28, 20
    cv2.rectangle(frame, (x0 - 8, y0 - 8), (x0 + w + 8, y0 + h + 10), (40, 40, 40), -1)
    cv2.putText(frame, text, (x0, y0 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, rows, status: dict[int, str], frame_index: int, fps: float,
                 draw_minimap: bool = True, note: str | None = None) -> None:
    """rows: iterable with player_id (0 = unknown), x1..y2, court_x, court_y, optional interpolated.

    `status` is updated in place: a drawn player becomes "visible".
    """
    positions = {}
    for r in rows:
        player = int(r.player_id)
        color = PLAYER_COLORS.get(player, UNKNOWN_COLOR)
        thickness = 1 if getattr(r, "interpolated", False) else (2 if player == 0 else 3)
        cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, thickness)
        _draw_label(frame, f"Player {player}" if player else "?", int(r.x1), int(r.y1) - 2, color)
        if player:
            positions[player] = (r.court_x, r.court_y)
            status[player] = "visible"
    if draw_minimap:
        _draw_minimap(frame, positions)
    _draw_status_panel(frame, status, frame_index, fps, note)
