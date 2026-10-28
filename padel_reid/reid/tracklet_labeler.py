"""Few-shot labelling of clean tracklets with their player number (1-4).

Two ways to label:
  * sheets  - `python -m padel_reid label` writes contact sheets (one row of
              thumbnails per tracklet, captioned with its id) to cache/thumbnails/
              and a template labels file; fill in `labels:` by hand.
  * window  - `python -m padel_reid.reid.tracklet_labeler` opens an OpenCV window,
              shows one tracklet at a time; press 1-4 to label, 0 to skip, q to quit.

Number the players so that partners share a team: 1 & 2 are one team, 3 & 4 the other.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from padel_reid.common.cache_paths import CachePaths
from padel_reid.common.video_io import iterate_frames
from padel_reid.reid.crop_quality import crop_box

THUMB_HEIGHT, THUMB_WIDTH = 160, 80
ROWS_PER_SHEET = 10


def pick_candidate_tracklets(summary: pd.DataFrame, min_frames: int, max_count: int = 120) -> pd.DataFrame:
    """Long, isolated tracklets spread over the whole match."""
    clean = summary[(summary["frames"] >= min_frames) & ~summary["in_interaction"]].copy()
    if len(clean) <= max_count:
        return clean.sort_values("start_frame")
    clean["time_bin"] = pd.qcut(clean["start_frame"], q=max_count // 4, labels=False, duplicates="drop")
    picked = clean.sort_values("frames", ascending=False).groupby("time_bin").head(4)
    return picked.sort_values("start_frame")


def collect_thumbnails(video_path: str, table: pd.DataFrame, tracklet_ids: list[int],
                       per_tracklet: int) -> dict[int, list[np.ndarray]]:
    wanted: dict[int, list[tuple[int, int]]] = {}
    for tracklet_id in tracklet_ids:
        rows = table.index[table["tracklet_id"] == tracklet_id].to_numpy()
        picks = rows[np.linspace(0, len(rows) - 1, per_tracklet).astype(int)]
        for row in picks:
            wanted.setdefault(int(table.at[row, "frame"]), []).append((tracklet_id, row))
    thumbs: dict[int, list[np.ndarray]] = {t: [] for t in tracklet_ids}
    for frame_index, frame in iterate_frames(video_path, start=min(wanted), wanted=set(wanted)):
        for tracklet_id, row in wanted[frame_index]:
            r = table.loc[row]
            crop = crop_box(frame, r.x1, r.y1, r.x2, r.y2)
            if crop is not None:
                thumbs[tracklet_id].append(cv2.resize(crop, (THUMB_WIDTH, THUMB_HEIGHT)))
    return thumbs


def _caption_strip(tracklet_id: int, info: str, thumbs: list[np.ndarray], per_tracklet: int) -> np.ndarray:
    strip = np.full((THUMB_HEIGHT, 150 + per_tracklet * THUMB_WIDTH, 3), 30, np.uint8)
    cv2.putText(strip, f"#{tracklet_id}", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    for i, line in enumerate(info.split("|")):
        cv2.putText(strip, line, (6, 75 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    for i, thumb in enumerate(thumbs[:per_tracklet]):
        strip[:, 150 + i * THUMB_WIDTH: 150 + (i + 1) * THUMB_WIDTH] = thumb
    return strip


def _info(row) -> str:
    side = "near" if row.near_fraction > 0.5 else "far"
    minutes, seconds = divmod(row.start_frame / 30.0, 60)
    return f"{int(minutes):02d}:{seconds:04.1f}|{side} x={row.court_x:.1f}|{row.frames} fr"


def export_thumbnail_sheets(cfg, paths: CachePaths) -> Path:
    table = pd.read_parquet(paths.tracklets)
    summary = pd.read_parquet(paths.tracklet_summary)
    candidates = pick_candidate_tracklets(summary, cfg.labeling.min_tracklet_frames)
    per = cfg.labeling.thumbnails_per_tracklet
    thumbs = collect_thumbnails(cfg.video_path, table, candidates["tracklet_id"].tolist(), per)
    strips = [_caption_strip(int(r.tracklet_id), _info(r), thumbs[int(r.tracklet_id)], per)
              for r in candidates.itertuples()]
    out_dir = paths.thumbnails_dir
    for old in out_dir.glob("sheet_*.jpg"):
        old.unlink()
    for i in range(0, len(strips), ROWS_PER_SHEET):
        cv2.imwrite(str(out_dir / f"sheet_{i // ROWS_PER_SHEET:02d}.jpg"), np.vstack(strips[i:i + ROWS_PER_SHEET]))

    labels_path = Path(cfg.labeling.labels_file)
    if not labels_path.exists():
        labels_path.write_text(
            "# Player number (1-4) for each labelled tracklet id; partners: 1 & 2, 3 & 4.\n"
            "# Candidates are shown in cache/thumbnails/sheet_*.jpg. Unlisted tracklets are unlabelled.\n"
            "teams: [[1, 2], [3, 4]]\nlabels:\n", encoding="utf-8")
    return out_dir


def load_labels(labels_file: str | Path) -> tuple[dict[int, int], list[list[int]]]:
    """Return ({tracklet_id: player 0-3}, teams as 0-based player lists)."""
    path = Path(labels_file)
    if not path.exists():
        return {}, [[0, 1], [2, 3]]
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    labels = {int(k): int(v) - 1 for k, v in (data.get("labels") or {}).items() if v}
    teams = [[p - 1 for p in team] for team in data.get("teams", [[1, 2], [3, 4]])]
    return labels, teams


def save_labels(labels_file: str | Path, labels: dict[int, int], teams: list[list[int]]) -> None:
    data = {"teams": [[p + 1 for p in team] for team in teams],
            "labels": {int(k): int(v) + 1 for k, v in sorted(labels.items())}}
    Path(labels_file).write_text(
        "# Player number (1-4) per tracklet id; partners: 1 & 2, 3 & 4.\n" + yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8")


def label_in_window(cfg) -> None:
    paths = CachePaths.from_config(cfg)
    table = pd.read_parquet(paths.tracklets)
    summary = pd.read_parquet(paths.tracklet_summary)
    labels, teams = load_labels(cfg.labeling.labels_file)
    candidates = pick_candidate_tracklets(summary, cfg.labeling.min_tracklet_frames)
    candidates = candidates[~candidates["tracklet_id"].isin(labels)]
    per = cfg.labeling.thumbnails_per_tracklet
    thumbs = collect_thumbnails(cfg.video_path, table, candidates["tracklet_id"].tolist(), per)
    for r in candidates.itertuples():
        strip = _caption_strip(int(r.tracklet_id), _info(r), thumbs[int(r.tracklet_id)], per)
        cv2.imshow("label tracklet: 1-4 = player, 0 = skip, q = quit", strip)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if ord("1") <= key <= ord("4"):
            labels[int(r.tracklet_id)] = key - ord("1")
            save_labels(cfg.labeling.labels_file, labels, teams)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from padel_reid.common.config_loader import load_config

    label_in_window(load_config())
