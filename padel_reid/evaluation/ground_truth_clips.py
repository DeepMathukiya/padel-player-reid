"""Ground-truth clips: export a clip + pre-annotation for CVAT, read MOT files back.

Workflow:
  1. `export_gt_clip` writes outputs/gt_clips/<name>/clip.mp4, clip.json (start
     frame in the full video) and gt/gt.txt pre-filled from the pipeline's output.
  2. Import clip.mp4 + gt.txt into CVAT ("MOT 1.1" format), fix every box and id
     by hand, export again as MOT 1.1 and replace gt/gt.txt.
  3. `python -m padel_reid evaluate --gt outputs/gt_clips/<name>/gt/gt.txt`.
MOT files are 1-based in frame number: mot_frame = clip_frame + 1.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from padel_reid.common.video_io import VideoWriter, iterate_frames, read_video_info

PREANNOTATION_NAME = "preannotation.txt"


def write_mot(table: pd.DataFrame, path: Path, id_column: str, frame_offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in table.sort_values(["frame", id_column]).itertuples():
        lines.append(f"{int(r.frame) - frame_offset + 1},{int(getattr(r, id_column))},{r.x1:.1f},{r.y1:.1f},"
                     f"{r.x2 - r.x1:.1f},{r.y2 - r.y1:.1f},1,1,1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_mot(path: str | Path, frame_offset: int = 0) -> pd.DataFrame:
    """MOT text -> DataFrame (frame in full-video numbering, id, x1, y1, x2, y2)."""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.size == 0:
        return pd.DataFrame(columns=["frame", "id", "x1", "y1", "x2", "y2"])
    if data.shape[1] > 6:                      # drop rows explicitly marked "not considered"
        data = data[data[:, 6] != 0]
    return pd.DataFrame({
        "frame": data[:, 0].astype(int) - 1 + frame_offset,
        "id": data[:, 1].astype(int),
        "x1": data[:, 2], "y1": data[:, 3],
        "x2": data[:, 2] + data[:, 4], "y2": data[:, 3] + data[:, 5],
    })


def clip_offset_for(gt_path: str | Path) -> int:
    """Start frame of the clip a gt.txt belongs to (from the clip.json next to it)."""
    for parent in Path(gt_path).resolve().parents[:3]:
        meta = parent / "clip.json"
        if meta.exists():
            return int(json.loads(meta.read_text())["start_frame"])
    return 0


def export_gt_clip(cfg, assignment: pd.DataFrame, start: int, end: int, name: str) -> Path:
    out_dir = Path(cfg.output_dir) / "gt_clips" / name
    info = read_video_info(cfg.video_path)
    with VideoWriter(out_dir / "clip.mp4", info.fps, (info.width, info.height)) as writer:
        for _, frame in iterate_frames(cfg.video_path, start, end):
            writer.write(frame)
    (out_dir / "clip.json").write_text(json.dumps({"video": cfg.video_path, "start_frame": start,
                                                   "end_frame": end}, indent=2))
    clip = assignment[(assignment["frame"] >= start) & (assignment["frame"] < end)]
    write_mot(clip, out_dir / "gt" / "gt.txt", "player_id", frame_offset=start)
    # Untouched copy of the draft: evaluate() refuses to treat an uncorrected gt.txt as ground truth.
    shutil.copyfile(out_dir / "gt" / "gt.txt", out_dir / "gt" / PREANNOTATION_NAME)
    (out_dir / "gt" / "labels.txt").write_text("player\n", encoding="utf-8")
    with zipfile.ZipFile(out_dir / "annotations_for_cvat.zip", "w") as archive:     # CVAT "MOT 1.1" import
        archive.write(out_dir / "gt" / "gt.txt", "gt/gt.txt")
        archive.write(out_dir / "gt" / "labels.txt", "gt/labels.txt")
    return out_dir


def is_uncorrected_preannotation(gt_path: str | Path) -> bool:
    draft = Path(gt_path).with_name(PREANNOTATION_NAME)
    return draft.exists() and draft.read_bytes() == Path(gt_path).read_bytes()
