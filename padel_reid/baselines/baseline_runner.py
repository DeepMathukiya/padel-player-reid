"""Produce every comparison row of the ablation table as a prediction parquet.

  bytetrack_raw            ByteTrack on the same detections (run on GT clip ranges)
  botsort_raw              our pass-1 BoT-SORT ids (already in detections.parquet)
  position_lock_baek_yun   Baek & Yun-style position lock (baselines/position_lock_tracker.py)
  ours_without_side_team   our pipeline without the court-side / team cost (ablation)
  ours_proposed            the proposed pipeline
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from padel_reid.common.cache_paths import CachePaths
from padel_reid.common.video_io import iterate_frames
from padel_reid.identity.identity_pipeline import assign_identities
from padel_reid.baselines.position_lock_tracker import run_position_lock


def gt_clip_ranges(cfg) -> list[tuple[int, int]]:
    ranges = []
    for meta in sorted((Path(cfg.output_dir) / "gt_clips").glob("*/clip.json")):
        data = json.loads(meta.read_text())
        ranges.append((int(data["start_frame"]), int(data["end_frame"])))
    return ranges


def run_bytetrack_baseline(cfg, paths: CachePaths) -> Path | None:
    from padel_reid.api import build_player_detector

    ranges = gt_clip_ranges(cfg)
    if not ranges:
        print("bytetrack baseline skipped: no outputs/gt_clips/*/clip.json yet")
        return None
    detection_cfg = type(cfg.detection)(dict(cfg.detection, tracker_config="config/bytetrack_padel.yaml"))
    detector = build_player_detector(cfg, detection_cfg)       # same detections as ours, only the tracker differs
    tables = []
    for i, (start, end) in enumerate(ranges):
        table = detector.run(iterate_frames(cfg.video_path, start, end))
        table["raw_track_id"] += 100000 * i          # ids must not collide across clips
        tables.append(table)
    out = paths.root / "baseline_bytetrack.parquet"
    pd.concat(tables).to_parquet(out)
    return out


def _variant(cfg, paths: CachePaths, name: str, overrides: dict) -> Path:
    """Run the global assignment with some config values changed."""
    variant_cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = variant_cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    table, _ = assign_identities(variant_cfg, paths)
    out = paths.root / f"assigned_{name}.parquet"
    table.to_parquet(out)
    return out


def run_all_baselines(cfg, paths: CachePaths) -> dict[str, Path]:
    outputs = {}
    lock = run_position_lock(pd.read_parquet(paths.detections))
    outputs["position_lock_baek_yun"] = paths.root / "baseline_position_lock.parquet"
    lock.to_parquet(outputs["position_lock_baek_yun"])

    outputs["ours_without_side_team"] = _variant(cfg, paths, "without_side_team",
                                                 {"identity.side_violation_cost": 0.0, "identity.side_flip_cost": 0.0})

    bytetrack = run_bytetrack_baseline(cfg, paths)
    if bytetrack:
        outputs["bytetrack_raw"] = bytetrack
    return outputs
