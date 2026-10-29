"""Produce every comparison row of the ablation table as a prediction parquet.

  bytetrack_raw            ByteTrack on the same detections (run on GT clip ranges)
  botsort_raw              our pass-1 BoT-SORT ids (already in detections.parquet)
  position_lock_baek_yun   Baek & Yun-style position lock (baselines/position_lock_tracker.py)
  ours_without_side_team   our pipeline without the court-side / team cost (ablation)
  ours_proposed            the proposed (label-free) pipeline
  label_seeded_oracle      the same assignment seeded and fixed with the 40 hand labels
                           (evaluation.seed_labels_file): an upper bound, not a production mode
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from padel_reid.common.cache_paths import CachePaths
from padel_reid.common.video_io import iterate_frames
from padel_reid.baselines.position_lock_tracker import run_position_lock
from padel_reid.evaluation.tracklet_annotation import load_labels
from padel_reid.identity.identity_pipeline import assign_identities


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


def _variant(cfg, paths: CachePaths, name: str, overrides: dict,
             seed_labels_override: dict[int, int] | None = None) -> Path:
    """Run the global assignment with some config values changed (or seeded with hand labels)."""
    variant_cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = variant_cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    table, _ = assign_identities(variant_cfg, paths, seed_labels_override)
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

    seed_labels, _ = load_labels(cfg.evaluation.seed_labels_file)
    if seed_labels:
        outputs["label_seeded_oracle"] = _variant(cfg, paths, "label_seeded_oracle", {}, seed_labels)

    bytetrack = run_bytetrack_baseline(cfg, paths)
    if bytetrack:
        outputs["bytetrack_raw"] = bytetrack
    return outputs
