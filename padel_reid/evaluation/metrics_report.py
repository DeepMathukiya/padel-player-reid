"""Evaluate every available method against a ground-truth clip and write a report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from padel_reid.common.cache_paths import CachePaths
from padel_reid.evaluation.ground_truth_clips import clip_offset_for, is_uncorrected_preannotation, read_mot
from padel_reid.evaluation.mot_metrics import evaluate_tracking

# Prediction files the pipeline / baselines write, with the column holding the id.
KNOWN_PREDICTIONS = {
    "bytetrack_raw": ("baseline_bytetrack.parquet", "raw_track_id"),
    "botsort_raw": ("detections.parquet", "raw_track_id"),
    "position_lock_baek_yun": ("baseline_position_lock.parquet", "player_id"),
    "ours_without_side_team": ("assigned_without_side_team.parquet", "player_id"),
    "ours_proposed": ("assigned_detections.parquet", "player_id"),
    "ours_live": ("assigned_live.parquet", "player_id"),                            # exactly what live mode showed
    "label_seeded_oracle": ("assigned_label_seeded_oracle.parquet", "player_id"),   # upper bound: saw labels
}


def _load_prediction(path: Path, id_column: str) -> pd.DataFrame:
    table = pd.read_parquet(path)
    return table.rename(columns={id_column: "id"})[["frame", "id", "x1", "y1", "x2", "y2"]]


def evaluate_against_ground_truth(cfg, paths: CachePaths, ground_truth: str,
                                  predictions: dict[str, str] | None = None) -> pd.DataFrame:
    if is_uncorrected_preannotation(ground_truth):
        raise ValueError(f"{ground_truth} is still the pipeline's own pre-annotation. Correct it in CVAT "
                         "first; scoring the pipeline against its own output would be meaningless.")
    gt = read_mot(ground_truth, clip_offset_for(ground_truth))
    frames = set(gt["frame"].unique())
    sources = {name: (paths.root / file, col) for name, (file, col) in KNOWN_PREDICTIONS.items()}
    for name, file in (predictions or {}).items():
        sources[name] = (Path(file), "player_id")

    rows, details = [], {}
    for name, (path, column) in sources.items():
        if not path.exists():
            continue
        pred = _load_prediction(path, column)
        pred = pred[pred["frame"].isin(frames)]
        result = evaluate_tracking(gt, pred)
        details[name] = {"maintained_fraction": result.pop("maintained_fraction"),
                         "id_mapping": result.pop("id_mapping")}
        rows.append({"method": name, **result})
    report = pd.DataFrame(rows).set_index("method")

    timing = json.loads(paths.timing.read_text()) if paths.timing.exists() else {}
    out_dir = Path(cfg.output_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ground_truth).parent.parent.name or "gt"
    report.to_csv(out_dir / f"metrics_{stem}.csv")
    (out_dir / f"metrics_{stem}.json").write_text(json.dumps(
        {"ground_truth": str(ground_truth), "frames": len(frames), "metrics": report.reset_index().to_dict("records"),
         "identity_details": details, "stage_timing": timing}, indent=2, default=float))
    return report
