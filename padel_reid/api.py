"""Public entry point of the padel Re-ID pipeline.

Stages (each reads/writes the cache folder, so they can be re-run independently):

    detect      pass 1: YOLO11s @960 + BoT-SORT               -> detections.parquet
    tracklets   cut raw tracks into pure tracklets            -> tracklet_detections.parquet
    embed       OSNet-AIN (MSMT17) embeddings, colour, quality -> embeddings.npz
    label       export thumbnails / read player labels        -> config/tracklet_labels.yaml
    assign     gallery + global CP-SAT identity assignment   -> assigned_detections.parquet
    render      output video with Player 1-4                  -> outputs/*.mp4
    evaluate    IDSW / IDF1 / MOTA / HOTA against MOT ground truth
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from padel_reid.common.cache_paths import CachePaths
from padel_reid.common.config_loader import ConfigNode, load_config
from padel_reid.common.stage_timer import timed_stage
from padel_reid.common.video_io import iterate_frames, prefetch_frames, read_video_info, resolve_frame_range
from padel_reid.court.homography import load_court_mapping

__all__ = [
    "load_config", "run_detection", "run_tracklets", "run_embedding", "export_labeling_sheets",
    "run_assignment", "run_render", "run_evaluation", "run_heldout_evaluation",
    "export_ground_truth_clip", "run_analysis",
    "run_baselines", "run_pipeline",
]


def _frame_range(cfg: ConfigNode) -> tuple[int, int]:
    return resolve_frame_range(read_video_info(cfg.video_path), cfg.start_frame, cfg.end_frame)


def build_player_detector(cfg: ConfigNode, detection_cfg: ConfigNode | None = None):
    """PlayerDetector with the court mapping and (optionally) the pre-detection court-ROI mask."""
    from padel_reid.common.config_loader import load_yaml
    from padel_reid.court.court_roi_mask import build_court_roi_mask
    from padel_reid.detection.player_detector import PlayerDetector

    detection_cfg = detection_cfg or cfg.detection
    info = read_video_info(cfg.video_path)
    court = load_yaml(cfg.court_config)
    mapping = load_court_mapping(cfg.court_config)
    roi_mask = None
    if detection_cfg.get("mask_outside_court", True):
        roi_mask = build_court_roi_mask(mapping, (info.width, info.height), court.roi_margin_m,
                                        tuple(court.person_height_px), court.roi_headroom,
                                        court.get("roi_standing_margin_m", 0.3))
    return PlayerDetector(detection_cfg, mapping, float(court.roi_margin_m), (info.width, info.height), roi_mask)


def run_detection(cfg: ConfigNode) -> pd.DataFrame:
    paths = CachePaths.from_config(cfg)
    detector = build_player_detector(cfg)
    start, end = _frame_range(cfg)
    with timed_stage(paths.timing, "detect_and_track", end - start):
        table = detector.run(prefetch_frames(iterate_frames(cfg.video_path, start, end)))
    table.to_parquet(paths.detections)
    return table


def run_tracklets(cfg: ConfigNode) -> pd.DataFrame:
    from padel_reid.tracking.interaction_detector import detect_interactions, group_interaction_events
    from padel_reid.tracking.tracklet_builder import build_tracklets, summarize_tracklets

    paths = CachePaths.from_config(cfg)
    info = read_video_info(cfg.video_path)
    detections = pd.read_parquet(paths.detections)
    with timed_stage(paths.timing, "tracklets", int(detections["frame"].nunique())):
        table = build_tracklets(detections, cfg.tracking)
        summary = summarize_tracklets(table, (info.width, info.height), cfg.identity.border_margin_fraction)
        _, _, pairs = detect_interactions(detections, cfg.tracking.interaction_iou,
                                          cfg.tracking.interaction_foot_distance_m)
    table.to_parquet(paths.tracklets)
    summary.to_parquet(paths.tracklet_summary)
    group_interaction_events(pairs).to_parquet(paths.interactions)
    return summary


def run_embedding(cfg: ConfigNode) -> None:
    from padel_reid.reid.tracklet_embedder import embed_tracklet_detections

    paths = CachePaths.from_config(cfg)
    table = pd.read_parquet(paths.tracklets)
    with timed_stage(paths.timing, "embed") as record:
        record["frames"] = embed_tracklet_detections(cfg, table, paths)


def export_labeling_sheets(cfg: ConfigNode) -> Path:
    from padel_reid.reid.tracklet_labeler import export_thumbnail_sheets

    return export_thumbnail_sheets(cfg, CachePaths.from_config(cfg))


def run_assignment(cfg: ConfigNode) -> pd.DataFrame:
    """Gallery + CP-SAT global identity assignment over the whole video."""
    from padel_reid.identity.identity_pipeline import assign_identities

    paths = CachePaths.from_config(cfg)
    with timed_stage(paths.timing, "assign") as record:
        table, events = assign_identities(cfg, paths)
        record["frames"] = int(table["frame"].nunique())
    table.to_parquet(paths.assignment)
    paths.events.write_text(json.dumps(events, indent=2, default=int), encoding="utf-8")
    return table


def run_render(cfg: ConfigNode, assignment_file: str | None = None, output_name: str = "padel_player_ids.mp4",
               start: int | None = None, end: int | None = None) -> Path:
    from padel_reid.rendering.video_renderer import render_video

    paths = CachePaths.from_config(cfg)
    table = pd.read_parquet(assignment_file or paths.assignment)
    output = Path(cfg.output_dir) / output_name
    render_video(cfg, table, output, start, end)
    return output


def run_evaluation(cfg: ConfigNode, ground_truth: str, predictions: dict[str, str] | None = None) -> pd.DataFrame:
    from padel_reid.evaluation.metrics_report import evaluate_against_ground_truth

    return evaluate_against_ground_truth(cfg, CachePaths.from_config(cfg), ground_truth, predictions)


def run_heldout_evaluation(cfg: ConfigNode, ground_truth_csv: str = "evaluation_data/heldout_ground_truth.csv") -> pd.DataFrame:
    from padel_reid.evaluation.heldout_tracklet_eval import evaluate_heldout

    return evaluate_heldout(cfg, CachePaths.from_config(cfg), ground_truth_csv)


def export_ground_truth_clip(cfg: ConfigNode, start: int, end: int, name: str) -> Path:
    from padel_reid.evaluation.ground_truth_clips import export_gt_clip

    paths = CachePaths.from_config(cfg)
    return export_gt_clip(cfg, pd.read_parquet(paths.assignment), start, end, name)


def write_report_charts(cfg: ConfigNode) -> list[Path]:
    from padel_reid.analysis.report_charts import write_report_charts as _write

    return _write(cfg, CachePaths.from_config(cfg))


def run_analysis(cfg: ConfigNode) -> Path:
    from padel_reid.analysis.gap_statistics import write_video_analysis

    return write_video_analysis(cfg, CachePaths.from_config(cfg))


def run_baselines(cfg: ConfigNode) -> dict[str, Path]:
    from padel_reid.baselines.baseline_runner import run_all_baselines

    return run_all_baselines(cfg, CachePaths.from_config(cfg))


def run_pipeline(cfg: ConfigNode) -> pd.DataFrame:
    """Everything after manual labelling: detect -> tracklets -> embed -> assign -> render."""
    paths = CachePaths.from_config(cfg)
    if not paths.detections.exists():
        run_detection(cfg)
    run_tracklets(cfg)
    run_embedding(cfg)
    table = run_assignment(cfg)
    run_render(cfg)
    return table
