"""Data-driven answer to "why do ID switches happen in this video?".

Writes outputs/analysis/: plots + summary.json covering
  * missing-detection gap lengths of the raw tracker tracks
  * constant-velocity prediction error vs. gap length (the re-link gate calibration)
  * interaction events (overlaps / close encounters) per minute
  * raw tracker fragmentation (how many ids a 4-player match produces)
  * exit / re-entry events and changeovers (after assignment, if available)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from padel_reid.common.cache_paths import CachePaths
from padel_reid.common.video_io import read_video_info
from padel_reid.identity.predictive_relinker import calibrate_gate


def _raw_track_gaps(detections: pd.DataFrame) -> np.ndarray:
    gaps = []
    for _, group in detections.groupby("raw_track_id"):
        d = np.diff(np.sort(group["frame"].to_numpy()))
        gaps.extend((d[d > 1] - 1).tolist())
    return np.asarray(gaps)


def write_video_analysis(cfg, paths: CachePaths) -> Path:
    out = Path(cfg.output_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fps = read_video_info(cfg.video_path).fps
    detections = pd.read_parquet(paths.detections)
    tracklets = pd.read_parquet(paths.tracklets)
    summary = pd.read_parquet(paths.tracklet_summary)
    interactions = pd.read_parquet(paths.interactions) if paths.interactions.exists() else pd.DataFrame()
    minutes = (detections["frame"].max() - detections["frame"].min() + 1) / fps / 60
    stats: dict = {"video_minutes": round(float(minutes), 2)}

    # 1. Missing-detection gaps inside raw tracks
    gaps = _raw_track_gaps(detections)
    stats["raw_track_gaps"] = {"count": int(len(gaps)),
                               "median_frames": float(np.median(gaps)) if len(gaps) else 0.0,
                               "share_under_10_frames": float((gaps < 10).mean()) if len(gaps) else 0.0}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.clip(gaps, 0, 60), bins=60, color="#3b6fb6")
    ax.set(xlabel="missing frames (clipped at 60)", ylabel="gaps", title="Detection gaps inside raw tracks")
    fig.tight_layout()
    fig.savefig(out / "gap_length_histogram.png", dpi=120)
    plt.close(fig)

    # 2. Prediction error vs gap length -> calibrated gate
    gate, errors = calibrate_gate(tracklets, fps, cfg.identity.max_speed_mps)
    if not errors.empty:
        grouped = errors.groupby("gap_frames")["error_m"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(grouped.median().index, grouped.median().values, label="median error")
        ax.plot(grouped.quantile(0.95).index, grouped.quantile(0.95).values, label="95th percentile")
        ax.plot(gate.gap_frames, gate.radius_m, "--", label="re-link gate")
        ax.set(xlabel="gap (frames)", ylabel="court metres", title="Constant-velocity prediction error")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "prediction_error_vs_gap.png", dpi=120)
        plt.close(fig)
        stats["gate_radius_m"] = {str(g): round(float(r), 2) for g, r in zip(gate.gap_frames, gate.radius_m)
                                  if g in (1, 5, 10, 15, 30, 60)}

    # 3. Interactions per minute
    if not interactions.empty:
        per_minute = (interactions["start_frame"] // int(60 * fps)).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(per_minute.index, per_minute.values, color="#c0504d")
        ax.set(xlabel="minute", ylabel="events", title="Interaction events (overlap or < 1 m apart)")
        fig.tight_layout()
        fig.savefig(out / "interactions_per_minute.png", dpi=120)
        plt.close(fig)
        durations = (interactions["end_frame"] - interactions["start_frame"] + 1) / fps
        stats["interactions"] = {"events": int(len(interactions)),
                                 "per_minute": round(len(interactions) / minutes, 2),
                                 "median_duration_s": round(float(durations.median()), 2)}

    # 4. Tracker fragmentation
    stats["raw_tracker"] = {"distinct_ids": int(detections["raw_track_id"].nunique()),
                            "frames_with_4_players": float((detections.groupby("frame").size() == 4).mean())}
    stats["tracklets"] = {"count": int(len(summary)), "median_frames": float(summary["frames"].median()),
                          "in_interaction": int(summary["in_interaction"].sum()),
                          "start_at_border": int(summary["starts_at_border"].sum()),
                          "end_at_border": int(summary["ends_at_border"].sum())}

    # 5. Events from the assignment stage
    if paths.events.exists():
        events = json.loads(paths.events.read_text())
        exits = events.get("exit_reentry", [])
        stats["exit_reentry"] = {"events": len(exits),
                                 "median_absence_s": float(np.median([e["absent_seconds"] for e in exits])) if exits else 0.0}
        stats["changeovers"] = events.get("changeovers", [])

    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    return out
