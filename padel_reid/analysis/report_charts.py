"""Static charts for the README and technical report (docs/images/*.png).

Reads the cache and outputs written by the pipeline, so the figures can be
regenerated after any re-run: python -m padel_reid charts
Colours follow one validated categorical palette; baselines are drawn in neutral
grey so our method is the only coloured series in comparison charts.
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

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BASELINE_GREY = "#b5b3ad"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # blue, orange, aqua (categorical slots 1-3)

METHOD_NAMES = {
    "botsort_raw": "BoT-SORT raw (tracker only)",
    "position_lock_baek_yun": "Position lock (Baek & Yun-style)",
    "ours_without_side_team": "Ours without side/team cost (ablation)",
    "ours_proposed": "Ours (proposed)",
}


def _style(ax, title: str, subtitle: str | None = None, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY)


def _figure(width: float = 8, height: float = 3.6):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _method_bars(report: pd.DataFrame, column: str, path: Path, title: str, subtitle: str,
                 fmt, xlim: float | None = None) -> None:
    order = [m for m in METHOD_NAMES if m in report.index]
    values = report.loc[order, column].to_numpy(float)
    labels = [METHOD_NAMES[m] for m in order]
    colors = [SERIES[0] if m.startswith("ours") else BASELINE_GREY for m in order]
    fig, ax = _figure(8, 0.55 * len(order) + 1.3)
    y = np.arange(len(order))[::-1]
    ax.barh(y, values, height=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
    limit = xlim or max(values.max() * 1.15, 1)
    for yi, v in zip(y, values):
        ax.text(v + limit * 0.01, yi, fmt(v), va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, labels, color=TEXT)
    ax.set_xlim(0, limit)
    _style(ax, title, subtitle)
    _save(fig, path)


def chart_heldout(cfg, out: Path) -> list[Path]:
    report_path = Path(cfg.output_dir) / "metrics" / "heldout_identity.csv"
    if not report_path.exists():
        return []
    report = pd.read_csv(report_path).set_index("method")
    accuracy = out / "heldout_identity_accuracy.png"
    switches = out / "heldout_id_switches.png"
    _method_bars(report.assign(pct=report["identity_accuracy"] * 100), "pct", accuracy,
                 "Identity accuracy on held-out tracklets",
                 f"{int(report['gt_boxes'].iloc[0]):,} hand-labelled boxes, best one-to-one id mapping (higher is better)",
                 lambda v: f"{v:.1f}%", xlim=112)
    _method_bars(report, "id_switches", switches, "ID switches on held-out tracklets",
                 "changes of the predicted id along each player's sequence (lower is better)",
                 lambda v: f"{int(v)}")
    return [accuracy, switches]


def chart_identity_timeline(cfg, paths: CachePaths, out: Path) -> Path | None:
    """Which half of the court each player is on over the match (shows the changeover)."""
    if not paths.assignment.exists():
        return None
    fps = read_video_info(cfg.video_path).fps
    table = pd.read_parquet(paths.assignment)
    table = table[table["court_y"].notna()]
    bin_frames = int(10 * fps)
    table = table.assign(bin=table["frame"] // bin_frames, near=table["court_y"] >= 10)
    side = table.groupby(["player_id", "bin"])["near"].mean().reset_index()
    fig, ax = _figure(9, 2.8)
    bin_minutes = bin_frames / fps / 60
    for player in range(1, 5):
        rows = side[side["player_id"] == player].sort_values("bin")
        bins = rows["bin"].to_numpy()
        near = rows["near"].to_numpy() >= 0.5
        # Merge consecutive bins with the same side into one run.
        breaks = np.where((np.diff(bins) != 1) | (near[1:] != near[:-1]))[0] + 1
        for run in np.split(np.arange(len(bins)), breaks):
            if len(run) == 0:
                continue
            start, length = bins[run[0]] * bin_minutes, len(run) * bin_minutes
            ax.broken_barh([(start, length)], (4.6 - player - 0.3, 0.6),
                           facecolors=SERIES[0] if near[run[0]] else SERIES[1], edgecolor=SURFACE, linewidth=1)
    events = json.loads(paths.events.read_text()) if paths.events.exists() else {}
    for change in events.get("changeovers", []):
        minute = change["time_s"] / 60
        ax.axvline(minute, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.text(minute + 0.2, 4.15, f"changeover {int(minute)}:{int(change['time_s'] % 60):02d}",
                fontsize=8.5, color=TEXT_SECONDARY, va="bottom")
    ax.set_yticks([3.6, 2.6, 1.6, 0.6], [f"Player {p}" for p in range(1, 5)], color=TEXT)
    ax.set_ylim(0.1, 4.5)
    ax.set_xlabel("match time (minutes)", color=TEXT_SECONDARY, fontsize=9)
    _style(ax, "Court half per player over the match",
           "10-second bins; gaps = player not visible (e.g. out of the bottom of the frame)", grid_axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])]
    ax.legend(handles, ["near half", "far half"], loc="upper right", bbox_to_anchor=(1, 1.22), ncol=2,
              frameon=False, fontsize=9, labelcolor=TEXT)
    path = out / "identity_timeline.png"
    _save(fig, path)
    return path


def chart_prediction_error(cfg, paths: CachePaths, out: Path) -> Path | None:
    fps = read_video_info(cfg.video_path).fps
    tracklets = pd.read_parquet(paths.tracklets)
    gate, errors = calibrate_gate(tracklets, fps, cfg.identity.max_speed_mps)
    if errors.empty:
        return None
    grouped = errors.groupby("gap_frames")["error_m"]
    series = [("median error", grouped.median()), ("95th percentile", grouped.quantile(0.95)),
              ("re-link gate", pd.Series(gate.radius_m, index=gate.gap_frames))]
    fig, ax = _figure(8, 3.6)
    label_offset = {"median error": 0.0, "95th percentile": -0.3, "re-link gate": 0.3}   # avoid collisions
    for (label, s), color in zip(series, SERIES):
        style = "--" if label == "re-link gate" else "-"
        ax.plot(s.index, s.values, style, color=color, linewidth=2, label=label)
        ax.text(s.index[-1] + 0.8, s.values[-1] + label_offset[label], label, fontsize=8.5, color=TEXT, va="center")
    ax.set_xlim(0, gate.gap_frames[-1] + 14)
    ax.set_xlabel("gap length (frames at 30 fps)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("court metres", color=TEXT_SECONDARY, fontsize=9)
    _style(ax, "Constant-velocity prediction error vs gap length",
           "frames deleted from clean tracklets; the gate is the 95th percentile (+0.3 m slack)", grid_axis="y")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    path = out / "prediction_error_vs_gap.png"
    _save(fig, path)
    return path


def chart_gap_histogram(paths: CachePaths, out: Path) -> Path:
    detections = pd.read_parquet(paths.detections)
    gaps = []
    for _, group in detections.groupby("raw_track_id"):
        d = np.diff(np.sort(group["frame"].to_numpy()))
        gaps.extend((d[d > 1] - 1).tolist())
    gaps = np.clip(np.asarray(gaps), 1, 30)
    fig, ax = _figure(8, 3.2)
    counts = np.bincount(gaps, minlength=31)[1:]
    ax.bar(np.arange(1, 31), counts, width=0.8, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.set_xlabel("missing frames inside a raw track (30 = 30 or more)", color=TEXT_SECONDARY, fontsize=9)
    share = (np.asarray(gaps) < 10).mean() * 100
    _style(ax, "Detection gaps inside raw tracker tracks", f"{len(gaps)} gaps; {share:.0f}% are shorter than 10 frames",
           grid_axis="y")
    path = out / "gap_length_histogram.png"
    _save(fig, path)
    return path


def chart_interactions(cfg, paths: CachePaths, out: Path) -> Path | None:
    if not paths.interactions.exists():
        return None
    fps = read_video_info(cfg.video_path).fps
    interactions = pd.read_parquet(paths.interactions)
    per_minute = (interactions["start_frame"] // int(60 * fps)).value_counts().reindex(range(30), fill_value=0)
    fig, ax = _figure(8, 3.2)
    ax.bar(per_minute.index, per_minute.values, width=0.8, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.set_xlabel("match minute", color=TEXT_SECONDARY, fontsize=9)
    events = json.loads(paths.events.read_text()) if paths.events.exists() else {}
    for change in events.get("changeovers", []):
        minute = int(change["time_s"] // 60)
        ax.annotate("changeover walk:\nteams pass each other", xy=(minute, per_minute.get(minute, 0)),
                    xytext=(minute + 2, per_minute.max() * 0.8), fontsize=8.5, color=TEXT_SECONDARY,
                    arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "linewidth": 0.8})
    _style(ax, "Interaction events per minute",
           f"{len(interactions)} events: boxes overlap or feet < 1 m apart; tracklets are cut at each one",
           grid_axis="y")
    path = out / "interactions_per_minute.png"
    _save(fig, path)
    return path


def write_report_charts(cfg, paths: CachePaths, out_dir: str | Path = "docs/images") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = chart_heldout(cfg, out)
    for path in (chart_identity_timeline(cfg, paths, out), chart_prediction_error(cfg, paths, out),
                 chart_gap_histogram(paths, out), chart_interactions(cfg, paths, out)):
        if path:
            written.append(path)
    return written
