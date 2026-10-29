"""Turn pure tracklets + embeddings into persistent Player 1-4 labels.

No manual labels are read here: the gallery is seeded from anchor sets of
coexisting players (identity/anchor_seeding.py). Hand labels exist only for
evaluation; the label-seeded oracle baseline passes them in explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from padel_reid.common.cache_paths import CachePaths
from padel_reid.common.video_io import read_video_info
from padel_reid.identity.anchor_seeding import SeedResult, discover_seed_labels
from padel_reid.identity.changeover_detector import changeovers_from_sides
from padel_reid.identity.exit_reentry_handler import exit_entry_flags, find_exit_reentry_events
from padel_reid.identity.gap_interpolation import fill_short_gaps, smooth_boxes
from padel_reid.identity.global_assignment import (AssignmentInputs, drop_conflicting_labels,
                                                   exclusivity_constraints, solve_global_assignment)
from padel_reid.identity.player_gallery import NUM_PLAYERS, PlayerGallery, scores_to_log_prob
from padel_reid.identity.predictive_relinker import calibrate_gate, find_relink_candidates
from padel_reid.identity.tracklet_descriptors import TrackletDescriptors, build_tracklet_descriptors

MAX_RELATIVE_COST = 6.0      # nats; caps how much one tracklet's appearance can dominate


def build_gallery(descriptors: TrackletDescriptors, labels: dict[int, int], summary: pd.DataFrame,
                  cfg) -> PlayerGallery:
    gallery = PlayerGallery(descriptors.embedding.shape[1], descriptors.color.shape[1],
                            cfg.reid.bank_size, cfg.reid.color_weight)
    for t, player in sorted(labels.items(), key=lambda kv: summary.at[kv[0], "start_frame"]):
        if t in descriptors.crop_rows:
            _, emb, quality = descriptors.crop_rows[t]
            side = int(summary.at[t, "near_fraction"] > 0.5)
            gallery.add(player, side, emb, quality, descriptors.color[t])
    return gallery


def appearance_log_prob(descriptors: TrackletDescriptors, gallery: PlayerGallery,
                        summary: pd.DataFrame) -> np.ndarray:
    """(T, 4) log P(player | tracklet appearance) from the gallery similarity."""
    num = len(descriptors.tracklet_ids)
    scores = np.zeros((num, NUM_PLAYERS))
    near = summary["near_fraction"].to_numpy() > 0.5
    for t in range(num):
        scores[t] = gallery.scores(descriptors.embedding[t], descriptors.color[t], int(near[t]))
    return scores_to_log_prob(scores)


def relative_cost(log_prob: np.ndarray) -> np.ndarray:
    return np.minimum(log_prob.max(axis=1, keepdims=True) - log_prob, MAX_RELATIVE_COST)


def window_side_counts(table: pd.DataFrame, window_frames: int) -> pd.DataFrame:
    counts = (table.assign(window=table["frame"] // window_frames)
              .groupby(["tracklet_id", "window"])["side"].agg(["sum", "size"]).reset_index())
    counts["near_frames"] = counts["sum"]
    counts["far_frames"] = counts["size"] - counts["sum"]
    return counts[["tracklet_id", "window", "near_frames", "far_frames"]]


def confident_assignments(labels: dict[int, int], cost: np.ndarray, summary: pd.DataFrame,
                          evidence_seconds: np.ndarray, margin: float = 1.0) -> dict[int, int]:
    """Tracklets whose assigned player is clearly the best appearance match."""
    out = {}
    for t, k in labels.items():
        others = np.delete(cost[t], k)
        if cost[t, k] == 0 and others.min() >= margin and evidence_seconds[t] > 0.5 \
                and not summary.at[t, "in_interaction"]:
            out[t] = k
    return out


def assign_identities(cfg, paths: CachePaths,
                      seed_labels_override: dict[int, int] | None = None) -> tuple[pd.DataFrame, dict]:
    """seed_labels_override (tracklet -> player 0-3) is for the label-seeded oracle baseline only:
    the labels then both seed the gallery and are fixed, as in the original few-shot setup."""
    info = read_video_info(cfg.video_path)
    fps = info.fps
    table = pd.read_parquet(paths.tracklets)
    summary = pd.read_parquet(paths.tracklet_summary)
    summary.index = summary["tracklet_id"].to_numpy()
    embeddings = np.load(paths.embeddings)
    num = len(summary)
    full = build_tracklet_descriptors(table, embeddings, num)
    evidence_seconds = full.evidence * cfg.reid.embed_stride / fps

    if seed_labels_override is None:
        seed = discover_seed_labels(table, summary, full, evidence_seconds, fps, cfg)
    else:
        labels = {t: k for t, k in seed_labels_override.items() if t < num}
        seed = SeedResult(labels, labels, diagnostics={"method": "label_override", "seed_tracklets": len(labels)})
    teams = seed.teams
    exits, entries = exit_entry_flags(summary)
    exclusivity = exclusivity_constraints(table, summary, fps, cfg.identity.max_speed_mps)
    labels = drop_conflicting_labels(seed.fixed_labels, summary, fps, cfg.identity.max_speed_mps,
                                     int(cfg.identity.relink_max_gap_seconds * fps), exits, entries,
                                     exclusivity.pairs)
    seed_labels = {**seed.seed_labels, **labels}
    gallery = build_gallery(full, seed_labels, summary, cfg)

    gate, _ = calibrate_gate(table, fps, cfg.identity.max_speed_mps)
    gate.to_json(paths.root / "relink_gate.json")
    candidates = find_relink_candidates(table, summary, gate, int(cfg.identity.relink_max_gap_seconds * fps))
    window_frames = int(cfg.identity.window_seconds * fps)
    counts = window_side_counts(table, window_frames)
    for round_index in range(cfg.identity.assignment_rounds):
        cost = relative_cost(appearance_log_prob(full, gallery, summary))
        result = solve_global_assignment(AssignmentInputs(
            summary=summary, appearance_cost=cost, evidence_seconds=evidence_seconds,
            window_counts=counts, relink_candidates=candidates, fixed_labels=labels, teams=teams,
            exits=exits, entries=entries, fps=fps, exclusivity=exclusivity.cliques), cfg.identity)
        print(f"round {round_index + 1}: {result.status}, objective {result.objective:.1f}, "
              f"{len(result.labels)}/{num} tracklets labelled")
        if round_index + 1 < cfg.identity.assignment_rounds:
            # Soft seeds the solver overruled are dropped so a seeding mistake does not persist.
            agreed = {t: k for t, k in seed_labels.items() if result.labels.get(t) == k}
            grown = {**confident_assignments(result.labels, cost, summary, evidence_seconds), **agreed, **labels}
            gallery = build_gallery(full, grown, summary, cfg)
    assigned = result.labels
    events: dict = {"changeovers": changeovers_from_sides(result.team_a_near, window_frames, fps, teams),
                    "solver_status": result.status, "seeding": seed.diagnostics}

    exit_events = find_exit_reentry_events(summary, assigned, fps)
    events["exit_reentry"] = exit_events
    events["unlabelled_tracklets"] = int(num - len(assigned))

    out = table[table["tracklet_id"].isin(assigned)].copy()
    out["player_id"] = out["tracklet_id"].map(assigned).astype(int) + 1
    exit_keys = {(e["player"], e["exit_frame"]) for e in exit_events}
    out = fill_short_gaps(out, cfg.identity.interpolate_max_gap_frames, exit_keys)
    out = smooth_boxes(out, cfg.identity.smoothing_window)
    # One box per player per frame: keep the most confident if the solver allowed a tie.
    out = out.sort_values(["frame", "player_id", "confidence"], ascending=[True, True, False])
    out = out.drop_duplicates(["frame", "player_id"]).reset_index(drop=True)
    return out, events
