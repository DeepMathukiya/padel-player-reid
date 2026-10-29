"""Global identity assignment over the whole video with OR-Tools CP-SAT.

Every tracklet gets exactly one of {Player 1..4, unknown}. The solver minimises

  appearance  - how badly the tracklet matches the player's gallery / classifier,
                weighted by how much good evidence the tracklet has
  side/team   - partners stay on the same half; per time window a boolean says
                which half team A is on, and changing it (a changeover) costs
                side_flip_cost, so the solver finds the changeovers itself
  continuity  - a bonus when a tracklet and the one that starts inside its
                predictive re-link gate get the same player
  unknown     - leaving a tracklet unlabelled costs unknown_cost nats x evidence

subject to
  exclusivity - tracklets with boxes in the same frame never share a player; tracklets
                whose time spans only interleave (a tracker flicker: old and new track
                alternating for a few frames) may, if the handover is short and reachable
  speed limit - two tracklets cannot share a player if moving from the end of one
                to the start of the other would exceed max_speed_mps (pairs that
                are an exit followed by an entry, or are cut off by the frame,
                are exempt)
  labels      - fixed tracklets keep their player (seed reference; in live mode also
                every tracklet already shown on screen)
  sides       - live mode: per-window team sides already shown are fixed
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from padel_reid.identity.predictive_relinker import RelinkCandidate

NUM_PLAYERS = 4
SCALE = 100          # CP-SAT needs integer costs
SPEED_SLACK_M = 2.5  # box / homography jitter allowed on top of the speed limit


@dataclass
class AssignmentInputs:
    summary: pd.DataFrame                 # one row per tracklet (dense ids 0..T-1)
    appearance_cost: np.ndarray           # (T, 4) relative cost, best player = 0
    evidence_seconds: np.ndarray          # (T,)
    window_counts: pd.DataFrame           # tracklet_id, window, near_frames, far_frames
    relink_candidates: list[RelinkCandidate]
    fixed_labels: dict[int, int]          # tracklet -> player (0-3)
    teams: list[list[int]]                # [[a, b], [c, d]] 0-based
    exits: set[int]
    entries: set[int]
    fps: float
    fixed_sides: dict[int, bool] = field(default_factory=dict)   # window -> team A near (live)
    exclusivity: list[list[int]] | None = None   # from exclusivity_constraints(); None = time-span overlap


@dataclass
class AssignmentResult:
    labels: dict[int, int]                # tracklet -> player (unknown tracklets omitted)
    team_a_near: dict[int, bool]          # window -> True if team A is on the near half
    objective: float
    status: str


def _overlap_cliques(summary: pd.DataFrame) -> list[list[int]]:
    """Maximal sets of tracklets alive at the same time (checked at every start frame)."""
    starts = summary["start_frame"].to_numpy()
    ends = summary["end_frame"].to_numpy()
    ids = summary["tracklet_id"].to_numpy()
    order = np.argsort(starts)
    cliques, seen = [], set()
    for i in order:
        alive = ids[(starts <= starts[i]) & (ends >= starts[i])]
        key = tuple(sorted(alive.tolist()))
        if len(key) > 1 and key not in seen:
            seen.add(key)
            cliques.append(list(key))
    return cliques


MAX_HANDOVER_FRAMES = 10   # interleaved time spans longer than this always count as two people


@dataclass
class Exclusivity:
    cliques: list[list[int]]              # each: at most one tracklet per player
    pairs: set[tuple[int, int]]           # every (a, b), a < b, that may not share a player


def exclusivity_constraints(table: pd.DataFrame, summary: pd.DataFrame, fps: float,
                            max_speed_mps: float) -> Exclusivity:
    """Who cannot be the same player, from actual co-presence instead of time-span overlap.

    Time-span overlap is too strict: when BoT-SORT flickers (old and new track ids
    alternating for a few frames), the two tracklets' spans overlap although they are
    never in the same frame, and a live run that already showed the old one as Player k
    could then never give the new one Player k. Such interleaved pairs may share a
    player if the overlap is short and the handover position is reachable.
    """
    cliques, seen = [], set()
    for ids in table.groupby("frame")["tracklet_id"].agg(lambda s: tuple(sorted(set(s)))):
        if len(ids) > 1 and ids not in seen:
            seen.add(ids)
            cliques.append(list(ids))
    pairs = {(a, b) for clique in cliques for i, a in enumerate(clique) for b in clique[i + 1:]}

    ordered = summary["tracklet_id"].to_numpy()[np.argsort(summary["start_frame"].to_numpy(), kind="stable")]
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if summary.at[b, "start_frame"] > summary.at[a, "end_frame"]:
                break
            pair = (min(a, b), max(a, b))
            if pair in pairs:
                continue
            overlap = int(min(summary.at[a, "end_frame"], summary.at[b, "end_frame"])
                          - summary.at[b, "start_frame"] + 1)
            if overlap > MAX_HANDOVER_FRAMES or speed_conflict(summary, a, b, overlap, fps, max_speed_mps,
                                                               set(), set()):
                pairs.add(pair)
                cliques.append(list(pair))
    return Exclusivity(cliques, pairs)


def speed_conflict(summary: pd.DataFrame, a: int, b: int, gap_frames: int, fps: float, max_speed_mps: float,
                   exits: set[int], entries: set[int]) -> bool:
    """True if one player cannot get from the end of tracklet a to the start of b in time.

    Exempt: real absences (exit -> entry) and ends where the box is cut off by the
    frame, because the foot position is unknown there.
    """
    if (a in exits and b in entries) or summary.at[a, "end_truncated"] or summary.at[b, "start_truncated"]:
        return False
    distance = float(np.hypot(summary.at[a, "end_court_x"] - summary.at[b, "start_court_x"],
                              summary.at[a, "end_court_y"] - summary.at[b, "start_court_y"]))
    return distance > max_speed_mps * gap_frames / fps + SPEED_SLACK_M


def drop_conflicting_labels(labels: dict[int, int], summary: pd.DataFrame, fps: float, max_speed_mps: float,
                            relink_max_gap: int, exits: set[int], entries: set[int],
                            exclusive_pairs: set[tuple[int, int]] | None = None) -> dict[int, int]:
    """Remove manual labels that contradict a hard constraint (keeps the longer tracklet).

    Two same-player labels on tracklets that overlap in time (usually a brief duplicate
    box) or that violate the speed limit would make the whole problem infeasible.
    """
    import warnings

    kept = dict(labels)
    ordered = sorted(kept, key=lambda t: summary.at[t, "start_frame"])
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if a not in kept or b not in kept or kept[a] != kept[b]:
                continue
            if exclusive_pairs is None:
                overlap = summary.at[b, "start_frame"] <= summary.at[a, "end_frame"]
            else:
                overlap = (min(a, b), max(a, b)) in exclusive_pairs
            gap = int(summary.at[b, "start_frame"] - summary.at[a, "end_frame"])
            too_fast = 0 < gap <= relink_max_gap and speed_conflict(summary, a, b, gap, fps, max_speed_mps,
                                                                    exits, entries)
            if overlap or too_fast:
                drop = a if summary.at[a, "frames"] < summary.at[b, "frames"] else b
                warnings.warn(f"label of tracklet {drop} (player {kept[drop] + 1}) conflicts with "
                              f"tracklet {b if drop == a else a}; ignoring it")
                kept.pop(drop)
    return kept


def solve_global_assignment(inputs: AssignmentInputs, identity_cfg) -> AssignmentResult:
    summary = inputs.summary
    model = cp_model.CpModel()
    tracklets = summary["tracklet_id"].to_numpy()
    team_of = np.zeros(NUM_PLAYERS, dtype=int)
    for team_index, team in enumerate(inputs.teams):
        team_of[team] = team_index

    x = {(t, k): model.NewBoolVar(f"x_{t}_{k}") for t in tracklets for k in range(NUM_PLAYERS)}
    unknown = {t: model.NewBoolVar(f"u_{t}") for t in tracklets}
    objective = []

    for t in tracklets:
        model.AddExactlyOne([x[t, k] for k in range(NUM_PLAYERS)] + [unknown[t]])
        evidence = min(inputs.evidence_seconds[t], identity_cfg.max_evidence_seconds)
        weight = identity_cfg.appearance_cost_scale * evidence
        for k in range(NUM_PLAYERS):
            cost = weight * inputs.appearance_cost[t, k]
            if cost > 0:
                objective.append(int(round(SCALE * cost)) * x[t, k])
        # Same units as appearance: leaving a tracklet unlabelled is only cheaper when
        # every allowed player is at least `unknown_cost` nats worse than its best match.
        objective.append(int(round(SCALE * max(weight, 0.1) * identity_cfg.unknown_cost)) * unknown[t])

    for t, player in inputs.fixed_labels.items():
        if t in unknown:
            model.Add(x[t, player] == 1)

    for clique in inputs.exclusivity if inputs.exclusivity is not None else _overlap_cliques(summary):
        for k in range(NUM_PLAYERS):
            model.AddAtMostOne([x[t, k] for t in clique])

    # Speed limit (hard) and re-link continuity (soft bonus).
    best_per_earlier: dict[int, list[RelinkCandidate]] = {}
    for cand in inputs.relink_candidates:
        a, b = cand.earlier, cand.later
        if speed_conflict(summary, a, b, cand.gap_frames, inputs.fps, identity_cfg.max_speed_mps,
                          inputs.exits, inputs.entries):
            for k in range(NUM_PLAYERS):
                model.AddBoolOr([x[a, k].Not(), x[b, k].Not()])
        elif cand.within_gate:
            best_per_earlier.setdefault(a, []).append(cand)
    for a, cands in best_per_earlier.items():
        for cand in sorted(cands, key=lambda c: c.distance_m)[:2]:
            bonus = identity_cfg.relink_bonus * (1.0 if len(cands) == 1 else 0.5)
            for k in range(NUM_PLAYERS):
                both = model.NewBoolVar(f"link_{a}_{cand.later}_{k}")
                model.AddImplication(both, x[a, k])
                model.AddImplication(both, x[cand.later, k])
                objective.append(-int(round(SCALE * bonus)) * both)

    # Team sides per window: team_a_near[w] == 1 means team A is on the near half.
    counts = inputs.window_counts
    # Fixed windows take part in the flip cost, so a live window continues the committed state.
    windows = sorted(set(counts["window"].unique().tolist()) | set(inputs.fixed_sides))
    team_a_near = {w: model.NewBoolVar(f"side_{w}") for w in windows}
    for w, a_near in inputs.fixed_sides.items():
        model.Add(team_a_near[w] == int(a_near))
    for prev, cur in zip(windows, windows[1:]):
        flip = model.NewBoolVar(f"flip_{cur}")
        model.Add(team_a_near[cur] - team_a_near[prev] <= flip)
        model.Add(team_a_near[prev] - team_a_near[cur] <= flip)
        objective.append(int(round(SCALE * identity_cfg.side_flip_cost)) * flip)

    per_frame = SCALE * identity_cfg.side_violation_cost / inputs.fps
    for row in counts.itertuples():
        t, s = int(row.tracklet_id), team_a_near[row.window]
        for k in range(NUM_PLAYERS):
            # Team A belongs on the near half when s = 1, team B when s = 0.
            if team_of[k] == 0:
                cost_if_s0, cost_if_s1 = row.near_frames, row.far_frames
            else:
                cost_if_s0, cost_if_s1 = row.far_frames, row.near_frames
            if cost_if_s0 == cost_if_s1 == 0:
                continue
            # cost = x*cost_if_s0 + (x AND s)*(cost_if_s1 - cost_if_s0)
            objective.append(int(round(per_frame * cost_if_s0)) * x[t, k])
            delta = int(round(per_frame * (cost_if_s1 - cost_if_s0)))
            if delta != 0:
                both = model.NewBoolVar(f"xs_{t}_{k}_{row.window}")
                model.AddImplication(both, x[t, k])
                model.AddImplication(both, s)
                model.AddBoolOr([x[t, k].Not(), s.Not(), both])
                objective.append(delta * both)

    model.Minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(identity_cfg.solver_time_limit_s)
    solver.parameters.num_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no solution: {solver.StatusName(status)}")

    labels = {int(t): k for t in tracklets for k in range(NUM_PLAYERS) if solver.Value(x[t, k])}
    sides = {int(w): bool(solver.Value(v)) for w, v in team_a_near.items()}
    return AssignmentResult(labels, sides, solver.ObjectiveValue() / SCALE, solver.StatusName(status))
