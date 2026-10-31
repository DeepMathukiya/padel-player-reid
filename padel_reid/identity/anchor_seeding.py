"""Label-free gallery seeds from "anchor sets" of coexisting players.

Padel gives structure that replaces manual labels: at any moment there are at most
4 players, partners share a half of the court, and tracklets alive at the same
time are different people. An anchor set is a moment where 4 clean tracklets
coexist, 2 on each half: a fully separated sample of all 4 players with the team
split known. Only 8 mappings of such a set to players {1, 2 | 3, 4} respect the
teams, so linking anchor sets over the match is a small, well-posed matching
problem:

  1. reference  - the best anchor set fixes the numbering by court region (team A =
                  its near pair, left = lower number inside a team; see region_mapping)
  2. growth     - repeatedly accept the anchor set whose best team-consistent
                  mapping beats the runner-up by the largest margin, matching
                  against a gallery grown from every set accepted so far; a set
                  sharing a tracklet with an accepted one must agree with it
  3. refinement - re-pick every set's mapping against a gallery built from all
                  *other* sets and vote per tracklet, until the votes are stable

Only the reference set becomes hard labels for the global assignment (to break
the player-permutation symmetry); every other seed only feeds the gallery, so the
solver can still overrule it. In live mode, later reseeds start from the labels
already shown (`known_labels`) instead of a reference, so numbering stays stable.
"""

from __future__ import annotations

import itertools
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from padel_reid.court.homography import COURT_WIDTH_M
from padel_reid.identity.player_gallery import NUM_PLAYERS, PlayerGallery
from padel_reid.identity.tracklet_descriptors import TrackletDescriptors

TEAMS = [[0, 1], [2, 3]]           # team A = players 1 & 2, team B = players 3 & 4 (0-based)


@dataclass
class AnchorSet:
    frame: int
    near: tuple[int, ...]           # tracklets on the near half at `frame`
    far: tuple[int, ...]
    score: float                    # weakest member's evidence (seconds)

    @property
    def members(self) -> tuple[int, ...]:
        return self.near + self.far


@dataclass
class SeedResult:
    seed_labels: dict[int, int]     # tracklet -> player (0-3), soft: gallery evidence only
    fixed_labels: dict[int, int]    # tracklet -> player, hard constraints for the solver
    teams: list[list[int]] = field(default_factory=lambda: [list(t) for t in TEAMS])
    diagnostics: dict = field(default_factory=dict)


def clean_tracklets(summary: pd.DataFrame, evidence_seconds: np.ndarray, fps: float, seeding_cfg) -> np.ndarray:
    """Long, untruncated tracklets outside interactions with enough crop evidence."""
    min_frames = seeding_cfg.min_tracklet_seconds * fps
    clean = summary[(summary["frames"] >= min_frames) & ~summary["in_interaction"]
                    & (summary["truncated_fraction"] < 0.5)
                    & (evidence_seconds[summary["tracklet_id"].to_numpy()] > seeding_cfg.min_evidence_seconds)]
    return clean["tracklet_id"].to_numpy()


def find_anchor_sets(table: pd.DataFrame, summary: pd.DataFrame, evidence_seconds: np.ndarray, fps: float,
                     seeding_cfg, min_members: int = 4) -> list[AnchorSet]:
    """Sampled moments where `min_members`+ clean tracklets coexist with at most 2 per half."""
    clean = clean_tracklets(summary, evidence_seconds, fps, seeding_cfg)
    step = max(int(round(seeding_cfg.sample_seconds * fps)), 1)
    rows = table[table["tracklet_id"].isin(clean) & (table["frame"] % step == 0)]

    sets, seen = [], set()
    for frame, group in rows.groupby("frame"):
        near = tuple(sorted(group.loc[group["side"] == 1, "tracklet_id"].astype(int)))
        far = tuple(sorted(group.loc[group["side"] == 0, "tracklet_id"].astype(int)))
        if len(near) > 2 or len(far) > 2 or len(near) + len(far) < min_members or (near, far) in seen:
            continue
        seen.add((near, far))
        sets.append(AnchorSet(int(frame), near, far, float(evidence_seconds[list(near + far)].min())))
    return sets


def team_consistent_mappings(anchor: AnchorSet) -> list[dict[int, int]]:
    """Every tracklet -> player mapping that keeps each half's tracklets on one team."""
    out = []
    for near_team in (0, 1):
        near_players, far_players = TEAMS[near_team], TEAMS[1 - near_team]
        for near_perm in itertools.permutations(near_players, len(anchor.near)):
            for far_perm in itertools.permutations(far_players, len(anchor.far)):
                out.append(dict(zip(anchor.members, near_perm + far_perm)))
    return out


def _team_a_near(anchor: AnchorSet, mapping: dict[int, int]) -> bool:
    return any(mapping[t] in TEAMS[0] for t in anchor.near) or any(mapping[t] in TEAMS[1] for t in anchor.far)


def region_mapping(near, far, summary: pd.DataFrame, free_players=range(NUM_PLAYERS),
                   team_a_near: bool = True) -> dict[int, int]:
    """Number tracklets by court region as seen from the camera: near-left, near-right,
    far-left, far-right = players 1-4 while team A is on the near half.

    Two tracklets on one half are ordered by court_x; a lone one goes by which side of
    the centre line it stands on. Only `free_players` are handed out: a tracklet whose
    own slot is taken gets its partner's slot if that is free, otherwise no number.
    """
    free = set(free_players)
    out = {}
    for half, team in ((near, TEAMS[0 if team_a_near else 1]), (far, TEAMS[1 if team_a_near else 0])):
        half = sorted(half, key=lambda t: summary.at[t, "court_x"])
        left = {t: i == 0 if len(half) == 2 else summary.at[t, "court_x"] < COURT_WIDTH_M / 2
                for i, t in enumerate(half)}
        for fallback in (False, True):            # own slots first, so a taken slot can't steal one
            for t in half:
                if t in out:
                    continue
                own, partner = team if left[t] else team[::-1]
                for p in ((own, partner) if fallback else (own,)):
                    if p in free:
                        out[t] = p
                        free.discard(p)
                        break
    return out


def _reference_mapping(anchor: AnchorSet, summary: pd.DataFrame) -> dict[int, int]:
    """Deterministic numbering by court region (team A = near pair, left = lower number)."""
    return region_mapping(anchor.near, anchor.far, summary)


class _AnchorLinker:
    def __init__(self, descriptors: TrackletDescriptors, summary: pd.DataFrame, fps: float, cfg):
        self.descriptors = descriptors
        self.summary = summary
        self.near = summary["near_fraction"].to_numpy() > 0.5
        self.cfg = cfg
        self.continuity_frames = cfg.identity.seeding.side_continuity_seconds * fps
        self.side_penalty = cfg.identity.seeding.side_switch_penalty

    def gallery(self, labels: dict[int, int]) -> PlayerGallery:
        gallery = PlayerGallery(self.descriptors.embedding.shape[1], self.descriptors.color.shape[1],
                                self.cfg.reid.bank_size, self.cfg.reid.color_weight)
        for t, player in sorted(labels.items(), key=lambda kv: self.summary.at[kv[0], "start_frame"]):
            if t in self.descriptors.crop_rows:
                _, emb, quality = self.descriptors.crop_rows[t]
                gallery.add(player, int(self.near[t]), emb, quality, self.descriptors.color[t])
        return gallery

    def similarity(self, gallery: PlayerGallery, tracklet: int) -> np.ndarray:
        return gallery.scores(self.descriptors.embedding[tracklet], self.descriptors.color[tracklet],
                              int(self.near[tracklet]))

    def mapping_costs(self, anchor: AnchorSet, gallery: PlayerGallery, decided: list[tuple[AnchorSet, bool]],
                      labels: dict[int, int] | None = None) -> list[tuple[float, dict[int, int]]]:
        """(cost, mapping) for each team-consistent mapping, cheapest first.

        cost = -summed gallery similarity, + side_switch_penalty if team A would be on
        the other half than in the nearest decided set within side_continuity_seconds.
        Mappings that contradict `labels` on a shared tracklet are left out.
        """
        sims = {t: self.similarity(gallery, t) for t in anchor.members}
        nearby = [(abs(other.frame - anchor.frame), a_near) for other, a_near in decided
                  if other is not anchor and abs(other.frame - anchor.frame) <= self.continuity_frames]
        expected_a_near = min(nearby)[1] if nearby else None
        out = []
        for mapping in team_consistent_mappings(anchor):
            if labels and any(t in labels and labels[t] != k for t, k in mapping.items()):
                continue
            cost = -sum(float(sims[t][k]) for t, k in mapping.items())
            if expected_a_near is not None and _team_a_near(anchor, mapping) != expected_a_near:
                cost += self.side_penalty
            out.append((cost, mapping))
        return sorted(out, key=lambda cm: cm[0])

    def grow(self, anchors: list[AnchorSet], labels: dict[int, int],
             accepted: dict[int, tuple]) -> tuple[dict[int, int], dict[int, tuple]]:
        """Confident-first growth from an initial label state (the reference set, or labels
        already shown live). Returns (labels, accepted set index -> (mapping, margin))."""
        min_margin = self.cfg.identity.seeding.min_margin
        labels, accepted = dict(labels), dict(accepted)
        pending = [i for i in range(len(anchors)) if i not in accepted]
        while pending:
            gallery = self.gallery(labels)
            decided = [(anchors[i], _team_a_near(anchors[i], m)) for i, (m, _) in accepted.items()]
            best = None
            for i in pending:
                costs = self.mapping_costs(anchors[i], gallery, decided, labels)
                if not costs:
                    continue
                margin = costs[1][0] - costs[0][0] if len(costs) > 1 else np.inf
                if best is None or margin > best[1]:
                    best = (i, margin, costs[0][1])
            if best is None or best[1] < min_margin:
                break
            i, margin, chosen = best
            accepted[i] = (chosen, margin)
            labels.update(chosen)
            pending.remove(i)
        return labels, accepted

    def refine(self, anchors: list[AnchorSet], accepted: dict[int, tuple], frozen: set[int],
               fixed: dict[int, int], iterations: int) -> tuple[dict[int, int], int]:
        """Leave-one-set-out re-picking with per-tracklet voting. Returns (labels, iterations run).

        Sets in `frozen` keep their mapping; every re-picked mapping must agree with `fixed`.
        """
        mappings = {i: m for i, (m, _) in accepted.items()}
        labels = _vote(mappings)
        for iteration in range(1, iterations + 1):
            decided = [(anchors[i], _team_a_near(anchors[i], m)) for i, m in mappings.items()]
            changed = False
            for i in list(mappings):
                if i in frozen:
                    continue
                others = {t: k for t, k in labels.items() if t not in anchors[i].members}
                costs = self.mapping_costs(anchors[i], self.gallery({**others, **fixed}), decided, fixed)
                if not costs:
                    mappings.pop(i)
                    changed = True
                elif costs[0][1] != mappings[i]:
                    mappings[i], changed = costs[0][1], True
            labels = _vote(mappings)
            if not changed:
                return labels, iteration
        return labels, iterations


def _vote(mappings: dict[int, dict[int, int]]) -> dict[int, int]:
    """Per-tracklet majority over the accepted sets; split votes are dropped."""
    votes: dict[int, Counter] = {}
    for mapping in mappings.values():
        for t, k in mapping.items():
            votes.setdefault(t, Counter())[k] += 1
    labels = {}
    for t, counter in votes.items():
        (top, n), *rest = counter.most_common()
        if not rest or n > rest[0][1]:
            labels[t] = top
    return labels


def cluster_seed_labels(descriptors: TrackletDescriptors, summary: pd.DataFrame, min_frames: int) -> dict[int, int]:
    """Last resort when no anchor sets exist: k-means of clean tracklets into 4 players.

    Clusters often split by near/far scale rather than person, and their numbering
    ignores teams, so they are only used as soft gallery seeds.
    """
    from sklearn.cluster import KMeans

    clean = summary[(summary["frames"] >= min_frames) & ~summary["in_interaction"]]["tracklet_id"].to_numpy()
    if len(clean) < NUM_PLAYERS:
        return {}
    features = np.hstack([descriptors.embedding[clean], 0.5 * descriptors.color[clean]])
    clusters = KMeans(NUM_PLAYERS, n_init=10, random_state=0).fit_predict(features)
    return {int(t): int(c) for t, c in zip(clean, clusters)}


def discover_seed_labels(table: pd.DataFrame, summary: pd.DataFrame, descriptors: TrackletDescriptors,
                         evidence_seconds: np.ndarray, fps: float, cfg,
                         known_labels: dict[int, int] | None = None) -> SeedResult:
    """Gallery seeds and symmetry-breaking fixed labels without any manual labels.

    known_labels (live reseeding): labels already shown on screen. Growth starts from
    them instead of a fresh reference set, so the numbering never changes, and no new
    fixed labels are returned.
    """
    seeding_cfg = cfg.identity.seeding
    anchors, members = [], 4
    for members in (4, 3, 2):     # smaller sets: short clips, breaks, a player out of view
        anchors = find_anchor_sets(table, summary, evidence_seconds, fps, seeding_cfg, members)
        if len(anchors) >= seeding_cfg.min_anchor_sets:
            break

    if not anchors:
        if known_labels:
            return SeedResult({}, {}, diagnostics={"method": "none", "seed_tracklets": 0})
        warnings.warn("No anchor sets (2+ clean coexisting players) found: seeding the gallery by clustering. "
                      "Identities may be wrong and teams arbitrary.")
        seeds = cluster_seed_labels(descriptors, summary, int(seeding_cfg.min_tracklet_seconds * fps))
        if not seeds:
            warnings.warn("Too few clean tracklets to seed the gallery; nothing can be labelled reliably.")
        return SeedResult(seeds, {}, diagnostics={"method": "clustering" if seeds else "none",
                                                  "seed_tracklets": len(seeds)})

    linker = _AnchorLinker(descriptors, summary, fps, cfg)
    reference = None
    if known_labels:
        fixed, frozen = {}, set()
        _, accepted = linker.grow(anchors, known_labels, {})
        labels, iterations = linker.refine(anchors, accepted, frozen, known_labels, seeding_cfg.refine_iterations)
        labels = {t: k for t, k in labels.items() if t not in known_labels}
    else:
        best_score = max(a.score for a in anchors)
        reference = min((a for a in anchors if a.score >= 0.9 * best_score), key=lambda a: a.frame)
        fixed = _reference_mapping(reference, summary)
        frozen = {anchors.index(reference)}
        _, accepted = linker.grow(anchors, fixed, {anchors.index(reference): (fixed, np.inf)})
        labels, iterations = linker.refine(anchors, accepted, frozen, {}, seeding_cfg.refine_iterations)
    margins = [m for i, (_, m) in accepted.items() if i not in frozen]
    forced = sum(1 for m in margins if not np.isfinite(m))      # fully determined by shared tracklets
    margins = [m for m in margins if np.isfinite(m)]
    return SeedResult(labels, fixed, diagnostics={
        "method": "anchor_sets",
        "anchor_set_size": members,
        "anchor_sets_found": len(anchors),
        "anchor_sets_accepted": len(accepted),
        "anchor_sets_forced_by_shared_tracklets": forced,
        "seed_tracklets": len(labels),
        "seed_tracklets_per_player": {k + 1: sum(1 for v in labels.values() if v == k) for k in range(NUM_PLAYERS)},
        "reference_frame": reference.frame if reference else None,
        "reference_tracklets": {int(t): k + 1 for t, k in fixed.items()},
        "median_margin": float(np.median(margins)) if margins else None,
        "refine_iterations": iterations,
    })
