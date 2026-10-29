"""Live identity: CP-SAT over a sliding window, with on-screen decisions fixed.

Every solve looks at the tracklets of the last `live.window_seconds`. Tracklets
already shown on screen keep their player (fixed labels) and team-side windows
already shown keep their side, so the solver can only decide what has not been
displayed yet; IDs never flip on screen.

Seeding is online too: until the first moment with 4 clean players (2 per half)
nothing is numbered and boxes show "?". That anchor set fixes Player 1-4. Every
`reseed_seconds` a separate thread re-runs the anchor-set seeding over the whole
session, growing from that reference set exactly as the offline seeding does, and
the gallery is rebuilt from the result. The solver's own decisions are not fed
back into the gallery (`live.gallery_from_shown`): in tests that let one early
mistake teach the gallery the wrong player and swap partners for minutes. The
re-link gate is recalibrated from the clean tracklets seen so far.
"""

from __future__ import annotations

import copy
import threading
import time
import traceback

import numpy as np

from padel_reid.common.config_loader import ConfigNode
from padel_reid.identity.anchor_seeding import TEAMS, discover_seed_labels, find_anchor_sets
from padel_reid.identity.exit_reentry_handler import exit_entry_flags
from padel_reid.identity.global_assignment import (AssignmentInputs, drop_conflicting_labels,
                                                   exclusivity_constraints, solve_global_assignment)
from padel_reid.identity.identity_pipeline import (appearance_log_prob, confident_assignments, relative_cost,
                                                   window_side_counts)
from padel_reid.identity.player_gallery import PlayerGallery
from padel_reid.identity.predictive_relinker import calibrate_gate, default_gate, find_relink_candidates
from padel_reid.live.live_history import LiveHistory, Snapshot


class SlidingWindowAssigner:
    def __init__(self, cfg, history: LiveHistory, fps: float):
        self.cfg, self.history, self.fps = cfg, history, fps
        live = cfg.live
        self.window_frames = int(live.window_seconds * fps)
        self.reseed_frames = int(live.reseed_seconds * fps)
        self.reseed_history_frames = int(live.reseed_history_seconds * fps) if live.reseed_history_seconds else 10 ** 9
        self.gallery_from_shown = live.gallery_from_shown
        self.gate_frames = int(live.gate_recalibrate_seconds * fps)
        self.side_window_frames = int(cfg.identity.window_seconds * fps)
        self.relink_gap = int(cfg.identity.relink_max_gap_seconds * fps)
        self.identity_cfg = ConfigNode({**cfg.identity, "solver_time_limit_s": live.solver_time_limit_s})
        # The first seeding may only have a single 4-player anchor set to go on.
        self.seed_cfg = copy.deepcopy(cfg)
        self.seed_cfg.identity.seeding.min_anchor_sets = 1
        self.gate = default_gate(fps, cfg.identity.max_speed_mps)
        self.last_reseed = self.last_gate = -10 ** 9
        self.solve_ms: list[float] = []
        self.reseed_ms: list[float] = []
        self.decisions: dict[int, tuple] = {}        # tracklet -> (player, appearance cost row, evidence s, interaction)
        self.failures = 0
        self._crop_cache: dict[int, tuple[int, tuple]] = {}

    # ---- seeding ----

    def _initial_seeding(self, snap: Snapshot) -> None:
        if not find_anchor_sets(snap.table, snap.summary, snap.evidence_seconds, self.fps,
                                self.cfg.identity.seeding, min_members=4):
            return
        seed = discover_seed_labels(snap.table, snap.summary, snap.descriptors, snap.evidence_seconds,
                                    self.fps, self.seed_cfg)
        with self.history.lock:
            state = self.history.state
            state.reference = snap.to_global(seed.fixed_labels)
            state.seeds = snap.to_global(seed.seed_labels)
            state.seeding.append({"frame": snap.frontier, "kind": "initial", **seed.diagnostics})
        self.last_reseed = snap.frontier

    def reseed(self) -> None:
        """Re-derive the gallery seeds from all anchor sets so far (runs on the seeding thread)."""
        start = time.perf_counter()
        snap = self.history.snapshot(self.reseed_history_frames)
        if snap is None:
            return
        # Grow from the reference only (not from what was shown), so a past mistake is not
        # built into the gallery: the seeds are re-derived from the anchor sets every time.
        with self.history.lock:
            known = snap.to_local(self.history.state.reference)
        if known:
            seed = discover_seed_labels(snap.table, snap.summary, snap.descriptors, snap.evidence_seconds,
                                        self.fps, self.cfg, known_labels=known)
            with self.history.lock:
                self.history.state.seeds = snap.to_global(seed.seed_labels)
                self.history.state.seeding.append({"frame": snap.frontier, "kind": "reseed", **seed.diagnostics})
        if snap.frontier - self.last_gate >= self.gate_frames:
            gate, errors = calibrate_gate(snap.table, self.fps, self.cfg.identity.max_speed_mps)
            if not errors.empty:
                self.gate = gate
            self.last_gate = snap.frontier
        self.last_reseed = snap.frontier
        self.reseed_ms.append((time.perf_counter() - start) * 1000)

    # ---- gallery ----

    def _gallery(self, labels: dict[int, int], descriptor_dims: tuple[int, int]) -> PlayerGallery:
        gallery = PlayerGallery(*descriptor_dims, self.cfg.reid.bank_size, self.cfg.reid.color_weight)
        evidence = self.history.gallery_evidence(labels)
        for t in sorted(evidence, key=lambda t: evidence[t][0]):
            _, near, crops = evidence[t]
            cached = self._crop_cache.get(t)
            if cached is None or cached[0] != len(crops):
                emb = np.stack([c[1] for c in crops]).astype(np.float32)
                quality = np.asarray([c[3] for c in crops], np.float32) + 1e-3
                color = (np.stack([c[2] for c in crops]).astype(np.float32) * quality[:, None]).sum(0) / quality.sum()
                cached = self._crop_cache[t] = (len(crops), (emb, quality, color / max(np.linalg.norm(color), 1e-6)))
            emb, quality, color = cached[1]
            gallery.add(labels[t], near, emb, quality, color)
        return gallery

    # ---- one live step ----

    def step(self) -> None:
        """Seed if needed, solve the current window and publish the latest decisions."""
        start = time.perf_counter()
        snap = self.history.snapshot(self.window_frames)
        if snap is None:
            with self.history.lock:
                self.history.state.solved_through = self.history.frontier
            return
        state = self.history.state
        if not state.reference:
            self._initial_seeding(snap)
        with self.history.lock:
            reference, seeds = dict(state.reference), dict(state.seeds)
            committed, confident = dict(state.committed), set(state.committed_confident)
            committed_sides = dict(state.committed_sides)
        if not reference:
            with self.history.lock:
                state.solved_through = snap.frontier
            return

        summary = snap.summary
        shown = {t: committed[t] for t in confident if t in committed} if self.gallery_from_shown else {}
        gallery_labels = {**seeds, **shown, **reference}
        gallery = self._gallery(gallery_labels, (snap.descriptors.embedding.shape[1], snap.descriptors.color.shape[1]))
        cost = relative_cost(appearance_log_prob(snap.descriptors, gallery, summary))
        exits, entries = exit_entry_flags(summary)
        exclusivity = exclusivity_constraints(snap.table, summary, self.fps, self.cfg.identity.max_speed_mps)
        fixed = drop_conflicting_labels(snap.to_local({**reference, **committed}), summary, self.fps,
                                        self.cfg.identity.max_speed_mps, self.relink_gap, exits, entries,
                                        exclusivity.pairs)
        counts = window_side_counts(snap.table, self.side_window_frames)
        first_window = int(counts["window"].min())
        try:
            result = solve_global_assignment(AssignmentInputs(
                summary=summary, appearance_cost=cost, evidence_seconds=snap.evidence_seconds,
                window_counts=counts,
                relink_candidates=find_relink_candidates(snap.table, summary, self.gate, self.relink_gap),
                fixed_labels=fixed, teams=TEAMS, exits=exits, entries=entries, fps=self.fps,
                exclusivity=exclusivity.cliques,
                fixed_sides={w: s for w, s in committed_sides.items() if w >= first_window - 1}), self.identity_cfg)
        except RuntimeError:
            self.failures += 1          # keep the previous decisions; the next window may be solvable
            with self.history.lock:
                state.solved_through = snap.frontier
            return
        confident_now = confident_assignments(result.labels, cost, summary, snap.evidence_seconds)
        for t, k in result.labels.items():          # why each label was chosen (commit rule, diagnostics)
            self.decisions[int(snap.global_ids[t])] = (k, cost[t].copy(), float(snap.evidence_seconds[t]),
                                                       bool(summary.at[t, "in_interaction"]))
        with self.history.lock:
            state.latest_labels = snap.to_global(result.labels)
            state.latest_confident = set(snap.to_global(confident_now))
            state.latest_sides.update(result.team_a_near)
            state.solved_through = snap.frontier
        self.solve_ms.append((time.perf_counter() - start) * 1000)


class SeedingWorker(threading.Thread):
    """Re-runs the seeding every `live.reseed_seconds` of video, off the solve thread
    (late in a match one reseed takes seconds; solves must keep their 1 s rhythm)."""

    def __init__(self, assigner: SlidingWindowAssigner, history: LiveHistory):
        super().__init__(daemon=True)
        self.assigner, self.history = assigner, history
        self.stopping = threading.Event()
        self.error: BaseException | None = None

    def run(self) -> None:
        while not self.stopping.is_set():
            with self.history.lock:
                has_reference = bool(self.history.state.reference)
            if not has_reference or self.history.frontier - self.assigner.last_reseed < self.assigner.reseed_frames:
                time.sleep(0.05)
                continue
            try:
                self.assigner.reseed()
            except Exception as exc:
                self.error = exc
                traceback.print_exc()
                return

    def stop(self) -> None:
        self.stopping.set()
        self.join()


class IdentityWorker(threading.Thread):
    """Runs `step` each time `live.solve_interval_seconds` of new video has become final."""

    def __init__(self, assigner: SlidingWindowAssigner, history: LiveHistory, interval_frames: int):
        super().__init__(daemon=True)
        self.assigner, self.history, self.interval = assigner, history, max(interval_frames, 1)
        self.stopping = threading.Event()
        self.error: BaseException | None = None

    def run(self) -> None:
        last = -10 ** 9
        while not self.stopping.is_set():
            frontier = self.history.frontier
            if frontier - last < self.interval:
                time.sleep(0.005)
                continue
            try:
                self.assigner.step()
            except Exception as exc:          # surface in the main thread instead of dying silently
                self.error = exc
                traceback.print_exc()
                return
            last = frontier

    def stop(self) -> None:
        self.stopping.set()
        self.join()
