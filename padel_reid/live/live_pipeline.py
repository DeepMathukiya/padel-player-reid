"""Live mode: every frame is decoded once and goes detect -> track -> tracklet -> embed
as it arrives; identities come from a sliding-window solve; frames are shown
`live.lag_seconds` late.

    FrameSource ─► perception (this thread) ─► LiveHistory ─► FrameEmitter ─► window / mp4
     (thread)      YOLO + BoT-SORT + online       ▲    ▲         (frame t - lag)
                   tracklets + crop embeddings    │    └── latest labels
                                                  └── IdentityWorker: CP-SAT every solve interval
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from padel_reid.common.cache_paths import CachePaths
from padel_reid.detection.player_detector import build_tracker
from padel_reid.identity.anchor_seeding import TEAMS
from padel_reid.identity.changeover_detector import changeovers_from_sides
from padel_reid.live.frame_embedder import FrameEmbedder
from padel_reid.live.frame_emitter import FrameEmitter
from padel_reid.live.frame_source import open_frame_source
from padel_reid.live.live_history import LiveHistory
from padel_reid.live.online_tracklet_builder import TRACKLET_COLUMNS, BuilderOutput, OnlineTrackletBuilder
from padel_reid.live.sliding_window_assigner import IdentityWorker, SeedingWorker, SlidingWindowAssigner

_FRAME = TRACKLET_COLUMNS.index("frame")


class _Perception:
    """Builder output -> history rows, crop embeddings and closed-tracklet summaries."""

    def __init__(self, builder: OnlineTrackletBuilder, embedder: FrameEmbedder, history: LiveHistory,
                 emitter: FrameEmitter):
        self.builder, self.embedder, self.history, self.emitter = builder, embedder, history, emitter

    def consume(self, out: BuilderOutput, newest_frame: int, final: bool = False) -> None:
        self.history.add_detections([], out.pairs)
        wanted = []
        for row_index, row, position in self.history.add_rows(out.rows):
            if self.embedder.wants(position):
                wanted.append((row_index, row))
        for tracklet, last_position, last_row in out.closed:
            if not self.embedder.wants(last_position):            # the batch embedder always takes the last row
                wanted.append((self.history.last_row_index(tracklet), last_row))
        for row_index, row in wanted:
            if (frame := self.emitter.frame(row[_FRAME])) is not None:
                self.embedder.add(row_index, row, frame)
        self.history.add_crops(self.embedder.flush(newest_frame, force=final))
        for tracklet, _, _ in out.closed:
            self.history.close(tracklet)


def _percentiles(values: list[float], scale: float = 1.0) -> dict:
    if not values:
        return {}
    return {"p50": round(float(np.percentile(values, 50)) * scale, 1),
            "p95": round(float(np.percentile(values, 95)) * scale, 1),
            "max": round(float(np.max(values)) * scale, 1)}


def run_live(cfg, source: str | None = None) -> dict:
    """Run the live pipeline until the source ends or the window is closed. Returns run statistics."""
    from padel_reid.api import build_player_detector

    paths = CachePaths.from_config(cfg)
    frames_source = open_frame_source(source, cfg)
    fps, frame_size = frames_source.fps, frames_source.frame_size
    detector = build_player_detector(cfg, frame_size=frame_size)
    tracker = build_tracker(cfg.detection.tracker_config)
    builder = OnlineTrackletBuilder(cfg.tracking)
    history = LiveHistory(fps, frame_size, cfg.identity.border_margin_fraction, cfg.reid.embed_stride,
                          cfg.reid.get("side_centering", False))
    output_video = Path(cfg.output_dir) / "padel_player_ids_live.mp4" if cfg.live.write_video else None
    emitter = FrameEmitter(cfg, history, fps, frame_size, output_video, cfg.live.display)
    perception = _Perception(builder, FrameEmbedder(cfg.reid, cfg.live, cfg.get("device", "cuda")), history, emitter)
    assigner = SlidingWindowAssigner(cfg, history, fps)
    worker = IdentityWorker(assigner, history, int(cfg.live.solve_interval_seconds * fps))
    seeding = SeedingWorker(assigner, history)
    max_buffered = emitter.lag + 3 * worker.interval        # frames in memory before perception waits

    # Warm up CUDA and both networks before frames arrive: the first calls take ~1 s,
    # during which a camera (or paced file) would already be dropping frames.
    blank = np.zeros((frame_size[1], frame_size[0], 3), np.uint8)
    detector.raw_detections_batch([blank])
    perception.embedder.embedder.embed([blank[:256, :128]])

    pad = cfg.tracking.interaction_pad_frames
    newest, processed, perception_ms = -1, 0, []
    first_frame, max_frames = None, (cfg.live.max_seconds * fps) if cfg.live.max_seconds else None
    started = time.perf_counter()
    worker.start()
    seeding.start()
    frames_source.start()
    try:
        while not emitter.stopped:
            batch = frames_source.get_batch(cfg.detection.batch_size)
            if not batch:
                break
            t0 = time.perf_counter()
            raw = detector.raw_detections_batch([frame for _, frame, _ in batch])
            final = BuilderOutput()
            for (frame_index, frame, arrival), (raw_xyxy, raw_conf) in zip(batch, raw):
                emitter.add_frame(frame_index, frame, arrival)
                xyxy, conf = detector.filter_detections(raw_xyxy, raw_conf)
                tracked = detector.track_frame(tracker, frame_index, frame, xyxy, conf)
                history.add_detections(tracked, [])
                final.extend(builder.push(frame_index, tracked))
                newest = frame_index
            perception.consume(final, newest)
            first_frame = batch[0][0] if first_frame is None else first_frame
            # Solves only see rows whose crops are embedded (crops wait for a batch, up to
            # live.embed_max_wait_frames); see LiveHistory.snapshot.
            waiting = perception.embedder.oldest_pending_frame()
            history.frontier = newest - pad if waiting is None else min(newest - pad, waiting - 1)
            perception_ms.append((time.perf_counter() - t0) * 1000 / len(batch))
            processed += len(batch)
            emitter.emit(emitter.ready(newest))
            emitter.poll_keys()
            while len(emitter.frames) > max_buffered and worker.is_alive() and not emitter.stopped:
                time.sleep(0.002)                            # identity thread is behind: wait for it
                emitter.emit(emitter.ready(newest))
                emitter.poll_keys()
            if max_frames is not None and newest - first_frame >= max_frames:
                break                                        # session length reached (cameras never end)
            if worker.error or seeding.error:
                raise RuntimeError("identity thread failed") from (worker.error or seeding.error)
    except KeyboardInterrupt:
        print("interrupted: finishing the frames already read")
    finally:
        frames_source.stop()
        perception.consume(builder.flush(), newest, final=True)
        history.frontier = newest
        worker.stop()
        if emitter.stopped:
            seeding.stopping.set()       # user quit: don't wait for a reseed in progress (can take ~20 s)
        else:
            seeding.stop()
        assigner.step()                                      # final solve over the tail
        emitter.emit(emitter.ready(newest, final=True))
        emitter.close()
    seconds = time.perf_counter() - started

    history.flush(paths)
    emitter.assigned_table().to_parquet(paths.root / "assigned_live.parquet")
    state = history.state
    stats = {
        "frames": processed, "seconds": round(seconds, 2), "fps": round(processed / seconds, 2) if seconds else None,
        "source_fps": fps, "dropped_frames": frames_source.dropped, "lag_seconds": cfg.live.lag_seconds,
        "display_delay_s": _percentiles(emitter.delays),
        "perception_ms_per_frame": _percentiles(perception_ms),
        "solve_ms": _percentiles(assigner.solve_ms), "solves": len(assigner.solve_ms),
        "reseed_ms": _percentiles(assigner.reseed_ms), "reseeds": len(assigner.reseed_ms),
        "failed_solves": assigner.failures,
        "tracklets": len(history.tracklet_rows), "tracklets_labelled": len(state.committed),
    }
    history.write_events(paths.root / "identity_events_live.json", {
        "changeovers": changeovers_from_sides(state.committed_sides, int(cfg.identity.window_seconds * fps), fps,
                                              TEAMS),
        "reference_tracklets": {int(t): k + 1 for t, k in state.reference.items()},
        "stats": stats,
    })
    timing = json.loads(paths.timing.read_text()) if paths.timing.exists() else {}
    timing["live"] = stats
    paths.timing.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return stats
