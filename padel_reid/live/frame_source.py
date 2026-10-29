"""Where live frames come from: a recorded file played as if live, or a camera.

Both read on a background thread and hand out (frame_index, BGR frame, arrival
time). A camera never waits for us: when processing falls behind, the oldest
queued frame is dropped (and counted). A file either runs as fast as possible
(blocking, nothing dropped) or is paced at its own fps and then behaves like a
camera.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import cv2

from padel_reid.common.video_io import iterate_frames, read_video_info, resolve_frame_range

_DONE = object()


class FrameSource:
    fps: float
    frame_size: tuple[int, int]          # (width, height)

    def __init__(self, queue_frames: int, drop_when_full: bool):
        self.queue: queue.Queue = queue.Queue(maxsize=queue_frames)
        self.drop_when_full = drop_when_full
        self.dropped = 0
        self.finished = False
        self._stop = threading.Event()

    def _produce(self) -> None:
        raise NotImplementedError

    def _put(self, item) -> bool:
        """Queue one item; False when the source was stopped."""
        while not self._stop.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                if self.drop_when_full:
                    try:
                        self.queue.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        pass
        return False

    def _run(self) -> None:
        try:
            self._produce()
        finally:
            self.queue.put(_DONE)

    def start(self) -> "FrameSource":
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def get_batch(self, max_frames: int) -> list[tuple[int, object, float]]:
        """Block for one frame, then take whatever else is already waiting (never waits for a full batch)."""
        if self.finished:
            return []
        batch = []
        while True:                    # short timeouts: a plain blocking get() delays Ctrl-C on Windows
            try:
                item = self.queue.get(timeout=0.2)
                break
            except queue.Empty:
                continue
        while item is not _DONE:
            batch.append(item)
            if len(batch) >= max_frames:
                break
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
        else:
            self.finished = True
        return batch

    def stop(self) -> None:
        self._stop.set()


class FileSource(FrameSource):
    def __init__(self, path: str | Path, start: int | None, end: int | None, realtime: bool, queue_frames: int,
                 speed: float = 1.0):
        # Only true real time behaves like a camera (drops when behind); a sped-up replay
        # keeps every frame and simply plays slower if processing can't keep up.
        super().__init__(queue_frames, drop_when_full=realtime and speed == 1.0)
        self.speed = speed
        info = read_video_info(path)
        self.path, self.realtime = path, realtime
        self.fps, self.frame_size = info.fps, (info.width, info.height)
        self.start_frame, self.end_frame = resolve_frame_range(info, start, end)

    def _produce(self) -> None:
        t0 = time.perf_counter()
        for frame_index, frame in iterate_frames(self.path, self.start_frame, self.end_frame):
            if self.realtime:
                wait = t0 + (frame_index - self.start_frame) / (self.fps * self.speed) - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            if not self._put((frame_index, frame, time.perf_counter())):
                return


class CameraSource(FrameSource):
    """RTSP/HTTP URL or a local camera index. Frame indices count captured frames, so
    a dropped frame leaves a gap and time stays consistent (index / fps)."""

    def __init__(self, url: str | int, queue_frames: int):
        super().__init__(queue_frames, drop_when_full=True)
        self.capture = cv2.VideoCapture(url)
        if not self.capture.isOpened():
            raise ConnectionError(f"Cannot open camera stream: {url}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _produce(self) -> None:
        frame_index = 0
        try:
            while not self._stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    return
                if not self._put((frame_index, frame, time.perf_counter())):
                    return
                frame_index += 1
        finally:
            self.capture.release()


def open_frame_source(source: str | None, cfg) -> FrameSource:
    """`source`: a file path (default: cfg.video_path), an rtsp/http(s) URL, or a camera index."""
    source = source if source is not None else cfg.video_path
    if str(source).isdigit():
        return CameraSource(int(source), cfg.live.camera_queue_frames)
    if "://" in str(source):
        return CameraSource(str(source), cfg.live.camera_queue_frames)
    return FileSource(source, cfg.start_frame, cfg.end_frame, cfg.live.realtime_playback,
                      cfg.live.camera_queue_frames if cfg.live.realtime_playback else 16,
                      float(cfg.live.get("playback_speed", 1.0)))
