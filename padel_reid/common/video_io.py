"""Video reading/writing helpers built on OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int


def read_video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = VideoInfo(
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def resolve_frame_range(info: VideoInfo, start: int | None, end: int | None) -> tuple[int, int]:
    start = max(0, start or 0)
    end = info.frame_count if end is None else min(end, info.frame_count)
    return start, end


def iterate_frames(path: str | Path, start: int = 0, end: int | None = None,
                   wanted: set[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) sequentially.

    If `wanted` is given, frames not in it are skipped with grab() (no decode),
    which is much faster than seeking for sparse but scattered frame sets.
    """
    cap = cv2.VideoCapture(str(path))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    index = start
    last_wanted = max(wanted) if wanted else None
    try:
        while end is None or index < end:
            if last_wanted is not None and index > last_wanted:
                break
            if wanted is not None and index not in wanted:
                if not cap.grab():
                    break
                index += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


def prefetch_frames(frames: Iterator[tuple[int, np.ndarray]], buffer_size: int = 16) -> Iterator[tuple[int, np.ndarray]]:
    """Decode frames on a background thread so decoding overlaps GPU inference."""
    import queue
    import threading

    buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
    done = object()

    def producer() -> None:
        try:
            for item in frames:
                buffer.put(item)
        finally:
            buffer.put(done)

    threading.Thread(target=producer, daemon=True).start()
    while (item := buffer.get()) is not done:
        yield item


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
