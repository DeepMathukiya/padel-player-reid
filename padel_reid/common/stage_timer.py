"""Record per-stage wall time and frames processed, for the FPS report."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@contextmanager
def timed_stage(timing_file: Path, stage: str, frames: int = 0):
    """Time a block and store {seconds, frames, fps} under `stage` in timing_file.

    `frames` may be updated by the caller through the yielded dict.
    """
    record = {"frames": frames}
    start = time.perf_counter()
    yield record
    seconds = time.perf_counter() - start
    data = _load(timing_file)
    data[stage] = {
        "seconds": round(seconds, 2),
        "frames": record["frames"],
        "fps": round(record["frames"] / seconds, 2) if seconds > 0 and record["frames"] else None,
    }
    timing_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
