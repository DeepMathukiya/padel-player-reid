"""Names of the intermediate files each pipeline stage writes to the cache folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachePaths:
    root: Path

    @classmethod
    def from_config(cls, cfg) -> "CachePaths":
        root = Path(cfg.cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    @property
    def detections(self) -> Path:          # pass 1: every kept detection with raw track id
        return self.root / "detections.parquet"

    @property
    def tracklets(self) -> Path:           # detections with pure tracklet ids
        return self.root / "tracklet_detections.parquet"

    @property
    def tracklet_summary(self) -> Path:
        return self.root / "tracklets.parquet"

    @property
    def interactions(self) -> Path:
        return self.root / "interactions.parquet"

    @property
    def embeddings(self) -> Path:          # OSNet-AIN (MSMT17) + colour + quality per embedded row
        return self.root / "embeddings.npz"

    @property
    def thumbnails_dir(self) -> Path:
        path = self.root / "thumbnails"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def assignment(self) -> Path:          # final per-detection player labels
        return self.root / "assigned_detections.parquet"

    @property
    def events(self) -> Path:
        return self.root / "identity_events.json"

    @property
    def timing(self) -> Path:
        return self.root / "stage_timing.json"
