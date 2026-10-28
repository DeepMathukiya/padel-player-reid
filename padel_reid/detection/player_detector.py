"""Pass 1 detection: one YOLO11 person-detection run per frame + Ultralytics BoT-SORT.

The frame is run once at 960 px, which finds both small far-side players as
reliably as a separate high-resolution far-court crop did (97% of frames on 300
sampled frames) at the same speed. Detections are filtered (court ROI, size,
partial duplicates, top-4 by confidence) and only then given to BoT-SORT, so
people on the neighbouring court never take a track id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from padel_reid.common.config_loader import load_yaml
from padel_reid.court.court_roi_mask import CourtRoiMask
from padel_reid.court.court_side import court_side, inside_court_roi
from padel_reid.court.homography import CourtMapping


class DetectionBatch:
    """The minimal Results-like object the Ultralytics trackers consume."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = xyxy.astype(np.float32).reshape(-1, 4)
        self.conf = conf.astype(np.float32).reshape(-1)
        self.cls = np.zeros(len(self.conf), dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        xy = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return np.concatenate([xy, wh], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index) -> "DetectionBatch":
        return DetectionBatch(self.xyxy[index], self.conf[index])


def build_tracker(tracker_config_path: str):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    args = IterableSimpleNamespace(**load_yaml(tracker_config_path))
    return BOTSORT(args) if args.tracker_type == "botsort" else BYTETracker(args)


def suppress_contained(xyxy: np.ndarray, conf: np.ndarray, threshold: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Drop boxes lying mostly (> threshold of their area) inside a bigger box.

    Catches partial duplicates of one person (e.g. upper body only) that plain
    IoU-based NMS keeps because their IoU with the full box is low.
    """
    if len(xyxy) < 2:
        return xyxy, conf
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    ix1 = np.maximum(xyxy[:, None, 0], xyxy[None, :, 0])
    iy1 = np.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
    ix2 = np.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
    iy2 = np.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    contained = (inter / np.maximum(area[:, None], 1e-6) > threshold) & (area[:, None] < area[None, :])
    keep = ~contained.any(axis=1)
    return xyxy[keep], conf[keep]


def foot_points(xyxy: np.ndarray) -> np.ndarray:
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)


class PlayerDetector:
    def __init__(self, detection_cfg, court_mapping: CourtMapping, roi_margin_m: float,
                 frame_size: tuple[int, int], roi_mask: CourtRoiMask | None = None):
        from ultralytics import YOLO

        self.cfg = detection_cfg
        self.mapping = court_mapping
        self.roi_margin_m = roi_margin_m
        self.frame_height = frame_size[1]
        self.roi = roi_mask
        self.batch_size = int(detection_cfg.get("batch_size", 1))
        self.gpu = None
        if detection_cfg.get("gpu_preprocessing", False):
            from padel_reid.detection.gpu_batched_yolo import GpuBatchedYolo

            self.gpu = GpuBatchedYolo(detection_cfg.model, detection_cfg.image_size, (frame_size[1], frame_size[0]),
                                      roi_mask, half=detection_cfg.half_precision)
        else:
            self.model = YOLO(detection_cfg.model)

    def raw_detections_batch(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """YOLO person boxes in full-frame pixels for each frame, on the court-ROI-masked crop."""
        if self.gpu is not None:
            return self.gpu(frames, self.cfg.min_confidence)
        out = []
        for frame in frames:
            image, (dx, dy) = frame, (0, 0)
            if self.roi is not None:
                image, (dx, dy) = self.roi.apply(frame), self.roi.bbox[:2]
            result = self.model.predict(image, imgsz=self.cfg.image_size, quantize=16 if self.cfg.half_precision else 32,
                                        device=0, conf=self.cfg.min_confidence, classes=[0], verbose=False)[0]
            out.append((result.boxes.xyxy.cpu().numpy() + np.array([dx, dy, dx, dy], dtype=np.float32),
                        result.boxes.conf.cpu().numpy()))
        return out

    def raw_detections(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.raw_detections_batch([frame])[0]

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filtered detections for one frame: (xyxy, conf)."""
        return self.filter_detections(*self.raw_detections(frame))

    def filter_detections(self, xyxy: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Partial duplicates, size, foot-point court ROI, then the top-4 by confidence."""
        if len(xyxy) == 0:
            return xyxy.reshape(0, 4), conf
        xyxy, conf = suppress_contained(xyxy, conf)

        heights = xyxy[:, 3] - xyxy[:, 1]
        mask = heights >= self.cfg.min_box_height_px
        mask &= inside_court_roi(self.mapping.image_to_court(foot_points(xyxy)), self.roi_margin_m)
        xyxy, conf = xyxy[mask], conf[mask]
        order = np.argsort(-conf)[: self.cfg.max_players]
        return xyxy[order], conf[order]

    def run(self, frames) -> pd.DataFrame:
        """frames: iterable of (frame_index, BGR image). Returns one row per tracked detection."""
        tracker = build_tracker(self.cfg.tracker_config)
        rows = []
        progress = tqdm(desc="detect+track", unit="frame")
        for batch in _batched(frames, self.batch_size):
            detections = self.raw_detections_batch([frame for _, frame in batch])
            # The tracker still sees one frame at a time, in order.
            for (frame_index, frame), (raw_xyxy, raw_conf) in zip(batch, detections):
                progress.update()
                xyxy, conf = self.filter_detections(raw_xyxy, raw_conf)
                rows.extend(self._track_frame(tracker, frame_index, frame, xyxy, conf))
        progress.close()
        table = pd.DataFrame(rows, columns=[
            "frame", "raw_track_id", "x1", "y1", "x2", "y2", "confidence",
            "foot_x", "foot_y", "court_x", "court_y", "truncated_bottom",
        ])
        table["side"] = court_side(table["court_y"].to_numpy())
        return table

    def _track_frame(self, tracker, frame_index: int, frame: np.ndarray, xyxy: np.ndarray,
                     conf: np.ndarray) -> list[tuple]:
        tracked = tracker.update(DetectionBatch(xyxy, conf), frame)
        if len(tracked) == 0:
            return []
        # Report the detector's own box (not the Kalman-smoothed one) for each
        # tracked detection; column 7 is the index into the detection set.
        det_index = tracked[:, 7].astype(int)
        boxes = xyxy[det_index]
        feet = foot_points(boxes)
        court_xy = self.mapping.image_to_court(feet)
        return [(frame_index, int(track_row[4]), *boxes[j], float(conf[det_index[j]]),
                 *feet[j], *court_xy[j], bool(boxes[j, 3] >= self.frame_height - 3))
                for j, track_row in enumerate(tracked)]


def _batched(items, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
