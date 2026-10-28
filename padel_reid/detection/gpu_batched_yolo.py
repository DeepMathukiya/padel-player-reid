"""YOLO person detection with GPU pre-processing and multi-frame batches.

Profiling the plain Ultralytics predict() path showed the GPU waiting on the CPU:
per frame ~1.3 ms court masking + ~2.5 ms letterboxing on the CPU around ~7.7 ms of
inference, one frame at a time (GPU at ~16 W). Here each batch of frames is
uploaded once; cropping to the court ROI, masking, BGR->RGB, resize and padding
all happen on the GPU, the raw network runs one forward pass for the whole batch,
and NMS runs on the person class only.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

from padel_reid.court.court_roi_mask import FILL_VALUE, CourtRoiMask

STRIDE = 32


class GpuBatchedYolo:
    def __init__(self, weights: str, image_size: int, frame_hw: tuple[int, int],
                 roi: CourtRoiMask | None, half: bool = True):
        from ultralytics import YOLO

        self.device = torch.device("cuda")
        self.dtype = torch.float16 if half else torch.float32
        self.net = YOLO(weights).model.fuse(verbose=False).eval().to(self.device, self.dtype)
        frame_h, frame_w = frame_hw
        self.x0, self.y0, x1, y1 = roi.bbox if roi is not None else (0, 0, frame_w, frame_h)
        crop_h, crop_w = y1 - self.y0, x1 - self.x0
        self.crop_hw = (crop_h, crop_w)
        self.scale = image_size / max(crop_h, crop_w)
        self.resized_hw = (round(crop_h * self.scale), round(crop_w * self.scale))
        self.padded_hw = tuple(math.ceil(s / STRIDE) * STRIDE for s in self.resized_hw)
        self.mask = None
        if roi is not None:
            x0, y0, x1, y1 = roi.bbox
            self.mask = torch.from_numpy(roi.mask[y0:y1, x0:x1]).to(self.device, self.dtype)[None, None]

    def _upload(self, frames: list[np.ndarray]) -> torch.Tensor:
        """(B, crop_h, crop_w, 3) uint8 on the GPU, without an intermediate CPU copy.

        A full-width row slice of a frame is already contiguous, so it is uploaded as is;
        np.stack-ing the crops first cost ~4 ms per frame.
        """
        crop_h, crop_w = self.crop_hw
        batch = torch.empty((len(frames), crop_h, crop_w, 3), dtype=torch.uint8, device=self.device)
        for i, frame in enumerate(frames):
            crop = frame[self.y0:self.y0 + crop_h, self.x0:self.x0 + crop_w]
            batch[i].copy_(torch.from_numpy(np.ascontiguousarray(crop)))
        return batch

    @staticmethod
    def _person_nms(predictions: torch.Tensor, conf: float, iou: float) -> list[torch.Tensor]:
        """NMS on the person class only (rows: cx, cy, w, h, class scores...).

        Ultralytics' generic NMS handles all 80 COCO classes and cost ~4.7 ms per frame;
        we only need class 0.
        """
        out = []
        for p in predictions:                      # (4 + num_classes, anchors)
            scores = p[4]
            keep = scores > conf
            boxes, scores = p[:4, keep].T, scores[keep]
            xyxy = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], dim=1)
            kept = nms(xyxy.float(), scores.float(), iou)
            out.append(torch.cat([xyxy[kept], scores[kept, None]], dim=1))
        return out

    @torch.no_grad()
    def __call__(self, frames: list[np.ndarray], conf: float, iou: float = 0.7) -> list[tuple[np.ndarray, np.ndarray]]:
        """Person boxes (full-frame pixels) and confidences for each frame of the batch."""
        x = self._upload(frames)
        x = x.permute(0, 3, 1, 2).flip(1).to(self.dtype).div_(255.0)        # BGR HWC uint8 -> RGB CHW
        if self.mask is not None:
            x = x * self.mask + (FILL_VALUE / 255.0) * (1 - self.mask)
        x = F.interpolate(x, size=self.resized_hw, mode="bilinear", align_corners=False)
        pad_h, pad_w = self.padded_hw[0] - self.resized_hw[0], self.padded_hw[1] - self.resized_hw[1]
        x = F.pad(x, (0, pad_w, 0, pad_h), value=FILL_VALUE / 255.0)
        predictions = self.net(x)
        predictions = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        results = self._person_nms(predictions, conf, iou)
        offset = np.array([self.x0, self.y0, self.x0, self.y0], dtype=np.float32)
        out = []
        for r in results:
            r = r.cpu().numpy()
            out.append((r[:, :4] / self.scale + offset, r[:, 4]))
        return out
