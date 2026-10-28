"""How much a single crop should be trusted for Re-ID.

quality = size x detector confidence x (1 - overlap with other players) x sharpness,
halved when the box is cut off by the bottom of the frame. Blurry, small, occluded
or truncated crops get little weight in every average they enter.
"""

from __future__ import annotations

import cv2
import numpy as np


def sharpness(crop: np.ndarray) -> float:
    """Variance of the Laplacian, squashed to [0, 1)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return variance / (variance + 60.0)


def crop_quality(crop: np.ndarray, box_height: float, confidence: float, overlap: float,
                 truncated: bool) -> float:
    size_term = min(box_height / 180.0, 1.0)
    quality = size_term * float(confidence) * (1.0 - min(overlap, 1.0)) * (0.3 + 0.7 * sharpness(crop))
    return quality * (0.5 if truncated else 1.0)


def crop_box(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(int(x2), width), min(int(y2), height)
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    return frame[y1:y2, x1:x2]
