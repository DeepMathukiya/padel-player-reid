"""Handcrafted colour descriptor: HSV histograms of head, upper body and legs.

Cheap and robust to motion blur. It helps where the deep embedding is weakest:
tiny far-side players, and telling a cap from a bald head or white hair.
"""

from __future__ import annotations

import cv2
import numpy as np

# Vertical body bands as fractions of the box height.
BODY_PARTS = {"head": (0.0, 0.18), "upper_body": (0.18, 0.55), "legs": (0.55, 1.0)}
HUE_BINS, SATURATION_BINS, VALUE_BINS = 12, 3, 3


def color_descriptor(crop: np.ndarray) -> np.ndarray:
    """L2-normalised concatenation of per-part HSV histograms (Hellinger-mapped)."""
    height, width = crop.shape[:2]
    # Keep the central 60% of the width to reduce court background.
    x0, x1 = int(width * 0.2), max(int(width * 0.8), int(width * 0.2) + 1)
    hsv = cv2.cvtColor(crop[:, x0:x1], cv2.COLOR_BGR2HSV)
    # Ignore court-blue pixels so the background does not dominate.
    court_blue = (hsv[..., 0] > 95) & (hsv[..., 0] < 130) & (hsv[..., 1] > 120)
    mask = (~court_blue).astype(np.uint8)
    parts = []
    for top, bottom in BODY_PARTS.values():
        y0, y1 = int(height * top), max(int(height * bottom), int(height * top) + 1)
        hist = cv2.calcHist([hsv[y0:y1]], [0, 1, 2], mask[y0:y1],
                            [HUE_BINS, SATURATION_BINS, VALUE_BINS], [0, 180, 0, 256, 0, 256]).ravel()
        hist = np.sqrt(hist / max(hist.sum(), 1.0))
        parts.append(hist)
    descriptor = np.concatenate(parts).astype(np.float32)
    return descriptor / max(np.linalg.norm(descriptor), 1e-6)
