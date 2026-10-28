"""Court-side logic: which half of the net a player is on, and the court ROI test."""

from __future__ import annotations

import numpy as np

from padel_reid.court.homography import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M


def court_side(court_y: np.ndarray) -> np.ndarray:
    """0 for the far half, 1 for the near half."""
    return (np.asarray(court_y) >= NET_Y_M).astype(np.int8)


def inside_court_roi(court_xy: np.ndarray, margin_m: float) -> np.ndarray:
    """True for foot points inside the court plus a margin.

    The near end is left open: the camera's bottom edge is around the near service
    line, so near players are often cut off and their foot estimate is unreliable.
    """
    court_xy = np.asarray(court_xy)
    x, y = court_xy[:, 0], court_xy[:, 1]
    return ((x >= -margin_m) & (x <= COURT_WIDTH_M + margin_m)
            & (y >= -margin_m) & (y <= COURT_LENGTH_M + 2 * margin_m))
