"""Image <-> court-metre mapping for a static fisheye camera.

The lens has strong barrel distortion, so a plain homography is ~1 m off near the
camera. We fit one radial distortion coefficient k1 jointly with the homography:
points are first undistorted (OpenCV pinhole + k1 model), then mapped by H.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

from padel_reid.common.config_loader import load_yaml

COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
NET_Y_M = 10.0


@dataclass
class CourtMapping:
    camera_matrix: np.ndarray
    k1: float
    image_to_court_h: np.ndarray

    @property
    def _dist(self) -> np.ndarray:
        return np.array([self.k1, 0.0, 0.0, 0.0])

    def image_to_court(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) pixel points -> (N, 2) court metres."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if len(points) == 0:
            return np.zeros((0, 2))
        undistorted = cv2.undistortPoints(points, self.camera_matrix, self._dist, P=self.camera_matrix)
        return cv2.perspectiveTransform(undistorted, self.image_to_court_h).reshape(-1, 2)


def _camera_matrix(width: int, height: int) -> np.ndarray:
    focal = 1000.0 * width / 1920.0
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)


def fit_court_mapping(image_points: np.ndarray, court_points: np.ndarray,
                      line_samples: list[tuple[np.ndarray, int, float]],
                      image_size: tuple[int, int]) -> CourtMapping:
    """Fit k1 and H.

    line_samples: (pixel points, court axis 0/1, court value) for points known to lie
    on a straight court line; they make the distortion estimate robust.
    """
    k_matrix = _camera_matrix(*image_size)

    def undistort(points: np.ndarray, k1: float) -> np.ndarray:
        return cv2.undistortPoints(points.reshape(-1, 1, 2).astype(np.float64), k_matrix,
                                   np.array([k1, 0, 0, 0]), P=k_matrix).reshape(-1, 2)

    def residual(k1: float) -> float:
        h, _ = cv2.findHomography(undistort(image_points, k1), court_points, 0)
        if h is None:
            return 1e9
        mapped = cv2.perspectiveTransform(undistort(image_points, k1).reshape(-1, 1, 2), h).reshape(-1, 2)
        total = float(((mapped - court_points) ** 2).sum())
        for pixels, axis, value in line_samples:
            on_line = cv2.perspectiveTransform(undistort(pixels, k1).reshape(-1, 1, 2), h).reshape(-1, 2)
            total += float(((on_line[:, axis] - value) ** 2).sum())
        return total

    best = minimize_scalar(residual, bounds=(-0.6, 0.3), method="bounded")
    k1 = float(best.x)
    h, _ = cv2.findHomography(undistort(image_points, k1), court_points, 0)
    return CourtMapping(k_matrix, k1, h)


def load_court_mapping(court_config_path: str) -> CourtMapping:
    cfg = load_yaml(court_config_path)
    rows = np.array(list(cfg.landmarks.values()), dtype=np.float64)
    line_samples = []
    if cfg.get("near_service_line_samples"):
        line_samples.append((np.array(cfg.near_service_line_samples, dtype=np.float64), 1, 16.95))
    return fit_court_mapping(rows[:, :2], rows[:, 2:], line_samples, tuple(cfg.image_size))
