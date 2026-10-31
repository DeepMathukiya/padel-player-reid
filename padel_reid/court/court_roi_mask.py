"""Image-space court ROI applied *before* detection.

The court rectangle (plus a margin) is projected into the image through the
fisheye-corrected mapping, and every column is extended upward by the height a
standing player has at that image row, so far players' bodies (which rise above
the court lines) stay inside. Everything outside the mask is painted grey and the
frame is cropped to the mask's bounding box before YOLO runs, so people on the
neighbouring court are never seen by the detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from padel_reid.court.homography import COURT_LENGTH_M, COURT_WIDTH_M, CourtMapping

FILL_VALUE = 114          # Ultralytics letterbox grey: neutral for the detector


@dataclass
class CourtRoiMask:
    mask: np.ndarray                     # (H, W) bool over the full frame
    bbox: tuple[int, int, int, int]      # x0, y0, x1, y1 crop of the mask

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.bbox
        self._crop_mask_u8 = self.mask[y0:y1, x0:x1].astype(np.uint8)
        self._grey = np.full((y1 - y0, x1 - x0, 3), FILL_VALUE, np.uint8)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Cropped copy of the frame with everything outside the ROI painted grey."""
        x0, y0, x1, y1 = self.bbox
        out = self._grey.copy()
        cv2.copyTo(frame[y0:y1, x0:x1], self._crop_mask_u8, out)     # SIMD; ~5x faster than bool indexing
        return out


def _ground_mask(mapping: CourtMapping, width: int, height: int, margin_m: float, step: int = 4) -> np.ndarray:
    """Pixels whose ground point lies on the court (+ margin).

    Built image -> court with the stable inverse mapping (undistort + homography);
    projecting a court outline the other way folds over under the fisheye model.
    Pixels above the horizon (homogeneous w with the wrong sign) are rejected.
    """
    xs, ys = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pixels, mapping.camera_matrix, np.array([mapping.k1, 0, 0, 0]),
                                      P=mapping.camera_matrix).reshape(-1, 2)
    homogeneous = np.hstack([undistorted, np.ones((len(undistorted), 1))]) @ mapping.image_to_court_h.T
    reference_w = (np.array([960.0, 600.0, 1.0]) @ mapping.image_to_court_h.T)[2]   # a point on the court
    valid = np.sign(homogeneous[:, 2]) == np.sign(reference_w)
    court = homogeneous[:, :2] / homogeneous[:, 2:3]
    inside = (valid & (court[:, 0] >= -margin_m) & (court[:, 0] <= COURT_WIDTH_M + margin_m)
              & (court[:, 1] >= -margin_m) & (court[:, 1] <= COURT_LENGTH_M + 4 * margin_m))
    coarse = inside.reshape(xs.shape).astype(np.uint8)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def build_court_roi_mask(mapping: CourtMapping, frame_size: tuple[int, int], margin_m: float,
                         person_height_px: tuple[float, float], headroom: float,
                         standing_margin_m: float = 0.3,
                         picture_box: tuple[int, int, int, int] | None = None) -> CourtRoiMask:
    """frame_size = (width, height); person_height_px = (slope, intercept) of height vs foot row.

    The ground area uses the full `margin_m` (foot-position jitter), but the upward
    extension for players' bodies starts from a tight outline `standing_margin_m`
    past the glass walls: nobody stands beyond the walls, and extending from the
    wide margin would re-include the neighbouring court above the sidelines.

    picture_box (x0, y0, x1, y1): the part of the frame that holds picture, for
    pillarboxed / letterboxed footage. The mask is clipped to it, so the crop YOLO
    sees (resized to detection.image_size) is not mostly empty bars.
    """
    width, height = frame_size
    mask = _ground_mask(mapping, width, height, margin_m)
    standing = _ground_mask(mapping, width, height, standing_margin_m)

    slope, intercept = person_height_px
    rows = np.arange(height)
    for x in np.where(standing.any(axis=0))[0]:
        top = int(rows[standing[:, x]].min())
        lift = int(headroom * max(slope * top + intercept, 0))
        mask[max(top - lift, 0):top, x] = True
    if picture_box is not None:
        px0, py0, px1, py1 = picture_box
        picture = np.zeros_like(mask)
        picture[py0:py1, px0:px1] = True
        mask &= picture

    ys, xs = np.where(mask)
    return CourtRoiMask(mask, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
