"""Per-frame posture gating.

Drops frames where the athlete is not in a 'standing/jumping' posture, i.e.,
their torso is not within ±tolerance° of image vertical. This catches:
  - Athlete lying down (recovery, prone, supine)
  - Athlete falling / mid-air rotation off-axis
  - Subject off-screen so the 'torso vector' is nonsense
  - Frames where the detector locked onto the wrong person

For Couro's deploy metrics (DJ landing, softball pitch, baseball swing) the
athlete is always within ~30° of vertical when the metric is meaningful. Frames
outside this cone are noise.

Torso angle is computed from shoulders-midpoint to hips-midpoint (or
hip_center if available). 0° = perfectly upright; ±90° = horizontal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_KP = {
    "L_sho": 5, "R_sho": 6, "L_hip": 11, "R_hip": 12,
    "neck": 18, "hip_center": 19,
}


# Default production tolerance — torso must be within ±DEFAULT_TOLERANCE_DEG
# of image vertical to count as "athlete-upright". 30° covers all standing,
# squatting, jumping, landing, swinging postures while excluding lying-down /
# fully-horizontal frames.
DEFAULT_TOLERANCE_DEG: float = 30.0


@dataclass(frozen=True)
class PostureGateResult:
    """Outcome of running the posture gate on one clip."""

    tolerance_deg: float
    n_total: int
    n_kept: int
    keep_mask: np.ndarray      # (n_total,) bool — True = keep frame
    torso_angle_deg: np.ndarray  # (n_total,) signed angle from vertical, NaN where indeterminate

    @property
    def retention(self) -> float:
        return self.n_kept / self.n_total if self.n_total else 0.0


def _torso_vec(xy: np.ndarray) -> np.ndarray:
    """Compute torso vector (hips_midpoint - shoulders_midpoint) per frame.

    Image coordinates: y increases downward; a standing athlete has the torso
    pointing image-down (positive y). Returns (n_frames, 2).
    """
    # Prefer the dedicated keypoints when available (DWPose gives both):
    # neck and hip_center are derived as midpoints of the shoulders / hips,
    # so they're more stable than recomputing midpoints when one shoulder
    # is occluded. Fall back to shoulder/hip midpoints if hip_center/neck
    # are entirely missing (e.g., NaN-filled).
    neck = xy[:, _KP["neck"], :]
    hip_c = xy[:, _KP["hip_center"], :]
    vec = hip_c - neck

    # Where neck or hip_center is NaN, fall back to shoulder/hip midpoints.
    bad = ~np.isfinite(vec).all(axis=1)
    if bad.any():
        l_sho = xy[:, _KP["L_sho"], :]
        r_sho = xy[:, _KP["R_sho"], :]
        l_hip = xy[:, _KP["L_hip"], :]
        r_hip = xy[:, _KP["R_hip"], :]
        sho_mid = 0.5 * (l_sho + r_sho)
        hip_mid = 0.5 * (l_hip + r_hip)
        fallback = hip_mid - sho_mid
        vec = np.where(bad[:, None], fallback, vec)
    return vec


def _signed_angle_from_vertical(vec: np.ndarray) -> np.ndarray:
    """Signed angle of vec relative to image-down vertical (positive = lean right).

    Image-down = (0, 1). atan2(dx, dy) gives 0 when vec aligns with image-down
    and ±90° when horizontal. ±180° = vec points up (subject upside-down or
    torso-reversed by detector).
    """
    dx = vec[..., 0]
    dy = vec[..., 1]
    return np.degrees(np.arctan2(dx, dy))


def posture_keep_mask(
    xy: np.ndarray,
    tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
) -> PostureGateResult:
    """Return a per-frame keep mask: True where |torso_angle_from_vertical| < tol.

    Args:
        xy: (n_frames, 26, 2) Halpe-26 keypoints in pixel coords (NaN ok).
        tolerance_deg: half-cone width around image-vertical. Defaults to 30°.

    Returns:
        PostureGateResult with the mask and the per-frame torso angle.
    """
    vec = _torso_vec(xy)
    torso = _signed_angle_from_vertical(vec)
    # Frames with NaN torso angle (both fallbacks missing) → drop.
    finite = np.isfinite(torso)
    within = np.abs(torso) <= tolerance_deg
    mask = finite & within
    return PostureGateResult(
        tolerance_deg=tolerance_deg,
        n_total=int(xy.shape[0]),
        n_kept=int(mask.sum()),
        keep_mask=mask,
        torso_angle_deg=torso,
    )
