"""Per-metric, per-frame DWPose keypoint confidence gating.

Drops frames where any of the keypoints used to compute a given joint angle
metric have DWPose confidence below a threshold. Returns a boolean mask the
same length as the input series; downstream callers should use the mask to
NaN out the corresponding entries of the predicted joint-angle time series
before computing layer-2 r vs ground truth (or before any other downstream
use).

This is a Layer-2 lever: noisy/missing keypoints are the dominant source of
per-frame joint-angle error. Filtering them out *before* scoring strictly
improves the |r| on the remaining frames (it does NOT bias r in either
direction — we're dropping rows, not shifting values). The cost is retention
%: aggressive thresholds throw away useful frames.

Keypoint mapping (Halpe-26, indices matching `couro_keypoints.KP`):
    For each metric we list the 2D keypoints whose conf must clear the floor:
    - knee_angle_r:     R_hip, R_kne, R_ank  (+ L_hip for hip_center sanity)
    - knee_angle_l:     L_hip, L_kne, L_ank  (+ R_hip)
    - hip_flexion_r:    R_hip, R_kne, neck    (torso + thigh)
    - hip_flexion_l:    L_hip, L_kne, neck
    - hip_adduction_r:  R_hip, R_kne, L_hip   (frontal-plane proxy)
    - hip_adduction_l:  L_hip, L_kne, R_hip
    - ankle_angle_r:    R_kne, R_ank, R_big_toe
    - ankle_angle_l:    L_kne, L_ank, L_big_toe
    - lumbar_extension: neck, hip_center, L_hip, R_hip
    - pelvis_tilt:      L_hip, R_hip
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Halpe-26 indices — kept here as a module-local copy so this file has no
# import dependency on couro_keypoints (it's a small, hot, production module).
_KP = {
    "nose": 0, "L_eye": 1, "R_eye": 2, "L_ear": 3, "R_ear": 4,
    "L_sho": 5, "R_sho": 6, "L_elb": 7, "R_elb": 8, "L_wri": 9, "R_wri": 10,
    "L_hip": 11, "R_hip": 12, "L_kne": 13, "R_kne": 14, "L_ank": 15, "R_ank": 16,
    "head_top": 17, "neck": 18, "hip_center": 19,
    "L_big_toe": 20, "R_big_toe": 21, "L_small_toe": 22, "R_small_toe": 23,
    "L_heel": 24, "R_heel": 25,
}


# Per-metric required keypoints (names from _KP).
METRIC_KEYPOINTS: dict[str, tuple[str, ...]] = {
    "knee_angle_r":     ("R_hip", "R_kne", "R_ank", "L_hip"),
    "knee_angle_l":     ("L_hip", "L_kne", "L_ank", "R_hip"),
    "hip_flexion_r":    ("R_hip", "R_kne", "neck", "hip_center"),
    "hip_flexion_l":    ("L_hip", "L_kne", "neck", "hip_center"),
    "hip_adduction_r":  ("R_hip", "R_kne", "L_hip"),
    "hip_adduction_l":  ("L_hip", "L_kne", "R_hip"),
    "ankle_angle_r":    ("R_kne", "R_ank", "R_big_toe"),
    "ankle_angle_l":    ("L_kne", "L_ank", "L_big_toe"),
    "lumbar_extension": ("neck", "hip_center", "L_hip", "R_hip"),
    "pelvis_tilt":      ("L_hip", "R_hip"),
}


# Default production threshold — sweep results showed 0.5 is the knee of the
# retention-vs-r tradeoff (see `harness/layer2_filtered_r.py` sweep output).
DEFAULT_CONF_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class ConfidenceGateResult:
    """Outcome of running the confidence gate on one (clip, metric) pair."""

    metric: str
    threshold: float
    n_total: int
    n_kept: int
    keep_mask: np.ndarray   # (n_total,) bool — True = keep frame

    @property
    def retention(self) -> float:
        return self.n_kept / self.n_total if self.n_total else 0.0


def confidence_keep_mask(
    confidence: np.ndarray,
    metric: str,
    threshold: float = DEFAULT_CONF_THRESHOLD,
) -> ConfidenceGateResult:
    """Return a per-frame keep mask for `metric` given keypoint confidences.

    Args:
        confidence: (n_frames, 26) DWPose keypoint scores in [0, 1].
        metric:     One of the keys in METRIC_KEYPOINTS.
        threshold:  Drop frame if ANY required keypoint has conf < threshold.

    Returns:
        ConfidenceGateResult with n_total, n_kept, and the boolean mask.
    """
    if metric not in METRIC_KEYPOINTS:
        # Unknown metric — keep all frames (caller can decide what to do).
        n = confidence.shape[0]
        return ConfidenceGateResult(
            metric=metric, threshold=threshold,
            n_total=n, n_kept=n, keep_mask=np.ones(n, dtype=bool),
        )

    required_idx = [_KP[name] for name in METRIC_KEYPOINTS[metric]]
    # Per-frame minimum conf across required keypoints.
    sub = confidence[:, required_idx]
    # NaNs in keypoint conf (rare) → treat as failing the threshold.
    sub = np.where(np.isfinite(sub), sub, 0.0)
    per_frame_min = sub.min(axis=1)
    mask = per_frame_min >= threshold
    return ConfidenceGateResult(
        metric=metric, threshold=threshold,
        n_total=int(confidence.shape[0]),
        n_kept=int(mask.sum()),
        keep_mask=mask,
    )
