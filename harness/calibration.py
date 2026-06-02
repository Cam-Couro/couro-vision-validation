"""Per-subject anthropometric calibration from observed video data.

The Winter 2009 ratios (femur=24.5% height, shank=24.6%, foot=15.2%) used
in `anthro_3d.py` are population averages. Real athletes vary ±5-10% from
these — and that error multiplies into every 2D-to-3D angle estimate.

This module computes per-subject anthropometric overrides by observing the
longest apparent segment length in the subject's video (in metres, scaled
via torso-pixel m_per_px). The longest observed segment is closest to the
true segment length because foreshortening only ever makes things shorter,
never longer.

For OpenCap: we have 5 cameras × 6+ trials per subject = many frames to pick
the best from. For production: would need a calibration capture (athlete
stands sideways, does a slow squat) or online accumulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .anthro_3d import ANTHRO_RATIOS, CameraModel, estimate_subject_depth_per_frame
from .couro_keypoints import KP, load_couro_output


@dataclass(frozen=True)
class SubjectAnthro:
    """Per-subject segment lengths in metres, calibrated from observed video."""
    subject: str
    height_m: float
    thigh_m: float
    shank_m: float
    foot_m: float
    torso_m: float
    # Source: "winter" (population avg) or "observed" (video-calibrated)
    thigh_source: str
    shank_source: str
    foot_source: str


def _segment_max_apparent_m(
    xy: np.ndarray,
    confidence: np.ndarray,
    m_per_px: np.ndarray,
    kp_a: int,
    kp_b: int,
    *,
    conf_threshold: float = 0.6,
) -> float:
    """Find the true-ish segment length in metres.

    Foreshortening shrinks apparent length; keypoint noise jitters it both
    ways. The high-end is dominated by NOISE (positive outliers) so 95th
    percentile overestimates. The plateau where the segment is closest to
    perpendicular-to-camera is what we want — use the 80th percentile of
    high-confidence frames as a robust middle ground.
    """
    seg_px = np.linalg.norm(xy[:, kp_b] - xy[:, kp_a], axis=1)
    conf_ok = (
        (confidence[:, kp_a] >= conf_threshold)
        & (confidence[:, kp_b] >= conf_threshold)
        & np.isfinite(m_per_px)
    )
    if not np.any(conf_ok):
        return float("nan")
    seg_m = seg_px[conf_ok] * m_per_px[conf_ok]
    seg_m = seg_m[np.isfinite(seg_m)]
    if seg_m.size < 10:
        return float("nan")
    return float(np.percentile(seg_m, 80))


def calibrate_subject(
    subject: str,
    kp_paths: list[Path],
    camera_pickles: dict[str, Path],
    height_m: float,
) -> SubjectAnthro:
    """Estimate per-subject segment lengths from all of their video data.

    Args:
        kp_paths: list of all Couro keypoint JSONs for this subject (all
                  cameras × all trials)
        camera_pickles: dict mapping camera_id ("Cam0") → camera pickle path
    """
    # Defaults from Winter ratios
    winter_thigh = height_m * ANTHRO_RATIOS["thigh"]
    winter_shank = height_m * ANTHRO_RATIOS["shank"]
    winter_foot = height_m * ANTHRO_RATIOS["foot"]
    winter_torso = height_m * ANTHRO_RATIOS["torso"]

    thigh_observations: list[float] = []
    shank_observations: list[float] = []
    foot_observations: list[float] = []

    for kp_path in kp_paths:
        try:
            cam_id = kp_path.stem.split("_")[-1]  # e.g. "subject2_DJ1_Cam2" → "Cam2"
            if cam_id not in camera_pickles:
                continue
            series = load_couro_output(kp_path)
            cam = CameraModel.from_pickle(camera_pickles[cam_id], series.width, series.height)
            m_per_px = estimate_subject_depth_per_frame(
                series.xy, series.confidence, cam,
                subject_height_m=height_m, KP=KP,
            )

            # Sample max segment lengths from each side
            for kp_a, kp_b, bucket in (
                ("R_hip", "R_kne", thigh_observations),
                ("L_hip", "L_kne", thigh_observations),
                ("R_kne", "R_ank", shank_observations),
                ("L_kne", "L_ank", shank_observations),
                ("R_heel", "R_big_toe", foot_observations),
                ("L_heel", "L_big_toe", foot_observations),
            ):
                v = _segment_max_apparent_m(
                    series.xy, series.confidence, m_per_px,
                    KP[kp_a], KP[kp_b],
                )
                if np.isfinite(v):
                    bucket.append(v)
        except Exception:
            continue

    def _consensus(observations: list[float], default: float, name: str) -> tuple[float, str]:
        if len(observations) >= 5:
            # Median across-trial estimates: more robust than mean, less biased
            # than max/95th percentile
            arr = np.array(observations)
            value = float(np.median(arr))
            # Tight sanity bounds — population variation is ~±10%, so anything
            # beyond ±15% of Winter is calibration noise, not real anthropometry
            if 0.85 * default <= value <= 1.15 * default:
                return value, "observed"
        return default, "winter"

    thigh, thigh_src = _consensus(thigh_observations, winter_thigh, "thigh")
    shank, shank_src = _consensus(shank_observations, winter_shank, "shank")
    foot, foot_src = _consensus(foot_observations, winter_foot, "foot")

    return SubjectAnthro(
        subject=subject,
        height_m=height_m,
        thigh_m=thigh,
        shank_m=shank,
        foot_m=foot,
        torso_m=winter_torso,  # torso is the reference, can't easily recalibrate
        thigh_source=thigh_src,
        shank_source=shank_src,
        foot_source=foot_src,
    )
