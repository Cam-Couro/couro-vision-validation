"""Parse Couro RTMPose-X Halpe-26 keypoint output → time-series joint angles.

The Couro inference pipeline emits per-frame 2D keypoints in Halpe-26 layout.
This module converts those into the same joint-angle conventions used by
OpenSim IK (knee_angle_r, hip_flexion_l, etc.) so they slot into the existing
harness alongside Vicon GT and academic-baseline IK outputs.

Halpe-26 keypoint order (from the Couro prod container):
    0:nose 1:L_eye 2:R_eye 3:L_ear 4:R_ear 5:L_sho 6:R_sho 7:L_elb 8:R_elb
    9:L_wri 10:R_wri 11:L_hip 12:R_hip 13:L_kne 14:R_kne 15:L_ank 16:R_ank
    17:head_top 18:neck 19:hip_center 20:L_big_toe 21:R_big_toe
    22:L_small_toe 23:R_small_toe 24:L_heel 25:R_heel
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .parsers import MotionData


KP = {
    "nose": 0, "L_eye": 1, "R_eye": 2, "L_ear": 3, "R_ear": 4,
    "L_sho": 5, "R_sho": 6, "L_elb": 7, "R_elb": 8, "L_wri": 9, "R_wri": 10,
    "L_hip": 11, "R_hip": 12, "L_kne": 13, "R_kne": 14, "L_ank": 15, "R_ank": 16,
    "head_top": 17, "neck": 18, "hip_center": 19,
    "L_big_toe": 20, "R_big_toe": 21, "L_small_toe": 22, "R_small_toe": 23,
    "L_heel": 24, "R_heel": 25,
}


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle between two 2D vectors per frame, in degrees [0, 180]."""
    dot = (v1 * v2).sum(axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.clip(dot / np.where(n1 * n2 > 1e-9, n1 * n2, 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _signed_angle_from_vertical(vec: np.ndarray) -> np.ndarray:
    """Signed angle of vec relative to image-down vertical (positive = lean forward).

    Image coordinates: y increases downward, so image-down = (0, 1).
    """
    dx, dy = vec[..., 0], vec[..., 1]
    return np.degrees(np.arctan2(dx, dy))


@dataclass(frozen=True)
class CouroKeypointSeries:
    """Time series of keypoints from one Couro inference run."""
    time: np.ndarray                   # (n_frames,) seconds
    xy: np.ndarray                     # (n_frames, 26, 2) pixel coords
    confidence: np.ndarray             # (n_frames, 26)
    width: int
    height: int
    fps: float


def load_couro_output(path: Path) -> CouroKeypointSeries:
    """Load Couro's KeypointsOutput JSON into a normalized array form.

    Schema observed from the rtm-pose-inference-prod container:
        {
          "keypoints_sequence": [
            {
              "frame_idx": int,
              "timestamp_ms": float,
              "keypoints": [[x, y], ...],      # 26 entries, halpe-26 order
              "keypoint_scores": [c, c, ...],  # 26 confidences in [0,1]
              "bbox": [...], "track_id": int
            }, ...
          ],
          "video_metadata": {"width", "height", "fps", "total_frames", "duration_sec"},
          ...
        }
    """
    with open(path) as f:
        d = json.load(f)
    frames = d.get("keypoints_sequence") or d.get("frames") or d.get("keypoint_frames") or []
    if not frames:
        raise ValueError(f"no keypoints_sequence in {path}")
    n = len(frames)
    xy = np.full((n, 26, 2), np.nan)
    conf = np.zeros((n, 26))
    times = np.zeros(n)
    for i, fr in enumerate(frames):
        kps = fr.get("keypoints") or fr.get("kps") or []
        scores = fr.get("keypoint_scores") or fr.get("scores") or []
        for j, kp in enumerate(kps[:26]):
            if isinstance(kp, dict):
                xy[i, j, 0] = kp.get("x", np.nan)
                xy[i, j, 1] = kp.get("y", np.nan)
                conf[i, j] = kp.get("confidence", kp.get("score", 0.0))
            elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
                xy[i, j, 0], xy[i, j, 1] = float(kp[0]), float(kp[1])
                if len(kp) >= 3:
                    conf[i, j] = float(kp[2])
        for j, c in enumerate(scores[:26]):
            conf[i, j] = float(c)
        ts_ms = fr.get("timestamp_ms")
        if ts_ms is not None:
            times[i] = float(ts_ms) / 1000.0
        else:
            times[i] = float(fr.get("timestamp", fr.get("time", i / max(d.get("fps", 30), 1))))
    meta = d.get("video_metadata") or d.get("metadata") or d
    return CouroKeypointSeries(
        time=times, xy=xy, confidence=conf,
        width=int(meta.get("width", 0)) or int(d.get("width", 0)),
        height=int(meta.get("height", 0)) or int(d.get("height", 0)),
        fps=float(meta.get("fps", 0)) or float(d.get("fps", 30)),
    )


def keypoints_to_motion_data(
    series: CouroKeypointSeries,
    conf_threshold: float = 0.3,
    *,
    subject_height_m: float | None = None,
    camera_pickle: Path | None = None,
    anthro_override: "dict[str, float] | None" = None,
    smooth_window: int = 0,
) -> MotionData:
    """Convert 2D keypoints to joint angles.

    If `subject_height_m` AND `camera_pickle` are provided, uses anthropometric
    foreshortening (anthro_3d module) to recover out-of-plane joint angles —
    materially improves front/oblique camera accuracy. Otherwise falls back to
    naive 2D vector math (only useful from a side camera).

    Angles match OpenSim convention where possible:
    - knee_angle_r/l: knee flexion (0° straight, positive = flexed)
    - hip_flexion_r/l: hip flexion (0° neutral, positive = forward)
    - hip_adduction_r/l, ankle_angle_r/l, lumbar_extension, pelvis_tilt
    """
    xy = series.xy.copy()
    bad = series.confidence < conf_threshold
    xy[bad] = np.nan

    if smooth_window and smooth_window >= 5 and xy.shape[0] >= smooth_window:
        # Savgol-smooth each keypoint's x/y trajectory over time. Reduces
        # per-frame jitter that propagates into angle calculations.
        from scipy.signal import savgol_filter
        smoothed = xy.copy()
        for kp_idx in range(xy.shape[1]):
            for dim in range(2):
                col = xy[:, kp_idx, dim]
                finite = np.isfinite(col)
                if finite.sum() >= smooth_window:
                    # Fill gaps first so Savgol doesn't propagate NaNs
                    col_filled = col.copy()
                    if not finite.all():
                        idx = np.arange(len(col))
                        col_filled[~finite] = np.interp(
                            idx[~finite], idx[finite], col[finite]
                        )
                    smoothed[:, kp_idx, dim] = savgol_filter(
                        col_filled, window_length=smooth_window, polyorder=2,
                        mode="interp",
                    )
                    # Re-mask frames that were originally low-confidence
                    smoothed[~finite, kp_idx, dim] = np.nan
        xy = smoothed

    def pt(name: str) -> np.ndarray:
        return xy[:, KP[name], :]

    # If we have height + camera intrinsics, compute the anthropometric 3D angles
    # AND combine them with naive 2D — the 3D path recovers out-of-plane angle
    # when segments are foreshortened (front view), but adds noise when they
    # aren't (side view). Hybrid: per-frame, per-joint, blend by foreshortening.
    have_3d = subject_height_m is not None and camera_pickle is not None and camera_pickle.exists()
    if have_3d:
        from .anthro_3d import (
            ANTHRO_RATIOS, CameraModel, estimate_subject_depth_per_frame,
            _angle_3d_from_segments,
        )
        cam = CameraModel.from_pickle(camera_pickle, series.width, series.height)
        m_per_px = estimate_subject_depth_per_frame(
            xy, series.confidence, cam, subject_height_m, KP=KP,
            conf_threshold=conf_threshold,
        )
        if anthro_override is not None:
            L_thigh = float(anthro_override.get("thigh_m", subject_height_m * ANTHRO_RATIOS["thigh"]))
            L_shank = float(anthro_override.get("shank_m", subject_height_m * ANTHRO_RATIOS["shank"]))
            L_foot = float(anthro_override.get("foot_m", subject_height_m * ANTHRO_RATIOS["foot"]))
            L_torso = float(anthro_override.get("torso_m", subject_height_m * ANTHRO_RATIOS["torso"]))
        else:
            L_thigh = subject_height_m * ANTHRO_RATIOS["thigh"]
            L_shank = subject_height_m * ANTHRO_RATIOS["shank"]
            L_foot = subject_height_m * ANTHRO_RATIOS["foot"]
            L_torso = subject_height_m * ANTHRO_RATIOS["torso"]

        # 3D angle estimate per joint (interior angle from segment vectors)
        knee_int_r_3d = _angle_3d_from_segments(pt("R_hip"), pt("R_kne"), pt("R_ank"), L_thigh, L_shank, m_per_px)
        knee_int_l_3d = _angle_3d_from_segments(pt("L_hip"), pt("L_kne"), pt("L_ank"), L_thigh, L_shank, m_per_px)
        hip_int_r_3d = _angle_3d_from_segments(pt("neck"), pt("R_hip"), pt("R_kne"), L_torso, L_thigh, m_per_px)
        hip_int_l_3d = _angle_3d_from_segments(pt("neck"), pt("L_hip"), pt("L_kne"), L_torso, L_thigh, m_per_px)
        ankle_int_r_3d = _angle_3d_from_segments(pt("R_kne"), pt("R_ank"), pt("R_big_toe"), L_shank, L_foot, m_per_px)
        ankle_int_l_3d = _angle_3d_from_segments(pt("L_kne"), pt("L_ank"), pt("L_big_toe"), L_shank, L_foot, m_per_px)

        # Naive 2D estimate per joint (good when segments not foreshortened)
        hip_r, kne_r, ank_r = pt("R_hip"), pt("R_kne"), pt("R_ank")
        hip_l, kne_l, ank_l = pt("L_hip"), pt("L_kne"), pt("L_ank")
        neck = pt("neck")
        toe_r, toe_l = pt("R_big_toe"), pt("L_big_toe")
        knee_int_r_2d = _angle_between(hip_r - kne_r, ank_r - kne_r)
        knee_int_l_2d = _angle_between(hip_l - kne_l, ank_l - kne_l)
        hip_int_r_2d = _angle_between(neck - hip_r, kne_r - hip_r)
        hip_int_l_2d = _angle_between(neck - hip_l, kne_l - hip_l)
        ankle_int_r_2d = _angle_between(kne_r - ank_r, toe_r - ank_r)
        ankle_int_l_2d = _angle_between(kne_l - ank_l, toe_l - ank_l)

        # Per-segment foreshortening — segments with r close to 1 are seen in
        # profile (in-plane), so we trust the 2D angle. Segments with r < ~0.8
        # are tilted toward/away from camera; we trust the 3D recovery.
        def _r(p1, p2, L_true):
            apparent = np.linalg.norm(p2 - p1, axis=1) * m_per_px
            return np.clip(apparent / L_true, 0.0, 1.0)

        r_thigh_r = _r(hip_r, kne_r, L_thigh)
        r_thigh_l = _r(hip_l, kne_l, L_thigh)
        r_shank_r = _r(kne_r, ank_r, L_shank)
        r_shank_l = _r(kne_l, ank_l, L_shank)
        r_torso = _r(neck, pt("hip_center"), L_torso)
        r_foot_r = _r(ank_r, toe_r, L_foot)
        r_foot_l = _r(ank_l, toe_l, L_foot)

        # Joint trust weight = (1 - smaller_segment_r). When r close to 1, weight
        # is near 0 → use 2D. When r close to 0 (heavily foreshortened), weight
        # is near 1 → use 3D. Smoothstep between 0.75 and 0.95 for stability.
        def _w(r1, r2):
            r_min = np.minimum(r1, r2)
            # smoothstep from 1.0 to 0.0 across r_min in [0.75, 0.95]
            t = np.clip((0.95 - r_min) / (0.95 - 0.75), 0.0, 1.0)
            return t * t * (3 - 2 * t)  # smoothstep

        w_knee_r = _w(r_thigh_r, r_shank_r)
        w_knee_l = _w(r_thigh_l, r_shank_l)
        w_hip_r = _w(r_torso, r_thigh_r)
        w_hip_l = _w(r_torso, r_thigh_l)
        w_ank_r = _w(r_shank_r, r_foot_r)
        w_ank_l = _w(r_shank_l, r_foot_l)

        knee_int_r = w_knee_r * knee_int_r_3d + (1 - w_knee_r) * knee_int_r_2d
        knee_int_l = w_knee_l * knee_int_l_3d + (1 - w_knee_l) * knee_int_l_2d
        hip_int_r = w_hip_r * hip_int_r_3d + (1 - w_hip_r) * hip_int_r_2d
        hip_int_l = w_hip_l * hip_int_l_3d + (1 - w_hip_l) * hip_int_l_2d
        ankle_int_r = w_ank_r * ankle_int_r_3d + (1 - w_ank_r) * ankle_int_r_2d
        ankle_int_l = w_ank_l * ankle_int_l_3d + (1 - w_ank_l) * ankle_int_l_2d

        # OpenSim conventions: flexion = 180 - interior angle for knee/hip,
        # 90 - interior for ankle.
        knee_angle_r = 180.0 - knee_int_r
        knee_angle_l = 180.0 - knee_int_l
        hip_flexion_r = 180.0 - hip_int_r
        hip_flexion_l = 180.0 - hip_int_l
        ankle_angle_r = 90.0 - ankle_int_r
        ankle_angle_l = 90.0 - ankle_int_l
        return _pack_motion(
            series.time, knee_angle_r, knee_angle_l,
            hip_flexion_r, hip_flexion_l,
            ankle_angle_r, ankle_angle_l,
            xy, pt,
        )

    # Knee flexion: angle at knee between (hip→knee) and (knee→ankle)
    hip_r, kne_r, ank_r = pt("R_hip"), pt("R_kne"), pt("R_ank")
    hip_l, kne_l, ank_l = pt("L_hip"), pt("L_kne"), pt("L_ank")
    knee_angle_r = _angle_between(hip_r - kne_r, ank_r - kne_r)
    knee_angle_l = _angle_between(hip_l - kne_l, ank_l - kne_l)
    # Convert from "interior angle" to "flexion": 180° straight, 0° fully bent.
    # OpenSim knee_angle convention: 0° = extended; we report 180 - measured so
    # downstream peak/ROM math is sign-aligned with OpenSim.
    knee_angle_r = 180.0 - knee_angle_r
    knee_angle_l = 180.0 - knee_angle_l

    # Hip flexion: angle at hip between (torso) and (hip→knee), sagittal sign
    # Torso direction = neck - hip (upward in image)
    neck = pt("neck")
    torso_r = neck - hip_r
    torso_l = neck - hip_l
    thigh_r = kne_r - hip_r
    thigh_l = kne_l - hip_l
    hip_flexion_r = _angle_between(torso_r, thigh_r)
    hip_flexion_l = _angle_between(torso_l, thigh_l)
    # Hip neutral = thigh aligned with torso (angle near 180°). Convert to flexion:
    hip_flexion_r = 180.0 - hip_flexion_r
    hip_flexion_l = 180.0 - hip_flexion_l

    # Hip adduction (frontal plane): horizontal angle from hip to knee
    # In 2D, this is a coarse approximation — only meaningful from front view.
    # arctan2 returns [-180, 180]; np.unwrap stitches across the boundary so
    # athletic motion that crosses near-vertical doesn't produce fake 360° ROMs.
    hip_adduction_r = np.degrees(np.unwrap(np.arctan2(kne_r[:, 0] - hip_r[:, 0], hip_r[:, 1] - kne_r[:, 1])))
    hip_adduction_l = np.degrees(np.unwrap(np.arctan2(hip_l[:, 0] - kne_l[:, 0], hip_l[:, 1] - kne_l[:, 1])))

    # Ankle flexion: angle at ankle between (knee→ankle) and (ankle→big_toe)
    toe_r = pt("R_big_toe")
    toe_l = pt("L_big_toe")
    ankle_angle_r = _angle_between(kne_r - ank_r, toe_r - ank_r)
    ankle_angle_l = _angle_between(kne_l - ank_l, toe_l - ank_l)
    # Neutral ankle = 90° between leg & foot; convert to dorsiflexion (positive)
    ankle_angle_r = 90.0 - ankle_angle_r
    ankle_angle_l = 90.0 - ankle_angle_l

    # Trunk lean (lumbar_extension proxy): signed angle of neck→hip_center vec from vertical
    hip_c = pt("hip_center")
    trunk_vec = hip_c - neck     # downward-ish in image when standing
    lumbar_extension = -_signed_angle_from_vertical(trunk_vec)

    # Pelvis tilt: signed angle of L_hip→R_hip vector from horizontal
    pelvis_vec = pt("R_hip") - pt("L_hip")
    pelvis_tilt = np.degrees(np.arctan2(pelvis_vec[:, 1], pelvis_vec[:, 0]))

    columns = (
        "time", "pelvis_tilt",
        "hip_flexion_r", "hip_adduction_r", "knee_angle_r", "ankle_angle_r",
        "hip_flexion_l", "hip_adduction_l", "knee_angle_l", "ankle_angle_l",
        "lumbar_extension",
    )
    values = np.column_stack([
        pelvis_tilt,
        hip_flexion_r, hip_adduction_r, knee_angle_r, ankle_angle_r,
        hip_flexion_l, hip_adduction_l, knee_angle_l, ankle_angle_l,
        lumbar_extension,
    ])
    return MotionData(
        columns=columns, time=series.time, values=values, in_degrees=True,
    )


def _pack_motion(
    time: np.ndarray,
    knee_r: np.ndarray, knee_l: np.ndarray,
    hip_flex_r: np.ndarray, hip_flex_l: np.ndarray,
    ankle_r: np.ndarray, ankle_l: np.ndarray,
    xy: np.ndarray, pt,
) -> MotionData:
    """Pack 3D-reconstructed angles plus the frontal-plane proxies into MotionData."""
    # Frontal-plane proxies (unchanged — these were already 2D-only and only
    # meaningful from a front view; not affected by foreshortening reconstruction).
    hip_r = pt("R_hip"); hip_l = pt("L_hip")
    kne_r = pt("R_kne"); kne_l = pt("L_kne")
    neck = pt("neck"); hip_c = pt("hip_center")
    # arctan2 returns [-180, 180]; np.unwrap stitches across the boundary so
    # athletic motion that crosses near-vertical doesn't produce fake 360° ROMs.
    hip_adduction_r = np.degrees(np.unwrap(np.arctan2(kne_r[:, 0] - hip_r[:, 0], hip_r[:, 1] - kne_r[:, 1])))
    hip_adduction_l = np.degrees(np.unwrap(np.arctan2(hip_l[:, 0] - kne_l[:, 0], hip_l[:, 1] - kne_l[:, 1])))
    trunk_vec = hip_c - neck
    lumbar_extension = -np.degrees(np.arctan2(trunk_vec[:, 0], trunk_vec[:, 1]))
    pelvis_vec = pt("R_hip") - pt("L_hip")
    pelvis_tilt = np.degrees(np.arctan2(pelvis_vec[:, 1], pelvis_vec[:, 0]))

    columns = (
        "time", "pelvis_tilt",
        "hip_flexion_r", "hip_adduction_r", "knee_angle_r", "ankle_angle_r",
        "hip_flexion_l", "hip_adduction_l", "knee_angle_l", "ankle_angle_l",
        "lumbar_extension",
    )
    values = np.column_stack([
        pelvis_tilt,
        hip_flex_r, hip_adduction_r, knee_r, ankle_r,
        hip_flex_l, hip_adduction_l, knee_l, ankle_l,
        lumbar_extension,
    ])
    return MotionData(columns=columns, time=time, values=values, in_degrees=True)


def write_as_mot(series: CouroKeypointSeries, out_path: Path) -> None:
    """Convert Couro keypoint output to OpenSim .mot format so the existing
    sweep machinery can consume it as another 'source' alongside HRNet/OpenPose.
    """
    motion = keypoints_to_motion_data(series)
    cols = motion.columns
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Coordinates\nversion=1\n")
        f.write(f"nRows={motion.time.size}\n")
        f.write(f"nColumns={len(cols)}\n")
        f.write("inDegrees=yes\n\n")
        f.write("endheader\n")
        f.write("\t".join(cols) + "\n")
        for i in range(motion.time.size):
            row = [f"{motion.time[i]:.4f}"]
            row.extend(f"{v:.4f}" if np.isfinite(v) else "nan" for v in motion.values[i])
            f.write("\t".join(row) + "\n")
