"""ASPset-510 dataset loader — parses C3D + cameras into the same shape as
our OpenCap pipeline so we can mix the two for training.

ASPset-510 schema:
  ASPset-510/
    splits.csv                                # subj, clip, split, segment
    trainval/joints_3d/{subj}/{subj}-{clip}.c3d   # 3D joint positions
    trainval/cameras/{subj}/{subj}-{view}.json    # intrinsics + extrinsics
    trainval/videos/{subj}/{subj}-{clip}-{view}.mkv   # raw video
    trainval/boxes/{subj}/{subj}-{clip}-{view}.csv    # athlete bbox per frame

  17 keypoints (no feet): right_ankle, right_knee, right_hip, right_wrist,
                          right_elbow, right_shoulder, left_ankle, ...,
                          head_top, head, neck, spine, pelvis
  50 Hz, 3D positions in mm
  3 cameras per subject (left, mid, right) — outdoor athletic clips

We convert the 3D positions into the same OpenSim-style joint angles we
already extract from OpenCap (hip_flexion_r, hip_adduction_r, knee_angle_r,
lumbar_extension) so the regression training code doesn't change.

ASPset HAS NO foot/toe keypoints → ankle_angle_r cannot be derived from
this dataset. That metric stays OpenCap-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import c3d
import numpy as np


ASPSET_KP = {
    "right_ankle": 0, "right_knee": 1, "right_hip": 2,
    "right_wrist": 3, "right_elbow": 4, "right_shoulder": 5,
    "left_ankle": 6, "left_knee": 7, "left_hip": 8,
    "left_wrist": 9, "left_elbow": 10, "left_shoulder": 11,
    "head_top": 12, "head": 13, "neck": 14, "spine": 15, "pelvis": 16,
}


@dataclass(frozen=True)
class ASPsetClip:
    subject: str
    clip_id: str           # e.g. "0066"
    joints_3d: np.ndarray  # (n_frames, 17, 3) in mm, world frame
    fps: float             # 50 Hz typically


@dataclass(frozen=True)
class ASPsetCamera:
    view: str              # "left", "mid", "right"
    intrinsic_matrix: np.ndarray   # 3x4
    extrinsic_matrix: np.ndarray   # 4x4 world→camera


def load_clip(c3d_path: Path) -> ASPsetClip:
    """Read one .c3d file into a fixed-shape (n, 17, 3) array."""
    with open(c3d_path, "rb") as f:
        reader = c3d.Reader(f)
        labels = [l.strip() for l in reader.point_labels]
        # Verify expected ordering
        for expected, actual in zip(ASPSET_KP.keys(), labels):
            if expected != actual:
                raise ValueError(f"Unexpected label order in {c3d_path}: {actual} vs {expected}")
        all_points = []
        for frame_no, points, analog in reader.read_frames():
            # points shape: (17, 5) where columns are x, y, z, residual, cameras
            all_points.append(points[:, :3])
        fps = float(reader.point_rate)
    arr = np.asarray(all_points, dtype=np.float64)
    # Parse subject/clip from filename: e.g. "04ac-0066.c3d" → subject "04ac", clip "0066"
    stem = c3d_path.stem
    subj, clip_id = stem.split("-", 1)
    return ASPsetClip(subject=subj, clip_id=clip_id, joints_3d=arr, fps=fps)


def load_camera(cam_path: Path) -> ASPsetCamera:
    """Read one camera JSON file."""
    with open(cam_path) as f:
        d = json.load(f)
    # intrinsic_matrix in ASPset is 3x4 flattened
    K = np.asarray(d["intrinsic_matrix"], dtype=np.float64).reshape(3, 4)
    # extrinsic_matrix is 4x4 flattened
    E = np.asarray(d["extrinsic_matrix"], dtype=np.float64).reshape(4, 4)
    stem = cam_path.stem  # e.g. "04ac-mid"
    view = stem.rsplit("-", 1)[1]
    return ASPsetCamera(view=view, intrinsic_matrix=K, extrinsic_matrix=E)


def project_to_2d(
    points_3d_world_mm: np.ndarray, cam: ASPsetCamera,
) -> np.ndarray:
    """Project world-frame 3D joint positions to 2D image pixels.

    points_3d_world_mm: (n_frames, 17, 3) in mm
    Returns: (n_frames, 17, 2) in pixels
    """
    n_frames, n_joints, _ = points_3d_world_mm.shape
    # World → camera frame
    pts_flat = points_3d_world_mm.reshape(-1, 3)               # (N, 3)
    pts_h = np.hstack([pts_flat, np.ones((pts_flat.shape[0], 1))])  # (N, 4)
    pts_cam = (cam.extrinsic_matrix @ pts_h.T).T               # (N, 4)
    # Camera → image (K is 3x4)
    pts_2d_h = (cam.intrinsic_matrix @ pts_cam.T).T            # (N, 3)
    # Perspective divide
    pts_2d = pts_2d_h[:, :2] / pts_2d_h[:, 2:3]                # (N, 2)
    return pts_2d.reshape(n_frames, n_joints, 2)


# ---------------------------------------------------------------------------
# Joint angle computation from 3D positions (in mm or any consistent unit)
# ---------------------------------------------------------------------------

def _angle_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle between two 3D vector time-series per frame, in degrees [0, 180]."""
    dot = (v1 * v2).sum(axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.clip(
        dot / np.where(n1 * n2 > 1e-9, n1 * n2, 1e-9), -1.0, 1.0,
    )
    return np.degrees(np.arccos(cos))


def joint_angles_from_aspset(clip: ASPsetClip) -> dict[str, np.ndarray]:
    """Convert ASPset 17-keypoint 3D positions into the same OpenSim-style
    joint angles we use elsewhere. Returns a dict of (n_frames,) arrays.

    Cannot produce ankle_angle_* — ASPset has no foot/toe keypoints.
    """
    pts = clip.joints_3d  # (n_frames, 17, 3)

    def kp(name: str) -> np.ndarray:
        return pts[:, ASPSET_KP[name], :]

    # Right hip flexion: angle between (shoulder→hip) and (hip→knee)
    torso_r = kp("right_hip") - kp("right_shoulder")
    thigh_r = kp("right_knee") - kp("right_hip")
    hip_flexion_r = _angle_3d(torso_r, thigh_r)

    # Left hip flexion
    torso_l = kp("left_hip") - kp("left_shoulder")
    thigh_l = kp("left_knee") - kp("left_hip")
    hip_flexion_l = _angle_3d(torso_l, thigh_l)

    # Right knee flexion: 180 - angle between (hip→knee) and (knee→ankle).
    # OpenSim convention: 0° = straight, positive = flexed.
    shank_r = kp("right_ankle") - kp("right_knee")
    knee_angle_r = 180.0 - _angle_3d(thigh_r, shank_r)

    # Left knee flexion
    shank_l = kp("left_ankle") - kp("left_knee")
    knee_angle_l = 180.0 - _angle_3d(thigh_l, shank_l)

    # Hip adduction (frontal plane): angle of thigh from body-vertical, projected
    # into the frontal plane. Wrap-free: arccos of the cosine bounded to [0, 180].
    body_vertical = kp("neck") - kp("pelvis")
    body_vertical = body_vertical / (np.linalg.norm(body_vertical, axis=-1, keepdims=True) + 1e-9)
    pelvis_lr = kp("right_hip") - kp("left_hip")
    pelvis_lr_norm = pelvis_lr / (np.linalg.norm(pelvis_lr, axis=-1, keepdims=True) + 1e-9)
    # Forward direction (anterior–posterior) = pelvis_lr × body_vertical
    forward = np.cross(pelvis_lr_norm, body_vertical)
    forward = forward / (np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-9)

    def _adduction(thigh: np.ndarray, lateral_axis: np.ndarray) -> np.ndarray:
        # Bounded definition: asin(lateral_component / thigh_length_in_frontal_plane).
        # Returns [-90°, +90°] — can't wrap. Adequate for any human motion where
        # the leg is roughly below the hip (i.e. not handstands/inversions).
        thigh_fp = thigh - (thigh * forward).sum(axis=-1, keepdims=True) * forward
        thigh_fp_len = np.linalg.norm(thigh_fp, axis=-1) + 1e-9
        lateral_component = (thigh * lateral_axis).sum(axis=-1)
        # For right leg: leg moving toward LEFT side (-lateral_axis direction) = adducted (positive)
        ratio = -lateral_component / thigh_fp_len
        return np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))

    hip_adduction_r = _adduction(thigh_r, pelvis_lr_norm)
    hip_adduction_l = _adduction(thigh_l, -pelvis_lr_norm)

    # Lumbar extension: angle between (pelvis→spine) and (spine→neck)
    lower_trunk = kp("spine") - kp("pelvis")
    upper_trunk = kp("neck") - kp("spine")
    lumbar_extension = 180.0 - _angle_3d(lower_trunk, upper_trunk)

    # Pelvis tilt: angle of (L_hip - R_hip) from horizontal (around vertical axis)
    # Use the projection of pelvis_lr onto a horizontal reference.
    pelvis_tilt = np.degrees(
        np.arctan2(pelvis_lr[:, 1], np.sqrt(pelvis_lr[:, 0] ** 2 + pelvis_lr[:, 2] ** 2))
    )

    return {
        "hip_flexion_r": hip_flexion_r,
        "hip_flexion_l": hip_flexion_l,
        "knee_angle_r": knee_angle_r,
        "knee_angle_l": knee_angle_l,
        "hip_adduction_r": hip_adduction_r,
        "hip_adduction_l": hip_adduction_l,
        "lumbar_extension": lumbar_extension,
        "pelvis_tilt": pelvis_tilt,
    }


# ---------------------------------------------------------------------------
# Bridge: ASPset 3D → synthetic Couro-Halpe26-style 2D keypoints
# ---------------------------------------------------------------------------

# Mapping ASPset 17 → Halpe 26 (NaN-fill where Halpe has a keypoint ASPset doesn't)
ASPSET_TO_HALPE26 = {
    # Halpe-26 idx : ASPset name (or None for unavailable)
    0:  None,           # nose             — not in ASPset
    1:  None,           # L_eye            — not in ASPset
    2:  None,           # R_eye            — not in ASPset
    3:  None,           # L_ear            — not in ASPset
    4:  None,           # R_ear            — not in ASPset
    5:  "left_shoulder",
    6:  "right_shoulder",
    7:  "left_elbow",
    8:  "right_elbow",
    9:  "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
    17: "head_top",
    18: "neck",
    19: "pelvis",       # Halpe "hip_center" ≈ ASPset "pelvis"
    20: None,           # L_big_toe        — not in ASPset
    21: None,           # R_big_toe        — not in ASPset
    22: None,           # L_small_toe      — not in ASPset
    23: None,           # R_small_toe      — not in ASPset
    24: None,           # L_heel           — not in ASPset
    25: None,           # R_heel           — not in ASPset
}


def aspset_to_halpe26_2d(
    aspset_xy_2d: np.ndarray,
) -> np.ndarray:
    """Convert ASPset 17-joint 2D output to Halpe-26 layout (NaN-padded).

    aspset_xy_2d: (n_frames, 17, 2)
    Returns:      (n_frames, 26, 2) with NaN for unavailable keypoints
    """
    n_frames = aspset_xy_2d.shape[0]
    halpe = np.full((n_frames, 26, 2), np.nan, dtype=np.float64)
    for halpe_idx, aspset_name in ASPSET_TO_HALPE26.items():
        if aspset_name is None:
            continue
        halpe[:, halpe_idx, :] = aspset_xy_2d[:, ASPSET_KP[aspset_name], :]
    return halpe
