"""Anthropometric 2D→3D joint-angle reconstruction.

Uses segment-length priors + camera intrinsics to recover the out-of-plane
component of joint angles that pure 2D vector math cannot see. Same insight
as OpenSim IK (rigid segment lengths constrain depth), implemented in pure
numpy without an OpenSim install.

Per-segment workflow:
1. Predict true segment length L_true in metres from subject height +
   standard anthropometric ratios (Winter 2009, de Leva 1996).
2. Measure apparent length L_apparent in pixels from the 2D keypoint output.
3. Estimate pixel-to-metre scale at the subject's distance from the camera
   using camera intrinsics + observed torso/segment.
4. Compute foreshortening ratio = (L_apparent / pixel_scale) / L_true.
5. Out-of-plane tilt = arccos(clip(ratio, 0, 1)).
6. Combine with in-plane angle from the 2D vectors to get the 3D joint angle.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Anthropometric ratios — fraction of total height (Winter 2009, table 4.1)
# Source: D.A. Winter, Biomechanics and Motor Control of Human Movement, 4th ed.
ANTHRO_RATIOS = {
    "thigh": 0.245,         # hip joint → knee joint
    "shank": 0.246,          # knee joint → ankle joint
    "foot": 0.152,           # heel → big toe (sole length)
    "torso": 0.288,          # hip → shoulder
    "head_neck": 0.130,
    "upper_arm": 0.186,
    "forearm": 0.146,
    "hand": 0.108,
}


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics extracted from OpenCap's cameraIntrinsicsExtrinsics.pickle."""
    fx: float                # focal length x (pixels)
    fy: float                # focal length y (pixels)
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_pickle(cls, path: Path, image_width: int, image_height: int) -> "CameraModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        K = np.asarray(data["intrinsicMat"], dtype=np.float64)
        # OpenCap intrinsics are stored at the native camera resolution; OpenCap's
        # sync'd video may be downscaled. Scale the focal lengths proportionally.
        stored_w, stored_h = data.get("imageSize", (image_width, image_height))
        if isinstance(stored_w, (list, tuple)):
            stored_w, stored_h = stored_w[0], stored_w[1]
        scale_x = image_width / float(stored_w) if stored_w else 1.0
        scale_y = image_height / float(stored_h) if stored_h else 1.0
        return cls(
            fx=float(K[0, 0]) * scale_x,
            fy=float(K[1, 1]) * scale_y,
            cx=float(K[0, 2]) * scale_x,
            cy=float(K[1, 2]) * scale_y,
            width=image_width,
            height=image_height,
        )

    @classmethod
    def from_aspset_json(cls, path: Path, image_width: int, image_height: int) -> "CameraModel":
        """ASPset cameras store intrinsics as 3x4 in JSON. Same idea as
        from_pickle: scale focal lengths if the inference image was downsampled.
        """
        import json as _json
        with open(path) as f:
            data = _json.load(f)
        K = np.asarray(data["intrinsic_matrix"], dtype=np.float64).reshape(3, 4)
        # ASPset videos are 4K (3840x2160). If Couro inference was run at
        # MAX_DIMENSION=1280, the keypoints are at the scaled resolution.
        # ASPset always native 4K so stored intrinsics assume 3840x2160 base.
        stored_w, stored_h = 3840, 2160
        scale_x = image_width / float(stored_w)
        scale_y = image_height / float(stored_h)
        return cls(
            fx=float(K[0, 0]) * scale_x,
            fy=float(K[1, 1]) * scale_y,
            cx=float(K[0, 2]) * scale_x,
            cy=float(K[1, 2]) * scale_y,
            width=image_width,
            height=image_height,
        )


def estimate_subject_depth_per_frame(
    xy: np.ndarray,
    confidence: np.ndarray,
    cam: CameraModel,
    subject_height_m: float,
    *,
    KP: dict[str, int],
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """Estimate the metres-per-pixel scale at the subject's distance for each frame.

    Uses the most reliable proxy segment (torso = hip_center → neck) and the
    anthropometric prediction of its true length. If torso confidence is poor,
    falls back to shoulder-to-shoulder width (less robust to lean but always
    visible).
    """
    n_frames = xy.shape[0]
    torso_true_m = subject_height_m * ANTHRO_RATIOS["torso"]

    hip = xy[:, KP["hip_center"], :]
    neck = xy[:, KP["neck"], :]
    hip_conf = confidence[:, KP["hip_center"]]
    neck_conf = confidence[:, KP["neck"]]

    torso_pixels = np.linalg.norm(neck - hip, axis=1)
    # Subject distance Z (metres) from camera: focal_length * (true_length / apparent_pixels).
    # Use fy since torso is mostly vertical in image.
    Z = (cam.fy * torso_true_m) / np.where(torso_pixels > 1, torso_pixels, np.nan)
    # metres-per-pixel at that depth (small-angle: m_per_pixel = Z / fy).
    m_per_px = Z / cam.fy
    # Mask out frames where torso confidence is poor.
    bad = (hip_conf < conf_threshold) | (neck_conf < conf_threshold) | ~np.isfinite(m_per_px)
    m_per_px[bad] = np.nan
    return m_per_px


def _seg_length_apparent_m(p1: np.ndarray, p2: np.ndarray, m_per_px: np.ndarray) -> np.ndarray:
    """Apparent segment length in metres, frame-by-frame."""
    return np.linalg.norm(p2 - p1, axis=1) * m_per_px


def _angle_3d_from_segments(
    proximal: np.ndarray,
    joint: np.ndarray,
    distal: np.ndarray,
    L_prox_true: float,
    L_dist_true: float,
    m_per_px: np.ndarray,
) -> np.ndarray:
    """3D joint angle from 2D keypoints + rigid segment-length constraints.

    Reconstructs each segment's 3D vector by combining in-plane direction
    (from 2D pixel positions) with out-of-plane tilt (from foreshortening
    against the true segment length).
    """
    seg_prox_2d = joint - proximal               # (n, 2)
    seg_dist_2d = distal - joint
    L_prox_apparent_m = np.linalg.norm(seg_prox_2d, axis=1) * m_per_px
    L_dist_apparent_m = np.linalg.norm(seg_dist_2d, axis=1) * m_per_px
    # Foreshortening ratio in [0, 1]; clip to handle keypoint noise that
    # makes apparent length exceed true length slightly.
    r_prox = np.clip(L_prox_apparent_m / L_prox_true, 0.0, 1.0)
    r_dist = np.clip(L_dist_apparent_m / L_dist_true, 0.0, 1.0)
    # Out-of-plane component of each segment (radians).
    tilt_prox = np.arccos(r_prox)
    tilt_dist = np.arccos(r_dist)
    # In-plane 3D directions (unit vectors with z=0):
    norm_p = np.linalg.norm(seg_prox_2d, axis=1)
    norm_d = np.linalg.norm(seg_dist_2d, axis=1)
    in_plane_prox = seg_prox_2d / np.where(norm_p[:, None] > 1, norm_p[:, None], 1.0)
    in_plane_dist = seg_dist_2d / np.where(norm_d[:, None] > 1, norm_d[:, None], 1.0)
    # 3D unit vectors: (in_plane * sin(out_of_plane_complement), z = cos(complement))
    # tilt is the angle from the image plane, so the in-plane component
    # has magnitude cos(tilt) (= r) and the z-component is sin(tilt).
    # Sign of z: we cannot disambiguate "toward camera" vs "away from camera"
    # without temporal continuity; pick consistent sign by assuming distal
    # endpoint of distal segment is FORWARD relative to joint when ratio < 1.
    v_prox = np.column_stack([
        in_plane_prox[:, 0] * r_prox,
        in_plane_prox[:, 1] * r_prox,
        np.sin(tilt_prox),
    ])
    v_dist = np.column_stack([
        in_plane_dist[:, 0] * r_dist,
        in_plane_dist[:, 1] * r_dist,
        np.sin(tilt_dist),
    ])
    # 3D joint angle is the angle between (proximal→joint) and (joint→distal).
    # Note: v_prox points joint→proximal direction; reverse it.
    v_prox_at_joint = -v_prox
    dot = (v_prox_at_joint * v_dist).sum(axis=1)
    norm_a = np.linalg.norm(v_prox_at_joint, axis=1)
    norm_b = np.linalg.norm(v_dist, axis=1)
    cos = np.clip(dot / np.where(norm_a * norm_b > 1e-9, norm_a * norm_b, 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cos))
