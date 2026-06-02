"""Ground-truth joint angle computation from 3D joint positions.

These functions compute the same 5 joint angles Couro's CV pipeline regresses
against (hip flexion, knee flexion, ankle dorsiflexion, hip adduction, lumbar
extension), but from the 3D skeleton ground truth — giving us perfect labels
for synthetic training data.

For real SMPL+H output, the same functions work because the input is just a
(144, 3) joint array. In the longer term we can also pull angles directly from
SMPL+H pose params (axis-angle), but computing from joint positions keeps the
proof-of-concept aligned with how Couro extracts angles from CV reconstruction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class JointAnglesDeg:
    """Joint angles in degrees, matching Couro's scoring config."""

    hip_flex_L: float
    hip_flex_R: float
    knee_flex_L: float
    knee_flex_R: float
    ankle_dorsiflex_L: float
    ankle_dorsiflex_R: float
    hip_adduct_L: float
    hip_adduct_R: float
    lumbar_extension: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Unsigned angle between two vectors, in radians."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos))


def compute_joint_angles(joints: np.ndarray) -> JointAnglesDeg:
    """Compute joint angles from a (144, 3) skeleton.

    Indices follow smplx.joint_names.JOINT_NAMES:
        0 pelvis, 1/2 L/R hip, 4/5 L/R knee, 7/8 L/R ankle,
        10/11 L/R foot, 12 neck, 9 spine3.
    """
    pelvis = joints[0]
    l_hip, r_hip = joints[1], joints[2]
    l_knee, r_knee = joints[4], joints[5]
    l_ank, r_ank = joints[7], joints[8]
    l_foot, r_foot = joints[10], joints[11]
    spine3, neck = joints[9], joints[12]

    # Hip flexion: angle between thigh and pelvis-down axis.
    pelvis_down = np.array([0.0, -1.0, 0.0])
    l_thigh = l_knee - l_hip
    r_thigh = r_knee - r_hip
    hip_flex_L = np.rad2deg(_angle_between(l_thigh, pelvis_down))
    hip_flex_R = np.rad2deg(_angle_between(r_thigh, pelvis_down))

    # Knee flexion: angle between thigh (hip->knee) and shank (knee->ankle).
    # Flexion = 180 - angle_between(thigh_down, shank_down).
    l_shank = l_ank - l_knee
    r_shank = r_ank - r_knee
    knee_flex_L = 180.0 - np.rad2deg(_angle_between(l_thigh, l_shank))
    knee_flex_R = 180.0 - np.rad2deg(_angle_between(r_thigh, r_shank))

    # Ankle dorsiflexion: angle between shank (ankle->knee) and foot vector.
    # Neutral standing ~90deg. Dorsiflexion reduces the angle below 90.
    l_foot_vec = l_foot - l_ank
    r_foot_vec = r_foot - r_ank
    ankle_dorsiflex_L = 90.0 - np.rad2deg(_angle_between(-l_shank, l_foot_vec))
    ankle_dorsiflex_R = 90.0 - np.rad2deg(_angle_between(-r_shank, r_foot_vec))

    # Hip adduction: lateral angle between thigh and pelvis-down in the
    # frontal (XY) plane.
    l_thigh_frontal = np.array([l_thigh[0], l_thigh[1], 0.0])
    r_thigh_frontal = np.array([r_thigh[0], r_thigh[1], 0.0])
    hip_adduct_L = np.rad2deg(_angle_between(l_thigh_frontal, pelvis_down))
    hip_adduct_R = np.rad2deg(_angle_between(r_thigh_frontal, pelvis_down))
    # Sign: positive when thigh tilts toward midline (x toward 0).
    if (l_hip[0] > 0 and l_thigh[0] < 0) or (l_hip[0] < 0 and l_thigh[0] > 0):
        hip_adduct_L = -hip_adduct_L
    if (r_hip[0] > 0 and r_thigh[0] < 0) or (r_hip[0] < 0 and r_thigh[0] > 0):
        hip_adduct_R = -hip_adduct_R

    # Lumbar extension: angle between spine (pelvis->spine3) and world up.
    spine_vec = spine3 - pelvis
    world_up = np.array([0.0, 1.0, 0.0])
    lumbar_extension = np.rad2deg(_angle_between(spine_vec, world_up))
    # Sign: positive when spine tilts forward (+Z component).
    if spine_vec[2] < 0:
        lumbar_extension = -lumbar_extension

    return JointAnglesDeg(
        hip_flex_L=round(hip_flex_L, 3),
        hip_flex_R=round(hip_flex_R, 3),
        knee_flex_L=round(knee_flex_L, 3),
        knee_flex_R=round(knee_flex_R, 3),
        ankle_dorsiflex_L=round(ankle_dorsiflex_L, 3),
        ankle_dorsiflex_R=round(ankle_dorsiflex_R, 3),
        hip_adduct_L=round(hip_adduct_L, 3),
        hip_adduct_R=round(hip_adduct_R, 3),
        lumbar_extension=round(lumbar_extension, 3),
    )
