"""Procedural synthetic skeleton in the smplx 144-joint output layout.

This module exists so the rest of the proof-of-concept pipeline (camera
projection, Halpe-26 mapping, GT angle computation) can be exercised end-to-end
without requiring registered SMPL+H model files.

Once SMPL+H is registered and downloaded, this module is replaced by:

    body_model = smplx.create(MODEL_PATH, model_type='smplh', gender='neutral',
                               use_pca=False, batch_size=1)
    output = body_model(betas=betas, body_pose=body_pose, global_orient=g_or)
    joints_world = output.joints[0].detach().numpy()  # (144, 3)

The downstream code consumes the same (144, 3) numpy array either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticPose:
    """Joint-angle parameters used to synthesize a body pose.

    All angles in radians. Positive flexion = forward bending.
    Used both to procedurally place joints AND as ground-truth.
    """

    hip_flex_L: float = 0.0
    hip_flex_R: float = 0.0
    knee_flex_L: float = 0.0
    knee_flex_R: float = 0.0
    ankle_dorsiflex_L: float = 0.0
    ankle_dorsiflex_R: float = 0.0
    hip_adduct_L: float = 0.0
    hip_adduct_R: float = 0.0
    lumbar_extension: float = 0.0


# Anthropometric defaults (meters) — average adult male, used only for the
# synthetic proof. Real pipeline uses SMPL+H's learned shape blend shapes.
SEG_LEN = {
    "pelvis_width": 0.30,
    "spine1_height": 0.12,
    "spine2_height": 0.12,
    "spine3_height": 0.10,
    "neck_height": 0.08,
    "head_height": 0.18,
    "shoulder_width": 0.42,
    "upper_arm_len": 0.30,
    "forearm_len": 0.27,
    "thigh_len": 0.45,
    "shank_len": 0.43,
    "foot_len": 0.25,
    "heel_offset": 0.07,
}


def _rotate_about_axis(point: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate point around axis by angle (Rodrigues)."""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    return (
        point * c
        + np.cross(axis, point) * s
        + axis * np.dot(axis, point) * (1 - c)
    )


def generate_synthetic_joints(pose: SyntheticPose) -> np.ndarray:
    """Return a (144, 3) array of joint positions in world coords.

    Hip_center sits at world origin. +Y up. Subject facing +Z (so a rear-view
    camera at azimuth=180 sees the subject's back).
    """
    joints = np.zeros((144, 3), dtype=np.float64)

    # --- Spine chain ---
    joints[0] = [0.0, 0.0, 0.0]  # pelvis
    joints[3] = [0.0, SEG_LEN["spine1_height"], 0.0]                          # spine1
    joints[6] = joints[3] + [0.0, SEG_LEN["spine2_height"], 0.0]              # spine2
    joints[9] = joints[6] + [0.0, SEG_LEN["spine3_height"], 0.0]              # spine3
    joints[12] = joints[9] + [0.0, SEG_LEN["neck_height"], 0.0]               # neck
    joints[15] = joints[12] + [0.0, SEG_LEN["head_height"] * 0.5, 0.05]       # head

    # Apply lumbar extension to spine chain above pelvis.
    # Tilt spine1..head forward by lumbar_extension around the pelvis X axis.
    if abs(pose.lumbar_extension) > 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
        for idx in [3, 6, 9, 12, 15]:
            joints[idx] = _rotate_about_axis(joints[idx], x_axis, pose.lumbar_extension)

    # --- Shoulders / arms (T-pose, arms straight out) ---
    sho_y = joints[9][1]
    half_sw = SEG_LEN["shoulder_width"] / 2.0
    joints[16] = [+half_sw, sho_y, 0.0]   # L shoulder
    joints[17] = [-half_sw, sho_y, 0.0]   # R shoulder
    joints[18] = joints[16] + [SEG_LEN["upper_arm_len"], 0.0, 0.0]  # L elbow
    joints[19] = joints[17] + [-SEG_LEN["upper_arm_len"], 0.0, 0.0] # R elbow
    joints[20] = joints[18] + [SEG_LEN["forearm_len"], 0.0, 0.0]    # L wrist
    joints[21] = joints[19] + [-SEG_LEN["forearm_len"], 0.0, 0.0]   # R wrist

    # Carry shoulder/arm joints up by spine tilt as well.
    if abs(pose.lumbar_extension) > 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
        for idx in [16, 17, 18, 19, 20, 21]:
            joints[idx] = _rotate_about_axis(joints[idx], x_axis, pose.lumbar_extension)

    # --- Hips/legs ---
    half_pw = SEG_LEN["pelvis_width"] / 2.0
    joints[1] = [+half_pw, 0.0, 0.0]   # L hip
    joints[2] = [-half_pw, 0.0, 0.0]   # R hip

    # Left leg chain
    l_thigh = np.array([0.0, -SEG_LEN["thigh_len"], 0.0])
    # Apply hip flexion (rotate thigh forward about +X axis through hip).
    l_thigh = _rotate_about_axis(l_thigh, np.array([1.0, 0.0, 0.0]), pose.hip_flex_L)
    # Apply hip adduction (rotate about +Z through hip; pos = toward midline).
    l_thigh = _rotate_about_axis(l_thigh, np.array([0.0, 0.0, 1.0]), -pose.hip_adduct_L)
    joints[4] = joints[1] + l_thigh

    l_shank_dir = l_thigh / np.linalg.norm(l_thigh)
    # Knee flexion rotates the shank backward about the knee X axis (relative).
    l_shank = _rotate_about_axis(
        l_shank_dir * SEG_LEN["shank_len"],
        np.array([1.0, 0.0, 0.0]),
        -pose.knee_flex_L,
    )
    joints[7] = joints[4] + l_shank

    # Foot: from ankle, project forward, with dorsiflexion adjusting tilt.
    l_foot_vec = np.array([0.0, -0.02, SEG_LEN["foot_len"]])
    l_foot_vec = _rotate_about_axis(
        l_foot_vec, np.array([1.0, 0.0, 0.0]), -pose.ankle_dorsiflex_L
    )
    joints[10] = joints[7] + l_foot_vec

    # Right leg (mirrored).
    r_thigh = np.array([0.0, -SEG_LEN["thigh_len"], 0.0])
    r_thigh = _rotate_about_axis(r_thigh, np.array([1.0, 0.0, 0.0]), pose.hip_flex_R)
    r_thigh = _rotate_about_axis(r_thigh, np.array([0.0, 0.0, 1.0]), pose.hip_adduct_R)
    joints[5] = joints[2] + r_thigh

    r_shank_dir = r_thigh / np.linalg.norm(r_thigh)
    r_shank = _rotate_about_axis(
        r_shank_dir * SEG_LEN["shank_len"],
        np.array([1.0, 0.0, 0.0]),
        -pose.knee_flex_R,
    )
    joints[8] = joints[5] + r_shank

    r_foot_vec = np.array([0.0, -0.02, SEG_LEN["foot_len"]])
    r_foot_vec = _rotate_about_axis(
        r_foot_vec, np.array([1.0, 0.0, 0.0]), -pose.ankle_dorsiflex_R
    )
    joints[11] = joints[8] + r_foot_vec

    # --- Face landmarks (indices 55..59) ---
    head = joints[15]
    joints[55] = head + [0.0, 0.0, 0.10]    # nose (forward of head center)
    joints[56] = head + [-0.03, 0.03, 0.08] # R eye
    joints[57] = head + [+0.03, 0.03, 0.08] # L eye
    joints[58] = head + [-0.08, 0.0, 0.02]  # R ear
    joints[59] = head + [+0.08, 0.0, 0.02]  # L ear

    # --- Foot landmarks (indices 60..65): big toe, small toe, heel L/R ---
    # Left foot
    l_ank = joints[7]
    l_toe_dir = joints[10] - l_ank
    l_toe_dir_norm = l_toe_dir / np.linalg.norm(l_toe_dir)
    joints[60] = joints[10] + l_toe_dir_norm * 0.02      # L big toe (slightly past foot tip)
    joints[61] = joints[60] + np.array([-0.04, 0.0, 0.0]) # L small toe (lateral)
    joints[62] = l_ank + np.array([0.0, -0.06, -SEG_LEN["heel_offset"]])  # L heel

    # Right foot
    r_ank = joints[8]
    r_toe_dir = joints[11] - r_ank
    r_toe_dir_norm = r_toe_dir / np.linalg.norm(r_toe_dir)
    joints[63] = joints[11] + r_toe_dir_norm * 0.02       # R big toe
    joints[64] = joints[63] + np.array([+0.04, 0.0, 0.0]) # R small toe
    joints[65] = r_ank + np.array([0.0, -0.06, -SEG_LEN["heel_offset"]])

    # Shift everything up so feet are roughly on the ground (y=0 floor).
    min_y = min(joints[62][1], joints[65][1])
    joints[:, 1] -= min_y

    return joints
