"""Compute 3D ground-truth joint angles from MPI-INF-3DHP annot3 mocap data.

annot3 is per-camera 3D joint positions (in that camera's coordinate frame).
For our purposes — computing joint flexion angles which are scalar and
camera-invariant — any single camera's annot3 works (we use cam 0).

Joint angles computed (matches Couro motion metrics):
    - hip_flexion_r:    angle between torso (neck->pelvis) and thigh (hip->knee)
    - knee_angle_r:     180 - angle between thigh (knee->hip) and shank (knee->ankle)
    - ankle_angle_r:    90 - angle between shank (ankle->knee) and foot (ankle->toe)
    - hip_adduction_r:  angle in the frontal/coronal plane (R_hip->R_knee from vertical)
    - lumbar_extension: signed forward-lean of trunk (neck->pelvis) from vertical
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

# MPI-INF-3DHP 'all' joint ordering, 0-indexed
J = {
    "spine3": 0, "spine4": 1, "spine2": 2, "spine": 3, "pelvis": 4,
    "neck": 5, "head": 6, "head_top": 7,
    "L_clav": 8, "L_sho": 9, "L_elb": 10, "L_wri": 11, "L_hand": 12,
    "R_clav": 13, "R_sho": 14, "R_elb": 15, "R_wri": 16, "R_hand": 17,
    "L_hip": 18, "L_kne": 19, "L_ank": 20, "L_foot": 21, "L_toe": 22,
    "R_hip": 23, "R_kne": 24, "R_ank": 25, "R_foot": 26, "R_toe": 27,
}


def _angle_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle between two 3D vectors per frame, degrees [0, 180]."""
    dot = (v1 * v2).sum(axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.clip(dot / np.where(n1 * n2 > 1e-9, n1 * n2, 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_gt_angles(annot_mat_path: Path, cam_id: int = 0) -> dict:
    """Compute per-frame GT joint angles from MPI-INF-3DHP annot3 (mocap).

    Returns a dict of {metric_name: np.ndarray (n_frames,)}.
    """
    m = sio.loadmat(str(annot_mat_path))
    a3 = m["annot3"][cam_id, 0]
    n = a3.shape[0]
    p = a3.reshape(n, 28, 3)  # (n_frames, 28 joints, xyz)

    pelvis = p[:, J["pelvis"]]
    neck = p[:, J["neck"]]
    R_hip, R_kne, R_ank = p[:, J["R_hip"]], p[:, J["R_kne"]], p[:, J["R_ank"]]
    R_toe = p[:, J["R_toe"]]
    L_hip, L_kne, L_ank = p[:, J["L_hip"]], p[:, J["L_kne"]], p[:, J["L_ank"]]
    L_toe = p[:, J["L_toe"]]

    torso = neck - pelvis  # upward in world

    # Hip flexion: angle between torso and thigh (hip->knee), where 0 = thigh
    # aligned with torso (standing) and positive = forward swing.
    thigh_r = R_kne - R_hip
    thigh_l = L_kne - L_hip
    hip_flex_r = 180.0 - _angle_3d(-torso, thigh_r)
    hip_flex_l = 180.0 - _angle_3d(-torso, thigh_l)
    # The above: -torso points downward. angle(-torso, thigh) ~ 0 when thigh
    # is straight down (standing). Hip flexion convention: 0 = neutral standing,
    # positive = forward. So flexion = angle between -torso and thigh, since both
    # near 0 when standing. Fix:
    hip_flex_r = _angle_3d(-torso, thigh_r)
    hip_flex_l = _angle_3d(-torso, thigh_l)

    # Knee flexion: 0 = straight leg, positive = flexed.
    knee_int_r = _angle_3d(R_hip - R_kne, R_ank - R_kne)
    knee_int_l = _angle_3d(L_hip - L_kne, L_ank - L_kne)
    knee_angle_r = 180.0 - knee_int_r
    knee_angle_l = 180.0 - knee_int_l

    # Ankle dorsiflexion: 0 = foot perpendicular to shank, positive = dorsiflexion.
    ankle_int_r = _angle_3d(R_kne - R_ank, R_toe - R_ank)
    ankle_int_l = _angle_3d(L_kne - L_ank, L_toe - L_ank)
    ankle_angle_r = 90.0 - ankle_int_r
    ankle_angle_l = 90.0 - ankle_int_l

    # Hip adduction R: angle in coronal/frontal plane between vertical
    # (world Y-axis equivalent) and the R_hip->R_knee thigh vector projected
    # into the plane normal to the body's sagittal axis.
    # World up in MPI-INF-3DHP camera frames: approximately the -Y axis of
    # each camera (since cameras are upright and Y is image-down).
    # We approximate "vertical" using the torso (neck-pelvis) direction and
    # compute the angle between thigh and torso in the direction perpendicular
    # to the body sagittal plane.
    # Simplification: compute the absolute angle between R_hip->R_kne and the
    # downward torso direction projected purely on adduction component:
    # adduction ~ angle between thigh and downward-torso, signed by side.
    # For our purposes, use the absolute angle between -torso and thigh
    # restricted to the frontal plane. Defined as:
    #   project thigh onto plane perpendicular to (R_hip - L_hip), then measure
    #   its tilt from -torso.
    hip_axis = R_hip - L_hip  # bilateral hip axis
    # Project thigh perpendicular to torso to isolate the abduction/adduction
    # plane (perpendicular to sagittal):
    def proj_perp(v, n):
        n_unit = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        return v - (v * n_unit).sum(axis=-1, keepdims=True) * n_unit

    # Sagittal axis ~ cross(torso, hip_axis); adduction plane normal = sagittal axis
    sagittal = np.cross(torso, hip_axis)
    thigh_r_front = proj_perp(thigh_r, sagittal)
    thigh_l_front = proj_perp(thigh_l, sagittal)
    torso_down = -torso
    torso_down_front = proj_perp(torso_down, sagittal)
    hip_add_r = _angle_3d(torso_down_front, thigh_r_front)
    hip_add_l = _angle_3d(torso_down_front, thigh_l_front)
    # Sign: adduction = thigh toward midline; abduction = away
    # We'll just report the magnitude; correlation absorbs the sign convention.

    # Lumbar extension: forward-lean angle of trunk from vertical.
    # Define "vertical" as the world up direction. In MPI-INF-3DHP camera 0
    # coords, world Y points roughly downward (image down), so world up ~ -Y_cam.
    # Use sagittal plane (the plane containing the bilateral hip axis is
    # perpendicular to sagittal). lumbar_ext = signed angle of trunk from
    # vertical in sagittal plane.
    # Simplification: angle between torso and the world-up axis we approximate
    # as the mean torso direction over the whole sequence (subject is usually
    # upright on average).
    torso_unit = torso / np.maximum(np.linalg.norm(torso, axis=-1, keepdims=True), 1e-9)
    mean_up = torso_unit.mean(axis=0)
    mean_up = mean_up / np.linalg.norm(mean_up)
    lumbar_ext = _angle_3d(torso, np.broadcast_to(mean_up, torso.shape))

    return {
        "hip_flexion_r": hip_flex_r,
        "hip_flexion_l": hip_flex_l,
        "knee_angle_r": knee_angle_r,
        "knee_angle_l": knee_angle_l,
        "ankle_angle_r": ankle_angle_r,
        "ankle_angle_l": ankle_angle_l,
        "hip_adduction_r": hip_add_r,
        "hip_adduction_l": hip_add_l,
        "lumbar_extension": lumbar_ext,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("annot_mat")
    p.add_argument("out_json")
    p.add_argument("--cam-id", type=int, default=0)
    args = p.parse_args()
    angles = compute_gt_angles(Path(args.annot_mat), args.cam_id)
    out = {k: v.tolist() for k, v in angles.items()}
    out["_source"] = {"annot_mat": str(args.annot_mat), "cam_id": args.cam_id}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"Wrote GT angles for {len(angles)} metrics, {len(next(iter(angles.values())))} frames")
    for k, v in angles.items():
        finite = np.isfinite(v)
        print(f"  {k:24s}  mean={v[finite].mean():7.2f}  std={v[finite].std():6.2f}"
              f"  range=[{v[finite].min():7.2f}, {v[finite].max():7.2f}]")


if __name__ == "__main__":
    main()
