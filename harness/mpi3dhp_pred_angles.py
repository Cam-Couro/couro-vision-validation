"""Compute 2D joint-angle predictions from DWPose Halpe-26 keypoints,
matching the Couro production motion-data conventions.

Output schema mirrors GT angles JSON from mpi3dhp_gt_angles.py so that
downstream Pearson r computation can pair them directly by frame_idx.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Halpe-26 indices
KP = {
    "nose": 0, "L_eye": 1, "R_eye": 2, "L_ear": 3, "R_ear": 4,
    "L_sho": 5, "R_sho": 6, "L_elb": 7, "R_elb": 8, "L_wri": 9, "R_wri": 10,
    "L_hip": 11, "R_hip": 12, "L_kne": 13, "R_kne": 14, "L_ank": 15, "R_ank": 16,
    "head_top": 17, "neck": 18, "hip_center": 19,
    "L_big_toe": 20, "R_big_toe": 21, "L_small_toe": 22, "R_small_toe": 23,
    "L_heel": 24, "R_heel": 25,
}


def _angle_2d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dot = (v1 * v2).sum(axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.clip(dot / np.where(n1 * n2 > 1e-9, n1 * n2, 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_pred_angles(dwpose_json: Path, conf_threshold: float = 0.3) -> dict:
    with open(dwpose_json) as f:
        d = json.load(f)
    frames = d["keypoints_sequence"]
    n = len(frames)
    xy = np.full((n, 26, 2), np.nan)
    conf = np.zeros((n, 26))
    frame_idx = np.zeros(n, dtype=int)
    for i, fr in enumerate(frames):
        kps = fr["keypoints"]
        scores = fr.get("keypoint_scores", [1.0] * 26)
        for j in range(min(26, len(kps))):
            kp = kps[j]
            xy[i, j, 0], xy[i, j, 1] = float(kp[0]), float(kp[1])
            conf[i, j] = float(scores[j])
        frame_idx[i] = int(fr["frame_idx"])

    # Mask low-confidence keypoints
    xy[conf < conf_threshold] = np.nan

    def p(name: str) -> np.ndarray:
        return xy[:, KP[name], :]

    # Key joints
    R_hip, R_kne, R_ank = p("R_hip"), p("R_kne"), p("R_ank")
    L_hip, L_kne, L_ank = p("L_hip"), p("L_kne"), p("L_ank")
    R_toe = p("R_big_toe"); L_toe = p("L_big_toe")
    neck = p("neck"); hip_c = p("hip_center")

    torso = neck - hip_c

    # Hip flexion (2D interior between torso and thigh)
    hip_flex_r = _angle_2d(torso, R_kne - R_hip)
    hip_flex_l = _angle_2d(torso, L_kne - L_hip)

    # Knee flexion: 180 - interior at knee
    knee_int_r = _angle_2d(R_hip - R_kne, R_ank - R_kne)
    knee_int_l = _angle_2d(L_hip - L_kne, L_ank - L_kne)
    knee_angle_r = 180.0 - knee_int_r
    knee_angle_l = 180.0 - knee_int_l

    # Ankle dorsiflexion: 90 - interior at ankle (shank vs foot)
    ankle_int_r = _angle_2d(R_kne - R_ank, R_toe - R_ank)
    ankle_int_l = _angle_2d(L_kne - L_ank, L_toe - L_ank)
    ankle_angle_r = 90.0 - ankle_int_r
    ankle_angle_l = 90.0 - ankle_int_l

    # Hip adduction R/L: angle in coronal plane (frontal) — approximation using
    # horizontal x-offset of knee relative to hip vs vertical drop.
    # Use unsigned absolute since 2D coronal plane signal is noisy from any
    # non-front camera. Correlation is sign-insensitive (Pearson allows negative).
    hip_add_r = np.degrees(np.arctan2(
        np.abs(R_kne[:, 0] - R_hip[:, 0]),
        np.abs(R_kne[:, 1] - R_hip[:, 1]) + 1e-9,
    ))
    hip_add_l = np.degrees(np.arctan2(
        np.abs(L_kne[:, 0] - L_hip[:, 0]),
        np.abs(L_kne[:, 1] - L_hip[:, 1]) + 1e-9,
    ))

    # Lumbar extension proxy: angle of neck->hip_center vector from image-vertical
    trunk = hip_c - neck
    lumbar_ext = np.degrees(np.arctan2(np.abs(trunk[:, 0]), np.abs(trunk[:, 1]) + 1e-9))

    return {
        "frame_idx": frame_idx,
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
    p.add_argument("dwpose_json")
    p.add_argument("out_json")
    p.add_argument("--conf", type=float, default=0.3)
    args = p.parse_args()
    angles = compute_pred_angles(Path(args.dwpose_json), conf_threshold=args.conf)
    out = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in angles.items()}
    out["_source"] = {"dwpose_json": str(args.dwpose_json)}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"Wrote pred angles to {args.out_json}")
    for k, v in angles.items():
        if k == "frame_idx":
            continue
        arr = np.asarray(v)
        finite = np.isfinite(arr)
        if finite.sum() == 0:
            print(f"  {k:24s}  NO FINITE VALUES")
            continue
        print(f"  {k:24s}  n_valid={finite.sum():5d}  mean={arr[finite].mean():7.2f}"
              f"  std={arr[finite].std():6.2f}")


if __name__ == "__main__":
    main()
