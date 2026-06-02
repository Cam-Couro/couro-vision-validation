"""Compute Layer-1 (keypoint position) Pearson r between Couro DWPose-L
predicted 2D keypoints and MPI-INF-3DHP ground-truth 2D joint annotations.

Approach
--------
MPI-INF-3DHP annot.mat already provides per-camera 2D projections of the
mocap 3D joints into image space (annot2). Each entry is a (n_frames, 56)
array — 28 joints x (x, y). We don't need to re-project from 3D; the dataset
authors did the projection using the studio's calibrated camera intrinsics
+ extrinsics for us.

We map a subset of the MPI-INF-3DHP 28-joint layout to Halpe-26 indices
(only the directly-comparable body joints — skipping synthesized joints like
head_top, neck, hip_center where the MPI joint name doesn't quite match,
and skipping joints MPI doesn't provide cleanly e.g. ears/eyes from a
markerless mocap subject).

For each subject/camera/keypoint we compute:
    - r_x : Pearson r between predicted x and GT x across valid frames
    - r_y : Pearson r between predicted y and GT y across valid frames
    - r_combined : mean(r_x, r_y)

Filtering:
    - confidence > 0.3 (matches the existing rear validation report)
    - 0 <= x <= width and 0 <= y <= height (in-bounds)
    - both pred and GT finite

Aggregation:
    - per-camera mean / median r across mapped keypoints
    - per-subject mean r (for cross-subject comparison)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

# MPI-INF-3DHP "all" joint ordering, 0-indexed (28 joints).
# Source: harness/mpi3dhp_gt_angles.py (J dict) — derived from official metadata.
J = {
    "spine3": 0, "spine4": 1, "spine2": 2, "spine": 3, "pelvis": 4,
    "neck": 5, "head": 6, "head_top": 7,
    "L_clav": 8, "L_sho": 9, "L_elb": 10, "L_wri": 11, "L_hand": 12,
    "R_clav": 13, "R_sho": 14, "R_elb": 15, "R_wri": 16, "R_hand": 17,
    "L_hip": 18, "L_kne": 19, "L_ank": 20, "L_foot": 21, "L_toe": 22,
    "R_hip": 23, "R_kne": 24, "R_ank": 25, "R_foot": 26, "R_toe": 27,
}

# Halpe-26 indices (from harness/couro_keypoints.py)
KP = {
    "nose": 0, "L_eye": 1, "R_eye": 2, "L_ear": 3, "R_ear": 4,
    "L_sho": 5, "R_sho": 6, "L_elb": 7, "R_elb": 8, "L_wri": 9, "R_wri": 10,
    "L_hip": 11, "R_hip": 12, "L_kne": 13, "R_kne": 14, "L_ank": 15, "R_ank": 16,
    "head_top": 17, "neck": 18, "hip_center": 19,
    "L_big_toe": 20, "R_big_toe": 21, "L_small_toe": 22, "R_small_toe": 23,
    "L_heel": 24, "R_heel": 25,
}

# Direct mapping: Halpe-26 keypoint name -> MPI joint name.
# Only the unambiguously-comparable body joints. We deliberately skip:
#   - eyes/ears: MPI has only "head" + "head_top" landmarks (no face detail)
#   - nose: closest is "head" but it's centroid of the head, not nose tip
#   - hip_center: pelvis would map but it's at sacrum not bilateral mid-hip
#   - toes/heels: MPI has L_toe and R_toe but not separate big-toe/small-toe/heel
# These maps cover the joints Couro's downstream metrics actually use.
MAP_HALPE_TO_MPI = {
    "L_sho": "L_sho", "R_sho": "R_sho",
    "L_elb": "L_elb", "R_elb": "R_elb",
    "L_wri": "L_wri", "R_wri": "R_wri",
    "L_hip": "L_hip", "R_hip": "R_hip",
    "L_kne": "L_kne", "R_kne": "R_kne",
    "L_ank": "L_ank", "R_ank": "R_ank",
    "head_top": "head_top",
    "neck": "neck",
    "hip_center": "pelvis",          # near-equivalent (MPI pelvis at sacrum)
    "L_big_toe": "L_toe", "R_big_toe": "R_toe",  # near-equivalent (MPI single toe)
    "nose": "head",                  # head centroid; weaker mapping
}


def load_dwpose(path: Path):
    """Load DWPose JSON, return (frame_idx array, xy [n,26,2], conf [n,26], w, h)."""
    with open(path) as f:
        d = json.load(f)
    seq = d["keypoints_sequence"]
    n = len(seq)
    xy = np.full((n, 26, 2), np.nan)
    conf = np.zeros((n, 26))
    fidx = np.zeros(n, dtype=int)
    for i, fr in enumerate(seq):
        fidx[i] = int(fr["frame_idx"])
        kps = fr.get("keypoints") or []
        scs = fr.get("keypoint_scores") or []
        for j in range(min(26, len(kps))):
            k = kps[j]
            xy[i, j, 0] = float(k[0])
            xy[i, j, 1] = float(k[1])
        for j in range(min(26, len(scs))):
            conf[i, j] = float(scs[j])
    meta = d.get("video_metadata") or {}
    w = int(meta.get("width", 2048))
    h = int(meta.get("height", 2048))
    return fidx, xy, conf, w, h


def load_gt_annot2(annot_path: Path, cam_id: int):
    """Return (n_frames, 28, 2) GT 2D keypoints for a given camera."""
    m = sio.loadmat(str(annot_path))
    a2 = m["annot2"][cam_id, 0]  # (n_frames, 56)
    n = a2.shape[0]
    return a2.reshape(n, 28, 2)


def pearson(x: np.ndarray, y: np.ndarray):
    """Pearson r with NaN safety. Returns nan if <2 valid pairs or zero variance."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), 0
    xv = x[mask]
    yv = y[mask]
    if xv.std() < 1e-9 or yv.std() < 1e-9:
        return float("nan"), int(mask.sum())
    r = np.corrcoef(xv, yv)[0, 1]
    return float(r), int(mask.sum())


def compute_r_one(subject: int, cam_id: int, base_dir: Path):
    """Compute per-keypoint Pearson r for one (subject, camera) pair."""
    dwpose_path = base_dir / "mpi_inf_3dhp_dwpose" / f"S{subject}_Seq2_cam{cam_id}.json"
    annot_path = base_dir / "mpi_inf_3dhp" / f"S{subject}" / "Seq2" / "annot.mat"
    if not dwpose_path.exists():
        return None
    if not annot_path.exists():
        return None

    fidx, pred_xy, conf, w, h = load_dwpose(dwpose_path)
    gt_xy_all = load_gt_annot2(annot_path, cam_id)
    # DWPose ran on stride 2 from frame 0. So frame_idx i in pred corresponds
    # to frame index fidx[i] in the full annot2 array.
    # Bounds-check (some seqs trim a frame at the end).
    valid_idx_mask = fidx < gt_xy_all.shape[0]
    if not valid_idx_mask.all():
        fidx = fidx[valid_idx_mask]
        pred_xy = pred_xy[valid_idx_mask]
        conf = conf[valid_idx_mask]
    gt_xy = gt_xy_all[fidx]  # (n_pred_frames, 28, 2)

    per_keypoint = {}
    for halpe_name, mpi_name in MAP_HALPE_TO_MPI.items():
        kp_idx = KP[halpe_name]
        j_idx = J[mpi_name]
        px = pred_xy[:, kp_idx, 0].copy()
        py = pred_xy[:, kp_idx, 1].copy()
        c = conf[:, kp_idx]
        gx = gt_xy[:, j_idx, 0].copy()
        gy = gt_xy[:, j_idx, 1].copy()

        # Filtering masks
        in_conf = c > 0.3
        in_bounds_pred = (px >= 0) & (px <= w) & (py >= 0) & (py <= h)
        in_bounds_gt = (gx >= 0) & (gx <= w) & (gy >= 0) & (gy <= h)
        finite_pred = np.isfinite(px) & np.isfinite(py)
        finite_gt = np.isfinite(gx) & np.isfinite(gy)
        mask = in_conf & in_bounds_pred & in_bounds_gt & finite_pred & finite_gt

        px_f = np.where(mask, px, np.nan)
        py_f = np.where(mask, py, np.nan)
        gx_f = np.where(mask, gx, np.nan)
        gy_f = np.where(mask, gy, np.nan)

        r_x, n_x = pearson(px_f, gx_f)
        r_y, n_y = pearson(py_f, gy_f)

        # Also compute the mean pixel error for sanity check vs the 10.3 px MPJPE.
        dx = px_f - gx_f
        dy = py_f - gy_f
        err = np.sqrt(dx * dx + dy * dy)
        err_finite = err[np.isfinite(err)]
        mean_err = float(np.mean(err_finite)) if err_finite.size else float("nan")

        per_keypoint[halpe_name] = {
            "mpi_joint": mpi_name,
            "r_x": r_x,
            "r_y": r_y,
            "r_combined": float(np.nanmean([r_x, r_y])) if np.isfinite(r_x) or np.isfinite(r_y) else float("nan"),
            "n_valid": int(min(n_x, n_y)),
            "mean_pixel_err": mean_err,
        }
    return {
        "subject": subject,
        "cam_id": cam_id,
        "n_frames_pred": int(pred_xy.shape[0]),
        "width": w,
        "height": h,
        "per_keypoint": per_keypoint,
    }


def aggregate_camera(results_for_cam: list[dict]):
    """Across subjects (for a fixed camera), compute mean r per keypoint and headline mean."""
    # Pool all r_combined values per keypoint across subjects (one value per subject).
    all_kp_names = list(MAP_HALPE_TO_MPI.keys())
    per_kp_means = {}
    for kp in all_kp_names:
        rs = []
        for res in results_for_cam:
            v = res["per_keypoint"].get(kp, {}).get("r_combined")
            if v is not None and np.isfinite(v):
                rs.append(v)
        per_kp_means[kp] = {
            "mean_r_combined": float(np.mean(rs)) if rs else float("nan"),
            "median_r_combined": float(np.median(rs)) if rs else float("nan"),
            "n_subjects": len(rs),
        }
    # Headline: mean across all (subject, keypoint) cells.
    cells = []
    for res in results_for_cam:
        for kp, kv in res["per_keypoint"].items():
            if np.isfinite(kv["r_combined"]):
                cells.append(kv["r_combined"])
    return {
        "headline_mean_r": float(np.mean(cells)) if cells else float("nan"),
        "headline_median_r": float(np.median(cells)) if cells else float("nan"),
        "n_cells": len(cells),
        "per_keypoint": per_kp_means,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data")
    p.add_argument("--out-dir", default="/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/mpi_inf_3dhp_keypoint_r")
    args = p.parse_args()

    base = Path(args.base_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    # S1 with cam 0 (front), 3 (side), 7 (rear) — the headline triple.
    for cam in (0, 3, 7):
        runs.append((1, cam))
    # S2-S8 rear cam 7 only (the cross-subject set).
    for s in range(2, 9):
        runs.append((s, 7))

    all_results = []
    for s, cam in runs:
        print(f"Computing S{s} cam{cam}...")
        r = compute_r_one(s, cam, base)
        if r is None:
            print(f"  SKIP: missing data")
            continue
        all_results.append(r)
        # Print quick summary
        kp_combined = [v["r_combined"] for v in r["per_keypoint"].values() if np.isfinite(v["r_combined"])]
        if kp_combined:
            print(f"  S{s} cam{cam}: mean r = {np.mean(kp_combined):.4f}, median r = {np.median(kp_combined):.4f}, n_kp = {len(kp_combined)}")

    # Per-camera aggregates
    by_cam = {}
    for cam in (0, 3, 7):
        results = [r for r in all_results if r["cam_id"] == cam]
        if results:
            by_cam[f"cam{cam}"] = aggregate_camera(results)
            print(f"\nCam{cam} ({len(results)} subjects): headline mean r = {by_cam[f'cam{cam}']['headline_mean_r']:.4f}")

    # Cross-subject (rear) aggregate: subset of by_cam already done.
    summary = {
        "runs": all_results,
        "by_camera": by_cam,
        "mapping": MAP_HALPE_TO_MPI,
        "methodology": {
            "filter": "conf>0.3 + in-image bounds + finite",
            "r": "Pearson r per (subject, cam, keypoint, x|y); combined = mean(r_x, r_y)",
            "headline": "Pooled mean r_combined across all (subject, keypoint) cells per camera",
        },
    }
    out_path = out / "keypoint_r_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
