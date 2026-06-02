"""Compute Pearson r between predicted joint angles (from DWPose) and GT
joint angles (from MPI-INF-3DHP mocap) for a single camera.

Inputs:
    - predicted angles JSON (from mpi3dhp_pred_angles.py)
    - GT angles JSON (from mpi3dhp_gt_angles.py)

Outputs Pearson r per metric, NaN-safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "hip_flexion_r", "hip_flexion_l",
    "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l",
    "hip_adduction_r", "hip_adduction_l",
    "lumbar_extension",
)


def pearson_r(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Pearson r over frames where both a and b are finite."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan"), int(mask.sum())
    a2, b2 = a[mask], b[mask]
    a2 = a2 - a2.mean()
    b2 = b2 - b2.mean()
    denom = np.linalg.norm(a2) * np.linalg.norm(b2)
    if denom < 1e-9:
        return float("nan"), int(mask.sum())
    return float((a2 * b2).sum() / denom), int(mask.sum())


def correlate(pred_path: Path, gt_path: Path) -> dict:
    with open(pred_path) as f:
        pred = json.load(f)
    with open(gt_path) as f:
        gt = json.load(f)

    frame_idx = np.asarray(pred["frame_idx"])
    results = {}
    for metric in METRICS:
        p = np.asarray(pred[metric])
        g_all = np.asarray(gt[metric])
        # Align GT to predicted frame indices
        g = g_all[frame_idx]
        r_signed, n = pearson_r(p, g)
        # For unsigned 2D adduction/lumbar, also report abs(r) and r vs |GT|
        r_abs_gt, _ = pearson_r(p, np.abs(g))
        results[metric] = {
            "pearson_r": r_signed,
            "pearson_r_abs_gt": r_abs_gt,
            "n_frames": n,
        }
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pred_json")
    p.add_argument("gt_json")
    p.add_argument("--label", default="cam")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()
    results = correlate(Path(args.pred_json), Path(args.gt_json))
    print(f"\n=== {args.label} ===")
    print(f"{'metric':<22} {'r':>8} {'r_abs_gt':>10} {'n':>8}")
    for m, r in results.items():
        print(f"{m:<22} {r['pearson_r']:>8.3f} {r['pearson_r_abs_gt']:>10.3f}"
              f" {r['n_frames']:>8d}")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"label": args.label, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
