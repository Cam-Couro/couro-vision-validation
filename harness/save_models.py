"""Train + save per-camera hip-flex regression models for drop-in to metrics_calculator.

Output: results/hip_flex_rom_regression_models.json with one model per camera
view bucket, structured exactly like knee_rotation.py's existing regression
coefficients.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.train_regression_poc import (
        FEATURE_NAMES, build_dataset, fit_ridge, loso_cv,
    )
else:
    from .train_regression_poc import FEATURE_NAMES, build_dataset, fit_ridge, loso_cv


FEATURE_NAMES = (
    "hip_flex_rom_r_2d3d",
    "knee_flex_rom_r_2d3d",
    "trunk_lean_rom",
    "pelvis_tilt_rom",
    "knee_flex_rom_diff_lr",
    "hip_y_range_px",
    "hip_flex_vel_peak",
)

VIEW_BUCKET_NAMES = {
    "Cam0": "side_left",
    "Cam1": "front_oblique_left",
    "Cam2": "front_center",
    "Cam3": "front_oblique_right",
    "Cam4": "side_right",
}


def main():
    out_dir = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, dict] = {}
    for cam in ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"]:
        X, y, subjects, tasks = build_dataset(cam=cam)
        if len(y) < 20:
            continue
        # LOSO-CV stats (held-out generalization estimate)
        cv = loso_cv(X, y, subjects, alpha=10.0)
        # Final model trained on all data (for production use)
        full = fit_ridge(X, y, alpha=10.0)

        models[VIEW_BUCKET_NAMES[cam]] = {
            "camera_id": cam,
            "view_bucket": VIEW_BUCKET_NAMES[cam],
            "target_metric": "hip_flexion_rom_r",
            "n_train_trials": int(len(y)),
            "n_subjects": len(set(subjects)),
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": full.feature_mean.tolist(),
            "feature_std": full.feature_std.tolist(),
            "weights_zscored": full.weights.tolist(),
            "intercept": full.bias,
            "loso_cv_stats": {
                "rmse_deg": cv["rmse"],
                "mae_deg": cv["mae"],
                "bias_deg": cv["bias"],
                "sd_of_difference_deg": cv["sd_diff"],
                "loa_95_lower_deg": cv["loa"][0],
                "loa_95_upper_deg": cv["loa"][1],
                "pearson_r": cv["pearson_r"],
            },
            "usage": (
                "Predict hip flexion ROM from a single drop-jump trial:\n"
                "  z_features = (raw_features - feature_mean) / feature_std\n"
                "  hip_flex_rom_pred = z_features @ weights_zscored + intercept"
            ),
            "trained_on": "OpenCap LabValidation_withVideos (drop jumps), "
                          "9 subjects per LOSO fold, Vicon mocap GT",
            "training_dataset": {
                "name": "OpenCap LabValidation",
                "citation": "Uhlrich et al. 2023, PLOS Comp Bio 19(10): e1011462",
                "license": "Apache 2.0",
            },
        }

    out_path = out_dir / "hip_flex_rom_regression_models.json"
    with open(out_path, "w") as f:
        json.dump({
            "version": "1.0",
            "produced_by": "harness.save_models",
            "produced_date": "2026-05-27",
            "description": (
                "Hip-flexion ROM regression models, one per camera view bucket. "
                "Trained on OpenCap drop-jump trials following the existing "
                "knee_rotation.py pattern. Drop into metrics_calculator as a new "
                "MetricCalculator subclass."
            ),
            "models": models,
        }, f, indent=2)
    print(f"wrote {out_path}")
    print()
    print("Per-view LOSO-CV performance:")
    print(f"{'view':<25} {'RMSE':>7} {'MAE':>7} {'bias':>7} {'r':>5}")
    for view, m in models.items():
        s = m["loso_cv_stats"]
        print(f"  {view:<23} {s['rmse_deg']:>6.2f}° {s['mae_deg']:>6.2f}° {s['bias_deg']:>+6.2f}° {s['pearson_r']:>4.2f}")


if __name__ == "__main__":
    main()
