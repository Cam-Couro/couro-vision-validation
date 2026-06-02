"""v13: same combined OpenCap+ASPset regression as v12 but with ASPset keypoints
inferred by DWPose-L (133-kp COCO-WholeBody remapped to Halpe-26) instead of
RTMPose-X Halpe-26.

Compares to v12 RTMPose so we can attribute any delta directly to the backbone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse v12 logic by monkey-patching its ASPSET_KEYPOINTS path before invoking
import harness.train_v12_combined as v12
from harness.train_regression_poc import fit_ridge, loso_cv
from harness.train_multi_metric import VIEW_BUCKET_NAMES


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
ASPSET_DWPOSE_KP = DATA_ROOT / "aspset510_dwpose_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


def main():
    if not ASPSET_DWPOSE_KP.exists() or not list(ASPSET_DWPOSE_KP.glob("*.json")):
        print(f"ERROR: DWPose keypoint JSONs not found in {ASPSET_DWPOSE_KP}")
        return

    v12.ASPSET_KEYPOINTS = ASPSET_DWPOSE_KP
    print(f"Using ASPset keypoints from: {v12.ASPSET_KEYPOINTS}")

    # Load v12 RTMPose result for direct baseline comparison
    rtm_baseline = {}
    v12_path = RESULTS_DIR / "v12_combined_models.json"
    if v12_path.exists():
        with open(v12_path) as f:
            d = json.load(f)
        for tgt, td in d.get("models", {}).items():
            rtm_baseline[tgt] = {
                v: m["loso_cv_stats"]["pearson_r"] for v, m in td.items()
            }

    views = ["side_left", "front_oblique_left", "front_center",
             "front_oblique_right", "side_right"]
    # ankle_angle_r excluded: ASPset has no foot 3D markers for GT, so DWPose's
    # better foot keypoints can't be trained against ASPset ankle ROM.
    targets = ["hip_flexion_r", "hip_adduction_r", "knee_angle_r", "lumbar_extension"]

    out = {
        "version": "1.0",
        "produced_by": "harness.train_v13_dwpose_combined",
        "produced_date": "2026-05-28",
        "approach": "v13 — same regression as v12 but ASPset keypoints from DWPose-L (133 COCO-WholeBody → Halpe-26)",
        "models": {},
    }

    print(f"{'TARGET':<22}{'VIEW':<22}{'n_oc':>6}{'n_asp':>7}{'n_tot':>7}"
          f"{'v12_r':>8}{'v13_r':>8}{'Δr':>7}")
    print("-" * 88)

    for target in targets:
        out["models"][target] = {}
        for view in views:
            result = v12.build_combined_dataset(target, view)
            if result is None:
                continue
            X, y, subjects, sources = result
            if len(y) < 20:
                continue
            n_oc = sum(1 for s in sources if s == "opencap")
            n_asp = sum(1 for s in sources if s == "aspset")
            cv = loso_cv(X, y, subjects, alpha=10.0)
            fit = fit_ridge(X, y, alpha=10.0)
            baseline = rtm_baseline.get(target, {}).get(view, float("nan"))
            d_r = cv["pearson_r"] - baseline if np.isfinite(baseline) else float("nan")
            print(f"  {target:<20}{view:<22}{n_oc:>6d}{n_asp:>7d}{len(y):>7d}"
                  f"{baseline:>+7.2f}{cv['pearson_r']:>+7.2f}{d_r:>+6.2f}")
            out["models"][target][view] = {
                "n_opencap": n_oc,
                "n_aspset": n_asp,
                "n_total": int(len(y)),
                "n_subjects": len(set(subjects)),
                "feature_mean": fit.feature_mean.tolist(),
                "feature_std": fit.feature_std.tolist(),
                "weights_zscored": fit.weights.tolist(),
                "intercept": fit.bias,
                "loso_cv_stats": {
                    "rmse_deg": cv["rmse"],
                    "pearson_r": cv["pearson_r"],
                    "bias_deg": cv["bias"],
                    "sd_of_difference_deg": cv["sd_diff"],
                },
                "v12_rtmpose_baseline_r": baseline,
                "delta_r_vs_v12": d_r,
            }

    out_path = RESULTS_DIR / "v13_dwpose_combined_models.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
