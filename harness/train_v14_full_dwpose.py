"""v14: same combined OpenCap+ASPset regression as v12, but BOTH datasets now use
DWPose-L keypoints. v13 was hybrid (OpenCap RTMPose + ASPset DWPose); v14 swaps
OpenCap to DWPose too.

This is the only combination that lets DWPose's better foot keypoints influence
ankle ROM — because OpenCap is the only source of ankle GT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.train_v9_phased as v9
import harness.train_v12_combined as v12
from harness.train_regression_poc import fit_ridge, loso_cv
from harness.train_multi_metric import VIEW_BUCKET_NAMES
from harness.train_v9_phased import build_dataset_v9


def build_opencap_only_dataset(target_gt: str, view: str):
    """For metrics ASPset can't provide GT for (ankle), train on OpenCap only —
    but with DWPose features instead of RTMPose."""
    cam_id = {v: k for k, v in VIEW_BUCKET_NAMES.items()}.get(view)
    if cam_id is None:
        return None
    X_oc, y_oc, subj_oc, _, _ = build_dataset_v9(target_gt, cam_id)
    if len(y_oc) == 0:
        return None
    subjects = [f"opencap_{s}" for s in subj_oc]
    sources = ["opencap"] * len(y_oc)
    return X_oc.astype(np.float64), np.asarray(y_oc, dtype=np.float64), subjects, sources


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
OPENCAP_DWPOSE_KP = DATA_ROOT / "opencap_dwpose_keypoints"
ASPSET_DWPOSE_KP = DATA_ROOT / "aspset510_dwpose_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


def main():
    if not OPENCAP_DWPOSE_KP.exists() or not list(OPENCAP_DWPOSE_KP.glob("*.json")):
        print(f"ERROR: OpenCap DWPose JSONs not found in {OPENCAP_DWPOSE_KP}")
        return
    if not ASPSET_DWPOSE_KP.exists() or not list(ASPSET_DWPOSE_KP.glob("*.json")):
        print(f"ERROR: ASPset DWPose JSONs not found in {ASPSET_DWPOSE_KP}")
        return

    # Repoint both data sources
    v9.KP_DIR = OPENCAP_DWPOSE_KP
    v12.ASPSET_KEYPOINTS = ASPSET_DWPOSE_KP
    print(f"OpenCap keypoints: {v9.KP_DIR}")
    print(f"ASPset  keypoints: {v12.ASPSET_KEYPOINTS}")

    # Load v13 RTM-OC + DW-ASP for comparison
    v13_baseline = {}
    v13_path = RESULTS_DIR / "v13_dwpose_combined_models.json"
    if v13_path.exists():
        with open(v13_path) as f:
            d = json.load(f)
        for tgt, td in d.get("models", {}).items():
            v13_baseline[tgt] = {
                v: m["loso_cv_stats"]["pearson_r"] for v, m in td.items()
            }
    v12_baseline = {}
    v12_path = RESULTS_DIR / "v12_combined_models.json"
    if v12_path.exists():
        with open(v12_path) as f:
            d = json.load(f)
        for tgt, td in d.get("models", {}).items():
            v12_baseline[tgt] = {
                v: m["loso_cv_stats"]["pearson_r"] for v, m in td.items()
            }

    views = ["side_left", "front_oblique_left", "front_center",
             "front_oblique_right", "side_right"]
    # v14 includes ankle_angle_r — the whole point of swapping OpenCap keypoints
    # is to give the ankle model better foot features.
    targets = ["hip_flexion_r", "hip_adduction_r", "knee_angle_r", "lumbar_extension",
               "ankle_angle_r"]

    out = {
        "version": "1.0",
        "produced_by": "harness.train_v14_full_dwpose",
        "produced_date": "2026-05-28",
        "approach": "v14 — full DWPose pipeline: both OpenCap and ASPset keypoints from DWPose-L (133 COCO-WholeBody → Halpe-26)",
        "models": {},
    }

    print(f"{'TARGET':<22}{'VIEW':<22}{'n_oc':>6}{'n_asp':>7}{'n_tot':>7}"
          f"{'v12_r':>8}{'v13_r':>8}{'v14_r':>8}{'Δ(14-12)':>10}")
    print("-" * 100)

    for target in targets:
        out["models"][target] = {}
        for view in views:
            if target == "ankle_angle_r":
                # ASPset has no foot 3D markers → OpenCap-only, but with DWPose features
                result = build_opencap_only_dataset(target, view)
            else:
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
            r12 = v12_baseline.get(target, {}).get(view, float("nan"))
            r13 = v13_baseline.get(target, {}).get(view, float("nan"))
            r14 = cv["pearson_r"]
            d_14_12 = r14 - r12 if np.isfinite(r12) else float("nan")
            print(f"  {target:<20}{view:<22}{n_oc:>6d}{n_asp:>7d}{len(y):>7d}"
                  f"{r12:>+7.2f}{r13:>+7.2f}{r14:>+7.2f}{d_14_12:>+9.2f}")
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
                "v12_rtmpose_r": r12,
                "v13_hybrid_r": r13,
                "delta_r_vs_v12": d_14_12,
            }

    out_path = RESULTS_DIR / "v14_full_dwpose_models.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
