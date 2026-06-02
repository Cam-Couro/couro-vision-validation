"""Event-anchored regression trainer for ALL 5 Couro DJ biomech metrics.

v5 showed event-anchoring (impact detection + loading-window features)
flipped ankle r from negative to positive. This trainer applies the same
recipe to every metric — most relevantly the v4 "scale-only" cases where
RMSE looked good but Pearson r was near zero (front_center hip flexion,
front_center knee, lumbar across the front views).

Features per target:
  - full-trial ROM (baseline — same as v4 naive)
  - target-specific event-anchored: rom_loading, at_contact, min_loading,
    max_loading, end_loading
  - one coupled metric's at_contact + rom_loading
  - loading duration (drop strategy proxy)

Same ridge alpha=10 + LOSO CV protocol.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from harness.parsers import parse_mot
    from harness.dj_events import detect_dj_events, event_anchored_for_metric
    from harness.train_regression_poc import fit_ridge, loso_cv
    from harness.train_multi_metric import _rom, _rom_unwrapped, VIEW_BUCKET_NAMES
else:
    from .couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .dj_events import detect_dj_events, event_anchored_for_metric
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import _rom, _rom_unwrapped, VIEW_BUCKET_NAMES


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
LAB = DATA_ROOT / "LabValidation_withVideos"
KP_DIR = DATA_ROOT / "couro_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


# Maps OpenSim GT column → (Couro motion-data column, primary coupling, secondary coupling)
TARGETS = {
    "hip_flexion_r":    ("hip_flexion_r",    "knee_angle_r",    "ankle_angle_r"),
    "hip_adduction_r":  ("hip_adduction_r",  "hip_adduction_l", "knee_angle_r"),
    "knee_angle_r":     ("knee_angle_r",     "hip_flexion_r",   "ankle_angle_r"),
    "ankle_angle_r":    ("ankle_angle_r",    "knee_angle_r",    "hip_flexion_r"),
    "lumbar_extension": ("lumbar_extension", "hip_flexion_r",   "pelvis_tilt"),
}


def _select_rom(col_name: str, series: np.ndarray) -> float:
    """Apply unwrap for the buggy hip_adduction proxy, plain ROM otherwise."""
    if col_name in ("hip_adduction_r", "hip_adduction_l"):
        return _rom_unwrapped(series)
    return _rom(series)


def extract_features_for_target(
    kp_path: Path, subject: str, cam: str, target_gt: str,
) -> dict[str, float]:
    series = load_couro_output(kp_path)
    meta = yaml.safe_load((LAB / subject / "sessionMetadata.yaml").read_text())
    cam_pkl = next((LAB / subject / "VideoData").glob(f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"))
    md = keypoints_to_motion_data(
        series,
        subject_height_m=float(meta["height_m"]),
        camera_pickle=cam_pkl,
    )
    couro_col, coup1, coup2 = TARGETS[target_gt]
    target_series = md.column(couro_col)
    coup1_series = md.column(coup1)
    coup2_series = md.column(coup2)

    ankle_y = series.xy[:, KP["R_ank"], 1]
    events = detect_dj_events(ankle_y, fps=series.fps)
    ev_t = event_anchored_for_metric(target_series, events)
    ev_c1 = event_anchored_for_metric(coup1_series, events)
    ev_c2 = event_anchored_for_metric(coup2_series, events)

    return {
        # Baseline (v4-equivalent feature)
        "target_rom_full": _select_rom(couro_col, target_series),
        # Target event-anchored
        "target_at_contact": ev_t["at_contact"],
        "target_rom_loading": ev_t["rom_loading"],
        "target_min_loading": ev_t["min_loading"],
        "target_max_loading": ev_t["max_loading"],
        "target_end_loading": ev_t["end_loading"],
        # Couplings (one feature each — keep total features modest)
        "coup1_at_contact": ev_c1["at_contact"],
        "coup1_rom_loading": ev_c1["rom_loading"],
        "coup2_at_contact": ev_c2["at_contact"],
        # Drop-strategy proxy
        "loading_duration_frames": ev_t["loading_duration_frames"],
    }


FEATURE_NAMES = (
    "target_rom_full",         # baseline first → naive_pred = X[:, 0]
    "target_at_contact",
    "target_rom_loading",
    "target_min_loading",
    "target_max_loading",
    "target_end_loading",
    "coup1_at_contact",
    "coup1_rom_loading",
    "coup2_at_contact",
    "loading_duration_frames",
)


def gt_rom(subject: str, task: str, gt_col: str) -> float:
    p = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not p.exists():
        return float("nan")
    return _rom(parse_mot(p).column(gt_col))


def build_dataset(target_gt: str, cam: str):
    X_rows, y, subjects, tasks = [], [], [], []
    skipped = 0
    for kp_path in sorted(KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        if not task.startswith("DJ"):
            continue
        gt = gt_rom(subject, task, target_gt)
        if not np.isfinite(gt):
            continue
        try:
            feats = extract_features_for_target(kp_path, subject, cam, target_gt)
        except Exception:
            skipped += 1
            continue
        values = list(feats.values())
        if not all(np.isfinite(values)):
            skipped += 1
            continue
        X_rows.append(values)
        y.append(gt)
        subjects.append(subject)
        tasks.append(task)
    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y, dtype=np.float64),
        subjects, tasks, skipped,
    )


def load_v4_stats() -> dict:
    """Pull the v4 LOSO stats so we can print a side-by-side comparison."""
    p = RESULTS_DIR / "all_metrics_regression_models.json"
    if not p.exists():
        return {}
    with open(p) as f:
        d = json.load(f)
    out = {}
    for tgt, td in d["targets"].items():
        out[tgt] = {}
        for view, m in td["models"].items():
            out[tgt][view] = {
                "rmse": m["loso_cv_stats"]["rmse_deg"],
                "r": m["loso_cv_stats"]["pearson_r"],
            }
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    v4 = load_v4_stats()

    out: dict = {
        "version": "1.0",
        "produced_by": "harness.train_event_anchored_all",
        "produced_date": "2026-05-27",
        "description": (
            "Event-anchored regression for all 5 DJ biomech metrics. DJ impact "
            "detected via ankle-y peak (scipy peak-find, same recipe Couro's "
            "gait_analyzer.py uses for sprint touchdown). Loading-window "
            "features replace full-trial ROM as inputs to ridge regression. "
            "9 features per metric × 5 camera views, LOSO cross-validated."
        ),
        "feature_names": list(FEATURE_NAMES),
        "training_dataset": {
            "name": "OpenCap LabValidation_withVideos (drop jumps)",
            "citation": "Uhlrich et al. 2023, PLOS Comp Bio 19(10): e1011462",
            "license": "Apache 2.0",
        },
        "alpha": 10.0,
        "targets": {},
    }

    print(f"\n{'TARGET':<22}{'VIEW':<22}{'n':>4}{'naive_RMSE':>11}"
          f"{'v4_RMSE':>10}{'v4_r':>7}{'v6_RMSE':>10}{'v6_r':>7}{'Δr':>7}")
    print("-" * 102)
    for target_gt in TARGETS:
        out["targets"][target_gt] = {"feature_names": list(FEATURE_NAMES), "models": {}}
        for cam in ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"]:
            X, y, subjects, tasks, skipped = build_dataset(target_gt, cam)
            view = VIEW_BUCKET_NAMES[cam]
            if len(y) < 20:
                continue
            naive_pred = X[:, 0]
            naive_rmse = float(np.sqrt(np.mean((naive_pred - y) ** 2)))
            naive_r = (
                float(np.corrcoef(naive_pred, y)[0, 1]) if y.std() > 0 else float("nan")
            )
            cv = loso_cv(X, y, subjects, alpha=10.0)
            full = fit_ridge(X, y, alpha=10.0)

            v4_view = v4.get(target_gt, {}).get(view, {})
            v4_rmse = v4_view.get("rmse", float("nan"))
            v4_r = v4_view.get("r", float("nan"))
            d_r = cv["pearson_r"] - v4_r if np.isfinite(v4_r) else float("nan")

            print(
                f"  {target_gt:<20}{view:<22}{len(y):>4d}"
                f"{naive_rmse:>10.2f}°{v4_rmse:>9.2f}°{v4_r:>+7.2f}"
                f"{cv['rmse']:>9.2f}°{cv['pearson_r']:>+7.2f}{d_r:>+7.2f}"
            )

            out["targets"][target_gt]["models"][view] = {
                "camera_id": cam,
                "view_bucket": view,
                "target_metric": target_gt,
                "n_train_trials": int(len(y)),
                "n_subjects": len(set(subjects)),
                "n_skipped_no_event": int(skipped),
                "feature_names": list(FEATURE_NAMES),
                "feature_mean": full.feature_mean.tolist(),
                "feature_std": full.feature_std.tolist(),
                "weights_zscored": full.weights.tolist(),
                "intercept": full.bias,
                "naive_baseline": {"rmse": naive_rmse, "pearson_r": naive_r},
                "loso_cv_stats": {
                    "rmse_deg": cv["rmse"],
                    "mae_deg": cv["mae"],
                    "bias_deg": cv["bias"],
                    "sd_of_difference_deg": cv["sd_diff"],
                    "loa_95_lower_deg": cv["loa"][0],
                    "loa_95_upper_deg": cv["loa"][1],
                    "pearson_r": cv["pearson_r"],
                },
                "v4_comparison": {
                    "v4_rmse_deg": v4_rmse,
                    "v4_pearson_r": v4_r,
                    "delta_r": d_r,
                },
            }

    out_path = RESULTS_DIR / "event_anchored_all_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
