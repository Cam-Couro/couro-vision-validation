"""Bilateral-feature ankle regression — use BOTH feet's features and let the
regression learn the best weighting per view.

Previous view-aware (hard-coded foot-side switch) underperformed because the
far ankle on side cameras is smaller in pixels, not necessarily cleaner.
Instead, give the regression a richer feature pool covering both sides and
let ridge select via learned weights.

Also adds: hip-center vertical as a fallback impact-detection signal for
Cam2 front-center, where ankle-vertical detection was failing on 80% of
trials due to foot foreshortening.
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
    from harness.dj_events import detect_dj_events_multi_signal, event_anchored_for_metric
    from harness.train_regression_poc import fit_ridge, loso_cv
    from harness.train_multi_metric import _rom, VIEW_BUCKET_NAMES
else:
    from .couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .dj_events import detect_dj_events_multi_signal, event_anchored_for_metric
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import _rom, VIEW_BUCKET_NAMES


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
LAB = DATA_ROOT / "LabValidation_withVideos"
KP_DIR = DATA_ROOT / "couro_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


FEATURE_NAMES = (
    # Baseline (full-trial right-ankle ROM — same as v4 naive)
    "ankle_r_rom_full",
    # Right-side event-anchored
    "ankle_r_at_contact",
    "ankle_r_rom_loading",
    "ankle_r_min_loading",
    # Left-side event-anchored (symmetry proxy)
    "ankle_l_at_contact",
    "ankle_l_rom_loading",
    "ankle_l_min_loading",
    # Couplings (both sides)
    "knee_r_at_contact",
    "knee_r_rom_loading",
    "knee_l_at_contact",
    "hip_r_at_contact",
    "hip_l_at_contact",
    # Drop-strategy proxy
    "loading_duration_frames",
)


def extract_bilateral(kp_path: Path, subject: str, cam: str) -> dict[str, float]:
    series = load_couro_output(kp_path)
    meta = yaml.safe_load((LAB / subject / "sessionMetadata.yaml").read_text())
    cam_pkl = next((LAB / subject / "VideoData").glob(f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"))
    md = keypoints_to_motion_data(
        series,
        subject_height_m=float(meta["height_m"]),
        camera_pickle=cam_pkl,
    )

    # Multi-signal event detection: try right-ankle, left-ankle, then hip-center
    r_ankle_y = series.xy[:, KP["R_ank"], 1]
    l_ankle_y = series.xy[:, KP["L_ank"], 1]
    hip_y = series.xy[:, KP["hip_center"], 1]
    events = detect_dj_events_multi_signal(
        [r_ankle_y, l_ankle_y, hip_y], fps=series.fps,
    )

    ankle_r = md.column("ankle_angle_r")
    ankle_l = md.column("ankle_angle_l")
    knee_r = md.column("knee_angle_r")
    knee_l = md.column("knee_angle_l")
    hip_r = md.column("hip_flexion_r")
    hip_l = md.column("hip_flexion_l")

    ev_ar = event_anchored_for_metric(ankle_r, events)
    ev_al = event_anchored_for_metric(ankle_l, events)
    ev_kr = event_anchored_for_metric(knee_r, events)
    ev_kl = event_anchored_for_metric(knee_l, events)
    ev_hr = event_anchored_for_metric(hip_r, events)
    ev_hl = event_anchored_for_metric(hip_l, events)

    return {
        "ankle_r_rom_full": _rom(ankle_r),
        "ankle_r_at_contact": ev_ar["at_contact"],
        "ankle_r_rom_loading": ev_ar["rom_loading"],
        "ankle_r_min_loading": ev_ar["min_loading"],
        "ankle_l_at_contact": ev_al["at_contact"],
        "ankle_l_rom_loading": ev_al["rom_loading"],
        "ankle_l_min_loading": ev_al["min_loading"],
        "knee_r_at_contact": ev_kr["at_contact"],
        "knee_r_rom_loading": ev_kr["rom_loading"],
        "knee_l_at_contact": ev_kl["at_contact"],
        "hip_r_at_contact": ev_hr["at_contact"],
        "hip_l_at_contact": ev_hl["at_contact"],
        "loading_duration_frames": ev_ar["loading_duration_frames"],
    }


def gt_ankle_r(subject: str, task: str) -> float:
    p = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not p.exists():
        return float("nan")
    return _rom(parse_mot(p).column("ankle_angle_r"))


def build_dataset(cam: str):
    X_rows, y, subjects, tasks = [], [], [], []
    skipped = 0
    for kp_path in sorted(KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        if not task.startswith("DJ"):
            continue
        gt = gt_ankle_r(subject, task)
        if not np.isfinite(gt):
            continue
        try:
            feats = extract_bilateral(kp_path, subject, cam)
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "version": "1.0",
        "produced_by": "harness.train_ankle_bilateral",
        "produced_date": "2026-05-27",
        "target_metric": "ankle_angle_r",
        "feature_names": list(FEATURE_NAMES),
        "approach": (
            "Bilateral feature pool — both feet's event-anchored angles fed "
            "to ridge; regression learns per-view weighting. Multi-signal "
            "impact detection (R-ankle → L-ankle → hip-center fallback)."
        ),
        "models": {},
    }

    # Load v6 ankle for comparison
    v6_ankle = {}
    v6_path = RESULTS_DIR / "deploy_ready_models.json"
    if v6_path.exists():
        with open(v6_path) as f:
            v6 = json.load(f)
        for view, m in v6.get("models", {}).get("ankle_angle_r", {}).items():
            v6_ankle[view] = {
                "rmse": m["loso_cv_stats"]["rmse_deg"],
                "r": m["loso_cv_stats"]["pearson_r"],
            }

    print(
        f"\n{'view':<22}{'n':>5}{'skip':>6}"
        f"{'naive_RMSE':>12}{'cv_RMSE':>10}{'cv_r':>7}"
        f"{'  v6_RMSE':>10}{'  v6_r':>8}{'  Δr':>7}"
    )
    print("-" * 95)

    for cam in ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"]:
        view = VIEW_BUCKET_NAMES[cam]
        X, y, subjects, tasks, skipped = build_dataset(cam)
        if len(y) < 20:
            print(f"  {view:<20} too few trials ({len(y)})")
            continue
        naive_pred = X[:, 0]
        naive_rmse = float(np.sqrt(np.mean((naive_pred - y) ** 2)))
        cv = loso_cv(X, y, subjects, alpha=10.0)
        full = fit_ridge(X, y, alpha=10.0)
        v6 = v6_ankle.get(view, {})
        v6_r = v6.get("r", float("nan"))
        v6_rmse = v6.get("rmse", float("nan"))
        d_r = cv["pearson_r"] - v6_r if np.isfinite(v6_r) else float("nan")
        print(
            f"  {view:<20}{len(y):>5d}{skipped:>6d}"
            f"{naive_rmse:>11.2f}°{cv['rmse']:>9.2f}°{cv['pearson_r']:>+6.2f}"
            f"{v6_rmse:>+9.2f}°{v6_r:>+7.2f}{d_r:>+6.2f}"
        )

        out["models"][view] = {
            "camera_id": cam,
            "view_bucket": view,
            "target_metric": "ankle_angle_r",
            "n_train_trials": int(len(y)),
            "n_subjects": len(set(subjects)),
            "n_skipped_no_event": int(skipped),
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": full.feature_mean.tolist(),
            "feature_std": full.feature_std.tolist(),
            "weights_zscored": full.weights.tolist(),
            "intercept": full.bias,
            "naive_baseline": {"rmse": naive_rmse},
            "loso_cv_stats": {
                "rmse_deg": cv["rmse"],
                "mae_deg": cv["mae"],
                "bias_deg": cv["bias"],
                "sd_of_difference_deg": cv["sd_diff"],
                "loa_95_lower_deg": cv["loa"][0],
                "loa_95_upper_deg": cv["loa"][1],
                "pearson_r": cv["pearson_r"],
            },
            "v6_comparison": {
                "v6_rmse_deg": v6_rmse,
                "v6_pearson_r": v6_r,
                "delta_r": d_r,
            },
        }

    out_path = RESULTS_DIR / "ankle_bilateral_models.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
