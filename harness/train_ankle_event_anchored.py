"""Retrain ankle dorsiflexion regression using event-anchored features.

v4 ankle result was: cv RMSE 9–13° but Pearson r NEGATIVE on every view
(model collapses to population mean). Honest diagnosis: full-trial ankle ROM
washes out the impact-phase signal that actually correlates with GT.

This script anchors features to the detected drop-jump impact event, the same
way Couro's existing track_mode/gait_analyzer.py anchors gait metrics to
heel-strike. Loading-phase ROM, contact-frame angle, and coupled knee/hip
angles at contact replace the noisy full-trial ROM as predictors.

If LOSO Pearson r goes positive across views, ankle is recoverable from 2D.
If r stays negative, the bottleneck is keypoint quality, not features.
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
    from harness.dj_events import detect_dj_events, event_anchored_features
    from harness.train_regression_poc import fit_ridge, loso_cv
    from harness.train_multi_metric import _rom, _rom_unwrapped, _vel_max, VIEW_BUCKET_NAMES
else:
    from .couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .dj_events import detect_dj_events, event_anchored_features
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import _rom, _rom_unwrapped, _vel_max, VIEW_BUCKET_NAMES


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
LAB = DATA_ROOT / "LabValidation_withVideos"
KP_DIR = DATA_ROOT / "couro_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


# Event-anchored features ankle regression uses.
# First feature = baseline (full-trial ankle ROM) so naive_pred = X[:, 0]
# matches the v4 numbers we're trying to beat.
FEATURE_NAMES = (
    "ankle_rom_full_trial",     # baseline (existing 2D+anthro ROM)
    "ankle_rom_loading",        # event-anchored ROM (impact → pushoff)
    "ankle_at_contact",         # ankle angle at initial contact
    "ankle_min_loading",        # deepest dorsiflexion during loading
    "knee_at_contact",          # coupled signal
    "knee_rom_loading",         # coupled ROM during loading
    "hip_at_contact",
    "loading_duration_frames",  # how long was foot on ground (loading phase length)
    "knee_flex_rom_r_full",     # coupled full-trial knee ROM
    "hip_flex_rom_r_full",      # coupled full-trial hip ROM
)


def extract_features(kp_path: Path, subject: str, cam: str) -> dict[str, float]:
    series = load_couro_output(kp_path)
    meta = yaml.safe_load((LAB / subject / "sessionMetadata.yaml").read_text())
    cam_pkl = next((LAB / subject / "VideoData").glob(f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"))
    md = keypoints_to_motion_data(
        series,
        subject_height_m=float(meta["height_m"]),
        camera_pickle=cam_pkl,
    )

    ankle_r = md.column("ankle_angle_r")
    knee_r = md.column("knee_angle_r")
    hip_r = md.column("hip_flexion_r")

    # Detect impact event from right ankle vertical
    ankle_y = series.xy[:, KP["R_ank"], 1]
    events = detect_dj_events(ankle_y, fps=series.fps)
    event_feats = event_anchored_features(ankle_r, knee_r, hip_r, events)

    return {
        "ankle_rom_full_trial": _rom(ankle_r),
        "ankle_rom_loading": event_feats["ankle_rom_loading"],
        "ankle_at_contact": event_feats["ankle_at_contact"],
        "ankle_min_loading": event_feats["ankle_min_loading"],
        "knee_at_contact": event_feats["knee_at_contact"],
        "knee_rom_loading": event_feats["knee_rom_loading"],
        "hip_at_contact": event_feats["hip_at_contact"],
        "loading_duration_frames": event_feats["loading_duration_frames"],
        "knee_flex_rom_r_full": _rom(knee_r),
        "hip_flex_rom_r_full": _rom(hip_r),
    }


def gt_ankle_rom(subject: str, task: str) -> float:
    p = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not p.exists():
        return float("nan")
    return _rom(parse_mot(p).column("ankle_angle_r"))


def build_cam_dataset(cam: str):
    X_rows, y, subjects, tasks = [], [], [], []
    skipped_event = 0
    skipped_other = 0
    for kp_path in sorted(KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        if not task.startswith("DJ"):
            continue
        gt = gt_ankle_rom(subject, task)
        if not np.isfinite(gt):
            continue
        try:
            feats = extract_features(kp_path, subject, cam)
        except Exception as e:
            skipped_other += 1
            continue
        values = list(feats.values())
        if not all(np.isfinite(values)):
            skipped_event += 1
            continue
        X_rows.append(values)
        y.append(gt)
        subjects.append(subject)
        tasks.append(task)
    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y, dtype=np.float64),
        subjects, tasks, skipped_event, skipped_other,
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "version": "1.0",
        "produced_by": "harness.train_ankle_event_anchored",
        "produced_date": "2026-05-27",
        "target_metric": "ankle_angle_r",
        "feature_names": list(FEATURE_NAMES),
        "approach": (
            "Event-anchored features. DJ initial-contact frame detected via "
            "ankle-y peak (Couro gait_analyzer technique). Loading-phase ROM "
            "and coupled knee/hip angles-at-contact replace full-trial ROM "
            "as predictors. Ridge alpha=10, LOSO CV."
        ),
        "models": {},
    }

    print(f"\n{'view':<22} {'n':>4} {'naive_RMSE':>11} {'cv_RMSE':>9} {'Δ%':>6} {'r':>6}  {'skip_evt':>9}")
    for cam in ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"]:
        X, y, subjects, tasks, skipped_event, skipped_other = build_cam_dataset(cam)
        if len(y) < 20:
            print(f"  {VIEW_BUCKET_NAMES[cam]:<20} too few trials ({len(y)})")
            continue
        naive_pred = X[:, 0]  # ankle_rom_full_trial — same baseline as v4
        naive_rmse = float(np.sqrt(np.mean((naive_pred - y) ** 2)))
        naive_r = float(np.corrcoef(naive_pred, y)[0, 1]) if y.std() > 0 else float("nan")
        cv = loso_cv(X, y, subjects, alpha=10.0)
        full = fit_ridge(X, y, alpha=10.0)

        delta = (naive_rmse - cv["rmse"]) / naive_rmse * 100
        view = VIEW_BUCKET_NAMES[cam]
        print(
            f"  {view:<20} {len(y):>4d} "
            f"{naive_rmse:>10.2f}° {cv['rmse']:>8.2f}° "
            f"{delta:>+5.0f}% {cv['pearson_r']:>+5.2f}  {skipped_event:>9d}"
        )

        out["models"][view] = {
            "camera_id": cam,
            "view_bucket": view,
            "target_metric": "ankle_angle_r",
            "n_train_trials": int(len(y)),
            "n_subjects": len(set(subjects)),
            "n_skipped_no_event": int(skipped_event),
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": full.feature_mean.tolist(),
            "feature_std": full.feature_std.tolist(),
            "weights_zscored": full.weights.tolist(),
            "intercept": full.bias,
            "naive_baseline": {
                "rmse": naive_rmse,
                "pearson_r": naive_r,
            },
            "loso_cv_stats": {
                "rmse_deg": cv["rmse"],
                "mae_deg": cv["mae"],
                "bias_deg": cv["bias"],
                "sd_of_difference_deg": cv["sd_diff"],
                "loa_95_lower_deg": cv["loa"][0],
                "loa_95_upper_deg": cv["loa"][1],
                "pearson_r": cv["pearson_r"],
            },
            "rmse_reduction_pct": float(delta),
        }

    out_path = RESULTS_DIR / "ankle_event_anchored_models.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
