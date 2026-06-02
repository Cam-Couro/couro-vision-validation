"""Ablation study: which v8 unlock actually helps?

The combined v8 (smoothing + conf + calibration + new features) underperformed
v7. Run each unlock separately to find which one(s) actually move the needle.

Each variant uses v7's same feature pool/extractor (train_event_anchored_all.py
style) but flips one knob at a time:
  v8a = smoothing only (window=3, the lighter variant)
  v8b = calibration only
  v8c = confidence threshold only (0.5 instead of 0.3)
  v8d = all three

Compare to v7 baseline by re-loading deploy_ready_models.json.
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
    from harness.train_multi_metric import _rom, _rom_unwrapped, VIEW_BUCKET_NAMES
    from harness.train_event_anchored_all import TARGETS, FEATURE_NAMES as V7_FEATURE_NAMES
    from harness.calibration import calibrate_subject, SubjectAnthro
else:
    from .couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .dj_events import detect_dj_events_multi_signal, event_anchored_for_metric
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import _rom, _rom_unwrapped, VIEW_BUCKET_NAMES
    from .train_event_anchored_all import TARGETS, FEATURE_NAMES as V7_FEATURE_NAMES
    from .calibration import calibrate_subject, SubjectAnthro


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
LAB = DATA_ROOT / "LabValidation_withVideos"
KP_DIR = DATA_ROOT / "couro_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


def _select_rom(col, series):
    if col in ("hip_adduction_r", "hip_adduction_l"):
        return _rom_unwrapped(series)
    return _rom(series)


def extract_v7_event_anchored(
    kp_path, subject, cam, target_gt,
    *,
    smooth_window=0,
    conf_threshold=0.3,
    anthro_override=None,
):
    """Same feature extractor as v7's train_event_anchored_all, with optional knobs."""
    series = load_couro_output(kp_path)
    meta = yaml.safe_load((LAB / subject / "sessionMetadata.yaml").read_text())
    cam_pkl = next((LAB / subject / "VideoData").glob(f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"))
    md = keypoints_to_motion_data(
        series,
        conf_threshold=conf_threshold,
        subject_height_m=float(meta["height_m"]),
        camera_pickle=cam_pkl,
        anthro_override=anthro_override,
        smooth_window=smooth_window,
    )
    couro_col, coup1, coup2 = TARGETS[target_gt]
    target_series = md.column(couro_col)
    coup1_series = md.column(coup1)
    coup2_series = md.column(coup2)

    r_ankle_y = series.xy[:, KP["R_ank"], 1]
    l_ankle_y = series.xy[:, KP["L_ank"], 1]
    hip_y = series.xy[:, KP["hip_center"], 1]
    events = detect_dj_events_multi_signal(
        [r_ankle_y, l_ankle_y, hip_y], fps=series.fps,
    )
    ev_t = event_anchored_for_metric(target_series, events)
    ev_c1 = event_anchored_for_metric(coup1_series, events)
    ev_c2 = event_anchored_for_metric(coup2_series, events)
    return {
        "target_rom_full": _select_rom(couro_col, target_series),
        "target_at_contact": ev_t["at_contact"],
        "target_rom_loading": ev_t["rom_loading"],
        "target_min_loading": ev_t["min_loading"],
        "target_max_loading": ev_t["max_loading"],
        "target_end_loading": ev_t["end_loading"],
        "coup1_at_contact": ev_c1["at_contact"],
        "coup1_rom_loading": ev_c1["rom_loading"],
        "coup2_at_contact": ev_c2["at_contact"],
        "loading_duration_frames": ev_t["loading_duration_frames"],
    }


def gt_rom(subject, task, gt_col):
    p = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not p.exists():
        return float("nan")
    return _rom(parse_mot(p).column(gt_col))


def build_dataset_ablation(target_gt, cam, calibrations, smooth_window, conf_threshold, use_cal):
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
        cal_override = None
        if use_cal and subject in calibrations:
            c = calibrations[subject]
            cal_override = {
                "thigh_m": c.thigh_m,
                "shank_m": c.shank_m,
                "foot_m": c.foot_m,
                "torso_m": c.torso_m,
            }
        try:
            feats = extract_v7_event_anchored(
                kp_path, subject, cam, target_gt,
                smooth_window=smooth_window,
                conf_threshold=conf_threshold,
                anthro_override=cal_override,
            )
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
    return (np.array(X_rows, dtype=np.float64),
            np.array(y, dtype=np.float64),
            subjects, tasks, skipped)


VARIANTS = {
    "v7 baseline":     {"smooth_window": 0, "conf_threshold": 0.3, "use_cal": False},
    "v8a smooth=3":    {"smooth_window": 3, "conf_threshold": 0.3, "use_cal": False},
    "v8b cal only":    {"smooth_window": 0, "conf_threshold": 0.3, "use_cal": True},
    "v8c conf=0.5":    {"smooth_window": 0, "conf_threshold": 0.5, "use_cal": False},
    "v8d all three":   {"smooth_window": 3, "conf_threshold": 0.5, "use_cal": True},
}


def main():
    # Pre-compute calibrations once
    subjects = sorted({p.stem.split("_")[0] for p in KP_DIR.glob("*.json")})
    cals: dict[str, SubjectAnthro] = {}
    print("Calibrating subjects...")
    for subj in subjects:
        meta_path = LAB / subj / "sessionMetadata.yaml"
        if not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text())
        h = float(meta["height_m"])
        kp_paths = sorted(KP_DIR.glob(f"{subj}_*.json"))
        cam_pkls = {}
        for cam_dir in (LAB / subj / "VideoData").glob("Session*/Cam*"):
            pkl = cam_dir / "cameraIntrinsicsExtrinsics.pickle"
            if pkl.exists():
                cam_pkls[cam_dir.name] = pkl
        if not kp_paths or not cam_pkls:
            continue
        cals[subj] = calibrate_subject(subj, kp_paths, cam_pkls, h)

    # For ablation focus on a few targets x views where v7 was weakest
    # (highest leverage to improve), plus a few strong ones (sanity check
    # we're not breaking what works).
    SLOTS = [
        ("hip_flexion_r", "Cam0"),     # v7 strong r=0.69
        ("hip_flexion_r", "Cam2"),     # v7 modest r=0.20
        ("hip_flexion_r", "Cam4"),     # v7 strong r=0.50
        ("knee_angle_r", "Cam0"),      # v7 strong r=0.55
        ("knee_angle_r", "Cam4"),      # v7 strong r=0.64
        ("ankle_angle_r", "Cam0"),     # v7 r=0.37 (bilateral)
        ("ankle_angle_r", "Cam2"),     # v7 r=0.15 (bilateral)
        ("lumbar_extension", "Cam0"),  # v7 strong
        ("lumbar_extension", "Cam2"),  # v7 strong
    ]

    header = f"\n{'target/view':<32}" + "".join(f"{v:>14}" for v in VARIANTS)
    print(header)
    print("-" * len(header))

    for target, cam in SLOTS:
        row = f"  {target} / {VIEW_BUCKET_NAMES[cam]:<14}"
        for vname, params in VARIANTS.items():
            X, y, subj, tasks, skipped = build_dataset_ablation(
                target, cam, cals,
                params["smooth_window"], params["conf_threshold"], params["use_cal"],
            )
            if len(y) < 20:
                row += f"{'  --':>14}"
                continue
            cv = loso_cv(X, y, subj, alpha=10.0)
            row += f"  r={cv['pearson_r']:>+5.2f} n={len(y):>2d}"
        print(row)


if __name__ == "__main__":
    main()
