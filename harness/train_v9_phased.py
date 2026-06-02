"""v9: time-phased features + DJAsym filtering + horizontal-flip augmentation.

Hypotheses tested:
  1. We're throwing away 95% of the temporal information by summarising each
     trial as ~10 scalars (ROM, at-contact, etc). Sample the target+coupling
     angles at 5 evenly-spaced phases of the loading window. ~22 features
     captures much richer dynamics.
  2. DJAsym trials are intentionally asymmetric — they violate the L/R
     symmetry our right-side regression implicitly assumes. Filtering them
     should reduce noise.
  3. For symmetric DJs, horizontally flipping the video (and swapping L/R
     keypoints + targets) doubles training data without collecting more.

Each can be toggled independently — easy ablation study.
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
else:
    from .couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .dj_events import detect_dj_events_multi_signal, event_anchored_for_metric
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import _rom, _rom_unwrapped, VIEW_BUCKET_NAMES


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
LAB = DATA_ROOT / "LabValidation_withVideos"
KP_DIR = DATA_ROOT / "couro_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


# Same target/coupling map as v6/v7
TARGETS = {
    "hip_flexion_r":    ("hip_flexion_r",    "knee_angle_r",    "ankle_angle_r"),
    "hip_adduction_r":  ("hip_adduction_r",  "hip_adduction_l", "knee_angle_r"),
    "knee_angle_r":     ("knee_angle_r",     "hip_flexion_r",   "ankle_angle_r"),
    "ankle_angle_r":    ("ankle_angle_r",    "knee_angle_r",    "hip_flexion_r"),
    "lumbar_extension": ("lumbar_extension", "hip_flexion_r",   "pelvis_tilt"),
}

# Left-side mirror columns — when we horizontally flip a video, swap these
MIRROR = {
    "hip_flexion_r": "hip_flexion_l",     "hip_flexion_l": "hip_flexion_r",
    "knee_angle_r": "knee_angle_l",       "knee_angle_l": "knee_angle_r",
    "ankle_angle_r": "ankle_angle_l",     "ankle_angle_l": "ankle_angle_r",
    "hip_adduction_r": "hip_adduction_l", "hip_adduction_l": "hip_adduction_r",
    "lumbar_extension": "lumbar_extension",
    "pelvis_tilt": "pelvis_tilt",
}


def _select_rom(col, series):
    if col in ("hip_adduction_r", "hip_adduction_l"):
        return _rom_unwrapped(series)
    return _rom(series)


def _phased_angles(series: np.ndarray, events, n_phases: int = 5) -> list[float]:
    """Sample angles at n_phases evenly-spaced points across the loading window.

    Returns NaN-filled list if events failed.
    """
    if events.loading_window is None:
        return [float("nan")] * n_phases
    a, b = events.loading_window
    a = max(0, int(a)); b = min(len(series), int(b))
    if b - a < n_phases:
        return [float("nan")] * n_phases
    phases = np.linspace(a, b - 1, n_phases).astype(int)
    return [
        float(series[i]) if np.isfinite(series[i]) else float("nan")
        for i in phases
    ]


def extract_v9_features(
    kp_path: Path, subject: str, cam: str, target_gt: str,
    *,
    flip: bool = False,
) -> dict[str, float]:
    """v9: 5-phase resampled features + summary scalars.

    flip=True: mirror the video horizontally and swap L↔R columns.
    The features extracted are then valid training samples whose GT is the
    ORIGINAL contralateral side (e.g. for target hip_flexion_r flipped,
    GT to predict is hip_flexion_l).
    """
    series = load_couro_output(kp_path)
    meta = yaml.safe_load((LAB / subject / "sessionMetadata.yaml").read_text())
    cam_pkl = next((LAB / subject / "VideoData").glob(f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"))

    if flip:
        # Mirror x-coordinates and swap L/R indices in the keypoint array.
        # Build a copy of the series with mirrored xy and remapped confidence.
        mirrored_xy = series.xy.copy()
        mirrored_xy[..., 0] = series.width - 1 - mirrored_xy[..., 0]
        mirrored_conf = series.confidence.copy()
        # Swap left/right keypoint indices: pairs from Halpe-26
        lr_pairs = [
            ("L_eye", "R_eye"), ("L_ear", "R_ear"),
            ("L_sho", "R_sho"), ("L_elb", "R_elb"), ("L_wri", "R_wri"),
            ("L_hip", "R_hip"), ("L_kne", "R_kne"), ("L_ank", "R_ank"),
            ("L_big_toe", "R_big_toe"), ("L_small_toe", "R_small_toe"),
            ("L_heel", "R_heel"),
        ]
        for l_name, r_name in lr_pairs:
            l, r = KP[l_name], KP[r_name]
            mirrored_xy[:, [l, r]] = mirrored_xy[:, [r, l]]
            mirrored_conf[:, [l, r]] = mirrored_conf[:, [r, l]]
        # Reconstruct a CouroKeypointSeries-like (the dataclass is frozen; rebuild)
        from harness.couro_keypoints import CouroKeypointSeries
        series = CouroKeypointSeries(
            time=series.time, xy=mirrored_xy, confidence=mirrored_conf,
            width=series.width, height=series.height, fps=series.fps,
        )

    md = keypoints_to_motion_data(
        series, conf_threshold=0.3,
        subject_height_m=float(meta["height_m"]),
        camera_pickle=cam_pkl,
    )
    couro_col, coup1, coup2 = TARGETS[target_gt]
    target_series = md.column(couro_col)
    coup1_series = md.column(coup1)
    coup2_series = md.column(coup2)

    # Event detection from multi-signal (ankle then hip fallback)
    r_y = series.xy[:, KP["R_ank"], 1]
    l_y = series.xy[:, KP["L_ank"], 1]
    hip_y = series.xy[:, KP["hip_center"], 1]
    events = detect_dj_events_multi_signal([r_y, l_y, hip_y], fps=series.fps)

    # 5-phase resampled angles for target and 2 couplings = 15 features
    phased_target = _phased_angles(target_series, events, 5)
    phased_coup1 = _phased_angles(coup1_series, events, 5)
    phased_coup2 = _phased_angles(coup2_series, events, 5)

    ev_t = event_anchored_for_metric(target_series, events)

    # Mean keypoint confidence in the loading window (input-quality proxy)
    if events.loading_window is not None:
        a, b = events.loading_window
        a = max(0, int(a)); b = min(series.confidence.shape[0], int(b))
        if b > a:
            confs = series.confidence[
                a:b, [KP["R_hip"], KP["R_kne"], KP["R_ank"]],
            ]
            mean_conf = float(np.nanmean(confs))
        else:
            mean_conf = float("nan")
    else:
        mean_conf = float("nan")

    feats = {
        "target_rom_full": _select_rom(couro_col, target_series),
        "target_rom_loading": ev_t["rom_loading"],
        "loading_duration_frames": ev_t["loading_duration_frames"],
        "knee_flex_diff_lr_full": abs(
            _rom(md.column("knee_angle_r")) - _rom(md.column("knee_angle_l"))
        ),
        "mean_kp_conf_loading": mean_conf,
    }
    for i, v in enumerate(phased_target):
        feats[f"target_phase{i}"] = v
    for i, v in enumerate(phased_coup1):
        feats[f"coup1_phase{i}"] = v
    for i, v in enumerate(phased_coup2):
        feats[f"coup2_phase{i}"] = v
    return feats


FEATURE_NAMES = (
    "target_rom_full", "target_rom_loading", "loading_duration_frames",
    "knee_flex_diff_lr_full", "mean_kp_conf_loading",
    "target_phase0", "target_phase1", "target_phase2", "target_phase3", "target_phase4",
    "coup1_phase0", "coup1_phase1", "coup1_phase2", "coup1_phase3", "coup1_phase4",
    "coup2_phase0", "coup2_phase1", "coup2_phase2", "coup2_phase3", "coup2_phase4",
)


def gt_rom_col(subject: str, task: str, col: str) -> float:
    p = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not p.exists():
        return float("nan")
    return _rom(parse_mot(p).column(col))


def build_dataset_v9(
    target_gt: str, cam: str,
    *,
    drop_asym: bool = False,
    flip_aug: bool = False,
):
    X_rows, y, subjects, tasks = [], [], [], []
    skipped = 0
    for kp_path in sorted(KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        if not task.startswith("DJ"):
            continue
        is_asym = "Asym" in task
        if drop_asym and is_asym:
            continue

        # Original sample: features as-is, target is original right-side metric
        gt = gt_rom_col(subject, task, target_gt)
        if not np.isfinite(gt):
            continue
        try:
            feats = extract_v9_features(kp_path, subject, cam, target_gt, flip=False)
        except Exception:
            skipped += 1
            continue
        vals = list(feats.values())
        if all(np.isfinite(vals)):
            X_rows.append(vals); y.append(gt); subjects.append(subject); tasks.append(task)

        # Augmentation sample (flipped): valid only for symmetric DJs to preserve symmetry
        if flip_aug and not is_asym:
            flip_target_col = MIRROR[target_gt]
            gt_flip = gt_rom_col(subject, task, flip_target_col)
            if not np.isfinite(gt_flip):
                continue
            try:
                feats_flip = extract_v9_features(kp_path, subject, cam, target_gt, flip=True)
            except Exception:
                continue
            vals_f = list(feats_flip.values())
            if all(np.isfinite(vals_f)):
                X_rows.append(vals_f); y.append(gt_flip)
                subjects.append(subject); tasks.append(f"{task}_flip")

    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y, dtype=np.float64),
        subjects, tasks, skipped,
    )


def main():
    # Focus the ablation on the weakest current deploys (highest leverage to lift)
    # plus a few strong ones to make sure we don't regress on those.
    SLOTS_TO_TEST = [
        ("hip_flexion_r",     "Cam0"),   # v7 strong r=0.69 — sanity check
        ("hip_flexion_r",     "Cam2"),   # v7 modest r=0.20 — leverage
        ("hip_flexion_r",     "Cam3"),   # v7 modest r=0.26 — leverage
        ("hip_adduction_r",   "Cam1"),   # v7 modest r=0.13 — biggest leverage
        ("hip_adduction_r",   "Cam2"),   # v7 modest r=0.18
        ("knee_angle_r",      "Cam2"),   # v7 modest r=0.15
        ("knee_angle_r",      "Cam4"),   # v7 strong r=0.64 — sanity
        ("ankle_angle_r",     "Cam0"),   # v7 r=0.37
        ("ankle_angle_r",     "Cam2"),   # v7 r=0.15
        ("ankle_angle_r",     "Cam3"),   # v7 r=-0.21 — biggest leverage
        ("lumbar_extension",  "Cam2"),   # v7 strong r=0.53 — sanity
    ]

    # Load v7 r values for comparison
    v7_path = RESULTS_DIR / "deploy_ready_models.json"
    v7_r = {}
    if v7_path.exists():
        with open(v7_path) as f:
            d = json.load(f)
        for tgt, td in d.get("models", {}).items():
            for view, m in td.items():
                v7_r[(tgt, view)] = m["loso_cv_stats"]["pearson_r"]

    VARIANTS = [
        ("v9a phased",         dict(drop_asym=False, flip_aug=False)),
        ("v9b phased+nosym",   dict(drop_asym=True,  flip_aug=False)),
        ("v9c phased+flipaug", dict(drop_asym=False, flip_aug=True)),
        ("v9d all on",         dict(drop_asym=True,  flip_aug=True)),
    ]

    print(f"\n{'slot':<38}{'v7_r':>7}", end="")
    for vname, _ in VARIANTS:
        print(f"{vname:>22}", end="")
    print()
    print("-" * (45 + 22 * len(VARIANTS)))

    for target, cam in SLOTS_TO_TEST:
        view = VIEW_BUCKET_NAMES[cam]
        baseline_r = v7_r.get((target, view), float("nan"))
        row = f"  {target} / {view:<20} {baseline_r:>+6.2f}"
        for vname, params in VARIANTS:
            X, y, subj, tasks, skipped = build_dataset_v9(target, cam, **params)
            if len(y) < 20:
                row += f"{'  --':>22}"
                continue
            cv = loso_cv(X, y, subj, alpha=10.0)
            d_r = cv["pearson_r"] - baseline_r if np.isfinite(baseline_r) else float("nan")
            sign = "+" if d_r >= 0 else ""
            row += f"  r={cv['pearson_r']:>+5.2f} n={len(y):>2d} Δ{sign}{d_r:>+5.2f}"[:22]
        print(row)


if __name__ == "__main__":
    main()
