"""v11: Expand training data with OpenCap squats / STS / walking trials.

Each subject has 16 trials but we've only used the 5-6 DJ ones. Other tasks
on disk per subject:
  - squats1, squatsAsym1 (~2 trials) — bilateral compression, biomech adjacent to DJ landing
  - STS1, STSweakLegs1 (~2 trials) — sit-to-stand, partial loading
  - walking1-3, walkingTS1-4 (~7 trials) — gait, biomechanically very different

Strategy: train one model per (target × view) using full-trial ROM features
on EXPANDED task sets, see if Pearson r lifts. Event-anchored / v9 phased
models are kept DJ-only since they require a single impact event.

Ablation variants:
  v11a: DJ only (baseline matches v4 full_trial)
  v11b: DJ + squats
  v11c: DJ + squats + STS
  v11d: ALL tasks including walking
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
    from harness.train_regression_poc import fit_ridge, loso_cv
    from harness.train_multi_metric import _rom, _rom_unwrapped, VIEW_BUCKET_NAMES, TARGETS
else:
    from .couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import _rom, _rom_unwrapped, VIEW_BUCKET_NAMES, TARGETS


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
LAB = DATA_ROOT / "LabValidation_withVideos"
KP_DIR = DATA_ROOT / "couro_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


TARGET_COL = {
    "hip_flexion_r":    "hip_flexion_r",
    "hip_adduction_r":  "hip_adduction_r",
    "knee_angle_r":     "knee_angle_r",
    "ankle_angle_r":    "ankle_angle_r",
    "lumbar_extension": "lumbar_extension",
}


def _select_rom(col, series):
    if col in ("hip_adduction_r", "hip_adduction_l"):
        return _rom_unwrapped(series)
    return _rom(series)


def task_prefix(task: str) -> str:
    """Bucket task into family: DJ, squat, STS, walking."""
    if task.startswith("DJ"):
        return "DJ"
    if task.startswith("squats"):
        return "squat"
    if task.startswith("STS"):
        return "STS"
    if task.startswith("walking"):
        return "walking"
    return "other"


def extract_features_full(kp_path: Path, subject: str, cam: str, target_gt: str) -> dict[str, float]:
    """v4-style full-trial features (no event detection — works on any task)."""
    series = load_couro_output(kp_path)
    meta = yaml.safe_load((LAB / subject / "sessionMetadata.yaml").read_text())
    cam_pkl = next((LAB / subject / "VideoData").glob(f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"))
    md = keypoints_to_motion_data(
        series, conf_threshold=0.3,
        subject_height_m=float(meta["height_m"]),
        camera_pickle=cam_pkl,
    )
    couro_col = TARGET_COL[target_gt]
    target_series = md.column(couro_col)
    hip_center_y = series.xy[:, KP["hip_center"], 1]

    feats: dict[str, float] = {
        "target_rom_full": _select_rom(couro_col, target_series),
        "hip_flex_rom_r": _rom(md.column("hip_flexion_r")),
        "knee_flex_rom_r": _rom(md.column("knee_angle_r")),
        "ankle_rom_r": _rom(md.column("ankle_angle_r")),
        "lumbar_rom": _rom(md.column("lumbar_extension")),
        "pelvis_tilt_rom": _rom(md.column("pelvis_tilt")),
        "knee_diff_lr": abs(_rom(md.column("knee_angle_r")) - _rom(md.column("knee_angle_l"))),
        "hip_y_range_px": float(np.nanmax(hip_center_y) - np.nanmin(hip_center_y)),
    }
    return feats


FEATURE_NAMES = (
    "target_rom_full", "hip_flex_rom_r", "knee_flex_rom_r", "ankle_rom_r",
    "lumbar_rom", "pelvis_tilt_rom", "knee_diff_lr", "hip_y_range_px",
)


def gt_rom(subject: str, task: str, gt_col: str) -> float:
    p = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not p.exists():
        return float("nan")
    return _rom(parse_mot(p).column(gt_col))


def build_dataset(target_gt: str, cam: str, allowed_families: set[str]):
    X_rows, y, subjects, tasks = [], [], [], []
    for kp_path in sorted(KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        fam = task_prefix(task)
        if fam not in allowed_families:
            continue
        gt = gt_rom(subject, task, target_gt)
        if not np.isfinite(gt):
            continue
        try:
            feats = extract_features_full(kp_path, subject, cam, target_gt)
        except Exception:
            continue
        vals = list(feats.values())
        if all(np.isfinite(vals)):
            X_rows.append(vals); y.append(gt)
            subjects.append(subject); tasks.append(task)
    return np.array(X_rows, dtype=np.float64), np.array(y, dtype=np.float64), subjects, tasks


VARIANTS = [
    ("v11a DJ only",              {"DJ"}),
    ("v11b DJ+squats",            {"DJ", "squat"}),
    ("v11c DJ+squats+STS",        {"DJ", "squat", "STS"}),
    ("v11d ALL incl walking",     {"DJ", "squat", "STS", "walking"}),
]


def main():
    # Load v9 deploy ridge stats for comparison
    v7_path = RESULTS_DIR / "deploy_ready_models.json"
    v7_r: dict = {}
    if v7_path.exists():
        with open(v7_path) as f:
            d = json.load(f)
        for tgt, td in d.get("models", {}).items():
            v7_r[tgt] = {
                v: m["loso_cv_stats"]["pearson_r"] for v, m in td.items()
            }

    print(f"\n{'slot':<36}", end="")
    for vname, _ in VARIANTS:
        print(f"{vname:>20}", end="")
    print(f"{'current_best_r':>20}")
    print("-" * 120)

    for target in TARGETS:
        for cam in ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"]:
            view = VIEW_BUCKET_NAMES[cam]
            row = f"  {target} / {view:<18}"
            for vname, families in VARIANTS:
                X, y, subj, tasks = build_dataset(target, cam, families)
                if len(y) < 20:
                    row += f"{'  --':>20}"
                    continue
                cv = loso_cv(X, y, subj, alpha=10.0)
                row += f"  r={cv['pearson_r']:>+5.2f} n={len(y):>3d}"[:20]
            cur_best = v7_r.get(target, {}).get(view, float("nan"))
            row += f"{cur_best:>+19.2f}"
            print(row)


if __name__ == "__main__":
    main()
