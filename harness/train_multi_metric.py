"""Extend the hip-flexion regression recipe to the rest of Couro's drop-jump
biomechanical metrics.

Targets (3D OpenSim GT columns in .mot files):
    hip_adduction_r        — frontal-plane hip motion (knee valgus risk proxy)
    knee_angle_r           — knee flexion ROM (sagittal)
    ankle_angle_r          — ankle dorsi/plantar flexion ROM
    lumbar_extension       — trunk forward-lean ROM

Pipeline: build a unified feature pool from Couro 2D+anthro reconstruction once
per trial, then fit one ridge regression per (target_metric × camera_view)
with leave-one-subject-out CV. Same recipe as `train_regression_poc.py`,
parameterised over target metric and feature subset.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from harness.parsers import parse_mot
    from harness.train_regression_poc import RidgeModel, fit_ridge, loso_cv
else:
    from .couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .train_regression_poc import RidgeModel, fit_ridge, loso_cv


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
LAB = DATA_ROOT / "LabValidation_withVideos"
KP_DIR = DATA_ROOT / "couro_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


VIEW_BUCKET_NAMES = {
    "Cam0": "side_left",
    "Cam1": "front_oblique_left",
    "Cam2": "front_center",
    "Cam3": "front_oblique_right",
    "Cam4": "side_right",
}


# ---------------------------------------------------------------------------
# Unified feature pool — every regressor draws from this set.
# ---------------------------------------------------------------------------

FEATURE_POOL = (
    "hip_flex_rom_r_2d3d",
    "hip_flex_rom_l_2d3d",
    "knee_flex_rom_r_2d3d",
    "knee_flex_rom_l_2d3d",
    "ankle_rom_r_2d3d",
    "ankle_rom_l_2d3d",
    "hip_add_rom_r",
    "hip_add_rom_l",
    "pelvis_tilt_rom",
    "lumbar_extension_rom",
    "knee_flex_rom_diff_lr",
    "hip_y_range_px",
    "hip_flex_vel_peak",
)


# Per-target feature subsets. The first feature is the naive 2D+anthro baseline
# (so naive_pred = X[:, 0] is the existing Couro estimate for that metric).
TARGETS = {
    "hip_flexion_r": (
        "hip_flex_rom_r_2d3d",        # baseline
        "knee_flex_rom_r_2d3d",
        "lumbar_extension_rom",
        "pelvis_tilt_rom",
        "knee_flex_rom_diff_lr",
        "hip_y_range_px",
        "hip_flex_vel_peak",
    ),
    "hip_adduction_r": (
        "hip_add_rom_r",              # baseline
        "hip_add_rom_l",
        "pelvis_tilt_rom",
        "knee_flex_rom_diff_lr",
        "knee_flex_rom_r_2d3d",
        "hip_flex_rom_r_2d3d",
        "hip_y_range_px",
    ),
    "knee_angle_r": (
        "knee_flex_rom_r_2d3d",       # baseline
        "hip_flex_rom_r_2d3d",
        "ankle_rom_r_2d3d",
        "hip_y_range_px",
        "pelvis_tilt_rom",
        "lumbar_extension_rom",
        "hip_flex_vel_peak",
    ),
    "ankle_angle_r": (
        "ankle_rom_r_2d3d",           # baseline
        "knee_flex_rom_r_2d3d",
        "hip_flex_rom_r_2d3d",
        "hip_y_range_px",
        "pelvis_tilt_rom",
        "ankle_rom_l_2d3d",
    ),
    "lumbar_extension": (
        "lumbar_extension_rom",       # baseline
        "hip_flex_rom_r_2d3d",
        "knee_flex_rom_r_2d3d",
        "pelvis_tilt_rom",
        "hip_y_range_px",
    ),
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _rom(arr: np.ndarray) -> float:
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def _rom_unwrapped(arr: np.ndarray) -> float:
    """ROM after unwrapping the angle signal (avoids arctan2 ±180 discontinuity).

    Couro's 2D `hip_adduction_*` proxy uses arctan2 which jumps from +179°
    to -179° as the leg crosses near-vertical, producing a fake ROM of ~360°.
    np.unwrap re-stitches the signal before we compute max-min.
    """
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    finite = arr[np.isfinite(arr)]
    unwrapped = np.degrees(np.unwrap(np.radians(finite)))
    return float(np.nanmax(unwrapped) - np.nanmin(unwrapped))


def _vel_max(arr: np.ndarray, fps: float = 60.0) -> float:
    if arr.size < 2:
        return float("nan")
    diff = np.diff(arr) * fps
    if not np.any(np.isfinite(diff)):
        return float("nan")
    return float(np.nanmax(np.abs(diff)))


def extract_feature_pool(kp_path: Path, subject: str, cam: str) -> dict[str, float]:
    """Compute the full feature pool from a single trial's Couro 2D output."""
    series = load_couro_output(kp_path)
    meta = yaml.safe_load((LAB / subject / "sessionMetadata.yaml").read_text())
    cam_pkl = next((LAB / subject / "VideoData").glob(f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"))
    md = keypoints_to_motion_data(
        series,
        subject_height_m=float(meta["height_m"]),
        camera_pickle=cam_pkl,
    )

    hip_center_y = series.xy[:, KP["hip_center"], 1]

    return {
        "hip_flex_rom_r_2d3d": _rom(md.column("hip_flexion_r")),
        "hip_flex_rom_l_2d3d": _rom(md.column("hip_flexion_l")),
        "knee_flex_rom_r_2d3d": _rom(md.column("knee_angle_r")),
        "knee_flex_rom_l_2d3d": _rom(md.column("knee_angle_l")),
        "ankle_rom_r_2d3d": _rom(md.column("ankle_angle_r")),
        "ankle_rom_l_2d3d": _rom(md.column("ankle_angle_l")),
        "hip_add_rom_r": _rom_unwrapped(md.column("hip_adduction_r")),
        "hip_add_rom_l": _rom_unwrapped(md.column("hip_adduction_l")),
        "pelvis_tilt_rom": _rom(md.column("pelvis_tilt")),
        "lumbar_extension_rom": _rom(md.column("lumbar_extension")),
        "knee_flex_rom_diff_lr": abs(
            _rom(md.column("knee_angle_r")) - _rom(md.column("knee_angle_l"))
        ),
        "hip_y_range_px": float(np.nanmax(hip_center_y) - np.nanmin(hip_center_y)),
        "hip_flex_vel_peak": _vel_max(md.column("hip_flexion_r"), fps=series.fps),
    }


def _gt_rom(subject: str, task: str, gt_column: str) -> float:
    gt_path = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not gt_path.exists():
        return float("nan")
    return _rom(parse_mot(gt_path).column(gt_column))


@dataclass(frozen=True)
class TrialRow:
    subject: str
    task: str
    features: dict[str, float]
    gt: dict[str, float]


def build_trial_rows(cam: str) -> list[TrialRow]:
    """Collect one TrialRow per trial — feature pool + every GT metric."""
    rows: list[TrialRow] = []
    for kp_path in sorted(KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        if not task.startswith("DJ"):
            continue
        try:
            feats = extract_feature_pool(kp_path, subject, cam)
        except Exception as e:
            print(f"  skip {subject}/{task}: {e}")
            continue
        if not all(np.isfinite(list(feats.values()))):
            continue
        gt = {target: _gt_rom(subject, task, target) for target in TARGETS}
        rows.append(TrialRow(subject=subject, task=task, features=feats, gt=gt))
    return rows


def select_features(rows: list[TrialRow], feature_names: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[r.features[name] for name in feature_names] for r in rows],
        dtype=np.float64,
    )


def select_target(rows: list[TrialRow], target: str) -> np.ndarray:
    return np.array([r.gt[target] for r in rows], dtype=np.float64)


# ---------------------------------------------------------------------------
# Train + report
# ---------------------------------------------------------------------------

def naive_stats(naive_pred: np.ndarray, y: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(np.mean((naive_pred - y) ** 2))),
        "mae": float(np.mean(np.abs(naive_pred - y))),
        "pearson_r": float(np.corrcoef(naive_pred, y)[0, 1]),
    }


def train_all_metrics(alpha: float = 10.0) -> dict:
    """Train all (target × camera) regressions and return a results dict."""
    out: dict = {"alpha": alpha, "targets": {}}
    for target, feature_names in TARGETS.items():
        out["targets"][target] = {"feature_names": list(feature_names), "models": {}}
        print(f"\n{'='*72}\nTARGET: {target}    features={list(feature_names)}\n{'='*72}")
        print(f"{'view':<22} {'n':>4} {'naive_RMSE':>11} {'cv_RMSE':>9} {'Δ%':>6} {'r':>5}")
        for cam in ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"]:
            rows = build_trial_rows(cam)
            # Drop rows with bad GT for THIS target
            rows_ok = [r for r in rows if np.isfinite(r.gt[target])]
            if len(rows_ok) < 20:
                continue
            X = select_features(rows_ok, feature_names)
            y = select_target(rows_ok, target)
            subjects = [r.subject for r in rows_ok]

            naive_pred = X[:, 0]  # first feature = existing 2D+anthro estimate
            naive = naive_stats(naive_pred, y)
            cv = loso_cv(X, y, subjects, alpha=alpha)
            full = fit_ridge(X, y, alpha=alpha)

            delta_pct = (naive["rmse"] - cv["rmse"]) / naive["rmse"] * 100
            view = VIEW_BUCKET_NAMES[cam]
            print(
                f"  {view:<20} {len(y):>4d} "
                f"{naive['rmse']:>10.2f}° {cv['rmse']:>8.2f}° "
                f"{delta_pct:>+5.0f}% {cv['pearson_r']:>4.2f}"
            )

            out["targets"][target]["models"][view] = {
                "camera_id": cam,
                "view_bucket": view,
                "target_metric": target,
                "n_train_trials": int(len(y)),
                "n_subjects": len(set(subjects)),
                "feature_names": list(feature_names),
                "feature_mean": full.feature_mean.tolist(),
                "feature_std": full.feature_std.tolist(),
                "weights_zscored": full.weights.tolist(),
                "intercept": full.bias,
                "naive_baseline": naive,
                "loso_cv_stats": {
                    "rmse_deg": cv["rmse"],
                    "mae_deg": cv["mae"],
                    "bias_deg": cv["bias"],
                    "sd_of_difference_deg": cv["sd_diff"],
                    "loa_95_lower_deg": cv["loa"][0],
                    "loa_95_upper_deg": cv["loa"][1],
                    "pearson_r": cv["pearson_r"],
                },
                "rmse_reduction_pct": float(delta_pct),
            }
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = train_all_metrics(alpha=10.0)
    out_path = RESULTS_DIR / "all_metrics_regression_models.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "version": "1.0",
                "produced_by": "harness.train_multi_metric",
                "produced_date": "2026-05-27",
                "description": (
                    "Per-(metric × camera-view) ridge regressions for Couro 2D→3D "
                    "biomechanical recovery. Trained on OpenCap LabValidation "
                    "drop-jump trials with Vicon mocap GT. Same recipe as "
                    "metrics_calculator/metrics/knee_rotation.py."
                ),
                "training_dataset": {
                    "name": "OpenCap LabValidation_withVideos (drop jumps)",
                    "citation": "Uhlrich et al. 2023, PLOS Comp Bio 19(10): e1011462",
                    "license": "Apache 2.0",
                },
                "usage": (
                    "For each metric × view bucket:\n"
                    "  z = (raw_features - feature_mean) / feature_std\n"
                    "  pred_rom_deg = z @ weights_zscored + intercept"
                ),
                **result,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
