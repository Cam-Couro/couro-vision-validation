"""Proof-of-concept: train a biomechanical regression to predict hip-flexion ROM
from a single front-view camera, using the OpenCap paired 2D-Couro / 3D-Vicon data.

Same recipe Couro already uses in knee_rotation.py:
  1. Pick 2D-observable features per trial
  2. Train regression on (features → 3D GT)
  3. Leave-one-subject-out CV to estimate accuracy
  4. If it beats the naive baseline, it goes into metrics_calculator
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
    from harness.couro_keypoints import load_couro_output, keypoints_to_motion_data
    from harness.parsers import parse_mot
else:
    from .couro_keypoints import load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot


FEATURE_NAMES = (
    "hip_flex_rom_r_2d3d",
    "knee_flex_rom_r_2d3d",
    "trunk_lean_rom",
    "pelvis_tilt_rom",
    "knee_flex_rom_diff_lr",
    "hip_y_range_px",
    "hip_flex_vel_peak",
)

DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
LAB = DATA_ROOT / "LabValidation_withVideos"
KP_DIR = DATA_ROOT / "couro_keypoints"


def _rom(arr: np.ndarray) -> float:
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def _peak_abs(arr: np.ndarray) -> float:
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmax(np.abs(arr)))


def _vel_max(arr: np.ndarray, fps: float = 60.0) -> float:
    """Peak absolute velocity (deg/sec)."""
    if arr.size < 2:
        return float("nan")
    diff = np.diff(arr) * fps
    return float(np.nanmax(np.abs(diff)))


def extract_features(kp_path: Path, subject: str, cam: str) -> dict:
    """7 features observable from front-view Couro 2D keypoints + GT label."""
    series = load_couro_output(kp_path)
    # Anthropometric 3D recovery (uses subject height + camera intrinsics)
    meta = yaml.safe_load((LAB / subject / "sessionMetadata.yaml").read_text())
    cam_pkl = next((LAB / subject / "VideoData").glob(f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"))
    md = keypoints_to_motion_data(
        series,
        subject_height_m=float(meta["height_m"]),
        camera_pickle=cam_pkl,
    )

    return {
        # Joint-angle ROMs (the metric we'd report normally)
        "hip_flex_rom_r_2d3d": _rom(md.column("hip_flexion_r")),
        "knee_flex_rom_r_2d3d": _rom(md.column("knee_angle_r")),
        # Trunk + pelvis (correlated with hip flexion during squat/landing)
        "trunk_lean_rom": _rom(md.column("lumbar_extension")),
        "pelvis_tilt_rom": _rom(md.column("pelvis_tilt")),
        # Knee/hip side asymmetry
        "knee_flex_rom_diff_lr": abs(_rom(md.column("knee_angle_r")) - _rom(md.column("knee_angle_l"))),
        # Hip-center vertical displacement (jump depth — strongly couples to hip flex)
        "hip_y_range_px": float(np.nanmax(series.xy[:, 19, 1]) - np.nanmin(series.xy[:, 19, 1])),
        # Peak hip flex velocity (deg/s) — coupling proxy
        "hip_flex_vel_peak": _vel_max(md.column("hip_flexion_r"), fps=series.fps),
    }


def _gt_hip_flex_rom(subject: str, task: str) -> float:
    gt_path = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not gt_path.exists():
        return float("nan")
    return _rom(parse_mot(gt_path).column("hip_flexion_r"))


def build_dataset(cam: str = "Cam2") -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build (X, y, subjects, tasks) over all OpenCap DJ trials at given camera."""
    X_rows, y, subjects, tasks = [], [], [], []
    for kp_path in sorted(KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        if not task.startswith("DJ"):
            continue
        gt = _gt_hip_flex_rom(subject, task)
        if not np.isfinite(gt):
            continue
        try:
            feats = extract_features(kp_path, subject, cam)
        except Exception as e:
            print(f"  skip {subject}/{task}: {e}")
            continue
        if not all(np.isfinite(list(feats.values()))):
            continue
        X_rows.append(list(feats.values()))
        y.append(gt)
        subjects.append(subject)
        tasks.append(task)
    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y, dtype=np.float64),
        subjects, tasks,
    )


@dataclass(frozen=True)
class RidgeModel:
    weights: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xz = (X - self.feature_mean) / self.feature_std
        return Xz @ self.weights + self.bias


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> RidgeModel:
    """Closed-form ridge regression with z-score-standardized features."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-9] = 1.0  # avoid divide-by-zero on constant features
    Xz = (X_train - mean) / std
    Xb = np.hstack([Xz, np.ones((Xz.shape[0], 1))])
    reg = alpha * np.eye(Xb.shape[1])
    reg[-1, -1] = 0  # don't regularize bias
    w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y_train)
    return RidgeModel(weights=w[:-1], bias=float(w[-1]), feature_mean=mean, feature_std=std)


def loso_cv(X, y, subjects, alpha: float = 1.0):
    """Leave-one-subject-out cross-validation."""
    subjects = np.array(subjects)
    unique = sorted(set(subjects))
    all_pred = np.full(len(y), np.nan)
    fold_rmses = []
    for s in unique:
        train_mask = subjects != s
        test_mask = subjects == s
        model = fit_ridge(X[train_mask], y[train_mask], alpha=alpha)
        pred = model.predict(X[test_mask])
        all_pred[test_mask] = pred
        rmse = float(np.sqrt(np.mean((pred - y[test_mask]) ** 2)))
        fold_rmses.append((s, rmse, int(test_mask.sum())))
    overall_rmse = float(np.sqrt(np.mean((all_pred - y) ** 2)))
    overall_mae = float(np.mean(np.abs(all_pred - y)))
    overall_bias = float(np.mean(all_pred - y))
    sd_diff = float(np.std(all_pred - y, ddof=1))
    r = float(np.corrcoef(all_pred, y)[0, 1])
    return {
        "all_pred": all_pred,
        "y": y,
        "fold_rmses": fold_rmses,
        "rmse": overall_rmse,
        "mae": overall_mae,
        "bias": overall_bias,
        "sd_diff": sd_diff,
        "pearson_r": r,
        "loa": (overall_bias - 1.96 * sd_diff, overall_bias + 1.96 * sd_diff),
    }


def main():
    for cam in ["Cam2", "Cam0", "Cam1", "Cam3", "Cam4"]:
        print(f"\n{'='*70}")
        print(f"Camera: {cam}")
        print('='*70)
        X, y, subjects, tasks = build_dataset(cam=cam)
        print(f"trials: {len(y)}  subjects: {len(set(subjects))}")
        if len(y) < 20:
            print("  too few trials, skipping")
            continue
        # Naive baseline: use the existing 2D+anthro hip_flex_rom_r directly
        naive_pred = X[:, 0]  # first feature is the existing 2D+anthro estimate
        naive_rmse = float(np.sqrt(np.mean((naive_pred - y) ** 2)))
        naive_mae = float(np.mean(np.abs(naive_pred - y)))
        naive_r = float(np.corrcoef(naive_pred, y)[0, 1])
        print(f"\nBASELINE (2D+anthro hip_flex_rom only):")
        print(f"  RMSE={naive_rmse:.2f}°  MAE={naive_mae:.2f}°  r={naive_r:.2f}")

        # Coupled regression with LOSO CV
        cv = loso_cv(X, y, subjects, alpha=10.0)
        print(f"\nCOUPLED REGRESSION (7 features, LOSO CV):")
        print(f"  RMSE={cv['rmse']:.2f}°  MAE={cv['mae']:.2f}°  r={cv['pearson_r']:.2f}")
        print(f"  bias={cv['bias']:+.2f}°  SD_diff={cv['sd_diff']:.2f}°")
        print(f"  95% LoA: [{cv['loa'][0]:+.1f}°, {cv['loa'][1]:+.1f}°]")
        delta = naive_rmse - cv['rmse']
        sign = "improvement" if delta > 0 else "regression"
        print(f"\n  -> {abs(delta):.1f}° {sign} vs baseline ({delta/naive_rmse*100:+.0f}%)")


if __name__ == "__main__":
    main()
