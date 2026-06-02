"""View-aware ankle dorsiflexion regression.

Hypothesis from v6: 4 of 5 ankle views fail because the right-ankle keypoint
is occluded or foreshortened on right-side cameras (Cam3 front_oblique_right,
Cam4 side_right) and on dead-on front_center (Cam2). The remaining views
(Cam0 side_left, Cam1 front_oblique_left) work because the right ankle is
the FAR foot and projects cleanly.

Strategy: for each camera view, pick the foot-side whose keypoint is
better-quality given the camera geometry. Use the chosen side's keypoints
to both:
  (a) detect the DJ impact event, and
  (b) extract loading-window ankle features

The OpenSim GT we're predicting is ankle_angle_r (right ankle). In symmetric
DJs both ankles move together, so the contralateral left-ankle keypoint is a
valid proxy. For asymmetric DJs (DJAsym tasks) this is a worse assumption,
but most of our trial set is symmetric DJ.

Per-view chosen side:
  Cam0 side_left          → R (right is far foot)
  Cam1 front_oblique_left → R
  Cam2 front_center       → R (try both at train time, pick best)
  Cam3 front_oblique_right→ L (left is far foot)
  Cam4 side_right         → L
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
    from harness.train_multi_metric import _rom, VIEW_BUCKET_NAMES
else:
    from .couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .dj_events import detect_dj_events, event_anchored_for_metric
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import _rom, VIEW_BUCKET_NAMES


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
LAB = DATA_ROOT / "LabValidation_withVideos"
KP_DIR = DATA_ROOT / "couro_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


# Pre-set measurement side per camera based on geometry.
# Front_center is ambiguous — we'll train both and pick.
CAM_PREFERRED_SIDE: dict[str, list[str]] = {
    "Cam0": ["R"],            # subject's right is the far foot
    "Cam1": ["R"],
    "Cam2": ["R", "L"],       # try both
    "Cam3": ["L"],            # subject's left is the far foot
    "Cam4": ["L"],
}


FEATURE_NAMES = (
    "ankle_rom_full",          # baseline (chosen-side full-trial ROM)
    "ankle_at_contact",
    "ankle_rom_loading",
    "ankle_min_loading",
    "ankle_max_loading",
    "ankle_end_loading",
    "knee_at_contact_same_side",
    "knee_rom_loading_same_side",
    "hip_at_contact_same_side",
    "loading_duration_frames",
)


def extract_features_side(
    kp_path: Path, subject: str, cam: str, side: str,
) -> dict[str, float]:
    """Extract event-anchored ankle features using the chosen body side.

    Args:
        side: "R" or "L" — which foot/ankle to use as the measurement signal.
              When "L", we anchor events to left-ankle vertical, and feed the
              left-ankle/left-knee/left-hip angles into the regression. The
              GT is still right-ankle (symmetry assumption for sym DJ).
    """
    series = load_couro_output(kp_path)
    meta = yaml.safe_load((LAB / subject / "sessionMetadata.yaml").read_text())
    cam_pkl = next((LAB / subject / "VideoData").glob(f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"))
    md = keypoints_to_motion_data(
        series,
        subject_height_m=float(meta["height_m"]),
        camera_pickle=cam_pkl,
    )

    if side == "R":
        ankle_col, knee_col, hip_col = "ankle_angle_r", "knee_angle_r", "hip_flexion_r"
        ankle_kp = KP["R_ank"]
    elif side == "L":
        ankle_col, knee_col, hip_col = "ankle_angle_l", "knee_angle_l", "hip_flexion_l"
        ankle_kp = KP["L_ank"]
    else:
        raise ValueError(f"side must be 'R' or 'L', got {side!r}")

    ankle = md.column(ankle_col)
    knee = md.column(knee_col)
    hip = md.column(hip_col)

    ankle_y = series.xy[:, ankle_kp, 1]
    events = detect_dj_events(ankle_y, fps=series.fps)
    ev_a = event_anchored_for_metric(ankle, events)
    ev_k = event_anchored_for_metric(knee, events)
    ev_h = event_anchored_for_metric(hip, events)

    return {
        "ankle_rom_full": _rom(ankle),
        "ankle_at_contact": ev_a["at_contact"],
        "ankle_rom_loading": ev_a["rom_loading"],
        "ankle_min_loading": ev_a["min_loading"],
        "ankle_max_loading": ev_a["max_loading"],
        "ankle_end_loading": ev_a["end_loading"],
        "knee_at_contact_same_side": ev_k["at_contact"],
        "knee_rom_loading_same_side": ev_k["rom_loading"],
        "hip_at_contact_same_side": ev_h["at_contact"],
        "loading_duration_frames": ev_a["loading_duration_frames"],
    }


def gt_ankle_rom_r(subject: str, task: str) -> float:
    p = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not p.exists():
        return float("nan")
    # Always predict right ankle (consistent target)
    return _rom(parse_mot(p).column("ankle_angle_r"))


def build_dataset(cam: str, side: str):
    X_rows, y, subjects, tasks = [], [], [], []
    skipped = 0
    for kp_path in sorted(KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        if not task.startswith("DJ"):
            continue
        gt = gt_ankle_rom_r(subject, task)
        if not np.isfinite(gt):
            continue
        try:
            feats = extract_features_side(kp_path, subject, cam, side)
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
        "produced_by": "harness.train_ankle_view_aware",
        "produced_date": "2026-05-27",
        "target_metric": "ankle_angle_r",
        "feature_names": list(FEATURE_NAMES),
        "approach": (
            "View-aware foot-side selection. For each camera view, the "
            "measurement side (L or R) is chosen based on camera geometry "
            "to use the far/least-occluded foot. GT is always right-ankle "
            "OpenSim IK; left-side features used as a symmetry proxy where "
            "needed."
        ),
        "models": {},
    }

    print(
        f"\n{'view':<22}{'side':>5}{'n':>5}"
        f"{'naive_RMSE':>12}{'cv_RMSE':>10}{'Δ%':>7}{'cv_r':>7}"
        f"{'  v6_r':>8}{'  Δr':>7}"
    )
    print("-" * 88)

    # v6 ankle baseline for comparison
    v6_path = RESULTS_DIR / "deploy_ready_models.json"
    v6_ankle = {}
    if v6_path.exists():
        with open(v6_path) as f:
            v6 = json.load(f)
        for view, m in v6.get("models", {}).get("ankle_angle_r", {}).items():
            v6_ankle[view] = m["loso_cv_stats"]["pearson_r"]

    for cam in ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"]:
        view = VIEW_BUCKET_NAMES[cam]
        best_for_view: dict | None = None
        for side in CAM_PREFERRED_SIDE[cam]:
            X, y, subjects, tasks, skipped = build_dataset(cam, side)
            if len(y) < 20:
                continue
            naive_pred = X[:, 0]
            naive_rmse = float(np.sqrt(np.mean((naive_pred - y) ** 2)))
            cv = loso_cv(X, y, subjects, alpha=10.0)
            full = fit_ridge(X, y, alpha=10.0)
            delta = (naive_rmse - cv["rmse"]) / naive_rmse * 100
            v6_r = v6_ankle.get(view, float("nan"))
            d_r = cv["pearson_r"] - v6_r if np.isfinite(v6_r) else float("nan")
            print(
                f"  {view:<20}{side:>5}{len(y):>5d}"
                f"{naive_rmse:>11.2f}°{cv['rmse']:>9.2f}°"
                f"{delta:>+5.0f}%{cv['pearson_r']:>+6.2f}"
                f"{v6_r:>+7.2f}{d_r:>+6.2f}"
            )
            candidate = {
                "camera_id": cam,
                "view_bucket": view,
                "measurement_side": side,
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
                "v6_comparison": {"v6_pearson_r": v6_r, "delta_r": d_r},
            }
            if best_for_view is None or candidate["loso_cv_stats"]["pearson_r"] > best_for_view["loso_cv_stats"]["pearson_r"]:
                best_for_view = candidate
        if best_for_view is not None:
            out["models"][view] = best_for_view

    out_path = RESULTS_DIR / "ankle_view_aware_models.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
