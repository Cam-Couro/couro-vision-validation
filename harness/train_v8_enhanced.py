"""v8 enhanced trainer — applies three single-camera precision unlocks together:

    A. Confidence-aware feature extraction (higher conf_threshold, mask low-conf
       frames from angle calculations so they don't poison ROM).
    B. Per-subject anthropometric calibration (replaces Winter 2009 population
       ratios with subject-specific segment lengths observed from the video).
    D. Temporal Savgol smoothing of 2D keypoints before angle computation.

Trains BOTH full-trial and event-anchored variants for each (metric × view)
and picks the winner per slot, comparable to v7's finalize_best_models.
Writes results to v8_deploy_ready_models.json.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
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
    from harness.calibration import calibrate_subject, SubjectAnthro
else:
    from .couro_keypoints import KP, load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .dj_events import detect_dj_events_multi_signal, event_anchored_for_metric
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import _rom, _rom_unwrapped, VIEW_BUCKET_NAMES
    from .calibration import calibrate_subject, SubjectAnthro


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
LAB = DATA_ROOT / "LabValidation_withVideos"
KP_DIR = DATA_ROOT / "couro_keypoints"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


# v8 enhancement parameters
SMOOTH_WINDOW = 7         # Savgol window (frames). 7 ≈ 117ms at 60fps.
CONF_THRESHOLD = 0.5      # was 0.3 by default; tighter for cleaner angles


TARGETS = {
    "hip_flexion_r":    ("hip_flexion_r",    "knee_angle_r",    "ankle_angle_r"),
    "hip_adduction_r":  ("hip_adduction_r",  "hip_adduction_l", "knee_angle_r"),
    "knee_angle_r":     ("knee_angle_r",     "hip_flexion_r",   "ankle_angle_r"),
    "ankle_angle_r":    ("ankle_angle_r",    "knee_angle_r",    "hip_flexion_r"),
    "lumbar_extension": ("lumbar_extension", "hip_flexion_r",   "pelvis_tilt"),
}

FEATURE_NAMES_FULL = (
    "target_rom_full", "coup1_rom_full", "coup2_rom_full",
    "target_p95_minus_p05", "loading_duration_proxy",
    "knee_flex_diff_lr_full", "hip_y_range_px",
)

FEATURE_NAMES_EVENT = (
    "target_rom_full",          # baseline first → naive comparison stays fair
    "target_at_contact", "target_rom_loading", "target_min_loading",
    "target_max_loading", "target_end_loading",
    "coup1_at_contact", "coup1_rom_loading", "coup2_at_contact",
    "loading_duration_frames",
)


def _select_rom(col_name: str, series: np.ndarray) -> float:
    if col_name in ("hip_adduction_r", "hip_adduction_l"):
        return _rom_unwrapped(series)
    return _rom(series)


def _rom_robust(series: np.ndarray) -> float:
    """5th-to-95th-percentile ROM — ignores outlier frames from keypoint noise."""
    finite = series[np.isfinite(series)]
    if finite.size < 5:
        return float("nan")
    return float(np.percentile(finite, 95) - np.percentile(finite, 5))


def _gather_subject_paths(subjects: list[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for subj in subjects:
        out[subj] = sorted(KP_DIR.glob(f"{subj}_*.json"))
    return out


def _camera_pickles_for(subject: str) -> dict[str, Path]:
    cams: dict[str, Path] = {}
    for cam_dir in (LAB / subject / "VideoData").glob("Session*/Cam*"):
        pkl = cam_dir / "cameraIntrinsicsExtrinsics.pickle"
        if pkl.exists():
            cams[cam_dir.name] = pkl
    return cams


def precompute_calibrations(subjects: list[str]) -> dict[str, SubjectAnthro]:
    """Build per-subject anthropometric calibration once, cache by subject."""
    print("Calibrating per-subject anthropometrics...")
    cache: dict[str, SubjectAnthro] = {}
    paths_by_subj = _gather_subject_paths(subjects)
    for subj in subjects:
        meta_path = LAB / subj / "sessionMetadata.yaml"
        if not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text())
        h = float(meta["height_m"])
        cam_pkls = _camera_pickles_for(subj)
        if not paths_by_subj.get(subj) or not cam_pkls:
            continue
        cal = calibrate_subject(subj, paths_by_subj[subj], cam_pkls, h)
        cache[subj] = cal
        print(
            f"  {subj}: thigh={cal.thigh_m:.3f}m ({cal.thigh_source})  "
            f"shank={cal.shank_m:.3f}m ({cal.shank_source})  "
            f"foot={cal.foot_m:.3f}m ({cal.foot_source})"
        )
    return cache


def extract_features_v8(
    kp_path: Path, subject: str, cam: str, target_gt: str,
    calibration: SubjectAnthro,
    *,
    approach: str,  # "full_trial" or "event_anchored"
) -> dict[str, float]:
    series = load_couro_output(kp_path)
    cam_pkl = next((LAB / subject / "VideoData").glob(f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"))

    anthro_override = {
        "thigh_m": calibration.thigh_m,
        "shank_m": calibration.shank_m,
        "foot_m": calibration.foot_m,
        "torso_m": calibration.torso_m,
    }
    md = keypoints_to_motion_data(
        series,
        conf_threshold=CONF_THRESHOLD,        # v8: stricter
        subject_height_m=calibration.height_m,
        camera_pickle=cam_pkl,
        anthro_override=anthro_override,       # v8: per-subject
        smooth_window=SMOOTH_WINDOW,           # v8: Savgol smoothing
    )

    couro_col, coup1, coup2 = TARGETS[target_gt]
    target_series = md.column(couro_col)
    coup1_series = md.column(coup1)
    coup2_series = md.column(coup2)

    if approach == "full_trial":
        hip_center_y = series.xy[:, KP["hip_center"], 1]
        return {
            "target_rom_full": _select_rom(couro_col, target_series),
            "coup1_rom_full": _select_rom(coup1, coup1_series),
            "coup2_rom_full": _select_rom(coup2, coup2_series),
            "target_p95_minus_p05": _rom_robust(target_series),
            "loading_duration_proxy": float(np.sum(np.isfinite(target_series))),
            "knee_flex_diff_lr_full": abs(
                _rom(md.column("knee_angle_r")) - _rom(md.column("knee_angle_l"))
            ),
            "hip_y_range_px": float(
                np.nanmax(hip_center_y) - np.nanmin(hip_center_y)
            ),
        }

    # event_anchored: multi-signal impact detection then loading-window features
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


def gt_rom(subject: str, task: str, gt_col: str) -> float:
    p = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not p.exists():
        return float("nan")
    return _rom(parse_mot(p).column(gt_col))


def build_dataset(
    target_gt: str, cam: str, calibrations: dict[str, SubjectAnthro], approach: str,
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
        if subject not in calibrations:
            continue
        gt = gt_rom(subject, task, target_gt)
        if not np.isfinite(gt):
            continue
        try:
            feats = extract_features_v8(
                kp_path, subject, cam, target_gt, calibrations[subject],
                approach=approach,
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
    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y, dtype=np.float64),
        subjects, tasks, skipped,
    )


def slot_score(rmse: float, r: float) -> float:
    if r is None or rmse is None:
        return -1e9
    r_term = r if r > 0 else r * 2.0
    return r_term - 0.01 * rmse


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subjects = sorted({p.stem.split("_")[0] for p in KP_DIR.glob("*.json")})
    calibrations = precompute_calibrations(subjects)

    # Load v7 deploy stats for comparison
    v7_path = RESULTS_DIR / "deploy_ready_models.json"
    v7_stats: dict = {}
    if v7_path.exists():
        with open(v7_path) as f:
            d = json.load(f)
        for tgt, td in d.get("models", {}).items():
            v7_stats[tgt] = {
                v: {
                    "rmse": m["loso_cv_stats"]["rmse_deg"],
                    "r": m["loso_cv_stats"]["pearson_r"],
                    "approach": m.get("approach", "?"),
                }
                for v, m in td.items()
            }

    out: dict = {
        "version": "1.0",
        "produced_by": "harness.train_v8_enhanced",
        "produced_date": "2026-05-27",
        "description": (
            "v8 = v7 + three single-camera precision unlocks: "
            "(A) confidence-aware feature extraction "
            "(B) per-subject anthropometric calibration "
            "(D) Savgol-smoothed 2D keypoints"
        ),
        "v8_parameters": {
            "savgol_window": SMOOTH_WINDOW,
            "conf_threshold": CONF_THRESHOLD,
        },
        "training_dataset": {
            "name": "OpenCap LabValidation_withVideos (drop jumps)",
            "citation": "Uhlrich et al. 2023, PLOS Comp Bio 19(10): e1011462",
        },
        "calibrations": {
            s: {"thigh_m": c.thigh_m, "shank_m": c.shank_m, "foot_m": c.foot_m,
                "torso_m": c.torso_m, "height_m": c.height_m,
                "thigh_source": c.thigh_source, "shank_source": c.shank_source,
                "foot_source": c.foot_source}
            for s, c in calibrations.items()
        },
        "models": {},
    }

    print(
        f"\n{'TARGET':<20}{'VIEW':<22}{'WINNER':<16}"
        f"{'v8_RMSE':>10}{'v8_r':>7}{'v7_RMSE':>10}{'v7_r':>7}{'Δr':>7}"
    )
    print("-" * 100)
    deploy = 0; scale = 0; nope = 0
    for target_gt in TARGETS:
        out["models"][target_gt] = {}
        for cam in ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"]:
            view = VIEW_BUCKET_NAMES[cam]
            cands = []
            for approach in ("full_trial", "event_anchored"):
                X, y, subjects_, tasks_, skipped = build_dataset(
                    target_gt, cam, calibrations, approach,
                )
                if len(y) < 20:
                    continue
                cv = loso_cv(X, y, subjects_, alpha=10.0)
                full = fit_ridge(X, y, alpha=10.0)
                naive_pred = X[:, 0]
                naive_rmse = float(np.sqrt(np.mean((naive_pred - y) ** 2)))
                cands.append({
                    "approach": approach,
                    "n": int(len(y)),
                    "rmse": cv["rmse"],
                    "r": cv["pearson_r"],
                    "bias": cv["bias"],
                    "sd_diff": cv["sd_diff"],
                    "loa": cv["loa"],
                    "mae": cv["mae"],
                    "naive_rmse": naive_rmse,
                    "skipped": skipped,
                    "fit": full,
                    "features": FEATURE_NAMES_FULL if approach == "full_trial" else FEATURE_NAMES_EVENT,
                })
            if not cands:
                continue
            cands.sort(key=lambda c: slot_score(c["rmse"], c["r"]), reverse=True)
            best = cands[0]
            v7 = v7_stats.get(target_gt, {}).get(view, {})
            v7_r = v7.get("r", float("nan"))
            v7_rmse = v7.get("rmse", float("nan"))
            d_r = best["r"] - v7_r if np.isfinite(v7_r) else float("nan")

            r = best["r"]
            if r >= 0.30:
                v = "DEPLOY"; deploy += 1
            elif r >= 0.10:
                v = "deploy(modest)"; deploy += 1
            elif r > 0:
                v = "scale-only"; scale += 1
            else:
                v = "DO NOT SHIP"; nope += 1

            print(
                f"  {target_gt:<18}{view:<22}{best['approach']:<16}"
                f"{best['rmse']:>9.2f}°{best['r']:>+6.2f}"
                f"{v7_rmse:>9.2f}°{v7_r:>+6.2f}{d_r:>+6.2f}  {v}"
            )

            out["models"][target_gt][view] = {
                "camera_id": cam,
                "view_bucket": view,
                "approach": best["approach"],
                "target_metric": target_gt,
                "n_train_trials": best["n"],
                "n_subjects": len(set(subjects_)),
                "feature_names": list(best["features"]),
                "feature_mean": best["fit"].feature_mean.tolist(),
                "feature_std": best["fit"].feature_std.tolist(),
                "weights_zscored": best["fit"].weights.tolist(),
                "intercept": best["fit"].bias,
                "loso_cv_stats": {
                    "rmse_deg": best["rmse"],
                    "mae_deg": best["mae"],
                    "bias_deg": best["bias"],
                    "sd_of_difference_deg": best["sd_diff"],
                    "loa_95_lower_deg": best["loa"][0],
                    "loa_95_upper_deg": best["loa"][1],
                    "pearson_r": best["r"],
                },
                "naive_baseline_rmse": best["naive_rmse"],
                "v7_comparison": {
                    "v7_rmse_deg": v7_rmse,
                    "v7_pearson_r": v7_r,
                    "delta_r": d_r,
                },
            }
    print(f"\nv8 Deploy: {deploy}  Scale: {scale}  Do not ship: {nope}")

    out_path = RESULTS_DIR / "v8_deploy_ready_models.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
