"""Preliminary v12 trainer using SYNTHETIC ASPset 2D (no inference required).

Strategy: project ASPset 3D joints through their camera intrinsics+extrinsics
to get 2D pixel coords, add Gaussian noise calibrated to Couro's published
per-joint MPJPE (knees 9.1px, ankles 5.6px, hips 20.7px), then use those as
"synthetic Couro keypoints" for v9 phased feature extraction.

Purpose: quick proof-of-concept — does adding 17 ASPset subjects move r?

Limitations:
  * Synthetic noise is i.i.d. Gaussian — real RTMPose has temporal correlation
    and bias patterns we don't replicate
  * Per-joint MPJPE used as σ across ALL views (real noise varies by view)
  * Subjects with extreme camera positions may have synthetic-2D far from
    what real RTMPose would produce on the matching video

If this run shows r lifting, the real-RTMPose run after AWS inference will
almost certainly show a bigger lift. If it doesn't show a lift, we know
the dataset diversity isn't enough on its own and we need to investigate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.train_regression_poc import fit_ridge, loso_cv
    from harness.train_multi_metric import VIEW_BUCKET_NAMES
    from harness.train_v9_phased import (
        TARGETS, FEATURE_NAMES, build_dataset_v9, extract_v9_features,
    )
    from harness.aspset_loader import (
        ASPSET_KP, load_clip, load_camera, joint_angles_from_aspset,
        project_to_2d, aspset_to_halpe26_2d,
    )
    from harness.couro_keypoints import KP as HALPE_KP, CouroKeypointSeries
    from harness.dj_events import detect_dj_events_multi_signal, event_anchored_for_metric
    from harness.train_v9_phased import _phased_angles, _select_rom
    from harness.train_multi_metric import _rom
else:
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import VIEW_BUCKET_NAMES, _rom
    from .train_v9_phased import (
        TARGETS, FEATURE_NAMES, build_dataset_v9, _phased_angles, _select_rom,
    )
    from .aspset_loader import (
        ASPSET_KP, load_clip, load_camera, joint_angles_from_aspset,
        project_to_2d, aspset_to_halpe26_2d,
    )
    from .couro_keypoints import KP as HALPE_KP, CouroKeypointSeries
    from .dj_events import detect_dj_events_multi_signal, event_anchored_for_metric


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
ASPSET = DATA_ROOT / "aspset510" / "ASPset-510" / "trainval"
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


# Couro per-joint MPJPE in pixels (from the validation summary doc, March 2026)
COURO_MPJPE_PX = {
    "shoulders": 5.4,   # 4.9-5.9 range
    "knees": 9.1,
    "ankles": 5.7,
    "hips": 20.9,       # 20.7-21.1
    "head": 8.0,        # estimated
    "elbows_wrists": 7.0,
}

# Map Halpe26 idx → MPJPE bucket
HALPE_TO_MPJPE = {
    0: "head", 1: "head", 2: "head", 3: "head", 4: "head",
    5: "shoulders", 6: "shoulders",
    7: "elbows_wrists", 8: "elbows_wrists",
    9: "elbows_wrists", 10: "elbows_wrists",
    11: "hips", 12: "hips",
    13: "knees", 14: "knees",
    15: "ankles", 16: "ankles",
    17: "head", 18: "shoulders", 19: "hips",
    # foot keypoints — unavailable from ASPset, will stay NaN
    20: "ankles", 21: "ankles", 22: "ankles", 23: "ankles", 24: "ankles", 25: "ankles",
}


def classify_view(cam_yaw_deg: float) -> str:
    ay = abs(cam_yaw_deg)
    if ay < 30:
        return "front_center"
    elif ay < 60:
        return "front_oblique_left" if cam_yaw_deg < 0 else "front_oblique_right"
    elif ay < 120:
        return "side_left" if cam_yaw_deg < 0 else "side_right"
    elif ay < 150:
        return "front_oblique_left" if cam_yaw_deg < 0 else "front_oblique_right"
    else:
        return "front_center"  # rear collapsed into front-center sagittal bucket


def compute_yaw(subject: str, view: str) -> float | None:
    cam_path = ASPSET / "cameras" / subject / f"{subject}-{view}.json"
    if not cam_path.exists():
        return None
    cam = load_camera(cam_path)
    clip_paths = sorted((ASPSET / "joints_3d" / subject).glob("*.c3d"))
    if not clip_paths:
        return None
    clip = load_clip(clip_paths[0])
    subj_pos = np.nanmedian(clip.joints_3d[:, 16, :], axis=0)
    lhip = np.nanmedian(clip.joints_3d[:, 8, :], axis=0)
    rhip = np.nanmedian(clip.joints_3d[:, 2, :], axis=0)
    body_right = rhip - lhip
    body_right_norm = body_right / (np.linalg.norm(body_right) + 1e-9)
    body_forward = np.array([-body_right_norm[2], 0, body_right_norm[0]])
    R = cam.extrinsic_matrix[:3, :3]
    t = cam.extrinsic_matrix[:3, 3]
    cam_pos = -R.T @ t
    to_cam = cam_pos - subj_pos
    fc = float(np.dot(to_cam, body_forward))
    rc = float(np.dot(to_cam, body_right_norm))
    return float(np.degrees(np.arctan2(rc, fc)))


def synthetic_couro_series_from_aspset(
    clip, cam, *, noise_seed: int = 42,
) -> CouroKeypointSeries:
    """Project ASPset 3D to 2D and add Couro-calibrated noise → fake Couro series."""
    xy_2d_clean = project_to_2d(clip.joints_3d, cam)  # (T, 17, 2)
    halpe_2d_clean = aspset_to_halpe26_2d(xy_2d_clean)  # (T, 26, 2), NaN where unavailable

    rng = np.random.default_rng(noise_seed)
    halpe_2d_noisy = halpe_2d_clean.copy()
    for kp_idx in range(26):
        bucket = HALPE_TO_MPJPE[kp_idx]
        sigma = COURO_MPJPE_PX[bucket] / np.sqrt(2)  # per-axis std (MPJPE is L2)
        finite_mask = np.isfinite(halpe_2d_clean[:, kp_idx, 0])
        noise = rng.normal(0, sigma, size=(finite_mask.sum(), 2))
        halpe_2d_noisy[finite_mask, kp_idx, :] += noise

    # Confidence = 1.0 where available, 0.0 where NaN
    confidence = np.where(
        np.isfinite(halpe_2d_noisy[..., 0]), 1.0, 0.0,
    ).astype(np.float64)
    # Replace NaN with zero so downstream code doesn't choke
    halpe_2d_noisy = np.where(np.isfinite(halpe_2d_noisy), halpe_2d_noisy, 0.0)

    # Camera intrinsics implicit — use rough estimate for width/height
    cx, cy = cam.intrinsic_matrix[0, 2], cam.intrinsic_matrix[1, 2]
    width = int(round(cx * 2))
    height = int(round(cy * 2))

    return CouroKeypointSeries(
        time=np.arange(halpe_2d_noisy.shape[0]) / clip.fps,
        xy=halpe_2d_noisy,
        confidence=confidence,
        width=width, height=height, fps=clip.fps,
    )


# ASPset couplings — substitute angles ASPset CAN produce (no ankle_angle).
# This matches the spirit of the v9 TARGETS dict but stays within ASPset's keypoint set.
ASPSET_TARGETS = {
    "hip_flexion_r":    ("hip_flexion_r",    "knee_angle_r",    "lumbar_extension"),
    "hip_adduction_r":  ("hip_adduction_r",  "hip_adduction_l", "knee_angle_r"),
    "knee_angle_r":     ("knee_angle_r",     "hip_flexion_r",   "lumbar_extension"),
    "lumbar_extension": ("lumbar_extension", "hip_flexion_r",   "pelvis_tilt"),
}


def extract_v9_synthetic(
    clip, cam, target_gt: str,
) -> dict[str, float] | None:
    """Synthetic-2D version of v9 phased feature extraction for one ASPset clip."""
    if target_gt not in ASPSET_TARGETS:
        return None
    couro_col, coup1, coup2 = ASPSET_TARGETS[target_gt]

    # Compute joint angles directly from ASPset 3D — these are our "computed motion data"
    angles = joint_angles_from_aspset(clip)
    if couro_col not in angles:
        return None
    target_series = angles[couro_col]
    coup1_series = angles.get(coup1)
    coup2_series = angles.get(coup2)
    if coup1_series is None or coup2_series is None:
        return None

    # Build a fake CouroKeypointSeries with noisy 2D for event detection
    series = synthetic_couro_series_from_aspset(clip, cam)

    # Event detection: ASPset motions don't have clean DJ-style impacts.
    # Use hip-center vertical as a "loading proxy" — wherever the body is lowest.
    r_y = series.xy[:, HALPE_KP["R_ank"], 1]
    l_y = series.xy[:, HALPE_KP["L_ank"], 1]
    hip_y = series.xy[:, HALPE_KP["hip_center"], 1]
    events = detect_dj_events_multi_signal([r_y, l_y, hip_y], fps=series.fps)

    phased_target = _phased_angles(target_series, events, 5)
    phased_coup1 = _phased_angles(coup1_series, events, 5)
    phased_coup2 = _phased_angles(coup2_series, events, 5)
    ev_t = event_anchored_for_metric(target_series, events)

    if events.loading_window is not None:
        a, b = events.loading_window
        a = max(0, int(a)); b = min(series.confidence.shape[0], int(b))
        if b > a:
            mean_conf = float(np.nanmean(
                series.confidence[a:b, [HALPE_KP["R_hip"], HALPE_KP["R_kne"], HALPE_KP["R_ank"]]]
            ))
        else:
            mean_conf = float("nan")
    else:
        mean_conf = float("nan")

    feats = {
        "target_rom_full": _select_rom(couro_col, target_series),
        "target_rom_loading": ev_t["rom_loading"],
        "loading_duration_frames": ev_t["loading_duration_frames"],
        "knee_flex_diff_lr_full": abs(
            _rom(angles["knee_angle_r"]) - _rom(angles["knee_angle_l"])
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


def gt_rom_aspset(clip, target_gt: str) -> float:
    """GT ROM directly from ASPset 3D joint angles."""
    angles = joint_angles_from_aspset(clip)
    return float(angles[target_gt].max() - angles[target_gt].min())


def build_aspset_part(target_gt: str, view: str):
    """Collect all ASPset clips that match this (target, view) slot."""
    X_rows, y, subjects, sources, clip_ids = [], [], [], [], []
    for subj_dir in sorted((ASPSET / "joints_3d").iterdir()):
        subject = subj_dir.name
        for clip_path in sorted(subj_dir.glob("*.c3d")):
            clip_id = clip_path.stem.split("-", 1)[1]
            clip = load_clip(clip_path)
            for asp_view in ["left", "mid", "right"]:
                yaw = compute_yaw(subject, asp_view)
                if yaw is None or classify_view(yaw) != view:
                    continue
                cam_path = ASPSET / "cameras" / subject / f"{subject}-{asp_view}.json"
                cam = load_camera(cam_path)
                try:
                    feats = extract_v9_synthetic(clip, cam, target_gt)
                except Exception as e:
                    continue
                if feats is None:
                    continue
                vals = list(feats.values())
                if not all(np.isfinite(vals)):
                    continue
                gt = gt_rom_aspset(clip, target_gt)
                # Filter pathological hip_add ROMs (gymnastic clips where asin saturates)
                if target_gt == "hip_adduction_r" and gt > 90:
                    continue
                X_rows.append(vals); y.append(gt)
                subjects.append(f"aspset_{subject}")
                sources.append("aspset")
                clip_ids.append(f"{subject}-{clip_id}-{asp_view}")
    return (
        np.array(X_rows, dtype=np.float64) if X_rows else np.empty((0, 20)),
        np.array(y, dtype=np.float64) if y else np.empty(0),
        subjects, sources, clip_ids,
    )


def build_opencap_part_aspset_compat(target_gt: str, view: str):
    """Build OpenCap features with ASPset-compatible couplings so the feature
    space matches the synthetic ASPset features. Re-uses v9 phased extraction
    with the ASPSET_TARGETS coupling map.
    """
    from harness.train_v9_phased import extract_v9_features, FEATURE_NAMES as V9_FEAT
    from harness.train_v9_phased import build_dataset_v9
    cam_id = {v: k for k, v in VIEW_BUCKET_NAMES.items()}.get(view)
    if cam_id is None:
        return np.empty((0, 20)), np.empty(0), [], [], []

    # Monkey-patch TARGETS to use the ASPset-compatible coupling for OpenCap
    # feature extraction in this slot.
    import harness.train_v9_phased as v9
    original = v9.TARGETS
    v9.TARGETS = {**original, target_gt: ASPSET_TARGETS[target_gt]}
    try:
        X, y, subj, _, _ = build_dataset_v9(target_gt, cam_id)
    finally:
        v9.TARGETS = original
    sources = ["opencap"] * len(subj)
    subjects_prefixed = [f"opencap_{s}" for s in subj]
    return X, y, subjects_prefixed, sources, []


def main():
    print("=== v12 SYNTHETIC ASPset preliminary trainer ===")
    print()
    print("Combines OpenCap (real RTMPose 2D) + ASPset (synthetic 2D with Couro-calibrated noise)")
    print()
    targets_to_test = ["hip_flexion_r", "knee_angle_r", "lumbar_extension", "hip_adduction_r"]
    views = ["side_left", "front_oblique_left", "front_center", "front_oblique_right", "side_right"]
    print(f"{'TARGET':<22}{'VIEW':<22}{'n_oc':>6}{'n_asp':>7}{'v9_r':>7}{'comb_r':>8}{'Δr':>7}")
    print("-" * 80)

    # Load v9 baseline for comparison
    v9_r = {}
    v9_path = RESULTS_DIR / "v9_phased_all_metrics.json"
    if v9_path.exists():
        with open(v9_path) as f:
            d = json.load(f)
        for tgt, td in d["models"].items():
            v9_r[tgt] = {v: m["loso_cv_stats"]["pearson_r"] for v, m in td.items()}

    out = {"models": {}}
    for target in targets_to_test:
        out["models"][target] = {}
        for view in views:
            X_oc, y_oc, subj_oc, _, _ = build_opencap_part_aspset_compat(target, view)
            X_as, y_as, subj_as, _, _ = build_aspset_part(target, view)
            if len(y_oc) + len(y_as) < 20:
                continue
            X = np.vstack([X_oc, X_as]) if len(y_oc) and len(y_as) else (X_oc if len(y_oc) else X_as)
            y = np.concatenate([y_oc, y_as]) if len(y_oc) and len(y_as) else (y_oc if len(y_oc) else y_as)
            subjects = subj_oc + subj_as
            cv = loso_cv(X, y, subjects, alpha=10.0)
            baseline_r = v9_r.get(target, {}).get(view, float("nan"))
            d_r = cv["pearson_r"] - baseline_r if np.isfinite(baseline_r) else float("nan")
            print(
                f"  {target:<20}{view:<22}{len(y_oc):>6d}{len(y_as):>7d}"
                f"{baseline_r:>+6.2f}{cv['pearson_r']:>+7.2f}{d_r:>+6.2f}"
            )
            out["models"][target][view] = {
                "n_opencap": int(len(y_oc)),
                "n_aspset": int(len(y_as)),
                "n_total": int(len(y)),
                "rmse": cv["rmse"],
                "pearson_r": cv["pearson_r"],
                "baseline_v9_r": baseline_r,
                "delta_r": d_r,
            }

    out_path = RESULTS_DIR / "v12_synthetic_preliminary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
