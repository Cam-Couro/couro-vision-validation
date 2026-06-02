"""v12: Combined OpenCap + ASPset-510 regression trainer.

Adds 17 new subjects + 1,350 athletic-motion trials to our 9 OpenCap subjects.
Uses the v9 phased-features approach (the strongest single-method winner) on
both datasets, with appropriate masking for the metrics ASPset can't supply
(ankle dorsiflexion — no foot keypoints in 17-joint layout).

Pipeline:
  1. For each subject in {OpenCap, ASPset}:
       extract phased features + GT for each (target × view) slot
  2. Combine into one big LOSO-friendly dataset
  3. LOSO by subject — held-out subject can be from either source
  4. Compare per-slot r values to v9 (OpenCap-only) baseline

View-bucket assignment for ASPset cameras is dynamic per-clip — each clip
has its athlete-relative yaw computed from the camera extrinsics and the
pelvis position, then bucketed into front/front-oblique/side/rear-oblique/rear.

To be run AFTER ASPset videos are processed through Couro RTMPose. Currently
stubbed — extract_aspset_features() is a placeholder until keypoint JSONs exist.
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
        load_clip, load_camera, joint_angles_from_aspset,
    )
else:
    from .train_regression_poc import fit_ridge, loso_cv
    from .train_multi_metric import VIEW_BUCKET_NAMES
    from .train_v9_phased import (
        TARGETS, FEATURE_NAMES, build_dataset_v9, extract_v9_features,
    )
    from .aspset_loader import load_clip, load_camera, joint_angles_from_aspset


DATA_ROOT = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data").expanduser()
ASPSET = DATA_ROOT / "aspset510" / "ASPset-510" / "trainval"
ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_couro_keypoints"  # placeholder for Couro RTMPose outputs
RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


# Metrics ASPset can supply GT for (no foot/toe → no ankle)
ASPSET_VALID_TARGETS = {"hip_flexion_r", "hip_adduction_r", "knee_angle_r", "lumbar_extension"}


def classify_view_aspset(cam_yaw_deg: float) -> str:
    """Map ASPset's athlete-relative camera yaw to our 5-bucket view scheme.

    Camera yaw is measured in the athlete's frame:
      0° = camera in front of athlete
      ±90° = camera to athlete's right/left side
      ±180° = camera behind athlete
    """
    ay = abs(cam_yaw_deg)
    if ay < 30:
        return "front_center"
    elif ay < 60:
        return "front_oblique_left" if cam_yaw_deg < 0 else "front_oblique_right"
    elif ay < 120:
        return "side_left" if cam_yaw_deg < 0 else "side_right"
    elif ay < 150:
        # Rear-oblique — fold into front_oblique bucket for symmetry
        return "front_oblique_left" if cam_yaw_deg < 0 else "front_oblique_right"
    else:
        # Pure rear — fold into front_center (symmetric sagittal view)
        return "front_center"


def compute_aspset_camera_yaws(
    subject: str, view: str,
) -> float | None:
    """Compute the athlete-relative yaw for one (subject, view) — averaged
    across all clips for that subject."""
    cam_path = ASPSET / "cameras" / subject / f"{subject}-{view}.json"
    if not cam_path.exists():
        return None
    cam = load_camera(cam_path)

    # Get subject median position from any clip
    clip_paths = sorted((ASPSET / "joints_3d" / subject).glob("*.c3d"))
    if not clip_paths:
        return None
    clip = load_clip(clip_paths[0])

    # Subject pelvis median
    subj_pos = np.nanmedian(clip.joints_3d[:, 16, :], axis=0)
    # Body forward = perpendicular to L-R hip vector in horizontal plane
    lhip = np.nanmedian(clip.joints_3d[:, 8, :], axis=0)
    rhip = np.nanmedian(clip.joints_3d[:, 2, :], axis=0)
    body_right = rhip - lhip
    body_right_norm = body_right / (np.linalg.norm(body_right) + 1e-9)
    body_forward = np.array([-body_right_norm[2], 0, body_right_norm[0]])

    # Camera world position
    R = cam.extrinsic_matrix[:3, :3]
    t = cam.extrinsic_matrix[:3, 3]
    cam_pos = -R.T @ t

    # Vector from subject to camera in athlete frame
    to_cam = cam_pos - subj_pos
    forward_component = float(np.dot(to_cam, body_forward))
    right_component = float(np.dot(to_cam, body_right_norm))

    return float(np.degrees(np.arctan2(right_component, forward_component)))


def extract_aspset_features(
    clip_id: str, subject: str, view: str, target_gt: str,
) -> tuple[dict[str, float], float] | None:
    """Load Couro keypoint JSON for ASPset clip and compute v9 phased features.

    Expects: ASPSET_KEYPOINTS/{subject}-{clip_id}-{view}.json (produced by
    aspset_infer.py running inside Couro's RTMPose container on AWS).

    GT is computed from ASPset's 3D markers (.c3d file in the data dir).
    Returns (features_dict, gt_rom_degrees) or None.
    """
    from harness.couro_keypoints import load_couro_output
    from harness.dj_events import detect_dj_events_multi_signal, event_anchored_for_metric
    from harness.train_v9_phased import _phased_angles, _select_rom
    from harness.train_multi_metric import _rom

    # Map "front_oblique_left" → which ASPset view file to look for is non-trivial.
    # Caller already determined the asp_view (left/mid/right) → so we receive it directly.
    # Naming convention: {subject}-{clip_id}-{asp_view}.json
    kp_path = ASPSET_KEYPOINTS / f"{subject}-{clip_id}-{view}.json"
    c3d_path = ASPSET / "joints_3d" / subject / f"{subject}-{clip_id}.c3d"
    if not kp_path.exists() or not c3d_path.exists():
        return None

    # GT comes from the c3d file (3D markers → joint angles)
    clip = load_clip(c3d_path)
    angles = joint_angles_from_aspset(clip)
    if target_gt not in angles:
        return None
    gt = float(angles[target_gt].max() - angles[target_gt].min())
    if not np.isfinite(gt):
        return None
    # Filter pathological hip_adduction values where asin saturates on extreme poses
    if target_gt == "hip_adduction_r" and gt > 90:
        return None

    # Features from the Couro keypoint output (real RTMPose result)
    series = load_couro_output(kp_path)
    # Map ASPSET coupling for this target — uses only the angles ASPset can produce
    couro_col_map = {
        "hip_flexion_r":    ("hip_flexion_r",    "knee_angle_r",    "lumbar_extension"),
        "hip_adduction_r":  ("hip_adduction_r",  "hip_adduction_l", "knee_angle_r"),
        "knee_angle_r":     ("knee_angle_r",     "hip_flexion_r",   "lumbar_extension"),
        "lumbar_extension": ("lumbar_extension", "hip_flexion_r",   "pelvis_tilt"),
    }
    if target_gt not in couro_col_map:
        return None  # ASPset has no foot keypoints → can't do ankle_angle_r
    couro_col, coup1, coup2 = couro_col_map[target_gt]

    # Use Couro keypoints + anthropometric reconstruction to get joint angles.
    # ASPset cameras have intrinsics; we don't have a subject height in ASPset
    # metadata, so estimate it from the marker bounding box (head_top to ankle).
    h_top = np.nanmedian(clip.joints_3d[:, 12, :], axis=0)  # head_top
    ankle = np.nanmedian(clip.joints_3d[:, 0, :], axis=0)   # right_ankle (z+ = up typically)
    subject_height_m = float(np.linalg.norm(h_top - ankle) / 1000.0)
    if not (1.4 <= subject_height_m <= 2.2):
        subject_height_m = 1.75  # plausible default
    # Use ASPset camera JSON for proper anthropometric reconstruction
    from harness.couro_keypoints import keypoints_to_motion_data
    aspset_cam_path = ASPSET / "cameras" / subject / f"{subject}-{view}.json"
    if aspset_cam_path.exists():
        # Pass a synthesized "pickle-like" path via JSON-aware loader.
        # We pre-build a CameraModel here and inject via the override hook —
        # but keypoints_to_motion_data takes a path argument, so we adapt by
        # monkey-patching CameraModel.from_pickle to handle ASPset JSON.
        import harness.anthro_3d as _anthro
        original_from_pickle = _anthro.CameraModel.from_pickle
        def _aspset_aware_from_pickle(path, w, h):
            return _anthro.CameraModel.from_aspset_json(path, w, h)
        try:
            _anthro.CameraModel.from_pickle = staticmethod(_aspset_aware_from_pickle)
            md = keypoints_to_motion_data(
                series, conf_threshold=0.3,
                subject_height_m=subject_height_m,
                camera_pickle=aspset_cam_path,
            )
        finally:
            _anthro.CameraModel.from_pickle = original_from_pickle
    else:
        md = keypoints_to_motion_data(series, conf_threshold=0.3)

    target_series = md.column(couro_col)
    coup1_series = md.column(coup1)
    coup2_series = md.column(coup2)

    # Event detection from ankle/hip vertical
    from harness.couro_keypoints import KP
    r_y = series.xy[:, KP["R_ank"], 1]
    l_y = series.xy[:, KP["L_ank"], 1]
    hip_y = series.xy[:, KP["hip_center"], 1]
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
                series.confidence[a:b, [KP["R_hip"], KP["R_kne"], KP["R_ank"]]]
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
    return feats, gt


ASPSET_COMPATIBLE_TARGETS = {
    "hip_flexion_r":    ("hip_flexion_r",    "knee_angle_r",    "lumbar_extension"),
    "hip_adduction_r":  ("hip_adduction_r",  "hip_adduction_l", "knee_angle_r"),
    "knee_angle_r":     ("knee_angle_r",     "hip_flexion_r",   "lumbar_extension"),
    "lumbar_extension": ("lumbar_extension", "hip_flexion_r",   "pelvis_tilt"),
}


def build_combined_dataset(target_gt: str, view: str):
    """Build LOSO dataset combining OpenCap and ASPset trials for one slot."""
    if target_gt not in ASPSET_COMPATIBLE_TARGETS:
        return None  # ankle_angle_r not trainable from ASPset

    X_rows, y, subjects, sources = [], [], [], []

    # OpenCap contribution — use ASPset-compatible couplings so the feature
    # space matches across both datasets. Monkey-patch v9's TARGETS during
    # extraction.
    cam_id = {v: k for k, v in VIEW_BUCKET_NAMES.items()}.get(view)
    if cam_id is not None:
        import harness.train_v9_phased as v9
        original = v9.TARGETS
        v9.TARGETS = {**original, target_gt: ASPSET_COMPATIBLE_TARGETS[target_gt]}
        try:
            X_oc, y_oc, subj_oc, _, _ = build_dataset_v9(target_gt, cam_id)
        finally:
            v9.TARGETS = original
        for i in range(len(y_oc)):
            X_rows.append(X_oc[i].tolist())
            y.append(float(y_oc[i]))
            subjects.append(f"opencap_{subj_oc[i]}")
            sources.append("opencap")

    # ASPset contribution — only if target is in valid set
    if target_gt in ASPSET_VALID_TARGETS:
        for subj_dir in sorted((ASPSET / "joints_3d").iterdir()):
            subject = subj_dir.name
            for clip_path in sorted(subj_dir.glob("*.c3d")):
                clip_id = clip_path.stem.split("-", 1)[1]
                for asp_view in ["left", "mid", "right"]:
                    # Map this clip's view to our 5-bucket scheme
                    yaw = compute_aspset_camera_yaws(subject, asp_view)
                    if yaw is None:
                        continue
                    bucket = classify_view_aspset(yaw)
                    if bucket != view:
                        continue
                    # Pull v9-shaped features from real Couro keypoint JSON
                    result = extract_aspset_features(
                        clip_id, subject, asp_view, target_gt,
                    )
                    if result is None:
                        continue
                    feats, gt = result
                    vals = list(feats.values())
                    if not all(np.isfinite(vals)):
                        continue
                    X_rows.append(vals)
                    y.append(gt)
                    subjects.append(f"aspset_{subject}")
                    sources.append("aspset")

    if not X_rows:
        return None
    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y, dtype=np.float64),
        subjects, sources,
    )


def main():
    """Train combined OpenCap + ASPset regression per (metric × view).

    Requires that Couro RTMPose has already been run on the ASPset videos and
    the resulting per-clip keypoint JSONs are saved to
    ASPSET_KEYPOINTS/{subject}-{clip}-{view}.json
    """
    import json
    from harness.train_regression_poc import fit_ridge, loso_cv

    if not ASPSET_KEYPOINTS.exists() or not list(ASPSET_KEYPOINTS.glob("*.json")):
        print(f"ERROR: ASPset keypoint JSONs not found in {ASPSET_KEYPOINTS}")
        print("Run RTMPose inference first (see harness/aspset_infer_aws.sh)")
        return

    # Load v9 deploy_ready_models for baseline comparison
    v9_baseline = {}
    v9_path = RESULTS_DIR / "deploy_ready_models.json"
    if v9_path.exists():
        with open(v9_path) as f:
            d = json.load(f)
        for tgt, td in d.get("models", {}).items():
            v9_baseline[tgt] = {v: m["loso_cv_stats"]["pearson_r"] for v, m in td.items()}

    views = ["side_left", "front_oblique_left", "front_center",
             "front_oblique_right", "side_right"]
    targets = ["hip_flexion_r", "hip_adduction_r", "knee_angle_r", "lumbar_extension"]

    out = {
        "version": "1.0",
        "produced_by": "harness.train_v12_combined",
        "produced_date": "2026-05-27",
        "approach": "v12 — OpenCap (9 subj) + ASPset-510 (17 subj) combined v9-style regression",
        "models": {},
    }

    print(f"{'TARGET':<22}{'VIEW':<22}{'n_oc':>6}{'n_asp':>7}{'n_tot':>7}"
          f"{'v9_r':>7}{'v12_r':>8}{'Δr':>7}")
    print("-" * 86)

    for target in targets:
        out["models"][target] = {}
        for view in views:
            result = build_combined_dataset(target, view)
            if result is None:
                continue
            X, y, subjects, sources = result
            if len(y) < 20:
                continue
            n_oc = sum(1 for s in sources if s == "opencap")
            n_asp = sum(1 for s in sources if s == "aspset")
            cv = loso_cv(X, y, subjects, alpha=10.0)
            fit = fit_ridge(X, y, alpha=10.0)
            baseline = v9_baseline.get(target, {}).get(view, float("nan"))
            d_r = cv["pearson_r"] - baseline if np.isfinite(baseline) else float("nan")
            print(f"  {target:<20}{view:<22}{n_oc:>6d}{n_asp:>7d}{len(y):>7d}"
                  f"{baseline:>+6.2f}{cv['pearson_r']:>+7.2f}{d_r:>+6.2f}")
            out["models"][target][view] = {
                "n_opencap": n_oc,
                "n_aspset": n_asp,
                "n_total": int(len(y)),
                "n_subjects": len(set(subjects)),
                "feature_mean": fit.feature_mean.tolist(),
                "feature_std": fit.feature_std.tolist(),
                "weights_zscored": fit.weights.tolist(),
                "intercept": fit.bias,
                "loso_cv_stats": {
                    "rmse_deg": cv["rmse"],
                    "pearson_r": cv["pearson_r"],
                    "bias_deg": cv["bias"],
                    "sd_of_difference_deg": cv["sd_diff"],
                },
                "v9_baseline_r": baseline,
                "delta_r": d_r,
            }

    out_path = RESULTS_DIR / "v12_combined_models.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
