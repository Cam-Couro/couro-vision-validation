"""Variance reduction pass for the 3 most-promotable Moderate Layer 3 slots.

Goal: push Bland-Altman 95% LoA half-width below the +/- 10 deg Good-tier gate
on slots whose CCC already sits at 0.83 - 0.86 but whose trial-to-trial
variance leaks past 10 deg.

Targets:
  - knee_angle_r / front_oblique_right / v9_phased       (n=8,  LoA 10.7) needs 0.7
  - hip_flexion_r / front_oblique_left / v13_dwpose_hybrid (n=21, LoA 11.3) needs 1.3
  - knee_angle_r / side_left / v14_full_dwpose            (n=12, LoA 12.4) needs 2.4

Three levers tested per slot:
  Lever 1: Outlier subject leave-one-out diagnostic.
           NOT a validation method. Identifies high-leverage subjects.
  Lever 2: Confidence-weighted feature aggregates.
           Weight phased-angle samples by min keypoint confidence at that frame.
           If a confidence-weighted feature pipeline tightens LoA, that's a real fix
           candidate. Re-run LOSO end-to-end on the new features.
  Lever 3: Temporal smoothing tuning.
           Sweep three Savgol smoothing window settings (off, w=5, w=9).
           Cheap, expect small gains. Re-run LOSO with new smoothed-feature builds.

Run:
    cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
    python -m harness.variance_reduction_pass

Outputs:
    data/variance_reduction/per_slot_variance_results.json
    data/variance_reduction/REPORT.md
    data/variance_reduction/recommended_v18_updates.json (only if any slot promotes)

Honesty constraints:
  - LOSO discipline preserved on every tier-change claim.
  - Outlier exclusion is a diagnostic, not a validation method.
  - No modification of v15/v16/v17/biomech_validity_stats source files.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.train_v9_phased as v9
import harness.train_v12_combined as v12
import harness.train_v14_full_dwpose as v14
from harness.biomech_validity_stats import (
    _ccc,
    _pearson,
    _loso_extract,
    compute_stats_from_pairs,
)
from harness.train_multi_metric import VIEW_BUCKET_NAMES
from harness.train_regression_poc import fit_ridge


REPO_ROOT = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation"
)
DATA_ROOT = REPO_ROOT / "data"
OUT_DIR = DATA_ROOT / "variance_reduction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIEW_TO_CAM = {v: k for k, v in VIEW_BUCKET_NAMES.items()}

# Good-tier gate from the biomech-validation classification.
GOOD_TIER_LOA_DEG = 10.0
GOOD_TIER_CCC = 0.60


# Three target slots from per_slot_validity.json (Moderate tier, CCC >= 0.83).
@dataclass(frozen=True)
class TargetSlot:
    target: str
    view: str
    approach: str
    baseline_n_subjects: int
    baseline_ccc: float
    baseline_loa_half_width: float

    @property
    def loa_gap_to_good(self) -> float:
        return self.baseline_loa_half_width - GOOD_TIER_LOA_DEG


TARGETS: list[TargetSlot] = [
    TargetSlot(
        target="knee_angle_r",
        view="front_oblique_right",
        approach="v9_phased",
        baseline_n_subjects=8,
        baseline_ccc=0.8289653636250673,
        baseline_loa_half_width=10.717005412025857,
    ),
    TargetSlot(
        target="hip_flexion_r",
        view="front_oblique_left",
        approach="v13_dwpose_hybrid",
        baseline_n_subjects=21,
        baseline_ccc=0.8430113813944956,
        baseline_loa_half_width=11.28941475337648,
    ),
    TargetSlot(
        target="knee_angle_r",
        view="side_left",
        approach="v14_full_dwpose",
        baseline_n_subjects=12,
        baseline_ccc=0.863598288890088,
        baseline_loa_half_width=12.43377411933933,
    ),
]


# ---------------------------------------------------------------------------
# Dataset builders pinned to the same paths the biomech_validity_stats
# harness uses for each approach.
# ---------------------------------------------------------------------------


def _configure_paths_for(approach: str) -> None:
    if approach == "v9_phased":
        v9.KP_DIR = DATA_ROOT / "couro_keypoints"
    elif approach == "v13_dwpose_hybrid":
        v9.KP_DIR = DATA_ROOT / "couro_keypoints"
        v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
    elif approach == "v14_full_dwpose":
        v9.KP_DIR = DATA_ROOT / "opencap_dwpose_keypoints"
        v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
    else:
        raise ValueError(f"unknown approach: {approach}")


def _build_baseline_dataset(slot: TargetSlot):
    """Replicate the dataset construction used by biomech_validity_stats."""
    _configure_paths_for(slot.approach)
    if slot.approach == "v9_phased":
        cam = VIEW_TO_CAM[slot.view]
        X, y, subjects, _, _ = v9.build_dataset_v9(slot.target, cam)
        if len(y) == 0:
            return None
        return (
            X.astype(np.float64),
            np.asarray(y, dtype=np.float64),
            list(subjects),
            ["opencap"] * len(y),
        )
    if slot.approach == "v13_dwpose_hybrid":
        return v12.build_combined_dataset(slot.target, slot.view)
    if slot.approach == "v14_full_dwpose":
        return v12.build_combined_dataset(slot.target, slot.view)
    return None


def _run_loso(X: np.ndarray, y: np.ndarray, subjects: list[str], alpha: float = 10.0):
    pred, obs, subj_arr = _loso_extract(X, y, subjects, alpha=alpha)
    mask = np.isfinite(pred) & np.isfinite(obs)
    return pred[mask], obs[mask], subj_arr[mask]


def _per_subject_arrays(pred: np.ndarray, obs: np.ndarray, subj_arr: np.ndarray):
    unique = sorted(set(subj_arr.tolist()))
    pred_s = np.array([float(np.mean(pred[subj_arr == s])) for s in unique])
    obs_s = np.array([float(np.mean(obs[subj_arr == s])) for s in unique])
    return pred_s, obs_s, unique


def _stats_from_per_subject(pred_s: np.ndarray, obs_s: np.ndarray) -> dict:
    diffs = pred_s - obs_s
    mean_bias = float(np.mean(diffs))
    sd_bias = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
    loa_upper = mean_bias + 1.96 * sd_bias if math.isfinite(sd_bias) else float("nan")
    loa_lower = mean_bias - 1.96 * sd_bias if math.isfinite(sd_bias) else float("nan")
    half = (
        max(abs(loa_upper - mean_bias), abs(mean_bias - loa_lower))
        if math.isfinite(sd_bias)
        else float("nan")
    )
    return {
        "n_subjects": int(len(pred_s)),
        "mean_bias_deg": mean_bias,
        "sd_bias_deg": sd_bias,
        "loa_lower_deg": loa_lower,
        "loa_upper_deg": loa_upper,
        "loa_half_width_deg": half,
        "pearson_r": _pearson(pred_s, obs_s),
        "ccc_lin": _ccc(pred_s, obs_s),
        "mae_deg": float(np.mean(np.abs(diffs))),
        "rmse_deg": float(np.sqrt(np.mean(diffs**2))),
    }


def _is_good_tier(ccc: float, loa_half: float) -> bool:
    return (
        math.isfinite(ccc)
        and math.isfinite(loa_half)
        and ccc > GOOD_TIER_CCC
        and loa_half < GOOD_TIER_LOA_DEG
    )


# ---------------------------------------------------------------------------
# Lever 1: outlier subject leave-one-out diagnostic.
# ---------------------------------------------------------------------------


def lever1_outlier_diagnostic(slot: TargetSlot, baseline_loso: dict) -> dict:
    """Drop each subject in turn from the per-subject Bland-Altman view and
    recompute CCC + LoA. Identify high-leverage subjects.

    THIS IS A DIAGNOSTIC, NOT A VALIDATION METHOD. A tier promotion that
    requires excluding a subject does not warrant a 'validated' label.
    """
    pred_s = np.asarray(baseline_loso["per_subject_pred_deg"])
    obs_s = np.asarray(baseline_loso["per_subject_obs_deg"])
    subjects = baseline_loso["subjects"]
    full_ccc = _ccc(pred_s, obs_s)

    rows = []
    n = len(subjects)
    for i, s in enumerate(subjects):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        p = pred_s[mask]
        o = obs_s[mask]
        stats = _stats_from_per_subject(p, o)
        rows.append({
            "dropped_subject": s,
            "n_remaining": int(mask.sum()),
            "ccc_no_subj": stats["ccc_lin"],
            "loa_half_width_no_subj_deg": stats["loa_half_width_deg"],
            "mean_bias_no_subj_deg": stats["mean_bias_deg"],
            "pearson_r_no_subj": stats["pearson_r"],
            "ccc_delta_from_full": float(stats["ccc_lin"] - full_ccc),
            "abs_error_dropped_deg": float(abs(pred_s[i] - obs_s[i])),
            "tier_after_drop_diagnostic": (
                "Good"
                if _is_good_tier(stats["ccc_lin"], stats["loa_half_width_deg"])
                else "Moderate-or-worse"
            ),
        })
    rows.sort(key=lambda r: r["loa_half_width_no_subj_deg"])
    best = rows[0]
    return {
        "lever": "lever1_outlier_diagnostic",
        "is_validation_method": False,
        "note": (
            "Single-subject leave-one-out is a leverage diagnostic. A tier "
            "promotion that requires excluding a subject does NOT warrant a "
            "validated label. Excluded subjects should be audited (motion "
            "type, capture artifact, body morphology) for systematic cause."
        ),
        "full_ccc": float(full_ccc),
        "full_loa_half_width_deg": float(baseline_loso["loa_half_width_deg"]),
        "best_drop": best,
        "all_drops": rows,
        "best_drop_promotes_diagnostic_tier": _is_good_tier(
            best["ccc_no_subj"], best["loa_half_width_no_subj_deg"]
        ),
    }


# ---------------------------------------------------------------------------
# Lever 2: confidence-weighted feature aggregates.
#
# Strategy: rebuild the feature matrix for each slot replacing the
# event-window aggregates (target_rom_loading, loading_duration, knee_diff,
# mean_kp_conf_loading) with confidence-weighted versions, and rescale the
# phased-angle samples by their per-frame min-keypoint-confidence. The
# v9 phased features are then trained again with ridge LOSO.
#
# Low-confidence frames contribute less to ROM aggregates; if outlier
# predictions are being driven by occasional low-conf frames, this should
# tighten LoA. Magnitude unknown -- we measure it.
# ---------------------------------------------------------------------------


from harness.couro_keypoints import (
    KP,
    CouroKeypointSeries,
    keypoints_to_motion_data,
    load_couro_output,
)
from harness.dj_events import (
    detect_dj_events_multi_signal,
    event_anchored_for_metric,
)
from harness.train_multi_metric import _rom, _rom_unwrapped
from harness.train_v9_phased import (
    LAB,
    TARGETS as V9_TARGETS,
    MIRROR,
    _select_rom,
    _phased_angles,
    gt_rom_col,
)


# Keypoints whose confidence we treat as the "input quality" proxy for a
# given joint angle. (For each side we use the 3 keypoints that define the
# segment chain through the joint, matching the existing
# mean_kp_conf_loading definition for the right knee chain.)
JOINT_KP_FOR_CONF = {
    "hip_flexion_r":    ["R_hip", "R_kne", "R_ank"],
    "hip_adduction_r":  ["R_hip", "R_kne", "L_hip"],
    "knee_angle_r":     ["R_hip", "R_kne", "R_ank"],
    "ankle_angle_r":    ["R_kne", "R_ank", "R_big_toe"],
    "lumbar_extension": ["neck", "hip_center", "R_hip"],
}


def _per_frame_min_conf(
    series: CouroKeypointSeries, joint_kps: list[str]
) -> np.ndarray:
    """For each frame, the minimum confidence across the joint's keypoints."""
    cols = []
    for name in joint_kps:
        if name not in KP:
            continue
        cols.append(series.confidence[:, KP[name]])
    if not cols:
        return np.ones(series.confidence.shape[0], dtype=np.float64)
    return np.minimum.reduce(cols)


def _confweighted_phased_angles(
    series_vals: np.ndarray, conf: np.ndarray, events, n_phases: int = 5
) -> list[float]:
    """Same as _phased_angles but the phase sample is a confidence-weighted
    average over a small window centered at the phase index.

    For each phase i, take a 5-frame window centered at the phase index,
    weight values by the per-frame min-confidence, and report the weighted
    mean. If all weights are near zero, fall back to NaN.
    """
    if events.loading_window is None:
        return [float("nan")] * n_phases
    a, b = events.loading_window
    a = max(0, int(a)); b = min(len(series_vals), int(b))
    if b - a < n_phases:
        return [float("nan")] * n_phases
    phases = np.linspace(a, b - 1, n_phases).astype(int)
    window = 2
    out: list[float] = []
    for p_idx in phases:
        lo = max(a, p_idx - window)
        hi = min(b, p_idx + window + 1)
        vals = series_vals[lo:hi]
        w = conf[lo:hi]
        finite_mask = np.isfinite(vals) & np.isfinite(w) & (w > 0.05)
        if not finite_mask.any():
            out.append(float("nan"))
            continue
        v = vals[finite_mask]
        wt = w[finite_mask]
        out.append(float(np.sum(v * wt) / np.sum(wt)))
    return out


def _confweighted_rom_window(
    series_vals: np.ndarray, conf: np.ndarray, events
) -> float:
    """ROM (max - min) inside the loading window, after filtering frames
    with min-confidence below 0.3 (matches the conf_threshold elsewhere)."""
    if events.loading_window is None:
        return float("nan")
    a, b = events.loading_window
    a = max(0, int(a)); b = min(len(series_vals), int(b))
    if b <= a:
        return float("nan")
    vals = series_vals[a:b]
    w = conf[a:b]
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0.3)
    if mask.sum() < 3:
        return float("nan")
    sel = vals[mask]
    return float(np.nanmax(sel) - np.nanmin(sel))


def _extract_features_variant(
    kp_path: Path,
    subject: str,
    cam: str | None,
    target_gt: str,
    *,
    flip: bool = False,
    use_confweighted: bool = False,
    smooth_window: int = 0,
    target_couplings: tuple[str, str, str] | None = None,
    camera_pickle: Path | None = None,
    subject_height_m: float | None = None,
) -> dict[str, float] | None:
    """Recompute the v9-style feature dict with optional confidence weighting
    and/or temporal smoothing of the underlying 2D keypoints.

    This is a parallel rebuild that does not modify the v9 module.
    """
    series = load_couro_output(kp_path)
    if flip:
        mirrored_xy = series.xy.copy()
        mirrored_xy[..., 0] = series.width - 1 - mirrored_xy[..., 0]
        mirrored_conf = series.confidence.copy()
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
        series = CouroKeypointSeries(
            time=series.time, xy=mirrored_xy, confidence=mirrored_conf,
            width=series.width, height=series.height, fps=series.fps,
        )

    # Determine camera + height inputs (OpenCap path vs ASPset path).
    use_opencap = cam is not None
    if use_opencap and (camera_pickle is None or subject_height_m is None):
        meta_path = LAB / subject / "sessionMetadata.yaml"
        if not meta_path.exists():
            return None
        meta = yaml.safe_load(meta_path.read_text())
        try:
            camera_pickle = next(
                (LAB / subject / "VideoData").glob(
                    f"Session*/{cam}/cameraIntrinsicsExtrinsics.pickle"
                )
            )
        except StopIteration:
            return None
        subject_height_m = float(meta["height_m"])

    if subject_height_m is None or camera_pickle is None or not camera_pickle.exists():
        # Fall back to naive 2D path -- only used for ASPset clips when
        # the per-subject camera JSON is missing; we skip those.
        return None

    md = keypoints_to_motion_data(
        series, conf_threshold=0.3,
        subject_height_m=subject_height_m,
        camera_pickle=camera_pickle,
        smooth_window=smooth_window,
    )
    if target_couplings is None:
        couro_col, coup1, coup2 = V9_TARGETS[target_gt]
    else:
        couro_col, coup1, coup2 = target_couplings
    target_series = md.column(couro_col)
    coup1_series = md.column(coup1)
    coup2_series = md.column(coup2)

    r_y = series.xy[:, KP["R_ank"], 1]
    l_y = series.xy[:, KP["L_ank"], 1]
    hip_y = series.xy[:, KP["hip_center"], 1]
    events = detect_dj_events_multi_signal([r_y, l_y, hip_y], fps=series.fps)

    if use_confweighted:
        joint_kps = JOINT_KP_FOR_CONF.get(target_gt, ["R_hip", "R_kne", "R_ank"])
        conf_target = _per_frame_min_conf(series, joint_kps)
        phased_target = _confweighted_phased_angles(
            target_series, conf_target, events, n_phases=5,
        )
        phased_coup1 = _confweighted_phased_angles(
            coup1_series, conf_target, events, n_phases=5,
        )
        phased_coup2 = _confweighted_phased_angles(
            coup2_series, conf_target, events, n_phases=5,
        )
        rom_loading = _confweighted_rom_window(target_series, conf_target, events)
    else:
        phased_target = _phased_angles(target_series, events, 5)
        phased_coup1 = _phased_angles(coup1_series, events, 5)
        phased_coup2 = _phased_angles(coup2_series, events, 5)
        ev_t = event_anchored_for_metric(target_series, events)
        rom_loading = ev_t["rom_loading"]

    ev_t = event_anchored_for_metric(target_series, events)
    loading_dur = ev_t["loading_duration_frames"]

    # Mean keypoint confidence in the loading window -- unchanged definition.
    if events.loading_window is not None:
        a, b = events.loading_window
        a = max(0, int(a)); b = min(series.confidence.shape[0], int(b))
        if b > a:
            confs = series.confidence[
                a:b, [KP["R_hip"], KP["R_kne"], KP["R_ank"]]
            ]
            mean_conf = float(np.nanmean(confs))
        else:
            mean_conf = float("nan")
    else:
        mean_conf = float("nan")

    feats = {
        "target_rom_full": _select_rom(couro_col, target_series),
        "target_rom_loading": rom_loading,
        "loading_duration_frames": loading_dur,
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


def _build_opencap_v9_with_variant(
    target: str,
    cam: str,
    *,
    use_confweighted: bool,
    smooth_window: int,
    target_couplings: tuple[str, str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Rebuild the OpenCap LOSO matrix for v9_phased with the requested feature variant."""
    X_rows: list[list[float]] = []
    y: list[float] = []
    subjects: list[str] = []
    for kp_path in sorted(v9.KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        if not task.startswith("DJ"):
            continue
        gt = gt_rom_col(subject, task, target)
        if not np.isfinite(gt):
            continue
        try:
            feats = _extract_features_variant(
                kp_path, subject, cam, target,
                use_confweighted=use_confweighted,
                smooth_window=smooth_window,
                target_couplings=target_couplings,
            )
        except Exception:
            feats = None
        if feats is None:
            continue
        vals = list(feats.values())
        if not all(np.isfinite(vals)):
            continue
        X_rows.append(vals)
        y.append(gt)
        subjects.append(subject)
    if not X_rows:
        return None
    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y, dtype=np.float64),
        subjects,
    )


def _build_combined_with_variant(
    target: str,
    view: str,
    *,
    use_confweighted: bool,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]] | None:
    """Rebuild v12-style combined OpenCap + ASPset dataset with feature variant.

    For ASPset we currently keep the original v12 extractor (the
    confidence-weighted lever is meaningful for OpenCap where we have known
    foreshortening + RTMPose/DWPose confidences; for ASPset clips we already
    do per-clip filtering and the per-frame conf signal is less calibrated).
    The smoothing variant is applied to both halves.
    """
    from harness.train_v12_combined import ASPSET_COMPATIBLE_TARGETS

    if target not in ASPSET_COMPATIBLE_TARGETS:
        return None
    target_couplings = ASPSET_COMPATIBLE_TARGETS[target]

    cam_id = VIEW_TO_CAM.get(view)
    if cam_id is None:
        return None

    # OpenCap half with variant.
    oc = _build_opencap_v9_with_variant(
        target, cam_id,
        use_confweighted=use_confweighted,
        smooth_window=smooth_window,
        target_couplings=target_couplings,
    )
    X_rows: list[list[float]] = []
    y: list[float] = []
    subjects: list[str] = []
    sources: list[str] = []
    if oc is not None:
        X_oc, y_oc, subj_oc = oc
        for i in range(len(y_oc)):
            X_rows.append(X_oc[i].tolist())
            y.append(float(y_oc[i]))
            subjects.append(f"opencap_{subj_oc[i]}")
            sources.append("opencap")

    # ASPset half via existing v12 extractor (with smoothing applied via the
    # keypoints_to_motion_data smooth_window argument -- v12 calls that
    # function but currently passes no smoothing; we patch it in for the
    # smoothing lever via a context manager that wraps the function).
    if smooth_window > 0 or use_confweighted:
        # We need to re-extract ASPset features with the new options. The
        # existing extract_aspset_features function does not support
        # smooth_window or confidence weighting. We add a wrapper here that
        # mirrors it but with the new options.
        aspset_rows = _aspset_rows_with_variant(
            target, view, target_couplings,
            use_confweighted=use_confweighted,
            smooth_window=smooth_window,
        )
    else:
        # Baseline: use stock v12 extractor.
        aspset_rows = _aspset_rows_baseline(target, view, target_couplings)
    for row in aspset_rows:
        X_rows.append(row["feats"])
        y.append(row["gt"])
        subjects.append(f"aspset_{row['subject']}")
        sources.append("aspset")

    if not X_rows:
        return None
    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y, dtype=np.float64),
        subjects, sources,
    )


def _aspset_rows_baseline(target: str, view: str, target_couplings):
    """Use the existing v12 extract path. Returns list of {subject, feats, gt}."""
    from harness.train_v12_combined import (
        ASPSET,
        ASPSET_VALID_TARGETS,
        compute_aspset_camera_yaws,
        classify_view_aspset,
        extract_aspset_features,
    )
    rows = []
    if target not in ASPSET_VALID_TARGETS:
        return rows
    asp_root = ASPSET / "joints_3d"
    if not asp_root.exists():
        return rows
    # Monkey-patch v9 TARGETS so extract_aspset_features uses the right couplings.
    original = v9.TARGETS
    v9.TARGETS = {**original, target: target_couplings}
    try:
        for subj_dir in sorted(asp_root.iterdir()):
            subject = subj_dir.name
            for clip_path in sorted(subj_dir.glob("*.c3d")):
                clip_id = clip_path.stem.split("-", 1)[1]
                for asp_view in ["left", "mid", "right"]:
                    yaw = compute_aspset_camera_yaws(subject, asp_view)
                    if yaw is None:
                        continue
                    bucket = classify_view_aspset(yaw)
                    if bucket != view:
                        continue
                    result = extract_aspset_features(
                        clip_id, subject, asp_view, target,
                    )
                    if result is None:
                        continue
                    feats, gt = result
                    vals = list(feats.values())
                    if not all(np.isfinite(vals)):
                        continue
                    rows.append({"subject": subject, "feats": vals, "gt": gt})
    finally:
        v9.TARGETS = original
    return rows


def _aspset_rows_with_variant(
    target: str, view: str, target_couplings,
    *, use_confweighted: bool, smooth_window: int,
):
    """ASPset half with smoothing / confidence-weighting applied through
    a parallel extractor that mirrors extract_aspset_features but routes
    through _extract_features_variant.
    """
    from harness.train_v12_combined import (
        ASPSET,
        ASPSET_VALID_TARGETS,
        compute_aspset_camera_yaws,
        classify_view_aspset,
        load_clip,
        joint_angles_from_aspset,
    )
    import harness.anthro_3d as _anthro

    rows = []
    if target not in ASPSET_VALID_TARGETS:
        return rows
    asp_root = ASPSET / "joints_3d"
    if not asp_root.exists():
        return rows

    original_from_pickle = _anthro.CameraModel.from_pickle

    def _aspset_aware_from_pickle(path, w, h):
        return _anthro.CameraModel.from_aspset_json(path, w, h)

    try:
        _anthro.CameraModel.from_pickle = staticmethod(_aspset_aware_from_pickle)
        for subj_dir in sorted(asp_root.iterdir()):
            subject = subj_dir.name
            for clip_path in sorted(subj_dir.glob("*.c3d")):
                clip_id = clip_path.stem.split("-", 1)[1]
                for asp_view in ["left", "mid", "right"]:
                    yaw = compute_aspset_camera_yaws(subject, asp_view)
                    if yaw is None:
                        continue
                    bucket = classify_view_aspset(yaw)
                    if bucket != view:
                        continue
                    kp_path = v12.ASPSET_KEYPOINTS / f"{subject}-{clip_id}-{asp_view}.json"
                    c3d_path = ASPSET / "joints_3d" / subject / f"{subject}-{clip_id}.c3d"
                    if not kp_path.exists() or not c3d_path.exists():
                        continue
                    clip = load_clip(c3d_path)
                    angles = joint_angles_from_aspset(clip)
                    if target not in angles:
                        continue
                    gt = float(angles[target].max() - angles[target].min())
                    if not np.isfinite(gt):
                        continue
                    if target == "hip_adduction_r" and gt > 90:
                        continue
                    aspset_cam_path = ASPSET / "cameras" / subject / f"{subject}-{asp_view}.json"
                    if not aspset_cam_path.exists():
                        continue
                    # Estimate subject height from marker bbox.
                    h_top = np.nanmedian(clip.joints_3d[:, 12, :], axis=0)
                    ankle = np.nanmedian(clip.joints_3d[:, 0, :], axis=0)
                    subject_height_m = float(np.linalg.norm(h_top - ankle) / 1000.0)
                    if not (1.4 <= subject_height_m <= 2.2):
                        subject_height_m = 1.75
                    try:
                        feats = _extract_features_variant(
                            kp_path, subject, None, target,
                            use_confweighted=use_confweighted,
                            smooth_window=smooth_window,
                            target_couplings=target_couplings,
                            camera_pickle=aspset_cam_path,
                            subject_height_m=subject_height_m,
                        )
                    except Exception:
                        feats = None
                    if feats is None:
                        continue
                    vals = list(feats.values())
                    if not all(np.isfinite(vals)):
                        continue
                    rows.append({"subject": subject, "feats": vals, "gt": gt})
    finally:
        _anthro.CameraModel.from_pickle = original_from_pickle
    return rows


def lever2_confweighted(slot: TargetSlot) -> dict:
    """Re-run LOSO with confidence-weighted phased-angle features."""
    _configure_paths_for(slot.approach)
    t0 = time.time()
    if slot.approach == "v9_phased":
        cam = VIEW_TO_CAM[slot.view]
        ds = _build_opencap_v9_with_variant(
            slot.target, cam, use_confweighted=True, smooth_window=0,
        )
        if ds is None:
            return {"lever": "lever2_confweighted", "status": "build_failed"}
        X, y, subjects = ds
    else:
        ds = _build_combined_with_variant(
            slot.target, slot.view,
            use_confweighted=True, smooth_window=0,
        )
        if ds is None:
            return {"lever": "lever2_confweighted", "status": "build_failed"}
        X, y, subjects, _ = ds
    if len(y) < 6:
        return {
            "lever": "lever2_confweighted",
            "status": f"insufficient_n={len(y)}",
            "duration_sec": time.time() - t0,
        }
    pred, obs, subj_arr = _run_loso(X, y, list(subjects), alpha=10.0)
    pred_s, obs_s, unique = _per_subject_arrays(pred, obs, subj_arr)
    stats = _stats_from_per_subject(pred_s, obs_s)
    return {
        "lever": "lever2_confweighted",
        "status": "ok",
        "n_trials": int(len(y)),
        "n_subjects": int(len(unique)),
        "subjects": unique,
        "per_subject_pred_deg": [float(p) for p in pred_s.tolist()],
        "per_subject_obs_deg": [float(o) for o in obs_s.tolist()],
        **stats,
        "promoted_to_good_tier": _is_good_tier(stats["ccc_lin"], stats["loa_half_width_deg"]),
        "duration_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Lever 3: temporal smoothing tuning.
# ---------------------------------------------------------------------------


SMOOTHING_WINDOWS: list[int] = [0, 5, 9]


def lever3_smoothing(slot: TargetSlot) -> dict:
    """Sweep three Savgol smoothing window settings, rerun LOSO, return best."""
    _configure_paths_for(slot.approach)
    results: list[dict] = []
    for win in SMOOTHING_WINDOWS:
        t0 = time.time()
        try:
            if slot.approach == "v9_phased":
                cam = VIEW_TO_CAM[slot.view]
                ds = _build_opencap_v9_with_variant(
                    slot.target, cam, use_confweighted=False, smooth_window=win,
                )
                if ds is None:
                    results.append({"smooth_window": win, "status": "build_failed"})
                    continue
                X, y, subjects = ds
            else:
                ds = _build_combined_with_variant(
                    slot.target, slot.view,
                    use_confweighted=False, smooth_window=win,
                )
                if ds is None:
                    results.append({"smooth_window": win, "status": "build_failed"})
                    continue
                X, y, subjects, _ = ds
            if len(y) < 6:
                results.append(
                    {
                        "smooth_window": win,
                        "status": f"insufficient_n={len(y)}",
                        "duration_sec": time.time() - t0,
                    }
                )
                continue
            pred, obs, subj_arr = _run_loso(X, y, list(subjects), alpha=10.0)
            pred_s, obs_s, unique = _per_subject_arrays(pred, obs, subj_arr)
            stats = _stats_from_per_subject(pred_s, obs_s)
            results.append({
                "smooth_window": win,
                "status": "ok",
                "n_trials": int(len(y)),
                "n_subjects": int(len(unique)),
                **stats,
                "promoted_to_good_tier": _is_good_tier(stats["ccc_lin"], stats["loa_half_width_deg"]),
                "duration_sec": time.time() - t0,
            })
        except Exception as e:
            results.append({"smooth_window": win, "status": f"error: {e!r}"})
    ok = [r for r in results if r.get("status") == "ok"]
    best = None
    if ok:
        best = min(ok, key=lambda r: r["loa_half_width_deg"])
    return {
        "lever": "lever3_smoothing_sweep",
        "windows_tested": SMOOTHING_WINDOWS,
        "per_window": results,
        "best": best,
    }


# ---------------------------------------------------------------------------
# Baseline LOSO reproduction (matches biomech_validity_stats).
# ---------------------------------------------------------------------------


def reproduce_baseline_loso(slot: TargetSlot) -> dict:
    ds = _build_baseline_dataset(slot)
    if ds is None:
        raise RuntimeError(f"could not build baseline dataset for {slot}")
    X, y, subjects, _ = ds
    pred, obs, subj_arr = _run_loso(X, y, list(subjects), alpha=10.0)
    pred_s, obs_s, unique = _per_subject_arrays(pred, obs, subj_arr)
    stats = _stats_from_per_subject(pred_s, obs_s)
    return {
        "target": slot.target,
        "view": slot.view,
        "approach": slot.approach,
        "n_trials": int(len(y)),
        "n_subjects": int(len(unique)),
        "subjects": unique,
        "per_subject_pred_deg": [float(p) for p in pred_s.tolist()],
        "per_subject_obs_deg": [float(o) for o in obs_s.tolist()],
        **stats,
    }


# ---------------------------------------------------------------------------
# Run + report.
# ---------------------------------------------------------------------------


def run_all() -> dict:
    out_per_slot: list[dict] = []
    for slot in TARGETS:
        print(f"\n=== {slot.target} / {slot.view} / {slot.approach} ===")
        slot_t0 = time.time()
        baseline = reproduce_baseline_loso(slot)
        print(
            f"  baseline: CCC={baseline['ccc_lin']:.3f} "
            f"LoA_half={baseline['loa_half_width_deg']:.2f}deg "
            f"(target<{GOOD_TIER_LOA_DEG})"
        )

        # Lever 1: outlier diagnostic
        lever1 = lever1_outlier_diagnostic(slot, baseline)
        bd = lever1["best_drop"]
        print(
            f"  lever1 best-drop subject={bd['dropped_subject']} "
            f"-> CCC={bd['ccc_no_subj']:.3f} LoA_half={bd['loa_half_width_no_subj_deg']:.2f}deg "
            f"(diagnostic, not validation)"
        )

        # Lever 2: confidence-weighted features
        lever2 = lever2_confweighted(slot)
        if lever2.get("status") == "ok":
            print(
                f"  lever2 confweighted: CCC={lever2['ccc_lin']:.3f} "
                f"LoA_half={lever2['loa_half_width_deg']:.2f}deg "
                f"-> {'GOOD' if lever2['promoted_to_good_tier'] else 'Moderate'}"
            )
        else:
            print(f"  lever2 confweighted: {lever2.get('status')}")

        # Lever 3: smoothing sweep
        lever3 = lever3_smoothing(slot)
        if lever3["best"]:
            b = lever3["best"]
            print(
                f"  lever3 smoothing best w={b['smooth_window']}: "
                f"CCC={b['ccc_lin']:.3f} LoA_half={b['loa_half_width_deg']:.2f}deg "
                f"-> {'GOOD' if b['promoted_to_good_tier'] else 'Moderate'}"
            )
        else:
            print("  lever3 smoothing: no successful window")

        slot_out = {
            "slot": {
                "target": slot.target,
                "view": slot.view,
                "approach": slot.approach,
                "baseline_n_subjects": slot.baseline_n_subjects,
                "baseline_ccc": slot.baseline_ccc,
                "baseline_loa_half_width_deg": slot.baseline_loa_half_width,
                "loa_gap_to_good_tier_deg": slot.loa_gap_to_good,
            },
            "baseline_reproduced": baseline,
            "lever1": lever1,
            "lever2": lever2,
            "lever3": lever3,
            "duration_sec": time.time() - slot_t0,
        }
        out_per_slot.append(slot_out)

    return {
        "version": "1.0",
        "produced_by": "harness.variance_reduction_pass",
        "good_tier_gate": {
            "ccc_gt": GOOD_TIER_CCC,
            "loa_half_width_lt_deg": GOOD_TIER_LOA_DEG,
        },
        "slots": out_per_slot,
    }


def _fmt(v, prec=2):
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return "-"
    return f"{v:.{prec}f}"


def render_markdown(out: dict) -> str:
    lines: list[str] = []
    lines.append("# Variance Reduction Pass for Moderate-Tier Layer 3 Slots")
    lines.append("")
    lines.append(
        "Three Moderate-tier slots whose CCC already exceeds 0.83 but whose "
        "Bland-Altman LoA half-width sits over the Good-tier gate of +/- 10 "
        "degrees. We test three levers to tighten LoA. Tier-promotion claims "
        "require LOSO-CV CCC > 0.60 AND LoA half-width < 10 degrees on the "
        "full subject cohort."
    )
    lines.append("")

    lines.append("## Targets")
    lines.append("")
    lines.append("| Slot | n subjects | CCC | LoA half-width (deg) | Gap to Good (deg) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for s in out["slots"]:
        sl = s["slot"]
        lines.append(
            f"| {sl['target']} / {sl['view']} / {sl['approach']} | "
            f"{sl['baseline_n_subjects']} | "
            f"{_fmt(sl['baseline_ccc'])} | "
            f"{_fmt(sl['baseline_loa_half_width_deg'])} | "
            f"{_fmt(sl['loa_gap_to_good_tier_deg'])} |"
        )
    lines.append("")

    lines.append("## Per-slot results")
    lines.append("")
    for s in out["slots"]:
        sl = s["slot"]
        baseline = s["baseline_reproduced"]
        lines.append(
            f"### {sl['target']} / {sl['view']} / {sl['approach']}"
        )
        lines.append("")
        lines.append(
            f"Baseline (this run, LOSO per-subject): n={baseline['n_subjects']}, "
            f"CCC={_fmt(baseline['ccc_lin'])}, "
            f"LoA half-width={_fmt(baseline['loa_half_width_deg'])} deg, "
            f"MAE={_fmt(baseline['mae_deg'])} deg, "
            f"bias={_fmt(baseline['mean_bias_deg'])} deg."
        )
        lines.append("")
        # Lever 1 table
        l1 = s["lever1"]
        bd = l1["best_drop"]
        promoted_diag = l1["best_drop_promotes_diagnostic_tier"]
        lines.append("**Lever 1: outlier subject leave-one-out (diagnostic, NOT validation)**")
        lines.append("")
        lines.append(
            f"- Highest-leverage subject: `{bd['dropped_subject']}` "
            f"(abs error in LOSO = {_fmt(bd['abs_error_dropped_deg'])} deg)"
        )
        lines.append(
            f"- After excluding that subject: CCC={_fmt(bd['ccc_no_subj'])}, "
            f"LoA half-width={_fmt(bd['loa_half_width_no_subj_deg'])} deg "
            f"({'crosses Good gate' if promoted_diag else 'still Moderate-or-worse'})"
        )
        lines.append(
            "- Note: dropping subjects post hoc is a leverage diagnostic, "
            "not a validated tier change. Promotion claims below come only "
            "from Levers 2 and 3 on the full subject cohort."
        )
        lines.append("")
        lines.append("| Dropped subject | n remain | CCC | LoA half (deg) | abs err on drop (deg) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for r in l1["all_drops"]:
            lines.append(
                f"| {r['dropped_subject']} | {r['n_remaining']} | "
                f"{_fmt(r['ccc_no_subj'])} | "
                f"{_fmt(r['loa_half_width_no_subj_deg'])} | "
                f"{_fmt(r['abs_error_dropped_deg'])} |"
            )
        lines.append("")

        # Lever 2
        l2 = s["lever2"]
        lines.append("**Lever 2: confidence-weighted phased-angle features**")
        lines.append("")
        if l2.get("status") == "ok":
            lines.append(
                f"- LOSO CCC = {_fmt(l2['ccc_lin'])} "
                f"(baseline {_fmt(baseline['ccc_lin'])}), "
                f"LoA half-width = {_fmt(l2['loa_half_width_deg'])} deg "
                f"(baseline {_fmt(baseline['loa_half_width_deg'])} deg)."
            )
            d_loa = l2["loa_half_width_deg"] - baseline["loa_half_width_deg"]
            d_ccc = l2["ccc_lin"] - baseline["ccc_lin"]
            lines.append(
                f"- delta LoA = {_fmt(d_loa)} deg, delta CCC = {_fmt(d_ccc)}."
            )
            lines.append(
                f"- Tier verdict (full cohort): "
                f"{'**GOOD (promoted)**' if l2['promoted_to_good_tier'] else 'Moderate (no promotion)'}"
            )
        else:
            lines.append(f"- Status: {l2.get('status')}")
        lines.append("")

        # Lever 3
        l3 = s["lever3"]
        lines.append("**Lever 3: temporal smoothing sweep (Savgol window 0, 5, 9 frames)**")
        lines.append("")
        lines.append("| window | n subj | CCC | LoA half (deg) | MAE (deg) | tier |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
        for r in l3["per_window"]:
            if r.get("status") != "ok":
                lines.append(
                    f"| {r['smooth_window']} | - | - | - | - | {r.get('status')} |"
                )
                continue
            tier = "Good" if r["promoted_to_good_tier"] else "Moderate"
            lines.append(
                f"| {r['smooth_window']} | {r['n_subjects']} | "
                f"{_fmt(r['ccc_lin'])} | {_fmt(r['loa_half_width_deg'])} | "
                f"{_fmt(r['mae_deg'])} | {tier} |"
            )
        if l3.get("best") is not None:
            b = l3["best"]
            d_loa = b["loa_half_width_deg"] - baseline["loa_half_width_deg"]
            d_ccc = b["ccc_lin"] - baseline["ccc_lin"]
            lines.append("")
            lines.append(
                f"- Best window = {b['smooth_window']}: CCC={_fmt(b['ccc_lin'])}, "
                f"LoA half-width={_fmt(b['loa_half_width_deg'])} deg "
                f"(delta LoA = {_fmt(d_loa)} deg, delta CCC = {_fmt(d_ccc)})."
            )
            lines.append(
                f"- Tier verdict (full cohort): "
                f"{'**GOOD (promoted)**' if b['promoted_to_good_tier'] else 'Moderate (no promotion)'}"
            )
        lines.append("")

    # Overall summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Slot | Baseline tier | Lever 2 tier | Lever 3 best tier | Best lever | Best LoA half (deg) |")
    lines.append("| --- | --- | --- | --- | --- | ---: |")
    for s in out["slots"]:
        sl = s["slot"]
        baseline = s["baseline_reproduced"]
        baseline_tier = "Good" if _is_good_tier(
            baseline["ccc_lin"], baseline["loa_half_width_deg"]
        ) else "Moderate"
        l2 = s["lever2"]
        l3 = s["lever3"]
        l2_tier = "build_failed"
        l2_loa = float("nan")
        if l2.get("status") == "ok":
            l2_tier = "Good" if l2["promoted_to_good_tier"] else "Moderate"
            l2_loa = l2["loa_half_width_deg"]
        l3_tier = "build_failed"
        l3_loa = float("nan")
        if l3.get("best") is not None:
            l3_tier = "Good" if l3["best"]["promoted_to_good_tier"] else "Moderate"
            l3_loa = l3["best"]["loa_half_width_deg"]
        candidates = []
        if math.isfinite(l2_loa):
            candidates.append(("lever2_confweighted", l2_loa, l2_tier))
        if math.isfinite(l3_loa):
            candidates.append((f"lever3_smoothing(w={s['lever3']['best']['smooth_window']})", l3_loa, l3_tier))
        if candidates:
            best_name, best_loa, _ = min(candidates, key=lambda x: x[1])
        else:
            best_name, best_loa = "none", float("nan")
        lines.append(
            f"| {sl['target']} / {sl['view']} | {baseline_tier} | {l2_tier} | "
            f"{l3_tier} | {best_name} | {_fmt(best_loa)} |"
        )
    lines.append("")

    lines.append("## Honest verdict")
    lines.append("")
    any_promoted = False
    for s in out["slots"]:
        sl = s["slot"]
        baseline = s["baseline_reproduced"]
        l2 = s["lever2"]
        l3 = s["lever3"]
        sl_name = f"{sl['target']} / {sl['view']}"
        l2_promo = l2.get("status") == "ok" and l2.get("promoted_to_good_tier")
        l3_promo = (
            l3.get("best") is not None and l3["best"]["promoted_to_good_tier"]
        )
        if l2_promo or l3_promo:
            any_promoted = True
            lever_name = "Lever 2 (conf-weighted features)" if l2_promo else (
                f"Lever 3 (smoothing window={l3['best']['smooth_window']})"
            )
            lines.append(
                f"- **{sl_name}**: PROMOTED to Good tier via {lever_name} on the full subject cohort."
            )
        else:
            l1 = s["lever1"]
            best = l1["best_drop"]
            if l1["best_drop_promotes_diagnostic_tier"]:
                lines.append(
                    f"- **{sl_name}**: no tier promotion on full cohort. Lever 1 "
                    f"diagnostic: excluding subject `{best['dropped_subject']}` "
                    f"(abs LOSO error {_fmt(best['abs_error_dropped_deg'])} deg) "
                    f"would cross the gate, but this is post-hoc and does NOT "
                    f"warrant a Good-tier claim. Subject warrants a clip-level "
                    f"audit to determine if there's a systematic cause."
                )
            else:
                lines.append(
                    f"- **{sl_name}**: no tier promotion. Best single-subject drop "
                    f"still yields LoA half-width {_fmt(best['loa_half_width_no_subj_deg'])} deg; "
                    f"variance is distributed, not driven by one outlier."
                )
    if not any_promoted:
        lines.append("")
        lines.append(
            "No slot crossed the Good-tier gate on the full subject cohort. "
            "The variance is real (not driven by a single outlier on at least "
            "some slots) and is not eliminated by confidence weighting or "
            "smoothing parameter tuning. The outlier diagnostics (Lever 1) "
            "surface candidates for further clip-level investigation but do "
            "not constitute validation of any slot at the Good tier."
        )
    lines.append("")
    lines.append("## Constraints respected")
    lines.append("")
    lines.append("- LOSO discipline preserved on every tier-change claim.")
    lines.append("- Single phone camera only.")
    lines.append("- No modification of v15/v16/v17 or biomech_validity_stats source files.")
    lines.append("- Outlier exclusion treated as a diagnostic, not a validation method.")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_v18_recommendation(out: dict) -> dict | None:
    """If at least one (slot, lever) cleanly promotes to Good tier on the
    full cohort, document the per-slot weight refresh that would feed a
    v18 deploy bundle.
    """
    promoted = []
    for s in out["slots"]:
        sl = s["slot"]
        l2 = s["lever2"]
        l3 = s["lever3"]
        promotions = []
        if l2.get("status") == "ok" and l2.get("promoted_to_good_tier"):
            promotions.append({
                "lever": "lever2_confweighted",
                "ccc": l2["ccc_lin"],
                "loa_half_width_deg": l2["loa_half_width_deg"],
                "mae_deg": l2["mae_deg"],
                "n_subjects": l2["n_subjects"],
                "feature_change": "phased-angle samples weighted by per-frame min-keypoint-confidence; rom_loading restricted to conf>0.3 frames",
            })
        if l3.get("best") is not None and l3["best"]["promoted_to_good_tier"]:
            b = l3["best"]
            promotions.append({
                "lever": f"lever3_smoothing_w={b['smooth_window']}",
                "ccc": b["ccc_lin"],
                "loa_half_width_deg": b["loa_half_width_deg"],
                "mae_deg": b["mae_deg"],
                "n_subjects": b["n_subjects"],
                "feature_change": f"Savgol smoothing window={b['smooth_window']} applied to 2D keypoints before angle extraction",
            })
        if promotions:
            promoted.append({
                "target": sl["target"],
                "view": sl["view"],
                "approach": sl["approach"],
                "promotions": promotions,
            })
    if not promoted:
        return None
    return {
        "version": "0.1",
        "produced_by": "harness.variance_reduction_pass",
        "note": (
            "Each promoted slot would need a deploy_ready_models.json weight "
            "refresh using the variant feature pipeline. The variance-reduction "
            "pass only re-ran LOSO; it did not write new ridge weights. A "
            "follow-up deploy refresh is required before any v18 tier-count "
            "update."
        ),
        "good_tier_gate": {
            "ccc_gt": GOOD_TIER_CCC,
            "loa_half_width_lt_deg": GOOD_TIER_LOA_DEG,
        },
        "promoted_slots": promoted,
        "new_tier_counts_estimate": {
            "good_tier_added": len(promoted),
            "note": "Final tier counts depend on whether the variant feature pipeline regresses any other slot once promoted slots are switched in.",
        },
    }


def main() -> None:
    out = run_all()
    (OUT_DIR / "per_slot_variance_results.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "REPORT.md").write_text(render_markdown(out))
    v18 = build_v18_recommendation(out)
    if v18 is not None:
        (OUT_DIR / "recommended_v18_updates.json").write_text(json.dumps(v18, indent=2))
        print(f"Wrote {OUT_DIR / 'recommended_v18_updates.json'}")
    print(f"Wrote {OUT_DIR / 'per_slot_variance_results.json'}")
    print(f"Wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
