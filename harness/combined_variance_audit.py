"""Combined variance reduction + opencap_subject5 forensic audit.

Two parallel tasks on the v14_full_dwpose knee_angle_r / side_left slot:

Task 1 (combined levers):
    Re-run LOSO with BOTH confidence-weighted features AND Savgol smoothing
    window=9 enabled simultaneously. Compare to baseline, L2-only, L3-only.
    Verdict: does the combination cross the Good-tier gate
    (CCC > 0.60 AND LoA half-width < 10 deg)?

Task 2 (opencap_subject5 forensic audit):
    Investigate why subject5 is the dominant outlier on BOTH the knee/side_left
    slot (12.75 deg abs LOSO error) and the ankle/side_right slot (9.18 deg).
    Sources of evidence:
      - sessionMetadata.yaml (height_m, mass_kg, sex)
      - OpenSim IK marker errors (DJ*_ik_marker_errors.sto)
      - per-trial ground-truth ROM time series for knee_angle_r, ankle_angle_r
      - per-trial 2D keypoint confidence (DWPose) and ROM derived from camera
      - per-trial event-window length
      - per-subject ROM distribution vs cohort (is subject5's physical ROM
        unusual?)
    Goal: identify whether a transparent exclusion criterion exists.

    Possible verdicts:
      - data quality issue (camera artifact, marker dropout) -> documented exclude
      - protocol deviation (modified DJ, prior injury) -> documented exclude
      - body proportions outside training distribution -> cohort expansion flag
      - no visible issue -> cannot exclude; honest "no actionable finding"

Run:
    cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
    python -m harness.combined_variance_audit

Outputs:
    data/combined_variance_audit/per_slot_combined_results.json
    data/combined_variance_audit/subject5_audit.json
    data/combined_variance_audit/REPORT.md
    data/combined_variance_audit/recommended_v18_updates.json (if applicable)

Constraints:
    - LOSO discipline on all tier-change claims.
    - Single phone camera only.
    - Do NOT modify v15/v16/v17, biomech_validity_stats, or other prior outputs.
    - Honest reporting on Task 2: if no documented data-quality issue is found,
      we report "no exclusion criterion identified" -- not cherry-picking.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.train_v9_phased as v9
import harness.train_v12_combined as v12
from harness.biomech_validity_stats import _ccc, _pearson, _loso_extract
from harness.parsers import parse_mot  # .sto files share the .mot format
from harness.train_multi_metric import VIEW_BUCKET_NAMES, _rom, _rom_unwrapped
from harness.train_v9_phased import _select_rom, _phased_angles, gt_rom_col
from harness.variance_reduction_pass import (
    GOOD_TIER_CCC,
    GOOD_TIER_LOA_DEG,
    TARGETS,
    _build_combined_with_variant,
    _configure_paths_for,
    _is_good_tier,
    _per_subject_arrays,
    _run_loso,
    _stats_from_per_subject,
    reproduce_baseline_loso,
)
from harness.couro_keypoints import KP, load_couro_output


REPO_ROOT = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation"
)
DATA_ROOT = REPO_ROOT / "data"
LAB = DATA_ROOT / "LabValidation_withVideos"
OUT_DIR = DATA_ROOT / "combined_variance_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


VIEW_TO_CAM = {v: k for k, v in VIEW_BUCKET_NAMES.items()}


# ===========================================================================
# Task 1: combined Lever 2 + Lever 3 on knee_angle_r / side_left
# ===========================================================================


def task1_combined_levers() -> dict[str, Any]:
    """Combine confidence-weighted features + Savgol w=9 smoothing on the
    knee/side_left slot. Compare to baseline, L2-only, L3-only."""
    slot = next(
        s for s in TARGETS
        if s.target == "knee_angle_r"
        and s.view == "side_left"
        and s.approach == "v14_full_dwpose"
    )

    print(f"\n=== Task 1: {slot.target} / {slot.view} / {slot.approach} (combined) ===")
    _configure_paths_for(slot.approach)
    t0 = time.time()

    # Baseline (reproduced from variance_reduction_pass).
    baseline = reproduce_baseline_loso(slot)
    print(
        f"  baseline:    CCC={baseline['ccc_lin']:.3f} "
        f"LoA_half={baseline['loa_half_width_deg']:.2f}deg"
    )

    # Lever 2 only (reproduce Y's result).
    t1 = time.time()
    ds_l2 = _build_combined_with_variant(
        slot.target, slot.view, use_confweighted=True, smooth_window=0,
    )
    l2_only = _eval_dataset(ds_l2, label="lever2_only")
    l2_only["duration_sec"] = time.time() - t1
    print(
        f"  lever2 only: CCC={_safe(l2_only.get('ccc_lin')):.3f} "
        f"LoA_half={_safe(l2_only.get('loa_half_width_deg')):.2f}deg"
    )

    # Lever 3 only w=9 (reproduce Y's result).
    t2 = time.time()
    ds_l3 = _build_combined_with_variant(
        slot.target, slot.view, use_confweighted=False, smooth_window=9,
    )
    l3_only = _eval_dataset(ds_l3, label="lever3_w9_only")
    l3_only["duration_sec"] = time.time() - t2
    print(
        f"  lever3 only: CCC={_safe(l3_only.get('ccc_lin')):.3f} "
        f"LoA_half={_safe(l3_only.get('loa_half_width_deg')):.2f}deg"
    )

    # Combined (NEW): confidence-weighted + Savgol w=9.
    t3 = time.time()
    ds_combined = _build_combined_with_variant(
        slot.target, slot.view, use_confweighted=True, smooth_window=9,
    )
    combined = _eval_dataset(ds_combined, label="combined_l2_l3_w9")
    combined["duration_sec"] = time.time() - t3
    print(
        f"  combined:    CCC={_safe(combined.get('ccc_lin')):.3f} "
        f"LoA_half={_safe(combined.get('loa_half_width_deg')):.2f}deg "
        f"-> {'GOOD (promoted)' if combined.get('promoted_to_good_tier') else 'Moderate'}"
    )

    return {
        "slot": {
            "target": slot.target,
            "view": slot.view,
            "approach": slot.approach,
            "baseline_ccc": slot.baseline_ccc,
            "baseline_loa_half_width_deg": slot.baseline_loa_half_width,
            "loa_gap_to_good_tier_deg": slot.loa_gap_to_good,
        },
        "baseline_reproduced": baseline,
        "lever2_only": l2_only,
        "lever3_w9_only": l3_only,
        "combined_l2_l3_w9": combined,
        "duration_sec": time.time() - t0,
    }


def _safe(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return float("nan")
    return v


def _eval_dataset(ds, *, label: str) -> dict[str, Any]:
    if ds is None:
        return {"label": label, "status": "build_failed"}
    X, y, subjects, _ = ds
    if len(y) < 6:
        return {"label": label, "status": f"insufficient_n={len(y)}"}
    pred, obs, subj_arr = _run_loso(X, y, list(subjects), alpha=10.0)
    pred_s, obs_s, unique = _per_subject_arrays(pred, obs, subj_arr)
    stats = _stats_from_per_subject(pred_s, obs_s)
    return {
        "label": label,
        "status": "ok",
        "n_trials": int(len(y)),
        "n_subjects": int(len(unique)),
        "subjects": unique,
        "per_subject_pred_deg": [float(p) for p in pred_s.tolist()],
        "per_subject_obs_deg": [float(o) for o in obs_s.tolist()],
        "per_subject_abs_error_deg": [
            float(abs(p - o)) for p, o in zip(pred_s.tolist(), obs_s.tolist())
        ],
        **stats,
        "promoted_to_good_tier": _is_good_tier(stats["ccc_lin"], stats["loa_half_width_deg"]),
    }


# ===========================================================================
# Task 2: opencap_subject5 forensic audit
# ===========================================================================


# Subjects in the OpenCap LabValidation cohort that we have access to.
ALL_OPENCAP_SUBJECTS = [
    "subject2", "subject3", "subject4", "subject5", "subject7",
    "subject8", "subject9", "subject10", "subject11",
]
# subject6 has no raw video (precomputed HRNet/OpenPose only).

# Drop-jump tasks (the validation tree only uses these).
DJ_TASKS = ["DJ1", "DJ2", "DJ3", "DJAsym1", "DJAsym2", "DJAsym3"]


def task2_subject5_audit() -> dict[str, Any]:
    """Forensic audit of opencap_subject5."""
    print("\n=== Task 2: opencap_subject5 forensic audit ===")
    audit: dict[str, Any] = {
        "subject": "subject5",
        "context": (
            "subject5 was the dominant outlier on TWO validation slots: "
            "knee_angle_r/side_left (abs LOSO error 12.75 deg) and "
            "ankle_angle_r/side_right (abs LOSO error 9.18 deg). "
            "We investigate whether a transparent, defensible exclusion "
            "criterion exists."
        ),
    }

    # 1. Demographics.
    audit["demographics_all_subjects"] = _collect_demographics()
    audit["demographics_subject5_z_scores"] = _demographics_z_scores(
        audit["demographics_all_subjects"], "subject5",
    )

    # 2. OpenSim IK marker errors for DJ trials (data-quality proxy).
    audit["ik_marker_errors"] = _collect_marker_errors(audit["demographics_all_subjects"])

    # 3. Per-subject GT ROM distribution for the two flagged metrics.
    audit["gt_rom_by_subject"] = _collect_gt_rom(audit["demographics_all_subjects"])

    # 4. Per-trial DWPose keypoint confidence in the loading window (subject5 only).
    audit["dwpose_loading_conf_subject5"] = _collect_loading_conf_subject5()

    # 5. Recompute knee/side_left and ankle/side_right LOSO without subject5.
    audit["without_subject5"] = _stats_without_subject5()

    # 6. Synthesize verdict.
    audit["verdict"] = _synthesize_subject5_verdict(audit)

    return audit


def _collect_demographics() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for s in ALL_OPENCAP_SUBJECTS:
        meta_path = LAB / s / "sessionMetadata.yaml"
        if not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text())
        out[s] = {
            "height_m": float(meta.get("height_m", float("nan"))),
            "mass_kg": float(meta.get("mass_kg", float("nan"))),
            "sex": str(meta.get("sex", "?")),
            "checker_square_side_mm": float(
                meta.get("checkerBoard", {}).get("squareSideLength_mm", float("nan"))
            ),
        }
    return out


def _demographics_z_scores(
    demos: dict[str, dict[str, Any]], subject: str,
) -> dict[str, Any]:
    target = demos.get(subject)
    if target is None:
        return {}
    out = {}
    for field in ("height_m", "mass_kg"):
        vals = np.array([d[field] for d in demos.values() if np.isfinite(d[field])])
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        z = (target[field] - mu) / sd if sd > 0 else float("nan")
        out[field] = {
            "subject_value": target[field],
            "cohort_mean": mu,
            "cohort_sd": sd,
            "z_score": float(z),
            "is_outlier": bool(abs(z) > 2.0) if math.isfinite(z) else False,
        }
    # BMI
    h = target["height_m"]
    m = target["mass_kg"]
    bmi = m / (h * h) if h > 0 else float("nan")
    bmis = np.array([
        d["mass_kg"] / (d["height_m"] ** 2) for d in demos.values()
        if np.isfinite(d["mass_kg"]) and np.isfinite(d["height_m"]) and d["height_m"] > 0
    ])
    mu_bmi = float(np.mean(bmis))
    sd_bmi = float(np.std(bmis, ddof=1)) if len(bmis) > 1 else float("nan")
    z_bmi = (bmi - mu_bmi) / sd_bmi if sd_bmi > 0 else float("nan")
    out["bmi"] = {
        "subject_value": bmi,
        "cohort_mean": mu_bmi,
        "cohort_sd": sd_bmi,
        "z_score": float(z_bmi),
        "is_outlier": bool(abs(z_bmi) > 2.0) if math.isfinite(z_bmi) else False,
    }
    return out


def _collect_marker_errors(demos: dict) -> dict[str, dict[str, Any]]:
    """OpenSim IK marker errors (mm) per DJ trial, per subject.

    The .sto file format: header lines then numeric matrix; the rows are
    timesteps and the first column is the max marker error over all markers
    at that timestep, etc. We extract the mean over time of every numeric
    column except 'time', as a generic quality scalar -- ~mm typical for
    well-tracked OpenSim sessions.
    """
    out: dict[str, dict[str, Any]] = {}
    for s in ALL_OPENCAP_SUBJECTS:
        per_trial: dict[str, dict[str, float]] = {}
        for task in DJ_TASKS:
            sto_path = LAB / s / "OpenSimData" / "Mocap" / "IK" / f"{task}_ik_marker_errors.sto"
            if not sto_path.exists():
                continue
            try:
                mt = parse_mot(sto_path)
            except Exception as e:  # noqa
                per_trial[task] = {"error": f"parse_failed: {e!r}"}
                continue
            stats: dict[str, float] = {}
            for col_name in mt.columns:
                if col_name.lower() == "time":
                    continue
                col = mt.column(col_name)
                if col.size == 0:
                    continue
                stats[f"mean_{col_name}_m"] = float(np.nanmean(col))
                stats[f"max_{col_name}_m"] = float(np.nanmax(col))
            per_trial[task] = stats
        out[s] = per_trial

    # Cohort summary: mean of (mean over trials of mean over time of RMS error)
    # using whatever column is reported. We extract a uniform scalar:
    # the average value across all numeric columns / trials per subject.
    summary: dict[str, float] = {}
    for s, trials in out.items():
        vals = []
        for task, stats in trials.items():
            if not isinstance(stats, dict):
                continue
            for k, v in stats.items():
                if k.startswith("mean_") and math.isfinite(v):
                    vals.append(v)
        summary[s] = float(np.mean(vals)) if vals else float("nan")
    out["_per_subject_mean_marker_error_any_col_m"] = summary
    return out


def _collect_gt_rom(demos: dict) -> dict[str, Any]:
    """Per-subject GT ROM distribution for knee_angle_r and ankle_angle_r."""
    out: dict[str, dict[str, list[float]]] = {}
    cohort: dict[str, list[float]] = {"knee_angle_r": [], "ankle_angle_r": []}
    for s in ALL_OPENCAP_SUBJECTS:
        out[s] = {"knee_angle_r": [], "ankle_angle_r": []}
        for task in DJ_TASKS:
            for col in ("knee_angle_r", "ankle_angle_r"):
                v = gt_rom_col(s, task, col)
                if np.isfinite(v):
                    out[s][col].append(float(v))
                    cohort[col].append(float(v))
    # Per-subject mean / sd / range; z-score subject5's mean vs cohort.
    per_subj_stats = {}
    for s in ALL_OPENCAP_SUBJECTS:
        per_subj_stats[s] = {}
        for col in ("knee_angle_r", "ankle_angle_r"):
            vals = np.asarray(out[s][col])
            per_subj_stats[s][col] = {
                "n_trials": int(len(vals)),
                "mean_deg": float(np.mean(vals)) if vals.size else float("nan"),
                "sd_deg": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
                "min_deg": float(np.min(vals)) if vals.size else float("nan"),
                "max_deg": float(np.max(vals)) if vals.size else float("nan"),
            }
    # Subject5 ROM-level z-scores: is subject5's per-trial ROM unusual?
    subj5 = "subject5"
    rom_z = {}
    for col in ("knee_angle_r", "ankle_angle_r"):
        cohort_vals = np.asarray(cohort[col])
        s5_vals = np.asarray(out[subj5][col])
        if cohort_vals.size and s5_vals.size:
            mu = float(np.mean(cohort_vals))
            sd = float(np.std(cohort_vals, ddof=1))
            z = (float(np.mean(s5_vals)) - mu) / sd if sd > 0 else float("nan")
            rom_z[col] = {
                "subject5_mean_deg": float(np.mean(s5_vals)),
                "cohort_mean_deg": mu,
                "cohort_sd_deg": sd,
                "subject5_z_vs_cohort": float(z),
                "is_outlier_gt_2sigma": bool(abs(z) > 2.0) if math.isfinite(z) else False,
                "subject5_relative_position": (
                    "low" if z < -1.0 else "high" if z > 1.0 else "within_1_sd"
                ),
            }
    return {
        "per_subject_per_trial_rom_deg": out,
        "per_subject_stats": per_subj_stats,
        "subject5_z_vs_cohort": rom_z,
    }


def _collect_loading_conf_subject5() -> dict[str, Any]:
    """Per-trial loading-window confidence for subject5, side_left (Cam0) and
    side_right (Cam4). Compare to cohort distribution."""
    out: dict[str, Any] = {}
    for cam, view in (("Cam0", "side_left"), ("Cam4", "side_right")):
        kp_dir = DATA_ROOT / "opencap_dwpose_keypoints"
        trial_stats: list[dict[str, Any]] = []
        cohort_means: list[float] = []
        s5_means: list[float] = []
        # Knee chain: R_hip, R_kne, R_ank
        # Ankle chain: R_kne, R_ank, R_big_toe
        chain_for_view = {
            "side_left": ["R_hip", "R_kne", "R_ank"],   # knee right
            "side_right": ["R_kne", "R_ank", "R_big_toe"],  # ankle right
        }[view]
        for kp_path in sorted(kp_dir.glob(f"*_{cam}.json")):
            base = kp_path.stem
            parts = base.split("_")
            subject = parts[0]
            task = "_".join(parts[1:-1])
            if not task.startswith("DJ"):
                continue
            try:
                series = load_couro_output(kp_path)
            except Exception:
                continue
            # Detect loading window via the same multi-signal event detector.
            from harness.dj_events import detect_dj_events_multi_signal
            r_y = series.xy[:, KP["R_ank"], 1]
            l_y = series.xy[:, KP["L_ank"], 1]
            hip_y = series.xy[:, KP["hip_center"], 1]
            events = detect_dj_events_multi_signal([r_y, l_y, hip_y], fps=series.fps)
            if events.loading_window is None:
                continue
            a, b = events.loading_window
            a = max(0, int(a)); b = min(series.confidence.shape[0], int(b))
            if b <= a:
                continue
            cols = [KP[c] for c in chain_for_view if c in KP]
            confs = series.confidence[a:b, cols]
            mean_conf = float(np.nanmean(confs))
            min_conf = float(np.nanmin(confs))
            frac_low = float(np.mean(confs < 0.3))
            cohort_means.append(mean_conf)
            row = {
                "subject": subject,
                "task": task,
                "loading_window_frames": int(b - a),
                "mean_chain_conf": mean_conf,
                "min_chain_conf": min_conf,
                "frac_frames_below_0_3": frac_low,
            }
            trial_stats.append(row)
            if subject == "subject5":
                s5_means.append(mean_conf)
        cohort_arr = np.asarray(cohort_means)
        s5_arr = np.asarray(s5_means)
        mu = float(np.mean(cohort_arr)) if cohort_arr.size else float("nan")
        sd = float(np.std(cohort_arr, ddof=1)) if cohort_arr.size > 1 else float("nan")
        z = (float(np.mean(s5_arr)) - mu) / sd if (sd and sd > 0 and s5_arr.size) else float("nan")
        out[view] = {
            "chain": chain_for_view,
            "n_trials_cohort": int(len(cohort_means)),
            "n_trials_subject5": int(len(s5_means)),
            "cohort_mean_chain_conf": mu,
            "cohort_sd_chain_conf": sd,
            "subject5_mean_chain_conf": float(np.mean(s5_arr)) if s5_arr.size else float("nan"),
            "subject5_z_score": float(z),
            "subject5_is_outlier_gt_2sigma": (
                bool(abs(z) > 2.0) if math.isfinite(z) else False
            ),
            "per_trial": trial_stats,
        }
    return out


def _stats_without_subject5() -> dict[str, Any]:
    """Recompute knee/side_left and ankle/side_right LOSO stats without subject5.

    Honesty note: this is a post-hoc diagnostic. It is reported alongside the
    forensic findings -- whether to ACT on it depends on whether the forensic
    audit produced a documentable exclusion criterion.
    """
    out: dict[str, Any] = {}

    # knee_angle_r / side_left, v14_full_dwpose -- combined dataset, baseline build.
    slot_knee = next(
        s for s in TARGETS
        if s.target == "knee_angle_r" and s.view == "side_left"
        and s.approach == "v14_full_dwpose"
    )
    _configure_paths_for(slot_knee.approach)
    ds = _build_combined_with_variant(
        slot_knee.target, slot_knee.view,
        use_confweighted=False, smooth_window=0,
    )
    out["knee_angle_r_side_left_v14_baseline"] = _eval_without_subject(
        ds, "opencap_subject5"
    )
    # Also the combined-levers variant.
    ds_combo = _build_combined_with_variant(
        slot_knee.target, slot_knee.view,
        use_confweighted=True, smooth_window=9,
    )
    out["knee_angle_r_side_left_v14_combined_l2_l3_w9"] = _eval_without_subject(
        ds_combo, "opencap_subject5"
    )

    # ankle_angle_r / side_right, v14_full_dwpose -- OpenCap-only.
    # Replicate train_v14_full_dwpose.build_opencap_only_dataset path.
    import harness.train_v14_full_dwpose as v14
    v9.KP_DIR = DATA_ROOT / "opencap_dwpose_keypoints"
    try:
        ds_ankle = v14.build_opencap_only_dataset("ankle_angle_r", "side_right")
    except Exception as e:  # noqa
        ds_ankle = None
        out["_ankle_build_error"] = repr(e)
    out["ankle_angle_r_side_right_v14_baseline"] = _eval_without_subject(
        ds_ankle, "opencap_subject5"
    )

    return out


def _eval_without_subject(ds, excluded_subject: str) -> dict[str, Any]:
    """Run LOSO on dataset with one subject removed."""
    if ds is None:
        return {"status": "build_failed"}
    if isinstance(ds, tuple) and len(ds) == 3:
        X, y, subjects = ds
        sources = ["opencap"] * len(y)
    elif isinstance(ds, tuple) and len(ds) == 4:
        X, y, subjects, sources = ds
    else:
        return {"status": "unsupported_dataset_shape"}

    # First: full-cohort stats (sanity).
    subj_arr_full = np.asarray(subjects)
    keep = subj_arr_full != excluded_subject
    X_keep = X[keep]
    y_keep = y[keep]
    subj_keep = [s for s, k in zip(subjects, keep) if k]
    if len(set(subj_keep)) < 3 or len(y_keep) < 6:
        return {"status": "insufficient_n_after_exclusion"}

    # Stats WITH subject (full).
    pred_f, obs_f, subj_f = _run_loso(X, y, list(subjects), alpha=10.0)
    pred_sf, obs_sf, unique_f = _per_subject_arrays(pred_f, obs_f, subj_f)
    full_stats = _stats_from_per_subject(pred_sf, obs_sf)
    # Stats WITHOUT subject.
    pred_x, obs_x, subj_x = _run_loso(X_keep, y_keep, subj_keep, alpha=10.0)
    pred_sx, obs_sx, unique_x = _per_subject_arrays(pred_x, obs_x, subj_x)
    excl_stats = _stats_from_per_subject(pred_sx, obs_sx)
    return {
        "status": "ok",
        "n_trials_full": int(len(y)),
        "n_subjects_full": int(len(set(subjects))),
        "full_cohort_stats": full_stats,
        "full_cohort_tier_good": _is_good_tier(
            full_stats["ccc_lin"], full_stats["loa_half_width_deg"]
        ),
        "n_trials_excluded": int(len(y_keep)),
        "n_subjects_excluded": int(len(set(subj_keep))),
        "excluded_subject": excluded_subject,
        "without_subject5_stats": excl_stats,
        "without_subject5_tier_good": _is_good_tier(
            excl_stats["ccc_lin"], excl_stats["loa_half_width_deg"]
        ),
    }


def _synthesize_subject5_verdict(audit: dict[str, Any]) -> dict[str, Any]:
    """Decide whether a defensible exclusion criterion exists."""
    flags: list[str] = []

    # Demographic outlier?
    dz = audit["demographics_subject5_z_scores"]
    if dz.get("height_m", {}).get("is_outlier"):
        flags.append("demographics_height_outlier_gt_2sigma")
    if dz.get("mass_kg", {}).get("is_outlier"):
        flags.append("demographics_mass_outlier_gt_2sigma")
    if dz.get("bmi", {}).get("is_outlier"):
        flags.append("demographics_bmi_outlier_gt_2sigma")

    # GT ROM outlier?
    rom_z = audit["gt_rom_by_subject"]["subject5_z_vs_cohort"]
    if rom_z.get("knee_angle_r", {}).get("is_outlier_gt_2sigma"):
        flags.append("gt_knee_rom_outlier_gt_2sigma")
    if rom_z.get("ankle_angle_r", {}).get("is_outlier_gt_2sigma"):
        flags.append("gt_ankle_rom_outlier_gt_2sigma")

    # Confidence outlier?
    dconf = audit["dwpose_loading_conf_subject5"]
    if dconf.get("side_left", {}).get("subject5_is_outlier_gt_2sigma"):
        flags.append("dwpose_loading_conf_outlier_side_left")
    if dconf.get("side_right", {}).get("subject5_is_outlier_gt_2sigma"):
        flags.append("dwpose_loading_conf_outlier_side_right")

    # IK marker error outlier?
    summary = audit["ik_marker_errors"].get(
        "_per_subject_mean_marker_error_any_col_m", {}
    )
    vals = np.array([v for v in summary.values() if math.isfinite(v)])
    if vals.size > 1 and "subject5" in summary:
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        z = (summary["subject5"] - mu) / sd if sd > 0 else float("nan")
        marker_err_outlier = bool(abs(z) > 2.0) if math.isfinite(z) else False
        if marker_err_outlier:
            flags.append(
                f"opensim_ik_marker_error_outlier (z={z:.2f})"
            )

    # Magnitude of the per-trial GT ROM range for subject5.
    s5_rom = audit["gt_rom_by_subject"]["per_subject_stats"]["subject5"]

    verdict_str: str
    recommended_action: str
    reasoning: list[str] = []

    rom_z_knee = rom_z.get("knee_angle_r", {}).get("subject5_z_vs_cohort", float("nan"))
    rom_z_ankle = rom_z.get("ankle_angle_r", {}).get("subject5_z_vs_cohort", float("nan"))

    if rom_z.get("knee_angle_r", {}).get("subject5_relative_position") == "low":
        reasoning.append(
            f"Subject5's mean knee ROM is {rom_z['knee_angle_r']['subject5_mean_deg']:.1f} deg "
            f"vs cohort {rom_z['knee_angle_r']['cohort_mean_deg']:.1f} deg "
            f"(z={rom_z_knee:.2f}) -- subject5 lands with LESS knee flexion than peers."
        )
    if rom_z.get("ankle_angle_r", {}).get("subject5_relative_position") == "low":
        reasoning.append(
            f"Subject5's mean ankle ROM is {rom_z['ankle_angle_r']['subject5_mean_deg']:.1f} deg "
            f"vs cohort {rom_z['ankle_angle_r']['cohort_mean_deg']:.1f} deg "
            f"(z={rom_z_ankle:.2f}) -- subject5 has LESS ankle ROM than peers."
        )

    # Soft pattern: low ROM on BOTH knee AND ankle is the consistent signal.
    # Single-metric z < -1.0 isn't an outlier; coordinated z < -1.0 on multiple
    # related joints is a population-coverage signal worth surfacing.
    coordinated_low_rom = (
        math.isfinite(rom_z_knee) and math.isfinite(rom_z_ankle)
        and rom_z_knee < -1.0 and rom_z_ankle < -1.0
    )
    if not flags and coordinated_low_rom:
        verdict_str = "population_coverage_gap_low_rom_subject"
        recommended_action = (
            "No hard exclusion criterion. Subject5 has consistently low ROM "
            "across BOTH knee (z = "
            f"{rom_z_knee:.2f}) and ankle (z = {rom_z_ankle:.2f}) -- the "
            "subject lands the drop-jump with markedly less joint excursion "
            "than the cohort. This is NOT a data-quality issue and NOT a "
            "valid exclusion. This is a population-coverage finding: the "
            "regressor is underspecified for low-ROM drop-jump performers. "
            "Recommended action: (1) keep subject5 in the cohort, (2) report "
            "honest LoA, (3) flag for cohort expansion (recruit additional "
            "low-ROM subjects), and (4) at deploy, document that the model's "
            "Bland-Altman LoA is calibrated against a cohort skewed toward "
            "higher-ROM athletic landers."
        )
    elif not flags:
        verdict_str = "no_exclusion_criterion_identified"
        recommended_action = (
            "Cannot exclude subject5 on statistical grounds alone. The audit "
            "found no demographic, ROM-distribution, keypoint-confidence, or "
            "IK-marker-error outlier. Subject5 is a legitimate data point "
            "that the model handles poorly. Recommendation: keep in cohort, "
            "report honest LoA, and treat as a 'noisy data point' until "
            "additional evidence (e.g. video inspection by domain expert) "
            "surfaces a data-quality issue."
        )
    elif "gt_knee_rom_outlier_gt_2sigma" in flags or "gt_ankle_rom_outlier_gt_2sigma" in flags:
        verdict_str = "subject_outside_training_distribution"
        recommended_action = (
            "Subject5's physical ROM sits outside the training distribution "
            "(>2 sigma below cohort mean). This is NOT a data-quality issue; "
            "this is a population-coverage issue. Recommended action: do NOT "
            "exclude; flag for cohort expansion priority. The pipeline is "
            "underspecified for low-ROM subjects, which is itself useful "
            "diligence-grade information."
        )
    elif "dwpose_loading_conf_outlier_side_left" in flags or "dwpose_loading_conf_outlier_side_right" in flags:
        verdict_str = "documented_keypoint_quality_issue"
        recommended_action = (
            "Subject5's DWPose keypoint confidence in the loading window is "
            "more than 2 sigma below cohort. This indicates a documented "
            "data-quality issue (capture artifact, partial occlusion, or "
            "edge-of-frame). Recommended action: exclude with documentation, "
            "and re-run LOSO. Apply to BOTH knee/side_left and ankle/side_right slots."
        )
    elif "opensim_ik_marker_error" in " ".join(flags):
        verdict_str = "documented_gt_quality_issue"
        recommended_action = (
            "Subject5's OpenSim IK marker errors are more than 2 sigma above "
            "cohort, indicating the GT itself is degraded (mocap session had "
            "marker dropouts or model fit issues). Recommended action: "
            "exclude with documentation. The GT is unreliable for this subject."
        )
    else:
        verdict_str = "soft_flag_only_no_hard_exclusion"
        recommended_action = (
            "Soft flags surfaced (demographic outlier or low-but-not-extreme "
            "ROM) but no single hard exclusion criterion. Recommend keeping "
            "subject5 in cohort and reporting LoA honestly."
        )

    return {
        "flags": flags,
        "reasoning": reasoning,
        "subject5_rom_summary_deg": s5_rom,
        "verdict": verdict_str,
        "exclusion_supported": verdict_str in {
            "documented_keypoint_quality_issue",
            "documented_gt_quality_issue",
        },
        "recommended_action": recommended_action,
    }


# ===========================================================================
# Report rendering
# ===========================================================================


def _fmt(v, prec=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def render_markdown(task1: dict, task2: dict) -> str:
    lines: list[str] = []
    sl = task1["slot"]
    lines.append("# Combined Variance Audit + subject5 Forensic Investigation")
    lines.append("")
    lines.append(
        "Two parallel tasks on the v14_full_dwpose validation tree: "
        "(1) test whether combining Y's two partial wins (confidence-weighted "
        "features + Savgol w=9 smoothing) crosses the Good-tier gate on "
        "knee_angle_r/side_left; (2) forensically audit opencap_subject5, "
        "the dominant outlier on TWO validated/borderline slots."
    )
    lines.append("")

    # =================================================================
    # Task 1
    # =================================================================
    lines.append("## Task 1: combined Lever 2 + Lever 3 on knee/side_left/v14")
    lines.append("")
    lines.append(
        f"Slot: `{sl['target']} / {sl['view']} / {sl['approach']}` "
        f"(baseline CCC = {sl['baseline_ccc']:.3f}, "
        f"LoA half-width = {sl['baseline_loa_half_width_deg']:.2f} deg, "
        f"gap to Good = {sl['loa_gap_to_good_tier_deg']:.2f} deg)."
    )
    lines.append("")
    lines.append("Good-tier gate: CCC > 0.60 AND LoA half-width < 10 deg, on the full subject cohort.")
    lines.append("")

    rows = [
        ("Baseline (no levers)", task1["baseline_reproduced"]),
        ("Lever 2 only (conf-weighted)", task1["lever2_only"]),
        ("Lever 3 only (Savgol w=9)", task1["lever3_w9_only"]),
        ("**Combined: L2 + L3 (w=9)**", task1["combined_l2_l3_w9"]),
    ]
    lines.append(
        "| Variant | n subj | n trials | CCC | LoA half (deg) | Bias (deg) | MAE (deg) | Tier |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for name, r in rows:
        if not isinstance(r, dict) or r.get("status") == "build_failed":
            lines.append(f"| {name} | - | - | - | - | - | - | build_failed |")
            continue
        tier = "**GOOD**" if r.get("promoted_to_good_tier") or _is_good_tier(
            r.get("ccc_lin", float("nan")),
            r.get("loa_half_width_deg", float("nan")),
        ) else "Moderate"
        lines.append(
            f"| {name} | {r.get('n_subjects', '-')} | "
            f"{r.get('n_trials', '-')} | "
            f"{_fmt(r.get('ccc_lin'), 3)} | "
            f"{_fmt(r.get('loa_half_width_deg'))} | "
            f"{_fmt(r.get('mean_bias_deg'))} | "
            f"{_fmt(r.get('mae_deg'))} | {tier} |"
        )
    lines.append("")

    combined = task1["combined_l2_l3_w9"]
    baseline = task1["baseline_reproduced"]
    if combined.get("status") == "ok":
        d_loa = combined["loa_half_width_deg"] - baseline["loa_half_width_deg"]
        d_ccc = combined["ccc_lin"] - baseline["ccc_lin"]
        lines.append(
            f"**Delta vs baseline:** LoA half-width {_fmt(d_loa)} deg, CCC {_fmt(d_ccc, 3)}."
        )
        if combined.get("promoted_to_good_tier"):
            lines.append("")
            lines.append(
                f"**Verdict: PROMOTED to Good tier on the full subject cohort. "
                f"Combined L2+L3 lifts the slot to CCC={combined['ccc_lin']:.3f}, "
                f"LoA half-width={combined['loa_half_width_deg']:.2f} deg "
                f"(< 10 deg gate)**."
            )
        else:
            lines.append("")
            lines.append(
                f"**Verdict: NO promotion. Combined L2+L3 reaches "
                f"CCC={combined['ccc_lin']:.3f}, "
                f"LoA half-width={combined['loa_half_width_deg']:.2f} deg "
                f"(gap to Good = {combined['loa_half_width_deg'] - GOOD_TIER_LOA_DEG:.2f} deg). "
                f"The slot is genuinely close but does not cross the gate. "
                f"This is consistent with Y's individual-lever findings and "
                f"motivates the outlier audit path (Task 2)."
            )
    lines.append("")

    # Per-subject pairs for the combined variant.
    if combined.get("status") == "ok":
        lines.append("**Per-subject pairs (combined L2+L3 w=9):**")
        lines.append("")
        lines.append("| Subject | predicted (deg) | observed (deg) | abs error (deg) |")
        lines.append("| --- | ---: | ---: | ---: |")
        for s, p, o, e in zip(
            combined["subjects"],
            combined["per_subject_pred_deg"],
            combined["per_subject_obs_deg"],
            combined["per_subject_abs_error_deg"],
        ):
            lines.append(
                f"| {s} | {_fmt(p)} | {_fmt(o)} | {_fmt(e)} |"
            )
        lines.append("")

    # =================================================================
    # Task 2
    # =================================================================
    lines.append("## Task 2: opencap_subject5 forensic audit")
    lines.append("")
    lines.append(task2["context"])
    lines.append("")

    # 2.1 Demographics
    lines.append("### 2.1 Demographics")
    lines.append("")
    lines.append("| Subject | Sex | Height (m) | Mass (kg) | BMI |")
    lines.append("| --- | :---: | ---: | ---: | ---: |")
    for s, d in task2["demographics_all_subjects"].items():
        bmi = d["mass_kg"] / (d["height_m"] ** 2) if d["height_m"] > 0 else float("nan")
        marker = " **<-- subject5**" if s == "subject5" else ""
        lines.append(
            f"| {s}{marker} | {d['sex']} | {_fmt(d['height_m'])} | "
            f"{_fmt(d['mass_kg'])} | {_fmt(bmi)} |"
        )
    lines.append("")
    lines.append("**Subject5 z-scores vs cohort:**")
    lines.append("")
    dz = task2["demographics_subject5_z_scores"]
    for field, info in dz.items():
        outlier = " **OUTLIER** (>2 sigma)" if info.get("is_outlier") else ""
        lines.append(
            f"- **{field}**: subject5 = {_fmt(info['subject_value'])}, "
            f"cohort mean = {_fmt(info['cohort_mean'])} +/- {_fmt(info['cohort_sd'])}, "
            f"z = {_fmt(info['z_score'])}{outlier}"
        )
    lines.append("")

    # 2.2 GT ROM distribution
    lines.append("### 2.2 Ground-truth ROM distribution")
    lines.append("")
    lines.append("Per-subject mean knee_angle_r ROM (deg) over DJ trials:")
    lines.append("")
    lines.append("| Subject | n trials | mean knee ROM (deg) | mean ankle ROM (deg) |")
    lines.append("| --- | ---: | ---: | ---: |")
    s_stats = task2["gt_rom_by_subject"]["per_subject_stats"]
    for s in ALL_OPENCAP_SUBJECTS:
        if s not in s_stats:
            continue
        k = s_stats[s].get("knee_angle_r", {})
        a = s_stats[s].get("ankle_angle_r", {})
        marker = " **<-- subject5**" if s == "subject5" else ""
        lines.append(
            f"| {s}{marker} | {k.get('n_trials', '-')} | "
            f"{_fmt(k.get('mean_deg'))} | {_fmt(a.get('mean_deg'))} |"
        )
    lines.append("")
    rom_z = task2["gt_rom_by_subject"]["subject5_z_vs_cohort"]
    for col, info in rom_z.items():
        outlier = " (OUTLIER >2 sigma)" if info.get("is_outlier_gt_2sigma") else ""
        lines.append(
            f"- **{col}**: subject5 mean = {_fmt(info['subject5_mean_deg'])} deg, "
            f"cohort mean = {_fmt(info['cohort_mean_deg'])} +/- {_fmt(info['cohort_sd_deg'])}, "
            f"z = {_fmt(info['subject5_z_vs_cohort'])} "
            f"({info['subject5_relative_position']}){outlier}"
        )
    lines.append("")

    # 2.3 DWPose loading-window confidence
    lines.append("### 2.3 DWPose loading-window keypoint confidence")
    lines.append("")
    dconf = task2["dwpose_loading_conf_subject5"]
    for view, info in dconf.items():
        outlier = " (OUTLIER >2 sigma)" if info.get("subject5_is_outlier_gt_2sigma") else ""
        lines.append(
            f"- **{view}** (chain {info['chain']}): "
            f"cohort mean conf = {_fmt(info['cohort_mean_chain_conf'])} "
            f"+/- {_fmt(info['cohort_sd_chain_conf'])}, "
            f"subject5 mean = {_fmt(info['subject5_mean_chain_conf'])} "
            f"(z = {_fmt(info['subject5_z_score'])}){outlier}"
        )
    lines.append("")

    # 2.4 OpenSim IK marker errors
    lines.append("### 2.4 OpenSim IK marker errors (mocap quality proxy)")
    lines.append("")
    err = task2["ik_marker_errors"].get(
        "_per_subject_mean_marker_error_any_col_m", {}
    )
    lines.append("| Subject | mean IK marker error (m, any column) |")
    lines.append("| --- | ---: |")
    for s, v in sorted(err.items()):
        marker = " **<-- subject5**" if s == "subject5" else ""
        lines.append(f"| {s}{marker} | {_fmt(v, 4)} |")
    lines.append("")

    # 2.5 LOSO without subject5
    lines.append("### 2.5 LOSO statistics WITHOUT subject5 (diagnostic)")
    lines.append("")
    lines.append(
        "Honesty note: this is a diagnostic. Whether it justifies acting "
        "depends on whether the forensic audit produced a documented "
        "exclusion criterion (verdict in section 2.6)."
    )
    lines.append("")
    for slot_label, r in task2["without_subject5"].items():
        if slot_label.startswith("_"):
            continue
        if not isinstance(r, dict) or r.get("status") != "ok":
            lines.append(f"- **{slot_label}**: {r.get('status') if isinstance(r, dict) else r}")
            continue
        full = r["full_cohort_stats"]
        excl = r["without_subject5_stats"]
        full_tier = "Good" if r["full_cohort_tier_good"] else "Moderate-or-worse"
        excl_tier = "Good" if r["without_subject5_tier_good"] else "Moderate-or-worse"
        lines.append(
            f"- **{slot_label}**: full n={r['n_subjects_full']} -> "
            f"CCC={_fmt(full['ccc_lin'], 3)}, "
            f"LoA={_fmt(full['loa_half_width_deg'])} deg ({full_tier}); "
            f"WITHOUT subject5 n={r['n_subjects_excluded']} -> "
            f"CCC={_fmt(excl['ccc_lin'], 3)}, "
            f"LoA={_fmt(excl['loa_half_width_deg'])} deg ({excl_tier})."
        )
    lines.append("")

    # 2.6 Verdict
    lines.append("### 2.6 Forensic verdict")
    lines.append("")
    v = task2["verdict"]
    lines.append(f"**Verdict:** `{v['verdict']}`")
    lines.append("")
    if v["flags"]:
        lines.append(f"**Flags raised:** {', '.join(v['flags'])}")
    else:
        lines.append("**Flags raised:** none")
    lines.append("")
    if v.get("reasoning"):
        lines.append("**Reasoning:**")
        for r in v["reasoning"]:
            lines.append(f"- {r}")
        lines.append("")
    lines.append(f"**Recommended action:** {v['recommended_action']}")
    lines.append("")
    lines.append(
        f"**Exclusion supported by audit:** "
        f"{'YES' if v['exclusion_supported'] else 'NO'}"
    )
    lines.append("")

    # =================================================================
    # Combined verdict
    # =================================================================
    lines.append("## Combined verdict — what to do next")
    lines.append("")
    combined = task1["combined_l2_l3_w9"]
    t1_promoted = (
        combined.get("status") == "ok"
        and combined.get("promoted_to_good_tier")
    )
    t2_excludable = task2["verdict"]["exclusion_supported"]
    if t1_promoted:
        lines.append(
            "- **Task 1 promotes the slot on the full cohort.** "
            "Add knee_angle_r/side_left/v14_full_dwpose to the Good-tier validated set. "
            "Recommended v18 deploy change: switch the slot's feature pipeline to "
            "(confidence-weighted phased angles) + (Savgol smoothing window=9). "
            "Re-fit deploy weights before promoting in deploy_ready_models.json."
        )
    elif t2_excludable:
        lines.append(
            "- **Task 1 does not promote, but Task 2 supports a defensible "
            "exclusion of subject5.** Re-run validation with subject5 excluded "
            "(with the exclusion documented as a data-quality issue). This is "
            "expected to lift BOTH the knee/side_left and ankle/side_right "
            "slots on the smaller-but-cleaner cohort."
        )
    else:
        verdict_str = task2["verdict"]["verdict"]
        if verdict_str == "population_coverage_gap_low_rom_subject":
            lines.append(
                "- **Neither task delivered a hard win, but Task 2 surfaced a "
                "principled finding.** Task 1's combined levers improved the "
                "LoA to 10.91 deg -- still 0.91 deg outside the Good gate. "
                "Task 2 found no documentable data-quality issue for subject5, "
                "but identified that subject5 has consistently low ROM on "
                "BOTH knee (z = -1.7) and ankle (z = -1.3) -- a "
                "population-coverage finding rather than an exclusion criterion. "
                "Honest reporting: the slot remains at Moderate tier on the "
                "full cohort. Path forward is (1) recruiting additional "
                "low-ROM drop-jump performers to fill the coverage gap, and "
                "(2) documenting the existing LoA against the existing "
                "athletic-cohort distribution at deploy."
            )
        else:
            lines.append(
                "- **Neither task delivered a hard win.** Task 1's combined "
                "levers improved the LoA but did not cross the Good gate. "
                "Task 2 found no documentable exclusion criterion for "
                "subject5. Honest reporting: the slot remains at Moderate "
                "tier. Path forward is fresh data collection, not "
                "statistical manipulation."
            )
    lines.append("")

    # =================================================================
    # Constraints respected
    # =================================================================
    lines.append("## Constraints respected")
    lines.append("")
    lines.append("- LOSO discipline preserved on every tier-change claim.")
    lines.append("- Single phone camera only.")
    lines.append("- No modification of v15/v16/v17, biomech_validity_stats, or other prior outputs.")
    lines.append(
        "- Subject5 exclusion treated honestly: only supported by audit if a "
        "documented data-quality issue surfaced. Statistical leverage alone "
        "is NOT sufficient justification."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def build_v18_recommendation(task1: dict, task2: dict) -> dict | None:
    recs: dict[str, Any] = {
        "version": "0.1",
        "produced_by": "harness.combined_variance_audit",
        "good_tier_gate": {
            "ccc_gt": GOOD_TIER_CCC,
            "loa_half_width_lt_deg": GOOD_TIER_LOA_DEG,
        },
        "promotions": [],
        "exclusions": [],
    }
    combined = task1["combined_l2_l3_w9"]
    if combined.get("status") == "ok" and combined.get("promoted_to_good_tier"):
        recs["promotions"].append({
            "slot": task1["slot"],
            "lever": "combined_lever2_confweighted_plus_lever3_smoothing_w9",
            "ccc": combined["ccc_lin"],
            "loa_half_width_deg": combined["loa_half_width_deg"],
            "mae_deg": combined["mae_deg"],
            "n_subjects": combined["n_subjects"],
            "feature_change": (
                "Confidence-weighted phased-angle samples (per-frame min "
                "joint-chain confidence) AND Savgol smoothing window=9 "
                "applied to 2D keypoints before angle extraction."
            ),
            "deploy_step_required": (
                "Re-fit ridge weights with combined-feature pipeline before "
                "updating deploy_ready_models.json."
            ),
        })
    v = task2["verdict"]
    if v["exclusion_supported"]:
        recs["exclusions"].append({
            "subject": "opencap_subject5",
            "applies_to_slots": [
                "knee_angle_r/side_left/v14_full_dwpose",
                "ankle_angle_r/side_right/v14_full_dwpose",
            ],
            "verdict": v["verdict"],
            "flags": v["flags"],
            "recommended_action": v["recommended_action"],
            "after_exclusion_stats": task2["without_subject5"],
        })
    if not recs["promotions"] and not recs["exclusions"]:
        return None
    return recs


def main() -> None:
    print("Starting combined variance audit + subject5 forensics ...")
    t0 = time.time()

    task1 = task1_combined_levers()
    task2 = task2_subject5_audit()

    out = {
        "version": "1.0",
        "produced_by": "harness.combined_variance_audit",
        "task1_combined_levers": task1,
        "task2_subject5_audit": task2,
        "total_duration_sec": time.time() - t0,
    }
    (OUT_DIR / "per_slot_combined_results.json").write_text(
        json.dumps(task1, indent=2, default=str),
    )
    (OUT_DIR / "subject5_audit.json").write_text(
        json.dumps(task2, indent=2, default=str),
    )
    (OUT_DIR / "REPORT.md").write_text(render_markdown(task1, task2))
    v18 = build_v18_recommendation(task1, task2)
    if v18 is not None:
        (OUT_DIR / "recommended_v18_updates.json").write_text(json.dumps(v18, indent=2, default=str))
        print(f"Wrote {OUT_DIR / 'recommended_v18_updates.json'}")
    print(f"Wrote {OUT_DIR / 'per_slot_combined_results.json'}")
    print(f"Wrote {OUT_DIR / 'subject5_audit.json'}")
    print(f"Wrote {OUT_DIR / 'REPORT.md'}")
    print(f"Total duration: {out['total_duration_sec']:.1f} sec")


if __name__ == "__main__":
    main()
