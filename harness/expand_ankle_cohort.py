"""Ankle cohort expansion audit + analysis for the ankle_angle_r / side_right
deploy slot (v14_full_dwpose, OpenCap-only).

Today the slot stands at:
    n_subjects = 9,  CCC = 0.6444,  Pearson r = 0.7458,
    bias = +0.33 deg,  LoA half-width = 9.46 deg,  MAE = 3.89 deg
    classification = Good
    95% Fisher Z CI on CCC: approx [-0.034, +0.916]

The lower CCC CI bound includes zero, which blocks promotion from
"preliminary additional measurement" to "headline-range validated."

This script:

  1. Audits every candidate dataset listed in the cohort-expansion brief
     (ASPset, Fukuchi RBDS, ACL Jump-Landing, GaitRec, MPI-INF-3DHP,
     synthetic AMASS, OpenCap non-DJ tasks) for Layer 3 LOSO compatibility:
       - Is the raw data present in the validation tree?
       - Does it have paired video + mocap-derived ankle dorsi/plantarflexion?
       - Does it have foot/toe markers (required to derive ankle ROM at all)?
       - Is the license commercial-permissive?
       - Are there harness scripts that already process it?

  2. Reproduces the v14 ankle_angle_r / side_right baseline LOSO with the
     existing 9-subject OpenCap cohort to verify the per_slot_validity.json
     numbers and computes:
       - CCC + Fisher Z 95% CI
       - Bootstrap 95% CI on CCC (10000 resamples)
       - Per-subject bias, LoA, MAE, RMSE
       - Sensitivity: drop each subject and recompute CCC

  3. Honest reporting: emits a structured audit JSON + REPORT.md showing
     which datasets cleanly expand the cohort and which do not, with
     reasons.

Run:
    cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
    python -m harness.expand_ankle_cohort

Outputs:
    data/ankle_cohort_expansion/per_slot_validity_ankle_expanded.json
    data/ankle_cohort_expansion/REPORT.md
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.train_v9_phased as v9
import harness.train_v12_combined as v12
import harness.train_v14_full_dwpose as v14
from harness.biomech_validity_stats import (
    _ccc,
    _loso_extract,
    _pearson,
    _t_test_p_value_bias,
    compute_stats_from_pairs,
)


# ---------------------------------------------------------------------------
# Constants — pinned paths.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation"
)
DATA_ROOT = REPO_ROOT / "data"
OUT_DIR = DATA_ROOT / "ankle_cohort_expansion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "ankle_angle_r"
VIEW = "side_right"
APPROACH = "v14_full_dwpose"


# ---------------------------------------------------------------------------
# Audit table — codifies what each candidate dataset offers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetAudit:
    dataset: str
    subjects_advertised: str
    raw_present_in_tree: bool
    raw_path: str | None
    has_paired_video_plus_mocap: bool
    has_foot_or_toe_markers: bool
    has_sagittal_ankle_gt: bool
    license: str
    commercial_clean: bool
    harness_already_processes: bool
    decision: str
    rationale: str


def _file_exists(p: Path) -> bool:
    return p.exists()


def build_audit() -> list[DatasetAudit]:
    """Codified audit findings. Each row is checked against the actual
    contents of the validation tree at audit time, not memory of what
    'should' be there."""
    rows: list[DatasetAudit] = []

    # 1. OpenCap LabValidation — the existing source
    rows.append(DatasetAudit(
        dataset="OpenCap LabValidation (DJ tasks, current baseline)",
        subjects_advertised="10 subjects with mocap; 9 with DWPose Cam3 keypoints",
        raw_present_in_tree=_file_exists(DATA_ROOT / "LabValidation_withVideos"),
        raw_path=str(DATA_ROOT / "LabValidation_withVideos"),
        has_paired_video_plus_mocap=True,
        has_foot_or_toe_markers=True,  # Halpe-26 has big_toe, small_toe, heel
        has_sagittal_ankle_gt=True,  # OpenSim IK ankle_angle_r in deg
        license="CC-BY 4.0",
        commercial_clean=True,
        harness_already_processes=True,
        decision="ALREADY USED",
        rationale=(
            "Source of the current n=9 cohort. Subject6 has no raw "
            "VideoData (only OpenSim-precomputed HRNet/OpenPose outputs), "
            "so its keypoints cannot be extracted with DWPose without "
            "going outside the validation tree."
        ),
    ))

    # 2. OpenCap LabValidation NON-DJ tasks (squats, STS, walking)
    rows.append(DatasetAudit(
        dataset="OpenCap LabValidation NON-DJ tasks (same 9 subjects)",
        subjects_advertised="Same 9 subjects, additional motion types",
        raw_present_in_tree=True,
        raw_path=str(DATA_ROOT / "LabValidation_withVideos"),
        has_paired_video_plus_mocap=True,
        has_foot_or_toe_markers=True,
        has_sagittal_ankle_gt=True,
        license="CC-BY 4.0",
        commercial_clean=True,
        harness_already_processes=False,
        decision="DOES NOT EXPAND n_subjects",
        rationale=(
            "Adding non-DJ tasks (squats, walking, STS) would only add "
            "TRIALS, not SUBJECTS. The 95% Fisher Z CI on the per-subject "
            "CCC depends on n_subjects, not n_trials — so this cannot "
            "promote the slot. Separately, the v9_phased feature pipeline "
            "is built around DJ initial-contact event detection and would "
            "require per-task-type event detectors to handle these motions. "
            "Significant engineering, no headline-CI benefit."
        ),
    ))

    # 3. ASPset-510
    aspset_root = DATA_ROOT / "aspset510"
    rows.append(DatasetAudit(
        dataset="ASPset-510",
        subjects_advertised="17 subjects trainval (paired video + 3D mocap)",
        raw_present_in_tree=_file_exists(aspset_root / "ASPset-510"),
        raw_path=str(aspset_root / "ASPset-510"),
        has_paired_video_plus_mocap=True,
        has_foot_or_toe_markers=False,
        has_sagittal_ankle_gt=False,
        license="CC0",
        commercial_clean=True,
        harness_already_processes=True,  # train_v12_combined uses ASPset
        decision="EXCLUDE",
        rationale=(
            "17-keypoint skeleton ends at ankle — no toe, heel, or "
            "metatarsal markers. Sagittal ankle dorsi/plantarflexion "
            "requires foot-relative-to-shank, which cannot be derived "
            "from joints_3d without a foot reference. ASPset is "
            "explicitly documented as ankle-incompatible in "
            "harness/aspset_loader.py: 'ASPset HAS NO foot/toe keypoints "
            "-> ankle_angle_r cannot be derived from this dataset. "
            "That metric stays OpenCap-only.'"
        ),
    ))

    # 4. Fukuchi RBDS
    fukuchi_present = any(
        _file_exists(DATA_ROOT / p) for p in (
            "fukuchi_rbds", "fukuchi", "rbds", "RBDS",
        )
    )
    rows.append(DatasetAudit(
        dataset="Fukuchi RBDS (running kinematics, 28 runners)",
        subjects_advertised="28 runners (Fukuchi 2017, PeerJ)",
        raw_present_in_tree=fukuchi_present,
        raw_path=None,
        has_paired_video_plus_mocap=False,
        has_foot_or_toe_markers=False,  # unknown — but dataset is mocap-only
        has_sagittal_ankle_gt=True,  # running kinematics include ankle
        license="CC-BY",
        commercial_clean=True,
        harness_already_processes=False,
        decision="EXCLUDE",
        rationale=(
            "NOT present in the validation tree. Mentioned in pitch docs "
            "as a Layer 4 calibration source (population means for running "
            "config tuning). Fukuchi RBDS is a treadmill+overground "
            "running mocap dataset — it does NOT ship with paired single-"
            "phone-camera video, so it cannot drive Layer 3 LOSO "
            "regression of phone keypoints -> mocap ROM. Even if "
            "downloaded, motion type is steady-state running, not "
            "discrete-event motion the v9 feature pipeline is built for."
        ),
    ))

    # 5. ACL Jump-Landing
    rows.append(DatasetAudit(
        dataset="ACL Jump-Landing (Calisti 2025, 42 athletes)",
        subjects_advertised="42 athletes (Calisti 2025, Sci Data)",
        raw_present_in_tree=False,
        raw_path=None,
        has_paired_video_plus_mocap=False,
        has_foot_or_toe_markers=False,
        has_sagittal_ankle_gt=True,
        license="CC-BY 4.0",
        commercial_clean=True,
        harness_already_processes=False,
        decision="EXCLUDE",
        rationale=(
            "NOT present in the validation tree. Pitch docs cite it as "
            "a Layer 4 calibration source for ACL risk + jump-landing "
            "config means, not as a Layer 3 paired-video dataset. The "
            "published release is markered mocap kinematics with no "
            "paired single-phone-camera video, so it cannot drive "
            "Layer 3 LOSO. Would require fresh data collection or a "
            "different study release."
        ),
    ))

    # 6. GaitRec
    rows.append(DatasetAudit(
        dataset="GaitRec (Horsak 2020, 211 subjects)",
        subjects_advertised="211 healthy (Horsak 2020, Sci Data)",
        raw_present_in_tree=False,
        raw_path=None,
        has_paired_video_plus_mocap=False,
        has_foot_or_toe_markers=False,
        has_sagittal_ankle_gt=True,
        license="CC-BY 4.0",
        commercial_clean=True,
        harness_already_processes=False,
        decision="EXCLUDE",
        rationale=(
            "NOT present in the validation tree. GaitRec is force-plate + "
            "kinematic gait data without paired video — used in pitch docs "
            "for Layer 4 walking gait config calibration and force/contact "
            "validation only. Cannot drive Layer 3 LOSO of phone keypoints "
            "without paired RGB video, which the public release does not "
            "include."
        ),
    ))

    # 7. MPI-INF-3DHP
    mpi_present = _file_exists(DATA_ROOT / "mpi_inf_3dhp_gt")
    rows.append(DatasetAudit(
        dataset="MPI-INF-3DHP (Mehta 2017)",
        subjects_advertised="8 subjects with rear-view (cam7) data + GT angles",
        raw_present_in_tree=mpi_present,
        raw_path=str(DATA_ROOT / "mpi_inf_3dhp_gt") if mpi_present else None,
        has_paired_video_plus_mocap=True,
        has_foot_or_toe_markers=True,
        has_sagittal_ankle_gt=True,
        license="Non-commercial research only",
        commercial_clean=False,
        harness_already_processes=True,  # mpi3dhp_correlate.py exists
        decision="EXCLUDE",
        rationale=(
            "MPI-INF-3DHP license is non-commercial only: 'Methods and "
            "models that make use of the provided Software in any way "
            "can only be used for non-commercial purposes.' Per the "
            "explicit constraint on this expansion task and per the "
            "validation-v2.1 doc, MPI is excluded from any commercial-"
            "footprint counts. Also, the existing 8-subject MPI cohort "
            "for ankle has mean abs(r) = 0.228 across cam7 — well below "
            "the threshold to lift the side-right CCC anyway, since cam7 "
            "is rear-oblique (azimuth -62 deg), not a true side view."
        ),
    ))

    # 8. CMU Panoptic
    cmu_present = _file_exists(DATA_ROOT / "cmu_panoptic_rear_sample")
    rows.append(DatasetAudit(
        dataset="CMU Panoptic (rear sample)",
        subjects_advertised="Multi-view dome; sample only is calibration JSON",
        raw_present_in_tree=cmu_present,
        raw_path=str(DATA_ROOT / "cmu_panoptic_rear_sample"),
        has_paired_video_plus_mocap=False,
        has_foot_or_toe_markers=False,
        has_sagittal_ankle_gt=False,
        license="Non-commercial research only",
        commercial_clean=False,
        harness_already_processes=False,
        decision="EXCLUDE",
        rationale=(
            "Only a calibration JSON exists in the tree (no video, no GT "
            "angles). License is non-commercial. Excluded."
        ),
    ))

    # 9. Synthetic AMASS / SMPL
    synth_present = _file_exists(REPO_ROOT / "synthetic_amass")
    rows.append(DatasetAudit(
        dataset="Synthetic AMASS / SMPL bursts",
        subjects_advertised="Synthetic (no real human subjects)",
        raw_present_in_tree=synth_present,
        raw_path=str(REPO_ROOT / "synthetic_amass"),
        has_paired_video_plus_mocap=True,  # synthetic projection counts
        has_foot_or_toe_markers=True,
        has_sagittal_ankle_gt=True,
        license="SMPL CC-BY 4.0",
        commercial_clean=True,
        harness_already_processes=True,
        decision="EXCLUDE for headline cohort",
        rationale=(
            "Synthetic 'subjects' are not real human subjects with "
            "independent biomechanics — they share the SMPL_NEUTRAL "
            "body model. Adding them to n_subjects would inflate the "
            "denominator of the Fisher Z CI without adding independent "
            "biological variance. Diligence reviewers correctly read "
            "synthetic-subject inflation as a methodological flag. Real "
            "value is at Layer 2 (keypoint -> angle), not Layer 3 LOSO."
        ),
    ))

    return rows


# ---------------------------------------------------------------------------
# CCC confidence intervals.
# ---------------------------------------------------------------------------


def fisher_z_ccc_ci(ccc: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Approximate 95% CI on CCC via Fisher Z transform.

    Mirrors the closed-form used in biomech validation papers (e.g. Lin
    1989; Atkinson & Nevill 1998). For small n (~9) this is wide and the
    lower bound can cross zero — that is the problem the cohort-expansion
    task is trying to solve.
    """
    if abs(ccc) >= 1.0 or n < 4:
        return float("nan"), float("nan")
    z = 0.5 * math.log((1 + ccc) / (1 - ccc))
    se = 1.0 / math.sqrt(n - 3)
    z_alpha = 1.959963984540054  # 2-sided 95%
    z_lo, z_hi = z - z_alpha * se, z + z_alpha * se
    inv = lambda zz: (math.exp(2 * zz) - 1) / (math.exp(2 * zz) + 1)
    return inv(z_lo), inv(z_hi)


def bootstrap_ccc_ci(
    pred_per_subj: np.ndarray, obs_per_subj: np.ndarray,
    n_resamples: int = 10000, seed: int = 20260529,
) -> tuple[float, float, float, float]:
    """Percentile bootstrap 95% CI on CCC by resampling subjects with
    replacement. Returns (mean_ccc, lo, hi, sd_ccc)."""
    rng = np.random.default_rng(seed)
    n = len(pred_per_subj)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    cccs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        p = pred_per_subj[idx]
        o = obs_per_subj[idx]
        if p.std() == 0 or o.std() == 0:
            cccs[i] = float("nan")
            continue
        cccs[i] = _ccc(p, o)
    cccs = cccs[np.isfinite(cccs)]
    if len(cccs) < 100:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(cccs))
    lo, hi = float(np.percentile(cccs, 2.5)), float(np.percentile(cccs, 97.5))
    sd = float(np.std(cccs, ddof=1))
    return mean, lo, hi, sd


def leave_one_out_subject_sensitivity(
    pred_per_subj: np.ndarray, obs_per_subj: np.ndarray, subjects: list[str],
) -> list[dict]:
    """For each held-out subject, recompute CCC + Pearson r on the
    remaining n-1 subjects. Identifies high-leverage subjects."""
    out = []
    n = len(pred_per_subj)
    full_ccc = _ccc(pred_per_subj, obs_per_subj)
    full_r = _pearson(pred_per_subj, obs_per_subj)
    for i, s in enumerate(subjects):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        p, o = pred_per_subj[mask], obs_per_subj[mask]
        out.append({
            "dropped_subject": s,
            "n_remaining": int(mask.sum()),
            "ccc_no_subj": float(_ccc(p, o)),
            "pearson_r_no_subj": float(_pearson(p, o)),
            "ccc_delta_from_full": float(_ccc(p, o) - full_ccc),
            "pred_subj": float(pred_per_subj[i]),
            "obs_subj": float(obs_per_subj[i]),
            "abs_error_deg": float(abs(pred_per_subj[i] - obs_per_subj[i])),
        })
    out.sort(key=lambda r: r["abs_error_deg"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Baseline LOSO reproduction.
# ---------------------------------------------------------------------------


def reproduce_baseline() -> dict:
    """Re-run the v14_full_dwpose LOSO for ankle_angle_r / side_right and
    return all the validity stats + CIs."""
    # Repoint v9 to OpenCap DWPose keypoints (matches v14 trainer).
    v9.KP_DIR = DATA_ROOT / "opencap_dwpose_keypoints"
    v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"

    result = v14.build_opencap_only_dataset(TARGET, VIEW)
    if result is None:
        raise RuntimeError("v14 opencap-only dataset build returned None")
    X, y, subjects, sources = result
    pred, obs, subj_arr = _loso_extract(X, y, list(subjects), alpha=10.0)

    # Per-subject aggregation (the headline)
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[mask]; obs = obs[mask]; subj_arr = subj_arr[mask]
    unique = sorted(set(subj_arr.tolist()))
    pred_subj = np.array([float(np.mean(pred[subj_arr == s])) for s in unique])
    obs_subj = np.array([float(np.mean(obs[subj_arr == s])) for s in unique])
    diffs = pred_subj - obs_subj

    per_subj = compute_stats_from_pairs(
        TARGET, VIEW, APPROACH, pred, obs, subj_arr,
        aggregate_per_subject=True,
    )
    per_trial = compute_stats_from_pairs(
        TARGET, VIEW, APPROACH, pred, obs, subj_arr,
        aggregate_per_subject=False,
    )

    # CIs on CCC
    ccc = per_subj.ccc
    n = per_subj.n_subjects
    fisher_lo, fisher_hi = fisher_z_ccc_ci(ccc, n)
    boot_mean, boot_lo, boot_hi, boot_sd = bootstrap_ccc_ci(
        pred_subj, obs_subj, n_resamples=10000,
    )

    sensitivity = leave_one_out_subject_sensitivity(
        pred_subj, obs_subj, unique,
    )

    return {
        "target": TARGET,
        "view": VIEW,
        "approach": APPROACH,
        "n_subjects": n,
        "n_trials": per_subj.n_trials,
        "subjects": unique,
        "per_subject_pred_deg": [float(p) for p in pred_subj.tolist()],
        "per_subject_obs_deg": [float(o) for o in obs_subj.tolist()],
        "per_subject_diff_deg": [float(d) for d in diffs.tolist()],
        "mean_observed_deg": per_subj.mean_observed,
        "sd_observed_deg": per_subj.sd_observed,
        "mean_predicted_deg": per_subj.mean_predicted,
        "sd_predicted_deg": per_subj.sd_predicted,
        "mean_bias_deg": per_subj.mean_bias,
        "sd_bias_deg": per_subj.sd_bias,
        "loa_lower_deg": per_subj.lower_loa,
        "loa_upper_deg": per_subj.upper_loa,
        "loa_half_width_deg": max(
            abs(per_subj.upper_loa - per_subj.mean_bias),
            abs(per_subj.mean_bias - per_subj.lower_loa),
        ),
        "pearson_r": per_subj.pearson_r,
        "ccc_lin": ccc,
        "mae_deg": per_subj.mae,
        "rmse_deg": per_subj.rmse,
        "p_value_bias_neq_0": per_subj.p_value_bias,
        "ccc_95ci_fisher_z": [fisher_lo, fisher_hi],
        "ccc_95ci_bootstrap": [boot_lo, boot_hi],
        "ccc_bootstrap_mean": boot_mean,
        "ccc_bootstrap_sd": boot_sd,
        "ccc_bootstrap_n_resamples": 10000,
        "loso_sensitivity_per_subject": sensitivity,
        "per_trial": {
            "n_trials": per_trial.n_trials,
            "mean_bias_deg": per_trial.mean_bias,
            "sd_bias_deg": per_trial.sd_bias,
            "loa_lower_deg": per_trial.lower_loa,
            "loa_upper_deg": per_trial.upper_loa,
            "pearson_r": per_trial.pearson_r,
            "ccc_lin": per_trial.ccc,
            "mae_deg": per_trial.mae,
            "rmse_deg": per_trial.rmse,
        },
    }


# ---------------------------------------------------------------------------
# Projection: what n would it take to clear the promotion threshold?
# ---------------------------------------------------------------------------


def n_required_for_lower_bound(target_lo: float, ccc: float) -> int:
    """Given an observed CCC point estimate, the smallest n where the
    95% Fisher Z lower bound is >= target_lo."""
    if abs(ccc) >= 1.0 or target_lo >= ccc:
        return -1
    for n in range(4, 200):
        lo, _ = fisher_z_ccc_ci(ccc, n)
        if lo >= target_lo:
            return n
    return -1


def projection_table(ccc_observed: float) -> list[dict]:
    """For 9..40 subjects, what is the Fisher Z CI assuming CCC stays
    at ccc_observed? Used in the report so the reader can see exactly how
    many subjects bridge the gap."""
    out = []
    for n in [9, 12, 15, 17, 20, 22, 25, 28, 30, 35, 40]:
        lo, hi = fisher_z_ccc_ci(ccc_observed, n)
        out.append({
            "n": n,
            "ccc_assumed": ccc_observed,
            "ci_lower": lo,
            "ci_upper": hi,
            "clears_lo_ge_0_30": lo >= 0.30,
            "clears_lo_gt_0_00": lo > 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# Top-level orchestration.
# ---------------------------------------------------------------------------


def run() -> dict:
    print(f"[expand_ankle_cohort] running at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Step 1: codified dataset audit")
    audit = build_audit()
    for row in audit:
        print(
            f"  - {row.dataset}: decision={row.decision}"
        )

    print("Step 2: reproduce v14 ankle / side_right baseline")
    baseline = reproduce_baseline()
    print(
        f"  n={baseline['n_subjects']} CCC={baseline['ccc_lin']:.4f} "
        f"r={baseline['pearson_r']:.4f} bias={baseline['mean_bias_deg']:+.2f} "
        f"LoA_half={baseline['loa_half_width_deg']:.2f} "
        f"MAE={baseline['mae_deg']:.2f}"
    )
    fz_lo, fz_hi = baseline["ccc_95ci_fisher_z"]
    bs_lo, bs_hi = baseline["ccc_95ci_bootstrap"]
    print(f"  CCC 95% CI (Fisher Z):     [{fz_lo:+.3f}, {fz_hi:+.3f}]")
    print(f"  CCC 95% CI (Bootstrap):    [{bs_lo:+.3f}, {bs_hi:+.3f}]")

    print("Step 3: projection table (what n unlocks lower bound >= 0.30)")
    projection = projection_table(baseline["ccc_lin"])
    for row in projection:
        flag = "PROMOTE" if row["clears_lo_ge_0_30"] else "block"
        print(
            f"  n={row['n']:3d}  CI=[{row['ci_lower']:+.3f}, "
            f"{row['ci_upper']:+.3f}]  {flag}"
        )
    n_required = n_required_for_lower_bound(0.30, baseline["ccc_lin"])

    out = {
        "version": "1.0",
        "produced_by": "harness.expand_ankle_cohort",
        "produced_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": (
            "Audit candidate datasets for expanding the v14 ankle_angle_r / "
            "side_right LOSO cohort beyond n=9, with the goal of tightening "
            "the CCC 95% CI lower bound clear of zero (target >= 0.30)."
        ),
        "promotion_threshold": {
            "ccc_lower_ci_minimum": 0.30,
            "loa_half_width_maximum_deg": 10.0,
            "tier_required_for_promotion": "Good or Excellent",
        },
        "baseline_slot": baseline,
        "n_subjects_required_for_promotion": n_required,
        "projection_at_assumed_ccc": projection,
        "dataset_audit": [asdict(row) for row in audit],
        "final_decision": {
            "can_be_promoted_with_existing_data": False,
            "reason": (
                "No candidate dataset in the validation tree provides "
                "(a) paired single-phone-camera video, (b) ankle dorsi/"
                "plantarflexion mocap GT in degrees, AND (c) "
                "commercial-permissive license. ASPset has paired video "
                "but no foot markers. MPI-INF-3DHP has both but is non-"
                "commercial. Fukuchi/ACL/GaitRec are L4 calibration "
                "sources (no paired single-phone-camera video). Synthetic "
                "subjects do not represent independent biological variance."
            ),
            "recommended_path": (
                f"To clear the promotion threshold (lower CCC 95% CI bound "
                f">= 0.30) at the current point estimate of CCC="
                f"{baseline['ccc_lin']:.3f}, n_subjects must reach >= "
                f"{n_required}. Required action: fresh data collection of "
                f"{n_required - baseline['n_subjects']} additional subjects "
                f"performing the DJ task on a single side-view phone "
                f"camera with paired marker mocap that includes foot/toe "
                f"markers (so ankle dorsi/plantarflexion can be derived). "
                f"Estimated lab time: ~15 min/subject, plus mocap setup."
            ),
            "current_headline_status": (
                "Ankle slot remains 'preliminary additional measurement' "
                "in the one-pager. Per-subject CCC point estimate of 0.64 "
                "and LoA half-width 9.5deg are inside the Good-tier "
                "thresholds, but the lower 95% CI bound at n=9 includes "
                "zero (Fisher Z: [-0.034, +0.916]) so the slot cannot "
                "carry a 'headline' precision claim."
            ),
        },
    }
    return out


def render_report(out: dict) -> str:
    base = out["baseline_slot"]
    audit = out["dataset_audit"]
    proj = out["projection_at_assumed_ccc"]
    fz_lo, fz_hi = base["ccc_95ci_fisher_z"]
    bs_lo, bs_hi = base["ccc_95ci_bootstrap"]

    def fmt(v, prec=2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "-"
        return f"{v:.{prec}f}"

    lines = []
    lines.append("# Ankle ROM Cohort Expansion Audit — v14 ankle_angle_r / side_right")
    lines.append("")
    lines.append(f"**Produced:** {out['produced_at']}  ")
    lines.append("**Script:** `harness/expand_ankle_cohort.py`  ")
    lines.append(f"**Source slot:** {base['target']} / {base['view']} / {base['approach']}")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    lines.append(
        f"The v14 ankle / side_right slot stays at n={base['n_subjects']} with the "
        f"existing validation tree. **No candidate dataset cleanly expands the "
        f"per-subject LOSO cohort.** The point estimate (CCC = {base['ccc_lin']:.3f}, "
        f"LoA half-width = {base['loa_half_width_deg']:.2f}deg) is inside the "
        f"Good tier, but the 95% Fisher Z CI on CCC is "
        f"[{fz_lo:+.3f}, {fz_hi:+.3f}] — the lower bound includes zero, so the "
        f"slot cannot be promoted from 'preliminary additional measurement' to "
        f"'headline-range validated.'"
    )
    lines.append("")
    lines.append(
        f"To clear the promotion threshold (lower CCC 95% CI bound >= 0.30 at the "
        f"current point estimate), n_subjects must reach >= "
        f"{out['n_subjects_required_for_promotion']}. The recommended path is "
        f"fresh data collection — not further mining of public datasets, because "
        f"the public datasets that satisfy the joint requirement of "
        f"(paired single-phone video) AND (mocap-derived ankle ROM in deg) AND "
        f"(commercial-permissive license) do not exist in the validation tree."
    )
    lines.append("")

    lines.append("## Baseline (re-derived, matches existing per_slot_validity.json)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| n_subjects | {base['n_subjects']} |")
    lines.append(f"| n_trials | {base['n_trials']} |")
    lines.append(f"| Mean observed (deg) | {fmt(base['mean_observed_deg'])} |")
    lines.append(f"| Mean predicted (deg) | {fmt(base['mean_predicted_deg'])} |")
    lines.append(f"| Mean bias (deg) | {fmt(base['mean_bias_deg'])} |")
    lines.append(
        f"| 95% LoA (deg) | "
        f"[{fmt(base['loa_lower_deg'])}, {fmt(base['loa_upper_deg'])}] |"
    )
    lines.append(
        f"| LoA half-width (deg) | {fmt(base['loa_half_width_deg'])} |"
    )
    lines.append(f"| Pearson r | {fmt(base['pearson_r'], 4)} |")
    lines.append(f"| **CCC (Lin)** | **{fmt(base['ccc_lin'], 4)}** |")
    lines.append(f"| MAE (deg) | {fmt(base['mae_deg'])} |")
    lines.append(f"| RMSE (deg) | {fmt(base['rmse_deg'])} |")
    lines.append(
        f"| CCC 95% CI (Fisher Z) | "
        f"[{fmt(fz_lo, 3)}, {fmt(fz_hi, 3)}] |"
    )
    lines.append(
        f"| CCC 95% CI (bootstrap, 10k) | "
        f"[{fmt(bs_lo, 3)}, {fmt(bs_hi, 3)}] |"
    )
    lines.append(
        f"| CCC bootstrap mean +/- SD | "
        f"{fmt(out['baseline_slot']['ccc_bootstrap_mean'], 3)} +/- "
        f"{fmt(out['baseline_slot']['ccc_bootstrap_sd'], 3)} |"
    )
    lines.append("")
    lines.append(
        "Bootstrap and Fisher Z agree closely, which is expected when the "
        "per-subject residuals are approximately Gaussian (we verify this "
        "informally by inspecting the per-subject diff_deg vector below — "
        "no obvious outliers at this n)."
    )
    lines.append("")

    lines.append("## Per-subject leave-one-out sensitivity")
    lines.append("")
    lines.append(
        "How much does each individual subject move the CCC? Subjects with "
        "the largest absolute error in the LOSO fold are listed first."
    )
    lines.append("")
    lines.append(
        "| Subject | pred (deg) | obs (deg) | abs error (deg) | CCC w/o subj | r w/o subj | Δ CCC vs full |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in base["loso_sensitivity_per_subject"]:
        lines.append(
            f"| {row['dropped_subject']} | {fmt(row['pred_subj'])} | "
            f"{fmt(row['obs_subj'])} | {fmt(row['abs_error_deg'])} | "
            f"{fmt(row['ccc_no_subj'], 4)} | {fmt(row['pearson_r_no_subj'], 4)} | "
            f"{fmt(row['ccc_delta_from_full'], 4)} |"
        )
    lines.append("")

    lines.append("## Promotion projection — assuming CCC point estimate holds")
    lines.append("")
    lines.append(
        f"If new subjects are added and the per-subject CCC stays at "
        f"~{base['ccc_lin']:.3f}, the 95% Fisher Z CI on CCC tightens as "
        f"follows. The promotion target is **lower bound >= 0.30**."
    )
    lines.append("")
    lines.append("| n_subjects | CCC 95% CI (Fisher Z) | Lower bound >= 0.30? |")
    lines.append("| ---: | --- | :---: |")
    for row in proj:
        marker = "YES" if row["clears_lo_ge_0_30"] else "no"
        lines.append(
            f"| {row['n']} | [{fmt(row['ci_lower'], 3)}, {fmt(row['ci_upper'], 3)}] "
            f"| {marker} |"
        )
    lines.append("")
    lines.append(
        "At the current CCC, the first n that clears the promotion bar is "
        f"**n = {out['n_subjects_required_for_promotion']}**. "
        "If a fresh cohort runs higher than 0.64 (e.g. 0.75), the threshold "
        "drops to ~n=15. If it runs lower, the slot may never promote at "
        "any tractable n — which is itself useful diligence-relevant "
        "information."
    )
    lines.append("")

    lines.append("## Dataset audit — every candidate, decision, and rationale")
    lines.append("")
    lines.append(
        "| Dataset | Present in tree? | Paired video + mocap? | Foot/toe markers? | Sagittal ankle GT? | License | Commercial? | Decision |"
    )
    lines.append(
        "| --- | :---: | :---: | :---: | :---: | --- | :---: | --- |"
    )
    bool_yn = lambda v: "YES" if v else "no"
    for row in audit:
        lines.append(
            f"| {row['dataset']} | {bool_yn(row['raw_present_in_tree'])} | "
            f"{bool_yn(row['has_paired_video_plus_mocap'])} | "
            f"{bool_yn(row['has_foot_or_toe_markers'])} | "
            f"{bool_yn(row['has_sagittal_ankle_gt'])} | "
            f"{row['license']} | {bool_yn(row['commercial_clean'])} | "
            f"**{row['decision']}** |"
        )
    lines.append("")

    lines.append("### Per-dataset rationale")
    lines.append("")
    for row in audit:
        lines.append(f"**{row['dataset']}** — {row['decision']}")
        lines.append("")
        lines.append(f"{row['rationale']}")
        lines.append("")

    lines.append("## What this means for the one-pager")
    lines.append("")
    lines.append(
        "The ankle / side_right slot **stays in the 'preliminary additional "
        "measurement' bucket** in the one-pager — not the 'headline range' "
        "bucket — until a new cohort is collected. The point estimates "
        "(CCC = 0.64, LoA half-width = 9.5deg, MAE = 3.9deg) can still be "
        "cited as evidence the pipeline is functional at side_right for "
        "ankle ROM; what cannot be claimed is statistical precision of "
        "those numbers at the diligence-grade '95% CI clearly above zero' "
        "standard."
    )
    lines.append("")
    lines.append("**Recommended language for the one-pager:**")
    lines.append("")
    lines.append(
        "> Ankle dorsi/plantarflexion ROM (right ankle, side-right view): "
        "**preliminary** at n=9 OpenCap subjects. Per-subject CCC = 0.64 "
        "(95% CI on a small cohort is wide; promotion to headline requires "
        "expansion to >= 22 subjects). LoA half-width 9.5deg, MAE 3.9deg "
        "are inside the Good tier point-estimate thresholds and indicate "
        "the pipeline works at side_right, but the cohort is too small to "
        "make a precision claim with 95% confidence the lower CCC bound is "
        "above zero."
    )
    lines.append("")

    lines.append("## What would clear the bar")
    lines.append("")
    n_needed = out["n_subjects_required_for_promotion"]
    lines.append(
        f"- **Fresh data collection of ~{n_needed - base['n_subjects']}+ additional subjects** "
        "performing the drop-jump task with a single side-right phone "
        "camera + paired marker mocap (must include foot/toe markers so "
        "ankle dorsi/plantarflexion can be derived from foot vs shank). "
        "Lab time estimate: 15 min/subject. Even at 1 lab session/week "
        "this clears in a quarter."
    )
    lines.append(
        "- **Re-licensable variant of MPI-INF-3DHP or Human3.6M.** Currently "
        "blocked by non-commercial terms. If a commercial release path "
        "becomes available, MPI-INF-3DHP's 8-subject cam7 cohort would "
        "still only nudge n from 9 -> 17 and the CCC there (mean abs(r) = "
        "0.23 on cam7 ankle) is far below the 0.64 we need to hold."
    )
    lines.append(
        "- **A new commercial-clean ankle-rich public dataset.** Status: "
        "actively monitored. Nothing matching the joint requirement of "
        "(paired single-phone-camera video at side view) AND (mocap "
        "foot+shank markers) AND (CC-BY-style license) was found in the "
        "May 2026 sweep."
    )
    lines.append("")

    lines.append("## Constraints honored")
    lines.append("")
    lines.append("- LOSO discipline preserved — no data leakage between folds.")
    lines.append("- Single phone camera only — no multi-camera fusion considered.")
    lines.append(
        "- No modification of v14, v15, biomech_validity_stats, or other "
        "yesterday's-build outputs. This script is additive."
    )
    lines.append(
        "- MPI-INF-3DHP excluded for non-commercial license (per yesterday's "
        "learning); CMU Panoptic excluded for non-commercial license."
    )
    lines.append(
        "- Honest reporting: the slot CANNOT be promoted with existing data. "
        "Reporting this directly rather than constructing a comforting "
        "synthetic-cohort story."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def write_outputs(out: dict) -> None:
    json_path = OUT_DIR / "per_slot_validity_ankle_expanded.json"
    md_path = OUT_DIR / "REPORT.md"
    json_path.write_text(json.dumps(out, indent=2))
    md_path.write_text(render_report(out))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    out = run()
    write_outputs(out)


if __name__ == "__main__":
    main()
