"""Apply Couro v15 calibration fix to the v14 deploy table.

Two corrections from Agent P's biomech-validity analysis are codified here:

Build 1 — LOSO-safe bias-intercept correction on 3 slots
    hip_adduction_r / front_center      (v12_combined, bias +3.60deg)
    hip_adduction_r / front_oblique_right (v12_combined, bias +3.98deg)
    knee_angle_r    / front_oblique_left  (v12_combined, bias -3.20deg)

  Each slot has Pearson r > 0.91 but per-subject CCC of 0.77-0.78 because
  of a consistent ROM overshoot. This is a calibration problem, not a
  model-quality problem.

  The bias intercept is learned per LOSO fold (held-out subject excluded)
  as mean(train_pred - train_obs). The held-out predictions are then
  corrected by subtracting that fold's intercept before computing
  per-subject Bland-Altman + CCC. The deploy-time scalar `bias_correction_deg`
  shipped in the v15 entry is the mean over folds of the per-fold intercept
  — a leak-free estimator of the model's systematic offset at inference time.

Build 2 — demote 2 broken front_center slots
    hip_flexion_r / front_center (per-subject r = -0.21, per-subject CCC = -0.15)
    knee_angle_r  / front_center (per-subject r = +0.22, per-subject CCC = +0.12)

  Both slots have positive per-trial r but negative or near-zero per-subject
  CCC — they track within-subject rep variation but fail to rank athletes
  against population norms. They cannot ship.

Outputs:
    results/deploy_ready_models_v15.json
    data/calibration_fix/per_slot_validity_v15.json

Methodological note: the bias intercept MUST be learned per-fold on the
train portion only. Computing it once on all data would leak the held-out
subject's offset into the correction term and would inflate the corrected
CCC by a constant. The harness here does it the right way.

Run:
    cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
    python -m harness.apply_calibration_fix
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.biomech_validity_stats import (
    BUILDERS,
    OUT_DIR as VALIDITY_OUT_DIR,
    classify,
    compute_stats_from_pairs,
    merged_slot_dict,
    stats_to_dict,
)
from harness.train_regression_poc import fit_ridge


# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------

REPO = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation"
)
RESULTS_DIR = REPO / "results"
V14_PATH = RESULTS_DIR / "deploy_ready_models.json"
V15_PATH = RESULTS_DIR / "deploy_ready_models_v15.json"

CAL_FIX_DIR = REPO / "data" / "calibration_fix"
CAL_FIX_DIR.mkdir(parents=True, exist_ok=True)
V15_VALIDITY_PATH = CAL_FIX_DIR / "per_slot_validity_v15.json"

V14_VALIDITY_PATH = REPO / "data" / "biomech_validity_stats" / "per_slot_validity.json"


# ---------------------------------------------------------------------------
# Build 1 + Build 2 configuration.
# ---------------------------------------------------------------------------

# Slots that get a LOSO-safe bias-intercept correction. Each entry is
# (target, view) — approach is read from the v14 deploy entry to keep the
# config small and unambiguous.
CORRECTED_SLOTS: tuple[tuple[str, str], ...] = (
    ("hip_adduction_r", "front_center"),
    ("hip_adduction_r", "front_oblique_right"),
    ("knee_angle_r", "front_oblique_left"),
)

# Slots removed from the v15 deploy table because per-subject CCC is at or
# below zero — they track reps but cannot rank athletes against norms.
DEMOTED_SLOTS: tuple[tuple[str, str], ...] = (
    ("hip_flexion_r", "front_center"),
    ("knee_angle_r", "front_center"),
)

LOSO_ALPHA = 10.0  # same as biomech_validity_stats._loso_extract default


# ---------------------------------------------------------------------------
# LOSO with per-fold bias-intercept correction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiasCorrectionResult:
    """Outputs of the LOSO-safe bias-intercept estimator for one slot."""

    target: str
    view: str
    approach: str
    fold_bias_intercepts: list[tuple[str, float]]  # (held_out_subject, train_mean_diff)
    deploy_bias_correction_deg: float  # mean across folds, shipped at inference
    pred_uncorrected: np.ndarray
    pred_corrected: np.ndarray
    obs: np.ndarray
    subjects: np.ndarray


def loso_with_bias_correction(
    X: np.ndarray, y: np.ndarray, subjects: list[str], alpha: float,
    target: str, view: str, approach: str,
) -> BiasCorrectionResult:
    """Run LOSO and compute a leak-free per-fold bias intercept.

    For each held-out subject s:
        1. Fit ridge on all-but-s.
        2. Predict on the train fold and compute
           bias_intercept_s = mean(pred_train - y_train).
        3. Predict on the held-out subject and store both uncorrected and
           corrected predictions (corrected = uncorrected - bias_intercept_s).

    The deploy-time `bias_correction_deg` is the mean of the per-fold
    intercepts — a held-out subject's offset never participates in its own
    correction term.
    """
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    n = len(y)
    pred_uncorrected = np.full(n, np.nan)
    pred_corrected = np.full(n, np.nan)
    fold_intercepts: list[tuple[str, float]] = []

    for s in unique:
        train_mask = subj_arr != s
        test_mask = subj_arr == s
        if train_mask.sum() < 2 or test_mask.sum() < 1:
            continue
        model = fit_ridge(X[train_mask], y[train_mask], alpha=alpha)
        train_pred = model.predict(X[train_mask])
        train_obs = y[train_mask]
        bias_intercept_s = float(np.mean(train_pred - train_obs))
        fold_intercepts.append((str(s), bias_intercept_s))

        test_pred = model.predict(X[test_mask])
        pred_uncorrected[test_mask] = test_pred
        pred_corrected[test_mask] = test_pred - bias_intercept_s

    if not fold_intercepts:
        deploy_bias = float("nan")
    else:
        deploy_bias = float(np.mean([b for _, b in fold_intercepts]))

    return BiasCorrectionResult(
        target=target,
        view=view,
        approach=approach,
        fold_bias_intercepts=fold_intercepts,
        deploy_bias_correction_deg=deploy_bias,
        pred_uncorrected=pred_uncorrected,
        pred_corrected=pred_corrected,
        obs=y,
        subjects=subj_arr,
    )


# ---------------------------------------------------------------------------
# v15 deploy-table construction.
# ---------------------------------------------------------------------------


def build_v15_deploy(v14: dict, corrections: dict[tuple[str, str], BiasCorrectionResult]) -> dict:
    """Produce v15 from v14: add bias_correction_deg to 3 slots, remove 2 slots."""
    v15 = copy.deepcopy(v14)
    v15["version"] = "v15"
    v15["produced_by"] = "harness.apply_calibration_fix"
    v15["produced_date"] = "2026-05-28"
    v15["description"] = (
        (v14.get("description", "") or "")
        + " | v15 = v14 + LOSO-safe bias-intercept correction on 3 slots"
        + " + 2 front_center slots demoted (per-subject CCC ≤ 0)."
    )
    v15["calibration_fix"] = {
        "applied_by": "harness.apply_calibration_fix",
        "from_version": "v14",
        "corrected_slots": [
            {
                "target": t,
                "view": v,
                "bias_correction_deg": corrections[(t, v)].deploy_bias_correction_deg,
                "n_folds": len(corrections[(t, v)].fold_bias_intercepts),
                "estimator": "mean over LOSO folds of mean(train_pred - train_obs); each held-out subject is excluded from its own correction term",
            }
            for (t, v) in CORRECTED_SLOTS if (t, v) in corrections
        ],
        "demoted_slots": [
            {
                "target": t,
                "view": v,
                "reason": "per-subject CCC <= 0; model tracks within-subject rep variation but fails to rank athletes against population norms",
            }
            for (t, v) in DEMOTED_SLOTS
        ],
    }

    # Apply Build 1: attach bias_correction_deg to each corrected slot's deploy entry.
    for (target, view) in CORRECTED_SLOTS:
        entry = v15["models"].get(target, {}).get(view)
        if entry is None:
            print(f"WARN: corrected slot {target}/{view} not present in v14 deploy")
            continue
        cor = corrections.get((target, view))
        if cor is None:
            print(f"WARN: no correction result for {target}/{view}")
            continue
        # Inference contract: prediction_corrected = prediction - bias_correction_deg
        entry["bias_correction_deg"] = cor.deploy_bias_correction_deg
        entry["bias_correction_metadata"] = {
            "estimator": "LOSO per-fold mean(train_pred - train_obs), averaged across folds",
            "n_folds": len(cor.fold_bias_intercepts),
            "fold_intercepts": [
                {"held_out_subject": s, "bias_intercept_deg": b}
                for (s, b) in cor.fold_bias_intercepts
            ],
            "inference_rule": "prediction_corrected_deg = prediction_deg - bias_correction_deg",
        }

    # Apply Build 2: drop demoted slots.
    for (target, view) in DEMOTED_SLOTS:
        tgt_block = v15["models"].get(target, {})
        if view in tgt_block:
            del tgt_block[view]
        else:
            print(f"WARN: demoted slot {target}/{view} not present in v14 deploy")

    return v15


# ---------------------------------------------------------------------------
# v15 per-slot validity stats.
# ---------------------------------------------------------------------------


def _slot_index(v14_validity_rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["target"], r["view"]): r for r in v14_validity_rows}


def build_v15_validity(
    v14_validity: dict, corrections: dict[tuple[str, str], BiasCorrectionResult],
    v15_deploy: dict,
) -> dict:
    """Produce per_slot_validity_v15.json.

    - For the 3 corrected slots: recompute per-subject + per-trial stats on
      the bias-corrected LOSO predictions.
    - For the 2 demoted slots: drop them (they are no longer in v15 deploy).
    - For the remaining 20 slots: copy v14 stats unchanged — the underlying
      LOSO predictions did not change.
    """
    src_rows = v14_validity.get("slots", [])
    by_slot = _slot_index(src_rows)

    out_rows: list[dict] = []
    for tgt_name, slots in v15_deploy["models"].items():
        for view_name in slots.keys():
            key = (tgt_name, view_name)
            if key in corrections:
                cor = corrections[key]
                per_subj = compute_stats_from_pairs(
                    target=tgt_name, view=view_name, approach=cor.approach,
                    pred=cor.pred_corrected, obs=cor.obs, subjects=cor.subjects,
                    aggregate_per_subject=True,
                )
                per_trial = compute_stats_from_pairs(
                    target=tgt_name, view=view_name, approach=cor.approach,
                    pred=cor.pred_corrected, obs=cor.obs, subjects=cor.subjects,
                    aggregate_per_subject=False,
                )
                row = merged_slot_dict(per_subj, per_trial)
                # Carry the uncorrected baseline so the report can do a clean diff.
                uncorrected_per_subj = compute_stats_from_pairs(
                    target=tgt_name, view=view_name, approach=cor.approach,
                    pred=cor.pred_uncorrected, obs=cor.obs, subjects=cor.subjects,
                    aggregate_per_subject=True,
                )
                row["bias_correction_applied"] = True
                row["bias_correction_deg"] = cor.deploy_bias_correction_deg
                row["pre_correction_per_subject"] = {
                    "mean_bias_deg": uncorrected_per_subj.mean_bias,
                    "loa_lower_deg": uncorrected_per_subj.lower_loa,
                    "loa_upper_deg": uncorrected_per_subj.upper_loa,
                    "loa_half_width_deg": max(
                        abs(uncorrected_per_subj.upper_loa - uncorrected_per_subj.mean_bias),
                        abs(uncorrected_per_subj.mean_bias - uncorrected_per_subj.lower_loa),
                    ),
                    "pearson_r": uncorrected_per_subj.pearson_r,
                    "ccc_lin": uncorrected_per_subj.ccc,
                    "mae_deg": uncorrected_per_subj.mae,
                    "rmse_deg": uncorrected_per_subj.rmse,
                    "classification": classify(uncorrected_per_subj),
                }
                out_rows.append(row)
            else:
                # Unchanged slot — copy v14 stats. The underlying ridge
                # weights/intercept were not modified.
                src = by_slot.get(key)
                if src is None:
                    print(f"WARN: v14 validity row missing for unchanged slot {key}")
                    continue
                out_rows.append(copy.deepcopy(src))

    out = copy.deepcopy(v14_validity)
    out["version"] = "1.0-v15-calibration_fix"
    out["produced_by"] = "harness.apply_calibration_fix"
    out["source_deploy"] = str(V15_PATH)
    out["v14_source_validity"] = str(V14_VALIDITY_PATH)
    out["calibration_fix"] = {
        "corrected_slots": [
            {
                "target": t,
                "view": v,
                "bias_correction_deg": corrections[(t, v)].deploy_bias_correction_deg,
            }
            for (t, v) in CORRECTED_SLOTS if (t, v) in corrections
        ],
        "demoted_slots": [{"target": t, "view": v} for (t, v) in DEMOTED_SLOTS],
        "methodology": (
            "Bias intercept is learned per LOSO fold as mean(train_pred - train_obs) "
            "on the training subjects; the held-out subject never participates in its "
            "own correction term. Held-out predictions are corrected by subtracting "
            "that fold's intercept before computing per-subject Bland-Altman + CCC. "
            "The deploy-time scalar shipped in deploy_ready_models_v15.json is the "
            "across-folds mean of the per-fold intercepts."
        ),
    }
    out["slots"] = out_rows
    return out


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------


def _run_correction_for_slot(target: str, view: str, approach: str) -> BiasCorrectionResult | None:
    builder = BUILDERS.get(approach)
    if builder is None:
        print(f"  no builder for approach={approach}")
        return None
    t0 = time.time()
    result = builder(target, view)
    if result is None:
        print(f"  builder returned None")
        return None
    X, y, subjects, _ = result
    if len(y) < 20:
        print(f"  only n={len(y)} samples — skipping")
        return None
    cor = loso_with_bias_correction(
        X=X.astype(np.float64), y=np.asarray(y, dtype=np.float64),
        subjects=list(subjects), alpha=LOSO_ALPHA,
        target=target, view=view, approach=approach,
    )
    dt = time.time() - t0
    print(
        f"  approach={approach} n_trials={len(y)} n_folds={len(cor.fold_bias_intercepts)} "
        f"deploy_bias_correction_deg={cor.deploy_bias_correction_deg:+.3f}  ({dt:.1f}s)"
    )
    return cor


def main() -> None:
    print("Loading v14 deploy + Agent P validity stats ...")
    v14 = json.loads(V14_PATH.read_text())
    v14_validity = json.loads(V14_VALIDITY_PATH.read_text())

    print(f"Re-running LOSO with bias correction on {len(CORRECTED_SLOTS)} slots:")
    corrections: dict[tuple[str, str], BiasCorrectionResult] = {}
    for (target, view) in CORRECTED_SLOTS:
        v14_entry = v14["models"].get(target, {}).get(view)
        if v14_entry is None:
            print(f"  {target}/{view}: NOT FOUND in v14 — skipping")
            continue
        approach = v14_entry.get("approach", "unknown")
        print(f"[{target} | {view}] approach={approach}")
        cor = _run_correction_for_slot(target, view, approach)
        if cor is not None:
            corrections[(target, view)] = cor

    print(f"\nDemoting {len(DEMOTED_SLOTS)} slots from v15 deploy:")
    for (t, v) in DEMOTED_SLOTS:
        print(f"  {t}/{v}")

    print("\nBuilding v15 deploy table ...")
    v15 = build_v15_deploy(v14, corrections)
    V15_PATH.write_text(json.dumps(v15, indent=2))
    print(f"  wrote {V15_PATH}")

    print("Building v15 per-slot validity stats ...")
    v15_validity = build_v15_validity(v14_validity, corrections, v15)
    V15_VALIDITY_PATH.write_text(json.dumps(v15_validity, indent=2))
    print(f"  wrote {V15_VALIDITY_PATH}")

    # Summary tier counts.
    def _tier_counts(rows: list[dict]) -> dict[str, int]:
        counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
        for r in rows:
            t = r.get("classification", "Poor")
            counts[t] = counts.get(t, 0) + 1
        return counts

    v14_counts = _tier_counts(v14_validity.get("slots", []))
    v15_counts = _tier_counts(v15_validity.get("slots", []))

    print("\nTier counts:")
    print(f"  v14 (n={len(v14_validity.get('slots', []))}): {v14_counts}")
    print(f"  v15 (n={len(v15_validity.get('slots', []))}): {v15_counts}")

    print("\nDone.")


if __name__ == "__main__":
    main()
