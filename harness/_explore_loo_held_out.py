"""Exploratory: leak-free leave-one-fold-out estimator on held-out residuals.

Spec-mandated estimator (mean(train_pred - train_obs) per fold) is
identically zero for ridge with an intercept term — the intercept is
fit so the train residual mean is zero. So the v15 deploy ships a
near-zero bias_correction_deg, which means the corrected stats equal
the uncorrected stats. That is the honest LOSO-safe answer to the
question the spec posed.

This exploratory script computes a DIFFERENT leak-free estimator:
for each held-out subject s, the bias intercept is the mean of the
*other* subjects' held-out residual means (the across-folds mean of
mean(pred_j - obs_j) for j != s). Subject s never contributes to its
own correction, so no leakage.

This estimator captures the systematic across-subject offset that the
ridge intercept cannot see, and it is the natural alternative if Saad
decides the spec-mandated estimator is too conservative.

Outputs:
    data/calibration_fix/loo_held_out_alternative.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation"
)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from harness.biomech_validity_stats import BUILDERS, compute_stats_from_pairs, classify
from harness.train_regression_poc import fit_ridge


SLOTS: tuple[tuple[str, str, str], ...] = (
    ("hip_adduction_r", "front_center", "v12_combined"),
    ("hip_adduction_r", "front_oblique_right", "v12_combined"),
    ("knee_angle_r", "front_oblique_left", "v12_combined"),
)


def half_loa(s) -> float:
    return max(abs(s.upper_loa - s.mean_bias), abs(s.mean_bias - s.lower_loa))


def stats_dict(s) -> dict:
    return {
        "pearson_r": s.pearson_r,
        "ccc_lin": s.ccc,
        "mean_bias_deg": s.mean_bias,
        "loa_lower_deg": s.lower_loa,
        "loa_upper_deg": s.upper_loa,
        "loa_half_width_deg": half_loa(s),
        "mae_deg": s.mae,
        "rmse_deg": s.rmse,
        "tier": classify(s),
    }


def run_slot(target: str, view: str, approach: str) -> dict:
    builder = BUILDERS[approach]
    X, y, subjects, _ = builder(target, view)
    X = X.astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))

    # Standard LOSO held-out predictions.
    pred_uncorr = np.full(len(y), np.nan)
    for s in unique:
        tr = subj_arr != s
        te = subj_arr == s
        if tr.sum() < 2 or te.sum() < 1:
            continue
        m = fit_ridge(X[tr], y[tr], alpha=10.0)
        pred_uncorr[te] = m.predict(X[te])

    # Per-subject mean residual on the held-out fold.
    subj_resid: dict[str, float] = {}
    for s in unique:
        idx = subj_arr == s
        if idx.sum() == 0:
            continue
        subj_resid[s] = float(np.mean(pred_uncorr[idx] - y[idx]))

    # Leave-one-fold-out: for each subject s, bias intercept = mean over j != s.
    fold_b: dict[str, float] = {}
    pred_corr = np.full(len(y), np.nan)
    for s in unique:
        others = [v for k, v in subj_resid.items() if k != s]
        b_s = float(np.mean(others)) if others else 0.0
        fold_b[s] = b_s
        pred_corr[subj_arr == s] = pred_uncorr[subj_arr == s] - b_s

    deploy_b = float(np.mean(list(fold_b.values()))) if fold_b else 0.0

    pre_subj = compute_stats_from_pairs(
        target, view, approach, pred_uncorr, y, subj_arr, aggregate_per_subject=True,
    )
    post_subj = compute_stats_from_pairs(
        target, view, approach, pred_corr, y, subj_arr, aggregate_per_subject=True,
    )
    pre_trial = compute_stats_from_pairs(
        target, view, approach, pred_uncorr, y, subj_arr, aggregate_per_subject=False,
    )
    post_trial = compute_stats_from_pairs(
        target, view, approach, pred_corr, y, subj_arr, aggregate_per_subject=False,
    )

    print(
        f"{target} / {view}: deploy_bias={deploy_b:+.3f}  "
        f"PRE  per-subj CCC={pre_subj.ccc:.3f} half-LoA={half_loa(pre_subj):.2f}  "
        f"POST per-subj CCC={post_subj.ccc:.3f} half-LoA={half_loa(post_subj):.2f}"
    )

    return {
        "target": target,
        "view": view,
        "approach": approach,
        "estimator": "leave-one-fold-out on held-out residuals (alternative to train-fold residual mean)",
        "deploy_bias_correction_deg": deploy_b,
        "fold_intercepts": [
            {"held_out_subject": s, "bias_intercept_deg": b}
            for s, b in fold_b.items()
        ],
        "per_subject_residual_means": subj_resid,
        "pre_per_subject": stats_dict(pre_subj),
        "post_per_subject": stats_dict(post_subj),
        "pre_per_trial": stats_dict(pre_trial),
        "post_per_trial": stats_dict(post_trial),
    }


def main() -> None:
    out = [run_slot(*s) for s in SLOTS]
    target_path = REPO / "data" / "calibration_fix" / "loo_held_out_alternative.json"
    target_path.write_text(json.dumps({
        "version": "1.0-exploratory",
        "produced_by": "harness._explore_loo_held_out",
        "estimator_description": (
            "For each held-out subject s, bias intercept = mean over j != s of "
            "(held-out-fold mean residual for subject j). Leak-free because s does "
            "not contribute to its own correction. Captures across-subject "
            "systematic offset that the ridge intercept cannot."
        ),
        "spec_mandated_estimator_returned_zero_because": (
            "Ridge with intercept fits train residual mean to identically zero by "
            "construction. The per-subject bias visible in Bland-Altman is across-"
            "held-out-subjects offset, not a within-train-fold offset."
        ),
        "slots": out,
    }, indent=2))
    print(f"Wrote {target_path}")


if __name__ == "__main__":
    main()
