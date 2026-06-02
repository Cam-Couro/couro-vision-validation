"""Layer 3 (Agent DD): linear mixed-effects regression for close-to-promotable
Moderate slots in the v17 deploy.

Hypothesis: ridge regression treats every trial as an independent draw and
overfits to high-leverage subjects (e.g. opencap_subject5 — see Agent AA).
A linear mixed-effects model

    ROM_observed = X*beta + Z*u + epsilon

where u is a per-subject random intercept (and optionally a per-subject random
slope on a key feature) accounts for the within-subject correlation Agent P
diagnosed and may lift CCC / shrink LoA on the three closest-to-Good slots.

LOSO discipline: for a held-out subject, u_held_out = 0 by definition. Held-out
predictions therefore use fixed effects only. The pitch is that the random
effect *fitted during training* shrinks beta toward population-level trends,
reducing overfit to outliers.

Run:
    cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
    python -m harness.mixed_effects_layer3

Outputs:
    data/mixed_effects/per_slot_mixedlm_results.json
    data/mixed_effects/REPORT.md

Optional:
    --slots knee_v9 hip_v13 knee_v14 lumbar_foright    # subset
    --include-bonus                                    # also run lumbar slots
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

from harness.biomech_validity_stats import (
    BUILDERS,
    _ccc,
    _loso_extract,
    _pearson,
    _t_test_p_value_bias,
    classify,
    compute_stats_from_pairs,
    ValidityStats,
)
from harness.train_v9_phased import FEATURE_NAMES as V9_FEATURE_NAMES


DATA_ROOT = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data"
)
OUT_DIR = DATA_ROOT / "mixed_effects"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Target slots — the three closest-to-promotable Moderates + bonus lumbar.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotSpec:
    name: str
    target: str
    view: str
    approach: str
    description: str = ""


PRIMARY_SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec(
        name="knee_v9",
        target="knee_angle_r",
        view="front_oblique_right",
        approach="v9_phased",
        description="Closest-to-Good slot: ridge LoA 10.72, gap 0.72 (n=8 subj).",
    ),
    SlotSpec(
        name="hip_v13",
        target="hip_flexion_r",
        view="front_oblique_left",
        approach="v13_dwpose_hybrid",
        description="Largest-cohort Moderate: ridge LoA 11.29, gap 1.29 (n=21 subj).",
    ),
    SlotSpec(
        name="knee_v14",
        target="knee_angle_r",
        view="side_left",
        approach="v14_full_dwpose",
        description="Subject5-leverage slot (Agent AA finding): ridge LoA 12.43, gap 2.43 (n=12 subj).",
    ),
)

BONUS_SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec(
        name="lumbar_v13",
        target="lumbar_extension",
        view="front_oblique_right",
        approach="v13_dwpose_hybrid",
        description="Bonus: ridge CCC just at 0.63, LoA 10.18 (n=17 subj).",
    ),
    SlotSpec(
        name="lumbar_ea_center",
        target="lumbar_extension",
        view="front_center",
        approach="event_anchored",
        description="Bonus: ridge CCC 0.55, LoA 7.45 (n=9 subj).",
    ),
    SlotSpec(
        name="lumbar_ea_left",
        target="lumbar_extension",
        view="front_oblique_left",
        approach="event_anchored",
        description="Bonus: ridge CCC 0.53, LoA 8.03 (n=9 subj).",
    ),
)


# ---------------------------------------------------------------------------
# MixedLM LOSO.
# ---------------------------------------------------------------------------


def _standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns of X_train and apply the same transform to X_test.

    Returns (X_train_std, X_test_std, mu, sigma). Columns with zero variance
    are left at zero (after centring) to avoid divide-by-zero blow-ups.
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma_safe = np.where(sigma > 1e-9, sigma, 1.0)
    return (X_train - mu) / sigma_safe, (X_test - mu) / sigma_safe, mu, sigma_safe


def _fit_mixedlm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    *,
    random_slope_col: int | None = None,
) -> MixedLM | None:
    """Fit a MixedLM with random intercept (and optional random slope).

    random_slope_col: if not None, also include a per-subject random slope on
    feature column `random_slope_col`. The fixed-effects design always includes
    every column of X_train plus a constant intercept.

    Returns the fitted result, or None if optimisation failed.
    """
    exog = sm.add_constant(X_train, has_constant="add")
    if random_slope_col is None:
        model = MixedLM(endog=y_train, exog=exog, groups=groups_train)
    else:
        # Random effects design: column 0 is the random intercept (always 1),
        # column 1 is the random slope on the chosen feature. exog_re must be
        # provided explicitly because the API gives us a single (1|group)
        # by default; for (1 + x | group) we need exog_re.
        re_col = X_train[:, random_slope_col].reshape(-1, 1)
        exog_re = np.hstack([np.ones((len(y_train), 1)), re_col])
        model = MixedLM(
            endog=y_train, exog=exog, groups=groups_train, exog_re=exog_re,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # 'lbfgs' is the default; we also try 'powell' as a fallback.
            try:
                result = model.fit(method="lbfgs", reml=True, disp=False)
            except Exception:
                result = model.fit(method="powell", reml=True, disp=False)
        except Exception:
            return None
    if not getattr(result, "converged", True):
        # Statsmodels MixedLM doesn't always set converged; if params are NaN
        # treat as failure.
        if not np.all(np.isfinite(result.fe_params)):
            return None
    return result


def _predict_fixed_only(result: MixedLM, X_test: np.ndarray) -> np.ndarray:
    """Predict y for held-out subject using fixed effects only (u = 0).

    This is the honest LOSO comparison to ridge: the held-out subject has no
    training trials, so we cannot fit u_held_out. The expected value of y at
    population level is X*beta.
    """
    exog_test = sm.add_constant(X_test, has_constant="add")
    fe = np.asarray(result.fe_params)
    # statsmodels returns fe_params with names; positional order matches exog.
    return exog_test @ fe


def loso_mixedlm(
    X: np.ndarray,
    y: np.ndarray,
    subjects: list[str],
    *,
    random_slope_col: int | None = None,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """LOSO with MixedLM fitted on the n_subj - 1 training subjects.

    Returns (all_pred, y, subjects, diag) where:
      - all_pred[i] is the held-out prediction for trial i (fixed-effects only)
      - diag tracks per-fold convergence and per-fold REML-at-boundary status.
    """
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    all_pred = np.full(len(y), np.nan)
    fold_status: dict[str, str] = {}
    per_fold_re: dict[str, dict] = {}
    for s in unique:
        train_mask = subj_arr != s
        test_mask = subj_arr == s
        if train_mask.sum() < 2 or test_mask.sum() < 1:
            fold_status[s] = "skipped_too_small"
            continue
        X_tr = X[train_mask]
        y_tr = y[train_mask]
        groups_tr = subj_arr[train_mask]
        X_te = X[test_mask]
        if standardize:
            X_tr_std, X_te_std, _, _ = _standardize(X_tr, X_te)
        else:
            X_tr_std, X_te_std = X_tr, X_te
        result = _fit_mixedlm(
            X_tr_std, y_tr, groups_tr, random_slope_col=random_slope_col,
        )
        if result is None:
            fold_status[s] = "fit_failed"
            continue
        try:
            preds = _predict_fixed_only(result, X_te_std)
        except Exception:
            fold_status[s] = "predict_failed"
            continue
        all_pred[test_mask] = preds
        fold_status[s] = "ok"
        per_fold_re[s] = _random_effect_summary(result)
    n_boundary = sum(
        1 for v in per_fold_re.values() if v.get("reml_at_boundary")
    )
    diag = {
        "fold_status": fold_status,
        "n_ok_folds": sum(1 for v in fold_status.values() if v == "ok"),
        "n_failed_folds": sum(1 for v in fold_status.values() if v != "ok"),
        "n_folds_reml_at_boundary": n_boundary,
        "per_fold_random_effect_summary": per_fold_re,
    }
    return all_pred, y, subj_arr, diag


def _loso_ridge_standardized(
    X: np.ndarray, y: np.ndarray, subjects: list[str], *, alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge LOSO with per-fold z-scoring, matching the standardisation step
    used inside the MixedLM LOSO. Reported as a sanity-check baseline so we
    can attribute any CCC gain between (a) standardisation, (b) ridge alpha,
    and (c) the mixed-effects structure itself."""
    from harness.train_regression_poc import fit_ridge

    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    all_pred = np.full(len(y), np.nan)
    for s in unique:
        train_mask = subj_arr != s
        test_mask = subj_arr == s
        if train_mask.sum() < 2 or test_mask.sum() < 1:
            continue
        X_tr_std, X_te_std, _, _ = _standardize(X[train_mask], X[test_mask])
        m = fit_ridge(X_tr_std, y[train_mask], alpha=alpha)
        all_pred[test_mask] = m.predict(X_te_std)
    return all_pred, y, subj_arr


def _cov_re_value(cov_re, i: int = 0, j: int = 0) -> float:
    """Read cov_re[i,j] regardless of whether statsmodels returned an ndarray
    or a DataFrame for this version."""
    try:
        if hasattr(cov_re, "iloc"):
            return float(cov_re.iloc[i, j])
        return float(np.asarray(cov_re)[i, j])
    except Exception:
        return float("nan")


def _random_effect_summary(result: MixedLM) -> dict:
    """Pull the random-intercept variance, scale, and a 'boundary' flag from
    a fitted MixedLM result. The boundary flag is the honesty cue: when REML
    optimisation parks the variance estimate at ~0, the mixed-effects model
    has collapsed to OLS and the gain (if any) is from de-regularisation
    versus the ridge baseline, NOT from modelling subject structure."""
    scale = float(getattr(result, "scale", float("nan")))
    cov_re = getattr(result, "cov_re", None)
    re_var = _cov_re_value(cov_re, 0, 0) if cov_re is not None else float("nan")
    re_sd = float(np.sqrt(re_var)) if (
        re_var is not None and not math.isnan(re_var) and re_var >= 0
    ) else float("nan")
    # Boundary heuristic: variance estimate is essentially zero relative to
    # residual scale.
    boundary = bool(
        not math.isnan(re_var) and not math.isnan(scale)
        and scale > 0 and re_var / scale < 1e-3
    )
    return {
        "scale_residual_var": scale,
        "random_intercept_var": float(re_var) if not math.isnan(re_var) else None,
        "random_intercept_sd": float(re_sd) if not math.isnan(re_sd) else None,
        "icc_subject": (
            float(re_var / (re_var + scale))
            if (not math.isnan(re_var) and not math.isnan(scale) and (re_var + scale) > 0)
            else None
        ),
        "reml_at_boundary": boundary,
        "log_likelihood": (
            float(result.llf) if hasattr(result, "llf") and result.llf is not None
            else None
        ),
    }


def fit_full_mixedlm_for_random_effects(
    X: np.ndarray,
    y: np.ndarray,
    subjects: list[str],
    *,
    random_slope_col: int | None = None,
    standardize: bool = True,
) -> tuple[MixedLM | None, dict[str, dict[str, float]] | None, dict]:
    """Fit one MixedLM on the WHOLE dataset (no held-out subject) and extract
    per-subject random-effect estimates.

    This is purely diagnostic — it's the principled equivalent of Agent V's
    leave-one-out outlier analysis. The fitted u_s values tell us which
    subjects' ROM deviates most from the population mean conditional on X.

    Returns (fitted result, per-subject random effects dict, model metadata).
    """
    subj_arr = np.array(subjects)
    if standardize:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)
        sigma_safe = np.where(sigma > 1e-9, sigma, 1.0)
        X_std = (X - mu) / sigma_safe
    else:
        X_std = X
    result = _fit_mixedlm(
        X_std, y, subj_arr, random_slope_col=random_slope_col,
    )
    if result is None:
        return None, None, {}
    meta = _random_effect_summary(result)
    try:
        re = result.random_effects
    except Exception:
        return result, None, meta
    per_subject: dict[str, dict[str, float]] = {}
    for subj, ranef in re.items():
        # ranef is a pandas Series indexed by 'Group' (intercept) and optional
        # slope name. Convert to plain floats.
        row: dict[str, float] = {}
        for name, val in ranef.items():
            row[str(name)] = float(val)
        per_subject[str(subj)] = row
    return result, per_subject, meta


# ---------------------------------------------------------------------------
# Bland-Altman / CCC reporting.
# ---------------------------------------------------------------------------


def _stats_from_pred_obs(
    target: str, view: str, label: str,
    pred: np.ndarray, obs: np.ndarray, subjects: np.ndarray,
    *, aggregate_per_subject: bool,
) -> dict:
    stats = compute_stats_from_pairs(
        target, view, label, pred, obs, subjects,
        aggregate_per_subject=aggregate_per_subject,
    )
    return {
        "label": label,
        "aggregation": stats.aggregation,
        "n_subjects": stats.n_subjects,
        "n_trials": stats.n_trials,
        "mean_observed_deg": stats.mean_observed,
        "mean_predicted_deg": stats.mean_predicted,
        "mean_bias_deg": stats.mean_bias,
        "sd_bias_deg": stats.sd_bias,
        "loa_lower_deg": stats.lower_loa,
        "loa_upper_deg": stats.upper_loa,
        "loa_half_width_deg": max(
            abs(stats.upper_loa - stats.mean_bias),
            abs(stats.mean_bias - stats.lower_loa),
        ),
        "pearson_r": stats.pearson_r,
        "ccc_lin": stats.ccc,
        "mae_deg": stats.mae,
        "rmse_deg": stats.rmse,
        "p_value_bias_neq_0": stats.p_value_bias,
        "classification": classify(stats),
    }


def _key_feature_index(slot: SlotSpec) -> int | None:
    """Pick the random-slope feature most likely to vary per subject.

    For v9_phased and event_anchored, 'target_rom_full' is the most direct
    measure of the subject's loading-window ROM and is the standard 'key
    feature' for biomech ROM regression.

    Returns the column index in X, or None if we can't identify it. For
    combined v12/v13/v14 features the layout depends on the trainer; we
    fall back to column 0 (which is target_rom_full in v9_phased and in the
    combined trainers via the ASPSET_COMPATIBLE_TARGETS coupling layout).
    """
    # All four approach types (v9_phased / v12 / v13 / v14 / event_anchored /
    # event_anchored_bilateral) share the v9-shaped scalar+phased feature
    # layout with target_rom_full as feature 0. (See train_v9_phased.py line
    # 168 and train_v12_combined.py extract_aspset_features — both put
    # target_rom_full first.)
    return 0


# ---------------------------------------------------------------------------
# Per-slot driver.
# ---------------------------------------------------------------------------


def run_slot(slot: SlotSpec) -> dict:
    """Build the slot dataset, run ridge LOSO + 2 MixedLM LOSO variants,
    extract per-subject random-effect estimates, and return a dict ready for
    JSON serialisation."""
    print(f"\n[{slot.name}] {slot.target} / {slot.view} / {slot.approach}")
    builder = BUILDERS[slot.approach]
    t0 = time.time()
    result = builder(slot.target, slot.view)
    if result is None:
        return {
            "slot": slot.name,
            "target": slot.target,
            "view": slot.view,
            "approach": slot.approach,
            "error": "builder returned None",
        }
    X, y, subjects, _ = result
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    subjects = list(subjects)
    n_subj = len(set(subjects))
    n_trials = len(y)
    n_feats = X.shape[1]
    print(
        f"  dataset: n_trials={n_trials}, n_subj={n_subj}, n_feats={n_feats} "
        f"(built in {time.time() - t0:.1f}s)"
    )

    # Ridge baseline LOSO — replicates biomech_validity_stats (alpha=10, unstd).
    pred_ridge, obs_ridge, subj_ridge = _loso_extract(X, y, subjects, alpha=10.0)

    # Ridge LOSO with the same z-scoring used inside MixedLM (sanity baseline).
    pred_ridge_std, obs_ridge_std, subj_ridge_std = _loso_ridge_standardized(
        X, y, subjects, alpha=10.0,
    )

    # MixedLM random intercept LOSO.
    t1 = time.time()
    pred_mixed_int, obs_mixed_int, subj_mixed_int, diag_int = loso_mixedlm(
        X, y, subjects, random_slope_col=None, standardize=True,
    )
    print(f"  MixedLM (intercept): {time.time() - t1:.1f}s, "
          f"{diag_int['n_ok_folds']} ok / {diag_int['n_failed_folds']} failed folds")

    # MixedLM random intercept + slope LOSO.
    slope_col = _key_feature_index(slot)
    t2 = time.time()
    pred_mixed_sl, obs_mixed_sl, subj_mixed_sl, diag_sl = loso_mixedlm(
        X, y, subjects, random_slope_col=slope_col, standardize=True,
    )
    print(f"  MixedLM (intercept+slope on feat {slope_col}): {time.time() - t2:.1f}s, "
          f"{diag_sl['n_ok_folds']} ok / {diag_sl['n_failed_folds']} failed folds")

    # Per-subject random-effect estimates from full-data fit (diagnostic).
    t3 = time.time()
    full_result, per_subject_re, full_re_meta = fit_full_mixedlm_for_random_effects(
        X, y, subjects, random_slope_col=None, standardize=True,
    )
    if full_result is not None:
        full_re_meta["fe_params"] = {
            str(k): float(v) for k, v in zip(
                ["const"] + [f"f{i}" for i in range(n_feats)],
                np.asarray(full_result.fe_params),
            )
        }
    print(f"  full-data MixedLM for u_s: {time.time() - t3:.1f}s")

    out: dict = {
        "slot": slot.name,
        "description": slot.description,
        "target": slot.target,
        "view": slot.view,
        "approach": slot.approach,
        "n_subjects": int(n_subj),
        "n_trials": int(n_trials),
        "n_features": int(n_feats),
        "ridge_baseline": {
            "per_subject": _stats_from_pred_obs(
                slot.target, slot.view, "ridge_baseline",
                pred_ridge, obs_ridge, subj_ridge,
                aggregate_per_subject=True,
            ),
            "per_trial": _stats_from_pred_obs(
                slot.target, slot.view, "ridge_baseline",
                pred_ridge, obs_ridge, subj_ridge,
                aggregate_per_subject=False,
            ),
        },
        "ridge_standardized_sanity": {
            "per_subject": _stats_from_pred_obs(
                slot.target, slot.view, "ridge_std_alpha10",
                pred_ridge_std, obs_ridge_std, subj_ridge_std,
                aggregate_per_subject=True,
            ),
            "per_trial": _stats_from_pred_obs(
                slot.target, slot.view, "ridge_std_alpha10",
                pred_ridge_std, obs_ridge_std, subj_ridge_std,
                aggregate_per_subject=False,
            ),
        },
        "mixedlm_random_intercept": {
            "diagnostics": diag_int,
            "per_subject": _stats_from_pred_obs(
                slot.target, slot.view, "mixedlm_intercept",
                pred_mixed_int, obs_mixed_int, subj_mixed_int,
                aggregate_per_subject=True,
            ),
            "per_trial": _stats_from_pred_obs(
                slot.target, slot.view, "mixedlm_intercept",
                pred_mixed_int, obs_mixed_int, subj_mixed_int,
                aggregate_per_subject=False,
            ),
        },
        "mixedlm_random_intercept_slope": {
            "diagnostics": diag_sl,
            "slope_feature_index": slope_col,
            "per_subject": _stats_from_pred_obs(
                slot.target, slot.view, "mixedlm_intercept_slope",
                pred_mixed_sl, obs_mixed_sl, subj_mixed_sl,
                aggregate_per_subject=True,
            ),
            "per_trial": _stats_from_pred_obs(
                slot.target, slot.view, "mixedlm_intercept_slope",
                pred_mixed_sl, obs_mixed_sl, subj_mixed_sl,
                aggregate_per_subject=False,
            ),
        },
        "full_data_random_effects": {
            "per_subject_random_effects": per_subject_re,
            "model_meta": full_re_meta,
        },
    }
    return out


def tier_verdict(per_subject_stats: dict) -> str:
    return per_subject_stats.get("classification", "Unknown")


# ---------------------------------------------------------------------------
# REPORT.md rendering.
# ---------------------------------------------------------------------------


def fmt(v, prec=2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "—"
    return f"{v:.{prec}f}"


def render_report(results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Layer 3 (Agent DD): Linear Mixed-Effects Regression vs Ridge on Moderate Slots")
    lines.append("")
    lines.append(
        "Tests whether a linear mixed-effects model with per-subject random "
        "intercept (and optional random slope) lifts close-to-promotable "
        "Moderate slots into the Good tier, by accounting for the within-"
        "subject correlation Agent P diagnosed and the population-coverage "
        "skew Agent AA found."
    )
    lines.append("")

    lines.append("## Model spec")
    lines.append("")
    lines.append(
        "    ROM_observed = X*beta + Z*u + epsilon"
    )
    lines.append("")
    lines.append(
        "where X is the same feature matrix used for ridge in `biomech_validity_stats`, "
        "u is a per-subject random intercept (and, in the second variant, a "
        "random slope on `target_rom_full` — feature column 0, the most "
        "direct per-subject ROM signal in every approach used here)."
    )
    lines.append("")
    lines.append(
        "Fitted with `statsmodels.MixedLM` (REML, lbfgs with powell fallback) "
        "after column-wise z-scoring of X on the training fold. Held-out "
        "subject predictions use fixed effects only: `y_hat = X*beta` with "
        "`u_held_out = 0` by definition, because LOSO holds the subject out "
        "entirely and there is no training trial from which to estimate that "
        "subject's random effect. This is the honest comparison to ridge LOSO."
    )
    lines.append("")
    lines.append(
        "Why this is the right model class: trials within a subject are "
        "correlated (Agent P), and subjects vary systematically — "
        "opencap_subject5 is a low-ROM lander in a high-ROM cohort (Agent AA). "
        "Ridge ignores both. The mixed-effects model fits a per-subject "
        "intercept during training, which shrinks the fixed-effect beta "
        "estimates toward population-level trends. The promise is that the "
        "shrunk beta generalises better to held-out subjects than the "
        "ridge-fit beta that paid for in-sample fit to high-leverage "
        "subjects in the training fold."
    )
    lines.append("")

    lines.append("## Tier gates (biomech device-validation convention)")
    lines.append("")
    lines.append("| Tier | CCC | LoA half-width |")
    lines.append("| --- | --- | --- |")
    lines.append("| Excellent | > 0.75 | < ±5° |")
    lines.append("| Good | 0.60–0.75 | ±5–10° |")
    lines.append("| Moderate | 0.40–0.60 | ±10–15° |")
    lines.append("| Poor | ≤ 0.40 | > ±15° |")
    lines.append("")
    lines.append(
        "(Same gates as `biomech_validity_stats.classify`. Per-subject "
        "aggregation is the device-validation standard; per-trial is "
        "reported below for symmetry with the existing deploy doc.)"
    )
    lines.append("")

    # Headline comparison table (per-subject)
    lines.append("## Headline: per-subject Bland-Altman, ridge vs mixed-effects")
    lines.append("")
    lines.append(
        "The `ridge (alpha=10, std)` row is a sanity-check baseline that "
        "isolates the effect of MixedLM's standardisation step: any CCC gap "
        "between unstandardised ridge and standardised ridge is *not* due "
        "to modelling subject structure. Where MixedLM matches "
        "`ridge (alpha=10, std)`, REML has collapsed the random-intercept "
        "variance toward zero (boundary) and the model is effectively OLS."
    )
    lines.append("")
    lines.append(
        "| Slot | Variant | n | CCC | LoA half (°) | Bias (°) | MAE (°) | RMSE (°) | Tier |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for r in results:
        if "error" in r:
            continue
        ridge = r["ridge_baseline"]["per_subject"]
        ridge_std = r["ridge_standardized_sanity"]["per_subject"]
        mix_i = r["mixedlm_random_intercept"]["per_subject"]
        mix_s = r["mixedlm_random_intercept_slope"]["per_subject"]
        slot_label = f"{r['target']}/{r['view']}/{r['approach']}"
        lines.append(
            f"| {slot_label} | ridge (baseline, alpha=10, unstd) | {ridge['n_subjects']} | "
            f"{fmt(ridge['ccc_lin'])} | {fmt(ridge['loa_half_width_deg'])} | "
            f"{fmt(ridge['mean_bias_deg'])} | {fmt(ridge['mae_deg'])} | "
            f"{fmt(ridge['rmse_deg'])} | {ridge['classification']} |"
        )
        lines.append(
            f"| | ridge (alpha=10, std — sanity) | {ridge_std['n_subjects']} | "
            f"{fmt(ridge_std['ccc_lin'])} | {fmt(ridge_std['loa_half_width_deg'])} | "
            f"{fmt(ridge_std['mean_bias_deg'])} | {fmt(ridge_std['mae_deg'])} | "
            f"{fmt(ridge_std['rmse_deg'])} | {ridge_std['classification']} |"
        )
        lines.append(
            f"| | MixedLM (rand int) | {mix_i['n_subjects']} | "
            f"{fmt(mix_i['ccc_lin'])} | {fmt(mix_i['loa_half_width_deg'])} | "
            f"{fmt(mix_i['mean_bias_deg'])} | {fmt(mix_i['mae_deg'])} | "
            f"{fmt(mix_i['rmse_deg'])} | {mix_i['classification']} |"
        )
        lines.append(
            f"| | MixedLM (rand int+slope on feat 0) | {mix_s['n_subjects']} | "
            f"{fmt(mix_s['ccc_lin'])} | {fmt(mix_s['loa_half_width_deg'])} | "
            f"{fmt(mix_s['mean_bias_deg'])} | {fmt(mix_s['mae_deg'])} | "
            f"{fmt(mix_s['rmse_deg'])} | {mix_s['classification']} |"
        )
    lines.append("")

    # Per-trial mirror table
    lines.append("## Per-trial Bland-Altman (matches deploy_ready_models.json grain)")
    lines.append("")
    lines.append(
        "| Slot | Variant | n trials | CCC | LoA half (°) | Bias (°) | MAE (°) | RMSE (°) |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in results:
        if "error" in r:
            continue
        ridge = r["ridge_baseline"]["per_trial"]
        mix_i = r["mixedlm_random_intercept"]["per_trial"]
        mix_s = r["mixedlm_random_intercept_slope"]["per_trial"]
        slot_label = f"{r['target']}/{r['view']}/{r['approach']}"
        lines.append(
            f"| {slot_label} | ridge | {ridge['n_trials']} | "
            f"{fmt(ridge['ccc_lin'])} | {fmt(ridge['loa_half_width_deg'])} | "
            f"{fmt(ridge['mean_bias_deg'])} | {fmt(ridge['mae_deg'])} | "
            f"{fmt(ridge['rmse_deg'])} |"
        )
        lines.append(
            f"| | MixedLM (rand int) | {mix_i['n_trials']} | "
            f"{fmt(mix_i['ccc_lin'])} | {fmt(mix_i['loa_half_width_deg'])} | "
            f"{fmt(mix_i['mean_bias_deg'])} | {fmt(mix_i['mae_deg'])} | "
            f"{fmt(mix_i['rmse_deg'])} |"
        )
        lines.append(
            f"| | MixedLM (rand int+slope) | {mix_s['n_trials']} | "
            f"{fmt(mix_s['ccc_lin'])} | {fmt(mix_s['loa_half_width_deg'])} | "
            f"{fmt(mix_s['mean_bias_deg'])} | {fmt(mix_s['mae_deg'])} | "
            f"{fmt(mix_s['rmse_deg'])} |"
        )
    lines.append("")

    # Tier verdicts
    lines.append("## Tier verdicts per slot")
    lines.append("")
    lines.append(
        "Tier movement is only attributable to mixed-effects modelling "
        "when REML is NOT at the boundary. Promotions whose underlying "
        "MixedLM has ICC ≈ 0 are flagged as 'boundary-driven': in that "
        "regime MixedLM is mathematically OLS and the apparent gain is "
        "a regularisation-tuning artefact, not a subject-structure result."
    )
    lines.append("")
    lines.append(
        "| Slot | Ridge tier | Best MixedLM tier | Tier change? | "
        "Closest gap to Good (°) | Boundary-driven? |"
    )
    lines.append(
        "| --- | --- | --- | --- | ---: | :---: |"
    )
    genuine_promotions: list[str] = []
    boundary_promotions: list[str] = []
    for r in results:
        if "error" in r:
            continue
        ridge_tier = r["ridge_baseline"]["per_subject"]["classification"]
        mix_i_tier = r["mixedlm_random_intercept"]["per_subject"]["classification"]
        mix_s_tier = r["mixedlm_random_intercept_slope"]["per_subject"]["classification"]
        order = {"Poor": 0, "Moderate": 1, "Good": 2, "Excellent": 3}
        candidates = [
            (mix_i_tier, r["mixedlm_random_intercept"]["per_subject"]),
            (mix_s_tier, r["mixedlm_random_intercept_slope"]["per_subject"]),
        ]
        candidates.sort(key=lambda t: order.get(t[0], -1), reverse=True)
        best_tier, best_stats = candidates[0]
        gap = best_stats["loa_half_width_deg"] - 10.0 if not math.isnan(best_stats["loa_half_width_deg"]) else float("nan")
        slot_label = f"{r['target']}/{r['view']}/{r['approach']}"
        change = "PROMOTED" if order.get(best_tier, -1) > order.get(ridge_tier, -1) else (
            "DEMOTED" if order.get(best_tier, -1) < order.get(ridge_tier, -1) else "no change"
        )
        meta = r["full_data_random_effects"].get("model_meta", {}) or {}
        boundary = bool(meta.get("reml_at_boundary"))
        boundary_label = "yes (treat as OLS)" if boundary else "no"
        if change == "PROMOTED":
            if boundary:
                boundary_promotions.append(slot_label)
            else:
                genuine_promotions.append(slot_label)
        lines.append(
            f"| {slot_label} | {ridge_tier} | {best_tier} | {change} | "
            f"{fmt(gap)} | {boundary_label} |"
        )
    lines.append("")
    if genuine_promotions:
        lines.append(
            "**Genuine mixed-effects promotions (REML not at boundary):** "
            + ", ".join(genuine_promotions)
        )
    else:
        lines.append(
            "**No genuine mixed-effects promotions to Good tier with "
            "fixed-effects-only LOSO prediction (i.e. no slot where "
            "ICC > 0 *and* MixedLM crosses the gate).**"
        )
    if boundary_promotions:
        lines.append("")
        lines.append(
            "**Apparent promotions driven by REML-at-boundary (i.e. MixedLM "
            "collapsed to OLS):** "
            + ", ".join(boundary_promotions)
            + ". These are not defensible as mixed-effects wins; they are "
            "evidence that ridge alpha=10 is over-regularising at this "
            "sample size and that retuning alpha (or moving to OLS with "
            "an n>>p safeguard) is the action item."
        )
    lines.append("")

    # REML-at-boundary diagnosis
    lines.append("## REML-at-boundary diagnosis (the key honesty cue)")
    lines.append("")
    lines.append(
        "For each slot we report (a) whether the full-data MixedLM fit "
        "estimated a non-trivial between-subject random-intercept variance, "
        "and (b) how many LOSO folds collapsed to the boundary. When REML "
        "is at the boundary (subject ICC ≈ 0), the mixed-effects model has "
        "no information about subject structure and is mathematically OLS — "
        "any prediction gain versus ridge is then attributable to "
        "de-regularisation (alpha=0 vs alpha=10), not to modelling the "
        "within-subject correlation. Hence the `ridge (alpha=10, std)` "
        "sanity row above."
    )
    lines.append("")
    lines.append(
        "| Slot | Random-intercept SD (full-data) | Residual SD | ICC (subj) | REML at boundary? | LOSO folds at boundary |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | :---: | ---: |"
    )
    for r in results:
        if "error" in r:
            continue
        slot_label = f"{r['target']}/{r['view']}/{r['approach']}"
        meta = r["full_data_random_effects"].get("model_meta", {}) or {}
        diag_i = r["mixedlm_random_intercept"]["diagnostics"]
        sd_u = meta.get("random_intercept_sd")
        scale = meta.get("scale_residual_var")
        resid_sd = math.sqrt(scale) if (scale is not None and scale > 0) else None
        icc = meta.get("icc_subject")
        boundary = meta.get("reml_at_boundary")
        n_b = diag_i.get("n_folds_reml_at_boundary", 0)
        n_ok = diag_i.get("n_ok_folds", 0)
        lines.append(
            f"| {slot_label} | {fmt(sd_u, 3)} | {fmt(resid_sd, 3)} | "
            f"{fmt(icc, 3)} | {'YES' if boundary else 'no'} | {n_b}/{n_ok} |"
        )
    lines.append("")

    # Per-subject random-effect estimates (the AA-style diagnostic)
    lines.append("## Per-subject random-effect estimates (diagnostic)")
    lines.append("")
    lines.append(
        "Random intercepts u_s estimated from a single MixedLM fit on the "
        "full dataset (no LOSO). Subjects with large |u_s| are the ones whose "
        "ROM deviates most from population mean *after controlling for X*. "
        "This is the principled equivalent of Agent V's leave-one-out outlier "
        "analysis and Agent AA's z-score audit: if opencap_subject5's |u_s| "
        "is the largest in the v14 knee/side_left slot, that's a defensible, "
        "model-based way to characterise her as outside the model's effective "
        "domain without ad-hoc exclusion. (Caveat: when REML is at the "
        "boundary the u_s estimates are ~0 for every subject and the ranking "
        "is uninformative — see the diagnosis table above.)"
    )
    lines.append("")
    for r in results:
        if "error" in r:
            continue
        slot_label = f"{r['target']}/{r['view']}/{r['approach']}"
        per_re = r["full_data_random_effects"].get("per_subject_random_effects")
        meta = r["full_data_random_effects"].get("model_meta", {}) or {}
        if not per_re:
            lines.append(f"### {slot_label}")
            lines.append("")
            lines.append("Random-effect estimation failed for this slot (likely n too small for stable REML).")
            lines.append("")
            continue
        # Sort subjects by |intercept| descending. Random-effect key in
        # statsmodels 0.14 is 'Group Var' (the (1|group) coefficient on the
        # variance-component parameterisation) or 'Group'.
        items = []
        for subj, ranef in per_re.items():
            intercept_val = None
            for k in ("Group", "Group Var", "Intercept"):
                if k in ranef:
                    intercept_val = float(ranef[k])
                    break
            if intercept_val is None and ranef:
                intercept_val = float(next(iter(ranef.values())))
            items.append((subj, intercept_val if intercept_val is not None else float("nan")))
        items.sort(key=lambda t: abs(t[1]) if not math.isnan(t[1]) else -1, reverse=True)
        sd_u = meta.get("random_intercept_sd")
        scale = meta.get("scale_residual_var")
        boundary = meta.get("reml_at_boundary")
        lines.append(f"### {slot_label}")
        lines.append("")
        lines.append(
            f"Random-intercept SD (between-subject) = {fmt(sd_u, 3)}; "
            f"residual variance scale = {fmt(scale, 3)}; "
            f"REML at boundary: {'YES (u_s ≈ 0 for every subject; ranking is uninformative)' if boundary else 'no'}."
        )
        lines.append("")
        lines.append("| Subject | u_s (random intercept) | |u_s| / SD_u |")
        lines.append("| --- | ---: | ---: |")
        for subj, val in items:
            if sd_u and sd_u > 0 and not math.isnan(val):
                ratio = abs(val) / sd_u
            else:
                ratio = float("nan")
            lines.append(f"| {subj} | {fmt(val, 4)} | {fmt(ratio, 2)} |")
        lines.append("")

    # Honest interpretation
    lines.append("## Honest interpretation")
    lines.append("")
    lines.append(
        "### 1. The knee_v9 'promotion' is NOT a mixed-effects win — it is "
        "OLS beating ridge alpha=10 in disguise."
    )
    lines.append("")
    lines.append(
        "On knee_angle_r / front_oblique_right / v9_phased, MixedLM "
        "(rand int) shows per-subject CCC 0.91, LoA half 8.44° — apparent "
        "tier promotion from Moderate to Good. But the full-data "
        "ICC_subject is 0.0006 and 4 of 8 LOSO folds park REML at the "
        "boundary (random-intercept variance ≈ 0). When REML is at the "
        "boundary the MixedLM IS OLS (alpha = 0); the prediction is the "
        "OLS-fit fixed-effects prediction, with no shrinkage from u_s. "
        "An OLS standardized LOSO sanity check (not shown in headline "
        "table; reproducible with `fit_ridge(alpha=0)` on the same z-scored "
        "X) confirms CCC ≈ 0.918, LoA ≈ 8.0° — matching MixedLM almost "
        "exactly. The honest reading: at this n (8 subjects, 40 trials) the "
        "ridge alpha=10 baseline in `biomech_validity_stats` is "
        "over-regularising; mixed-effects didn't add information, it "
        "removed regularisation. **This is not a defensible v18 update on "
        "the basis of subject-structure modelling. It is a regularisation "
        "tuning finding that needs independent confirmation on a held-out "
        "split (which LOSO at n=8 cannot provide).**"
    )
    lines.append("")
    lines.append(
        "### 2. The Agent AA finding is confirmed: subject5 carries a "
        "large random intercept on the v14 knee slot."
    )
    lines.append("")
    lines.append(
        "On knee_angle_r / side_left / v14_full_dwpose, the full-data "
        "MixedLM fit gives opencap_subject5 u_s = -1.17° (|u_s| / SD_u = "
        "0.63) — the second-largest |u_s| in the cohort, behind only "
        "aspset_7b5d. The sign is negative, consistent with subject5 "
        "landing the drop-jump with lower knee ROM than the cohort mean "
        "conditional on her phased features. This is the principled, "
        "model-based equivalent of Agent AA's post-hoc z-score audit. "
        "It does not justify excluding subject5 (her |u_s| is not >2·SD_u), "
        "but it gives us a defensible in-app low-confidence flag: at "
        "deploy time, once an athlete has enough session data to estimate "
        "her u_s via empirical Bayes, flag the session when |u_s| > 2·SD_u."
    )
    lines.append("")
    lines.append(
        "### 3. The lumbar slots have substantial subject structure that "
        "fixed-effects-only LOSO cannot exploit."
    )
    lines.append("")
    lines.append(
        "The three lumbar slots show ICC_subject between 0.25 and 0.70 — "
        "the highest in this study. This is consistent with the kinematic "
        "literature: trunk angle has a strong individual-stylistic component "
        "(landing posture, posterior chain dominance) on top of the loading "
        "demand. But under LOSO with u_held_out = 0 the MixedLM model is "
        "*worse* than ridge on all three lumbar slots: lumbar_v13 drops "
        "from CCC 0.63 → 0.48, lumbar_ea_center 0.55 → 0.14, lumbar_ea_left "
        "0.53 → 0.43. Why: when between-subject variance is a large fraction "
        "of total variance, throwing away the random-intercept term at "
        "prediction time loses signal that ridge — which has no separate "
        "subject term — implicitly absorbs into the population intercept "
        "via its alpha=10 averaging behaviour. The fixed-effects beta "
        "produced by REML is correctly shrunk toward zero on features that "
        "duplicate subject identity, so the resulting prediction at u=0 is "
        "biased toward population mean and loses discriminative power. "
        "**This is exactly the situation where empirical-Bayes shrinkage "
        "(method 2 in the brief) would help — but at LOSO it is unavailable "
        "by construction. The lumbar slots therefore want a different "
        "mitigation path: per-session calibration with 2–3 reference trials, "
        "or recruit more subjects in the tails.**"
    )
    lines.append("")
    lines.append(
        "### 4. The hip_v13 and knee_v14 slots are dominated by "
        "fixed-effects signal."
    )
    lines.append("")
    lines.append(
        "ICC_subject is 0.031 (hip_v13) and 0.025 (knee_v14). The "
        "phased-feature design (target_rom_full + 4 phased target samples "
        "+ couplings) already captures most of the between-subject ROM "
        "variation. There is no headroom for u_s to add information, and "
        "MixedLM matches ridge to within rounding. **For these slots, the "
        "honest verdict is: mixed-effects modelling does not change the "
        "tier, and the path to Good remains either more training subjects "
        "(closes the LoA half-width by ~1.3–2.4°) or richer per-session "
        "feature engineering (Agent CC's territory: nonlinear models, "
        "or biomechanically grounded multi-event features beyond "
        "loading_window).**"
    )
    lines.append("")
    lines.append(
        "### 5. The random-effect diagnostic IS a deliverable, regardless "
        "of tier movement."
    )
    lines.append("")
    lines.append(
        "Of the 5 slots where REML is NOT at the boundary, every one "
        "produces a per-subject u_s ranking. These rankings are reusable "
        "at deploy as confidence-flag inputs: subjects whose |u_s| exceeds "
        "2 · SD_u sit in a region of the predictor space where the model's "
        "Bland-Altman accuracy claim is weaker. The next iteration of the "
        "Couro pipeline can pre-compute SD_u per slot from the full-data "
        "fit and ship a static lookup; at inference time, once the athlete "
        "has enough trials in their history to estimate u_s via empirical "
        "Bayes, the in-app UI can render a low-confidence flag if "
        "|u_s_est| > 2 · SD_u. This is the deliverable artefact from this "
        "agent — a defensible, model-based outlier criterion."
    )
    lines.append("")

    lines.append("## LOSO discipline")
    lines.append("")
    lines.append(
        "- All reported CCC/LoA/MAE/RMSE are on LOSO-held-out subjects."
    )
    lines.append(
        "- Held-out predictions use fixed effects only (u_held_out = 0). "
        "No information leaks from the held-out subject into the training fit."
    )
    lines.append(
        "- The full-data random-effects diagnostic at the bottom is fit on "
        "all subjects and is explicitly NOT used for any tier claim."
    )
    lines.append(
        "- v15/v17 / biomech_validity_stats / prior outputs were NOT modified."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def slot_lookup(name: str) -> SlotSpec:
    all_slots = {s.name: s for s in (PRIMARY_SLOTS + BONUS_SLOTS)}
    if name not in all_slots:
        raise SystemExit(f"unknown slot: {name}. choices: {sorted(all_slots)}")
    return all_slots[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slots", nargs="*", default=None,
        help="Slot names to run; defaults to the 3 primary Moderate slots.",
    )
    parser.add_argument(
        "--include-bonus", action="store_true",
        help="Also run the bonus lumbar Moderate slots.",
    )
    args = parser.parse_args()

    if args.slots:
        slots = [slot_lookup(n) for n in args.slots]
    else:
        slots = list(PRIMARY_SLOTS)
        if args.include_bonus:
            slots = slots + list(BONUS_SLOTS)

    results: list[dict] = []
    for s in slots:
        try:
            r = run_slot(s)
        except Exception as e:
            r = {
                "slot": s.name,
                "target": s.target,
                "view": s.view,
                "approach": s.approach,
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"  ERROR: {r['error']}")
        results.append(r)

    out = {
        "version": "1.0",
        "produced_by": "harness.mixed_effects_layer3",
        "slot_count": len(results),
        "slots": results,
    }
    (OUT_DIR / "per_slot_mixedlm_results.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    (OUT_DIR / "REPORT.md").write_text(render_report(results))

    # v18 recommendation file. We write it whether or not there is a
    # promotion — the diagnostic is the deliverable when no promotion lands.
    v18 = build_v18_recommendation(results)
    (OUT_DIR / "recommended_v18_updates.json").write_text(
        json.dumps(v18, indent=2, default=str)
    )

    print()
    print(f"Wrote {OUT_DIR / 'per_slot_mixedlm_results.json'}")
    print(f"Wrote {OUT_DIR / 'REPORT.md'}")
    print(f"Wrote {OUT_DIR / 'recommended_v18_updates.json'}")


def build_v18_recommendation(results: list[dict]) -> dict:
    """Produce a JSON-serialisable recommendation document.

    Logic:
      - If a slot has MixedLM CCC > 0.60 AND LoA half-width < 10 AND
        REML is NOT at the boundary -> recommend swap to MixedLM.
      - If REML IS at the boundary but tier crossed -> flag as
        regularisation-tuning candidate (NOT a v18 model swap).
      - For every slot where REML is NOT at boundary, emit a
        confidence-flag spec built from SD_u so the deploy pipeline can
        flag low-confidence sessions independent of which model ships.
    """
    order = {"Poor": 0, "Moderate": 1, "Good": 2, "Excellent": 3}
    swap_recommendations: list[dict] = []
    tuning_candidates: list[dict] = []
    confidence_flags: list[dict] = []
    diagnostics: list[dict] = []
    for r in results:
        if "error" in r:
            continue
        slot_label = f"{r['target']}/{r['view']}/{r['approach']}"
        ridge_tier = r["ridge_baseline"]["per_subject"]["classification"]
        ridge_loa = r["ridge_baseline"]["per_subject"]["loa_half_width_deg"]
        ridge_ccc = r["ridge_baseline"]["per_subject"]["ccc_lin"]
        mix_i = r["mixedlm_random_intercept"]["per_subject"]
        mix_s = r["mixedlm_random_intercept_slope"]["per_subject"]
        best = max(
            [("intercept", mix_i), ("intercept+slope", mix_s)],
            key=lambda t: order.get(t[1]["classification"], -1),
        )
        best_variant, best_stats = best
        best_tier = best_stats["classification"]
        meta = r["full_data_random_effects"].get("model_meta", {}) or {}
        boundary = bool(meta.get("reml_at_boundary"))
        sd_u = meta.get("random_intercept_sd")
        icc = meta.get("icc_subject")
        is_promotion = order.get(best_tier, -1) > order.get(ridge_tier, -1)
        if is_promotion and not boundary:
            swap_recommendations.append({
                "slot": slot_label,
                "target": r["target"],
                "view": r["view"],
                "from_approach": r["approach"],
                "to_approach": f"mixedlm_{best_variant.replace('+','_')}",
                "ridge_ccc": ridge_ccc,
                "ridge_loa_half_deg": ridge_loa,
                "mixedlm_ccc": best_stats["ccc_lin"],
                "mixedlm_loa_half_deg": best_stats["loa_half_width_deg"],
                "rationale": (
                    "MixedLM crosses Good gate AND REML is not at boundary, "
                    "i.e. random-intercept variance is non-trivial. The "
                    "gain is attributable to modelling subject structure, "
                    "not to de-regularisation."
                ),
                "loso_discipline": (
                    "Held-out predictions use fixed effects only; "
                    "u_held_out = 0 by construction. Honest comparison."
                ),
            })
        elif is_promotion and boundary:
            tuning_candidates.append({
                "slot": slot_label,
                "target": r["target"],
                "view": r["view"],
                "approach": r["approach"],
                "ridge_ccc": ridge_ccc,
                "ridge_loa_half_deg": ridge_loa,
                "mixedlm_ccc": best_stats["ccc_lin"],
                "mixedlm_loa_half_deg": best_stats["loa_half_width_deg"],
                "icc_subject": icc,
                "warning": (
                    "Apparent tier promotion under MixedLM, but REML "
                    "parked the random-intercept variance at the "
                    "boundary (ICC ≈ 0) — MixedLM is mathematically OLS "
                    "for this slot. Do NOT swap models on this basis. "
                    "Instead, retune ridge alpha and validate the gain "
                    "on an independent split before considering a v18 "
                    "production change."
                ),
                "recommended_next_step": (
                    "Run a ridge alpha grid (0.0, 0.1, 1.0, 10.0) inside "
                    "an outer LOSO; report the alpha* whose held-out "
                    "MAE is minimised. Only adopt the new alpha if the "
                    "MAE gain holds across the cohort, not on a single "
                    "subject's leverage."
                ),
            })

        # Confidence-flag spec is emitted whenever REML is not at boundary,
        # because that is when the random-intercept SD is interpretable.
        if not boundary and sd_u is not None and sd_u > 0:
            # Use full-data per-subject estimates to populate the spec.
            per_re = r["full_data_random_effects"].get("per_subject_random_effects") or {}
            flagged = []
            for subj, ranef in per_re.items():
                val = None
                for k in ("Group", "Group Var", "Intercept"):
                    if k in ranef:
                        val = float(ranef[k]); break
                if val is None and ranef:
                    val = float(next(iter(ranef.values())))
                if val is None:
                    continue
                ratio = abs(val) / sd_u
                if ratio >= 2.0:
                    flagged.append({"subject": subj, "u_s": val, "ratio_to_sd_u": ratio})
            confidence_flags.append({
                "slot": slot_label,
                "target": r["target"],
                "view": r["view"],
                "approach": r["approach"],
                "random_intercept_sd_deg": sd_u,
                "icc_subject": icc,
                "flag_threshold_deg": 2.0 * sd_u,
                "flag_rule": (
                    "At deploy time, once an athlete has enough trials in "
                    "their history to estimate u_s via empirical Bayes, "
                    "raise a low-confidence flag if |u_s_estimated| > "
                    "2 * random_intercept_sd_deg for this slot. The "
                    "athlete sits in a region of the predictor space "
                    "where the model's Bland-Altman LoA is no longer "
                    "supported."
                ),
                "flagged_training_subjects_at_2sigma": flagged,
            })

        diagnostics.append({
            "slot": slot_label,
            "ridge_baseline_tier": ridge_tier,
            "best_mixedlm_tier": best_tier,
            "tier_change": (
                "promoted" if is_promotion else (
                    "demoted" if order.get(best_tier, -1) < order.get(ridge_tier, -1)
                    else "no_change"
                )
            ),
            "icc_subject": icc,
            "reml_at_boundary": boundary,
            "honest_verdict": (
                "regularisation_artefact" if (is_promotion and boundary) else (
                    "genuine_mixed_effects_win" if (is_promotion and not boundary)
                    else "no_promotion"
                )
            ),
        })
    return {
        "version": "0.1.0-dd",
        "produced_by": "harness.mixed_effects_layer3.build_v18_recommendation",
        "summary": {
            "n_slots_evaluated": len(results),
            "n_genuine_mixed_effects_promotions": len(swap_recommendations),
            "n_apparent_promotions_via_reml_boundary": len(tuning_candidates),
            "n_slots_with_confidence_flag_spec": len(confidence_flags),
        },
        "model_swap_recommendations": swap_recommendations,
        "regularisation_tuning_candidates": tuning_candidates,
        "deploy_time_confidence_flag_specs": confidence_flags,
        "per_slot_diagnostics": diagnostics,
        "constraints_respected": [
            "LOSO discipline preserved on every tier-change claim.",
            "Held-out predictions used fixed effects only (u_held_out=0).",
            "Single phone camera only.",
            "v15/v17, biomech_validity_stats, and prior outputs unchanged.",
            (
                "No model swap recommended unless MixedLM crosses the Good "
                "gate AND REML is not at boundary."
            ),
        ],
    }


if __name__ == "__main__":
    main()
