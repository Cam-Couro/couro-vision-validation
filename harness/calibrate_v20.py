"""Agent QQ -- Nested-LOSO residual calibration on the v20 reader.

v20 = Agent GG2's ROM-aware OpenCap-only learned Layer 2 + ridge L3
(``models/rom_aware_layer2_alldata_v1.pt`` -> FF's ridge pipeline).

This is the v20 sibling of Agent PP's calibration build (v37 = v23+cal,
v38 = v31+cal, v39 = v17+cal). PP did NOT calibrate v20, which is the
reader that wins ``hip_adduction_r / front_oblique_left`` in v40's
oracle. That slot has the **tightest LoA in the validation table
(+/-3.3 deg)** but a modest CCC of 0.69 -- the textbook bias-dominated
pattern that residual calibration is designed to fix.

# Outputs

  * ``results/deploy_ready_models_v41_v20_calibrated.json``
  * ``data/residual_calibration/per_slot_validity_v41_v20_calibrated.json``

# Algorithm

Identical to PP's nested-LOSO calibration (see
``harness/residual_calibration.py``):

  Outer LOSO: for each subject S in cohort:
    Inner LOSO: for each training subject T (T != S):
      Train ridge alpha=10 on cohort minus {S, T}
      Predict T's trials -> store (pred_pseudo_T, y_T)
    Stack inner predictions across all T -> (pred_pseudo, y_pseudo)
    Fit calibration  y_pseudo = a * pred_pseudo + b
                     using closed-form least squares
    Train outer ridge on cohort minus {S}
    pred_S = outer_ridge.predict(X_S)
    pred_S_cal = a * pred_S + b
  Aggregate (pred_S_cal, y_S) across all S; compute Bland-Altman + CCC
  on per-subject means.

# Per-slot fallback

If calibration's LoA half-width is >= uncalibrated LoA half-width
(i.e. delta >= 0), or calibration's CCC underperforms uncalibrated by
more than 0.05 CCC, we fall back to the uncalibrated v20 prediction
for that slot. Same fallback rule as PP.

# LOSO discipline

* Layer 2 (GG2 ROM-aware): trained on all 9 OpenCap subjects, NO LOSO
  at L2 (same as v20 baseline -- upper bound for OpenCap subjects).
* Layer 3 (ridge): outer LOSO at subject level.
* Calibration fit: inner LOSO over training-only subjects. The outer
  held-out subject S is NEVER used to choose (a, b).
"""
from __future__ import annotations

import builtins
import json
import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

# numpy 2.x compat for any chumpy-backed loads.
for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.train_regression_poc import fit_ridge  # noqa: E402
from harness.per_slot_prediction_dump import DumpAccumulator  # noqa: E402


# Agent SS per-clip dump infrastructure for v41 (v20 + nested calibration).
DUMP_ACCUMULATOR: DumpAccumulator | None = None

# Silence ridge runtime warnings.
warnings_filter = logging.getLogger("py.warnings")
warnings_filter.setLevel(logging.ERROR)
import warnings  # noqa: E402
warnings.filterwarnings("ignore", category=RuntimeWarning)


# -------------------------------------------------------------------------
# Paths.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
OUT_DIR: Final[Path] = DATA_ROOT / "residual_calibration"

V17_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v17_selective.json"
)
V20_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v20_rom_aware.json"
)
BASELINE_PATH: Final[Path] = (
    DATA_ROOT / "biomech_validity_stats" / "per_slot_validity.json"
)
V20_PER_SLOT_PATH: Final[Path] = (
    DATA_ROOT / "rom_aware_layer2" / "per_slot_validity_v20.json"
)

# GG2 ROM-aware all-data L2 checkpoint.
L2_GG2_CKPT: Final[Path] = (
    REPO_ROOT / "models" / "rom_aware_layer2_alldata_v1.pt"
)

# Tag for outputs.
V41_TAG: Final[str] = "v41_v20_calibrated"
V41_DEPLOY_PATH: Final[Path] = (
    RESULTS_DIR / f"deploy_ready_models_{V41_TAG}.json"
)
V41_PER_SLOT_PATH: Final[Path] = OUT_DIR / f"per_slot_validity_{V41_TAG}.json"

# CCC degradation fallback threshold (per Cameron's spec).
CCC_FALLBACK_DELTA: Final[float] = 0.05


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[QQ-cal t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Metrics (mirrors PP for byte-identical reporting).
# -------------------------------------------------------------------------


def _ccc(p: np.ndarray, o: np.ndarray) -> float:
    if len(p) < 2:
        return float("nan")
    mp, mo = float(np.mean(p)), float(np.mean(o))
    vp, vo = float(np.var(p, ddof=0)), float(np.var(o, ddof=0))
    cov = float(np.mean((p - mp) * (o - mo)))
    denom = vp + vo + (mp - mo) ** 2
    if denom == 0:
        return float("nan")
    return 2.0 * cov / denom


def _pearson(p: np.ndarray, o: np.ndarray) -> float:
    if len(p) < 2 or p.std(ddof=1) == 0 or o.std(ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(p, o)[0, 1])


def _classify(ccc: float, loa_half: float) -> str:
    if math.isnan(ccc):
        return "Poor"
    if ccc > 0.75 and loa_half < 5:
        return "Excellent"
    if ccc > 0.60 and loa_half < 10:
        return "Good"
    if ccc > 0.40 and loa_half < 15:
        return "Moderate"
    return "Poor"


def _stats_per_subject(
    pred: np.ndarray, obs: np.ndarray, subjects: np.ndarray
) -> dict:
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[mask]
    obs = obs[mask]
    subjects = subjects[mask]
    n_trials = int(len(pred))
    n_subj_pre = len(set(subjects.tolist()))
    unique = sorted(set(subjects.tolist()))
    agg_pred = np.array([float(np.mean(pred[subjects == s])) for s in unique])
    agg_obs = np.array([float(np.mean(obs[subjects == s])) for s in unique])
    if len(agg_pred) < 2:
        return {
            "n_subjects": n_subj_pre, "n_trials": n_trials,
            "mean_bias_deg": float("nan"), "sd_bias_deg": float("nan"),
            "loa_upper_deg": float("nan"), "loa_lower_deg": float("nan"),
            "loa_half_width_deg": float("nan"),
            "pearson_r": float("nan"), "ccc_lin": float("nan"),
            "mae_deg": float("nan"), "rmse_deg": float("nan"),
            "classification": "Poor",
        }
    diffs = agg_pred - agg_obs
    mean_bias = float(np.mean(diffs))
    sd_bias = float(np.std(diffs, ddof=1))
    upper = mean_bias + 1.96 * sd_bias
    lower = mean_bias - 1.96 * sd_bias
    loa_half = max(abs(upper - mean_bias), abs(mean_bias - lower))
    r = _pearson(agg_pred, agg_obs)
    ccc = _ccc(agg_pred, agg_obs)
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    return {
        "n_subjects": n_subj_pre, "n_trials": n_trials,
        "mean_observed_deg": float(np.mean(agg_obs)),
        "sd_observed_deg": float(np.std(agg_obs, ddof=1)),
        "mean_predicted_deg": float(np.mean(agg_pred)),
        "sd_predicted_deg": float(np.std(agg_pred, ddof=1)),
        "mean_bias_deg": mean_bias,
        "sd_bias_deg": sd_bias,
        "loa_upper_deg": upper,
        "loa_lower_deg": lower,
        "loa_half_width_deg": loa_half,
        "pearson_r": r,
        "ccc_lin": ccc,
        "mae_deg": mae,
        "rmse_deg": rmse,
        "classification": _classify(ccc, loa_half),
    }


# -------------------------------------------------------------------------
# Calibration core (identical to PP).
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """Linear ``y_calibrated = a * pred + b`` calibration."""
    a: float
    b: float
    n_pseudo: int

    def apply(self, pred: np.ndarray) -> np.ndarray:
        return self.a * pred + self.b


def _fit_calibration(
    pseudo_pred: np.ndarray, pseudo_y: np.ndarray
) -> Calibration:
    """Fit a 1-D linear calibration via closed-form least squares.

    Falls back to identity (a=1, b=0) if input is degenerate.
    """
    mask = np.isfinite(pseudo_pred) & np.isfinite(pseudo_y)
    p = pseudo_pred[mask].astype(np.float64)
    y = pseudo_y[mask].astype(np.float64)
    if len(p) < 3 or float(np.std(p, ddof=1)) < 1e-9:
        return Calibration(a=1.0, b=0.0, n_pseudo=int(len(p)))
    var_p = float(np.var(p, ddof=0))
    if var_p < 1e-12:
        return Calibration(a=1.0, b=0.0, n_pseudo=int(len(p)))
    cov_py = float(np.mean((p - p.mean()) * (y - y.mean())))
    a = cov_py / var_p
    b = float(y.mean() - a * p.mean())
    if not math.isfinite(a) or not math.isfinite(b):
        return Calibration(a=1.0, b=0.0, n_pseudo=int(len(p)))
    return Calibration(a=float(a), b=float(b), n_pseudo=int(len(p)))


def nested_loso_with_calibration(
    X: np.ndarray, y: np.ndarray, subjects: list[str],
    *, alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Nested-LOSO ridge + per-fold calibration. Identical to PP."""
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    pred_uncal = np.full(len(y), np.nan)
    pred_cal = np.full(len(y), np.nan)
    fold_records: list[dict] = []

    for outer_s in unique:
        outer_train_mask = subj_arr != outer_s
        outer_test_mask = subj_arr == outer_s
        if outer_train_mask.sum() < 4 or outer_test_mask.sum() < 1:
            continue

        inner_subjects = sorted(set(subj_arr[outer_train_mask].tolist()))
        pseudo_pred_list: list[np.ndarray] = []
        pseudo_y_list: list[np.ndarray] = []
        for inner_s in inner_subjects:
            inner_train_mask = (
                outer_train_mask & (subj_arr != inner_s)
            )
            inner_test_mask = outer_train_mask & (subj_arr == inner_s)
            if (
                inner_train_mask.sum() < 4
                or inner_test_mask.sum() < 1
            ):
                continue
            try:
                inner_model = fit_ridge(
                    X[inner_train_mask], y[inner_train_mask], alpha=alpha,
                )
                p_inner = inner_model.predict(X[inner_test_mask])
            except Exception:
                continue
            pseudo_pred_list.append(p_inner)
            pseudo_y_list.append(y[inner_test_mask])

        if not pseudo_pred_list:
            cal = Calibration(a=1.0, b=0.0, n_pseudo=0)
        else:
            pp = np.concatenate(pseudo_pred_list)
            py = np.concatenate(pseudo_y_list)
            cal = _fit_calibration(pp, py)

        try:
            outer_model = fit_ridge(
                X[outer_train_mask], y[outer_train_mask], alpha=alpha,
            )
            p_outer = outer_model.predict(X[outer_test_mask])
        except Exception:
            continue
        pred_uncal[outer_test_mask] = p_outer
        pred_cal[outer_test_mask] = cal.apply(p_outer)
        fold_records.append({
            "subject": outer_s,
            "a": cal.a, "b": cal.b, "n_pseudo": cal.n_pseudo,
            "n_test_trials": int(outer_test_mask.sum()),
        })

    return pred_uncal, pred_cal, y, subj_arr, fold_records


# -------------------------------------------------------------------------
# L2 patching for GG2 (mirrors FF / GG2 layer3 setup).
# -------------------------------------------------------------------------


def _setup_l2_gg2() -> tuple:
    """Load the GG2 ROM-aware all-data L2 checkpoint."""
    from harness.learned_layer2_real_gt import (
        TemporalKeypointCNNConf, HALPE_KEYS_WITH_SMPL,
    )
    from harness.smpl_layer2_poc import HALPE26_INDEX

    ckpt = torch.load(L2_GG2_CKPT, weights_only=False)
    model = TemporalKeypointCNNConf()
    model.load_state_dict(ckpt["model_state"])
    model.y_mean = ckpt["y_mean"]  # type: ignore[attr-defined]
    model.y_std = ckpt["y_std"]    # type: ignore[attr-defined]
    halpe_idx_arr = np.array(
        [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL],
        dtype=np.int64,
    )
    log(f"loaded GG2 L2 from {L2_GG2_CKPT}")
    return model, halpe_idx_arr


def _patch_motion_data(model, halpe_idx_arr) -> None:
    """Patch keypoints_to_motion_data with FF's factory using GG2 model."""
    from harness import couro_keypoints as ck_mod
    from harness import layer3_retrain_on_learned_l2 as ff
    import harness.train_regression_poc as trp
    import harness.train_v9_phased as v9
    import harness.train_v12_combined as v12
    import harness.train_v14_full_dwpose as v14
    import harness.train_event_anchored_all as ea
    import harness.train_ankle_bilateral as ea_ank

    original_fn = ck_mod.keypoints_to_motion_data
    patched_fn = ff._patched_keypoints_to_motion_data_factory(
        model, halpe_idx_arr, original_fn,
    )
    ck_mod.keypoints_to_motion_data = patched_fn
    for mod in (trp, v9, v12, v14, ea, ea_ank):
        if hasattr(mod, "keypoints_to_motion_data"):
            mod.keypoints_to_motion_data = patched_fn
    log("monkey-patched keypoints_to_motion_data in 6 modules")


def _unpatch_motion_data(original_fn) -> None:
    from harness import couro_keypoints as ck_mod
    import harness.train_regression_poc as trp
    import harness.train_v9_phased as v9
    import harness.train_v12_combined as v12
    import harness.train_v14_full_dwpose as v14
    import harness.train_event_anchored_all as ea
    import harness.train_ankle_bilateral as ea_ank

    ck_mod.keypoints_to_motion_data = original_fn
    for mod in (trp, v9, v12, v14, ea, ea_ank):
        if hasattr(mod, "keypoints_to_motion_data"):
            mod.keypoints_to_motion_data = original_fn


# -------------------------------------------------------------------------
# Per-slot feature builder (mirrors PP / FF).
# -------------------------------------------------------------------------


def _collect_features_for_slot(target: str, view: str, approach: str):
    from harness.train_multi_metric import VIEW_BUCKET_NAMES
    import harness.train_v9_phased as v9
    import harness.train_v12_combined as v12
    import harness.train_v14_full_dwpose as v14
    import harness.train_event_anchored_all as ea
    import harness.train_ankle_bilateral as ea_ank

    VIEW_TO_CAM = {v: k for k, v in VIEW_BUCKET_NAMES.items()}

    def _build_v12(target, view):
        v9.KP_DIR = DATA_ROOT / "couro_keypoints"
        v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_couro_keypoints"
        return v12.build_combined_dataset(target, view)

    def _build_v13(target, view):
        v9.KP_DIR = DATA_ROOT / "couro_keypoints"
        v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
        return v12.build_combined_dataset(target, view)

    def _build_v14(target, view):
        v9.KP_DIR = DATA_ROOT / "opencap_dwpose_keypoints"
        v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
        if target == "ankle_angle_r":
            return v14.build_opencap_only_dataset(target, view)
        return v12.build_combined_dataset(target, view)

    def _build_v9p(target, view):
        v9.KP_DIR = DATA_ROOT / "couro_keypoints"
        cam = VIEW_TO_CAM.get(view)
        if cam is None:
            return None
        X, y, subjects, _, _ = v9.build_dataset_v9(target, cam)
        if len(y) == 0:
            return None
        return (
            X.astype(np.float64),
            np.asarray(y, dtype=np.float64),
            subjects,
            ["opencap"] * len(y),
        )

    def _build_ea(target, view):
        ea.KP_DIR = DATA_ROOT / "couro_keypoints"
        cam = VIEW_TO_CAM.get(view)
        if cam is None:
            return None
        X, y, subjects, _, _ = ea.build_dataset(target, cam)
        if len(y) == 0:
            return None
        return (
            X.astype(np.float64),
            np.asarray(y, dtype=np.float64),
            subjects,
            ["opencap"] * len(y),
        )

    def _build_ea_bilat(target, view):
        if target != "ankle_angle_r":
            return None
        ea_ank.KP_DIR = DATA_ROOT / "couro_keypoints"
        cam = VIEW_TO_CAM.get(view)
        if cam is None:
            return None
        if hasattr(ea_ank, "build_dataset"):
            try:
                res = ea_ank.build_dataset(cam)
            except TypeError:
                res = ea_ank.build_dataset(target, cam)
            if isinstance(res, tuple):
                X, y, subjects = res[0], res[1], res[2]
                if len(y) == 0:
                    return None
                return (
                    np.asarray(X, dtype=np.float64),
                    np.asarray(y, dtype=np.float64),
                    list(subjects),
                    ["opencap"] * len(y),
                )
        return None

    BUILDERS = {
        "v12_combined": _build_v12,
        "v13_dwpose_hybrid": _build_v13,
        "v14_full_dwpose": _build_v14,
        "v9_phased": _build_v9p,
        "event_anchored": _build_ea,
        "event_anchored_bilateral": _build_ea_bilat,
    }
    builder = BUILDERS.get(approach)
    if builder is None:
        return None
    try:
        return builder(target, view)
    except Exception as e:
        log(f"  builder for {target}/{view}/{approach} failed: {e!r}")
        return None


# -------------------------------------------------------------------------
# Orchestration.
# -------------------------------------------------------------------------


def _ccc_finite(x: float | None) -> float:
    if x is None:
        return float("-inf")
    if isinstance(x, float) and math.isnan(x):
        return float("-inf")
    return float(x)


def _loa_finite(x: float | None) -> float:
    if x is None:
        return float("inf")
    if isinstance(x, float) and math.isnan(x):
        return float("inf")
    return float(x)


def run() -> dict:
    """Run nested-LOSO calibration on the v20 reader."""
    log(f"=== reader=v20  tag={V41_TAG}  l2=GG2 ROM-aware ===")

    # Patch L2 with GG2 model.
    from harness import couro_keypoints as ck_mod
    original_fn = ck_mod.keypoints_to_motion_data
    model, halpe_idx_arr = _setup_l2_gg2()
    _patch_motion_data(model, halpe_idx_arr)

    try:
        v17 = json.loads(V17_PATH.read_text())
        baseline = json.loads(BASELINE_PATH.read_text())
        baseline_by_slot = {
            (s["target"], s["view"]): s for s in baseline["slots"]
        }
        v20_deploy = json.loads(V20_PATH.read_text())
        v20_per_slot_raw = json.loads(V20_PER_SLOT_PATH.read_text())
        v20_per_slot_by_key: dict[tuple[str, str], dict] = {}
        for s in v20_per_slot_raw.get("slots", []):
            if "target" in s and "view" in s:
                v20_per_slot_by_key[(s["target"], s["view"])] = s

        slot_keys = []
        for target, slots in v17["models"].items():
            for view, entry in slots.items():
                slot_keys.append((target, view, entry))
        log(f"=== {len(slot_keys)} slots to process ===")

        per_slot_validity: list[dict] = []
        deploy_models: dict[str, dict[str, dict]] = {}
        tier_counter = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
        cal_helped: list[dict] = []
        cal_hurt: list[dict] = []
        ccc_regressed_slots: list[dict] = []

        for i, (target, view, entry) in enumerate(slot_keys):
            approach = entry.get("approach", "unknown")
            t0 = time.time()
            log(
                f"--- slot {i+1}/{len(slot_keys)} {target}/{view} "
                f"approach={approach} ---"
            )

            result = _collect_features_for_slot(target, view, approach)
            if result is None:
                log("  no builder result; carrying baseline")
                base = baseline_by_slot.get((target, view), {})
                per_slot_validity.append({
                    **base, f"{V41_TAG}_skipped": True,
                    "skip_reason": f"no builder for {approach}",
                    "target": target, "view": view,
                })
                # Carry the v20 entry through as-is.
                v20_entry = (
                    v20_deploy.get("models", {})
                    .get(target, {}).get(view) or entry
                )
                deploy_models.setdefault(target, {})[view] = v20_entry
                continue

            X, y, subjects, sources = result
            if len(y) < 20:
                log(f"  only n={len(y)} samples; carrying baseline")
                base = baseline_by_slot.get((target, view), {})
                per_slot_validity.append({
                    **base, f"{V41_TAG}_skipped": True,
                    "skip_reason": f"only n={len(y)} samples",
                    "target": target, "view": view,
                })
                v20_entry = (
                    v20_deploy.get("models", {})
                    .get(target, {}).get(view) or entry
                )
                deploy_models.setdefault(target, {})[view] = v20_entry
                continue

            try:
                pred_unc, pred_cal, y_out, subj_out, fold_recs = (
                    nested_loso_with_calibration(
                        X.astype(np.float64),
                        y.astype(np.float64),
                        list(subjects),
                        alpha=10.0,
                    )
                )
            except Exception as e:
                log(f"  nested LOSO failed: {e!r}")
                traceback.print_exc(file=sys.stdout)
                base = baseline_by_slot.get((target, view), {})
                per_slot_validity.append({
                    **base, f"{V41_TAG}_skipped": True,
                    "skip_reason": f"nested LOSO error: {e!r}",
                    "target": target, "view": view,
                })
                v20_entry = (
                    v20_deploy.get("models", {})
                    .get(target, {}).get(view) or entry
                )
                deploy_models.setdefault(target, {})[view] = v20_entry
                continue

            stats_unc = _stats_per_subject(pred_unc, y_out, subj_out)
            stats_cal = _stats_per_subject(pred_cal, y_out, subj_out)

            if fold_recs:
                a_vals = [f["a"] for f in fold_recs]
                b_vals = [f["b"] for f in fold_recs]
                np_vals = [f["n_pseudo"] for f in fold_recs]
                cal_summary = {
                    "a_mean": float(np.mean(a_vals)),
                    "a_std": float(np.std(a_vals, ddof=0)),
                    "b_mean": float(np.mean(b_vals)),
                    "b_std": float(np.std(b_vals, ddof=0)),
                    "n_pseudo_mean": float(np.mean(np_vals)),
                    "n_folds": len(fold_recs),
                }
            else:
                cal_summary = {
                    "a_mean": 1.0, "a_std": 0.0, "b_mean": 0.0, "b_std": 0.0,
                    "n_pseudo_mean": 0.0, "n_folds": 0,
                }

            loa_unc = stats_unc.get("loa_half_width_deg")
            loa_cal = stats_cal.get("loa_half_width_deg")
            ccc_unc = stats_unc.get("ccc_lin")
            ccc_cal = stats_cal.get("ccc_lin")
            try:
                loa_delta = float(loa_cal) - float(loa_unc)
            except (TypeError, ValueError):
                loa_delta = float("nan")
            try:
                ccc_delta = (
                    _ccc_finite(ccc_cal) - _ccc_finite(ccc_unc)
                )
            except (TypeError, ValueError):
                ccc_delta = float("nan")

            # Fallback rule (per Cameron's spec):
            #   * choose calibrated if it tightens LoA AND does not regress
            #     CCC by more than CCC_FALLBACK_DELTA;
            #   * otherwise fall back to uncalibrated.
            ccc_regressed = (
                math.isfinite(ccc_delta)
                and ccc_delta < -CCC_FALLBACK_DELTA
            )
            loa_tightened = (
                math.isfinite(loa_delta) and loa_delta < 0
            )
            if loa_tightened and not ccc_regressed:
                chosen = "calibrated"
                chosen_stats = stats_cal
                cal_helped.append({
                    "slot": f"{target}|{view}",
                    "loa_unc": loa_unc, "loa_cal": loa_cal,
                    "loa_delta": loa_delta,
                    "ccc_unc": ccc_unc, "ccc_cal": ccc_cal,
                    "ccc_delta": ccc_delta,
                })
            else:
                chosen = "uncalibrated"
                chosen_stats = stats_unc
                cal_hurt.append({
                    "slot": f"{target}|{view}",
                    "loa_unc": loa_unc, "loa_cal": loa_cal,
                    "loa_delta": loa_delta,
                    "ccc_unc": ccc_unc, "ccc_cal": ccc_cal,
                    "ccc_delta": ccc_delta,
                    "fallback_reason": (
                        "ccc_regressed" if ccc_regressed
                        else ("loa_did_not_tighten" if not loa_tightened
                              else "unknown")
                    ),
                })
                if ccc_regressed:
                    ccc_regressed_slots.append({
                        "slot": f"{target}|{view}",
                        "ccc_unc": ccc_unc, "ccc_cal": ccc_cal,
                        "ccc_delta": ccc_delta,
                    })

            new_tier = chosen_stats.get("classification", "Poor") or "Poor"
            tier_counter[new_tier] = tier_counter.get(new_tier, 0) + 1

            if DUMP_ACCUMULATOR is not None:
                _pred_dump = pred_cal if chosen == "calibrated" else pred_unc
                DUMP_ACCUMULATOR.add_slot(
                    target=target, view=view,
                    approach=f"{approach}|{chosen}",
                    pred=_pred_dump, obs=y_out, subjects=subj_out,
                )

            # The "baseline" v20 stats we should match when we fall back.
            v20_baseline_stats = v20_per_slot_by_key.get((target, view), {})
            v20_baseline_ccc = v20_baseline_stats.get("ccc_lin")
            v20_baseline_loa = v20_baseline_stats.get("loa_half_width_deg")

            slot_record = {
                "target": target,
                "view": view,
                "approach": approach,
                "base_reader": "v20",
                "n_subjects": chosen_stats.get("n_subjects"),
                "n_trials": chosen_stats.get("n_trials"),
                "pearson_r": chosen_stats.get("pearson_r"),
                "ccc_lin": chosen_stats.get("ccc_lin"),
                "loa_half_width_deg": chosen_stats.get("loa_half_width_deg"),
                "mean_bias_deg": chosen_stats.get("mean_bias_deg"),
                "sd_bias_deg": chosen_stats.get("sd_bias_deg"),
                "mae_deg": chosen_stats.get("mae_deg"),
                "rmse_deg": chosen_stats.get("rmse_deg"),
                "classification": new_tier,
                "calibration": chosen,
                "uncalibrated_ccc": stats_unc.get("ccc_lin"),
                "uncalibrated_loa_half": stats_unc.get("loa_half_width_deg"),
                "uncalibrated_classification": stats_unc.get(
                    "classification"
                ),
                "calibrated_ccc": stats_cal.get("ccc_lin"),
                "calibrated_loa_half": stats_cal.get("loa_half_width_deg"),
                "calibrated_classification": stats_cal.get("classification"),
                "loa_delta_cal_minus_unc": loa_delta,
                "ccc_delta_cal_minus_unc": ccc_delta,
                "calibration_summary": cal_summary,
                "v20_baseline_ccc": v20_baseline_ccc,
                "v20_baseline_loa": v20_baseline_loa,
            }
            per_slot_validity.append(slot_record)

            # Build deploy entry: start from v20 entry, layer cal block on top.
            v20_entry = (
                v20_deploy.get("models", {})
                .get(target, {}).get(view) or entry
            )
            deploy_entry = dict(v20_entry)
            deploy_entry[f"{V41_TAG}_block"] = {
                "l3_type": "ridge_with_calibration",
                "base_reader": "v20",
                "calibration_per_fold": fold_recs,
                "calibration_summary": cal_summary,
                "chosen_at_deploy": chosen,
            }
            deploy_entry[f"{V41_TAG}_loso_stats"] = {
                k: chosen_stats.get(k) for k in [
                    "n_subjects", "n_trials", "mean_bias_deg",
                    "sd_bias_deg", "loa_upper_deg", "loa_lower_deg",
                    "loa_half_width_deg", "pearson_r", "ccc_lin",
                    "mae_deg", "rmse_deg", "classification",
                ]
            }
            deploy_entry[f"{V41_TAG}_base_reader"] = "v20"
            deploy_entry[f"{V41_TAG}_calibration_chosen"] = chosen
            deploy_models.setdefault(target, {})[view] = deploy_entry

            log(
                f"  uncal CCC={_ccc_finite(ccc_unc):+.3f} "
                f"LoA={_loa_finite(loa_unc):.2f} | "
                f"cal CCC={_ccc_finite(ccc_cal):+.3f} "
                f"LoA={_loa_finite(loa_cal):.2f} "
                f"(dLoA={loa_delta:+.3f}, dCCC={ccc_delta:+.3f}) "
                f"-> {chosen} tier={new_tier} "
                f"({time.time()-t0:.1f}s)"
            )

        log(
            f"=== {V41_TAG} complete: tier counts {tier_counter} ==="
        )
        log(
            f"   calibration helped: {len(cal_helped)} slots, hurt/no-op: "
            f"{len(cal_hurt)} slots"
        )

        # Write per-slot validity.
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        per_slot_obj = {
            "version": V41_TAG,
            "produced_by": (
                "harness.calibrate_v20 (Agent QQ -- nested-LOSO "
                "calibration on v20 reader)"
            ),
            "base_reader": "v20",
            "l2_kind": "gg2_rom_aware",
            "label": (
                "v20 (GG2 ROM-aware OpenCap-only learned L2) + ridge L3 "
                "+ nested-LOSO calibration"
            ),
            "loso_discipline": (
                "L2 (GG2 ROM-aware): all 9 OpenCap subjects, NO LOSO at "
                "L2 (upper bound for OpenCap subjects). L3: outer LOSO "
                "at subject level. Calibration fit: inner LOSO over "
                "training-only subjects; outer held-out subject NEVER "
                "seen during calibration fitting."
            ),
            "fallback_rule": {
                "tighten_loa_required": True,
                "ccc_regression_threshold": CCC_FALLBACK_DELTA,
                "description": (
                    "Choose calibrated only if LoA tightens (delta < 0) "
                    f"AND CCC does not regress by more than "
                    f"{CCC_FALLBACK_DELTA:.2f}. Otherwise fall back to "
                    "uncalibrated."
                ),
            },
            "tier_counts": tier_counter,
            "calibration_helped": cal_helped,
            "calibration_hurt": cal_hurt,
            "ccc_regression_fallbacks": ccc_regressed_slots,
            "slots": per_slot_validity,
        }
        V41_PER_SLOT_PATH.write_text(
            json.dumps(per_slot_obj, indent=2, default=str)
        )
        log(f"wrote {V41_PER_SLOT_PATH}")

        deploy_obj = {
            "version": V41_TAG,
            "produced_by": (
                "harness.calibrate_v20 (Agent QQ -- nested-LOSO "
                "calibration on v20)"
            ),
            "produced_date": time.strftime("%Y-%m-%d"),
            "base_reader": "v20",
            "description": (
                "v20 (GG2 ROM-aware OpenCap-only learned L2) + ridge L3 "
                "+ per-slot linear calibration "
                "(pred_cal = a * pred + b) fit on nested-LOSO "
                "pseudo-residuals and applied to outer LOSO predictions. "
                "Per-slot fallback to uncalibrated ridge if calibration "
                "fails to tighten LoA or regresses CCC by more than "
                f"{CCC_FALLBACK_DELTA:.2f} (no-regression guarantee). "
                "Single camera, single DWPose stream at inference."
            ),
            "loso_discipline": (
                "L2 all-data (no LOSO at L2); L3 outer LOSO; "
                "calibration inner LOSO over training-only subjects"
            ),
            "v17_base_path": str(V17_PATH),
            "v20_base_path": str(V20_PATH),
            "approaches": v17.get("approaches"),
            "training_dataset": v17.get("training_dataset"),
            "models": deploy_models,
            "calibration_fix": v17.get("calibration_fix"),
            "selective_adoption": v17.get("selective_adoption"),
        }
        V41_DEPLOY_PATH.write_text(
            json.dumps(deploy_obj, indent=2, default=str)
        )
        log(f"wrote {V41_DEPLOY_PATH}")

    finally:
        _unpatch_motion_data(original_fn)
        log("unpatched motion data")

    return per_slot_obj


# -------------------------------------------------------------------------
# CLI.
# -------------------------------------------------------------------------


def main() -> None:
    global DUMP_ACCUMULATOR
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-predictions", action="store_true",
        help="If set, write per-subject (pred, gt) dumps under "
             "data/per_slot_predictions/per_slot_predictions_v41.json.",
    )
    args = parser.parse_args()
    log("=== Agent QQ (calibrate v20) START ===")
    if args.dump_predictions:
        DUMP_ACCUMULATOR = DumpAccumulator(
            reader="v41",
            loso_discipline=(
                "Nested LOSO with residual calibration (QQ: base=v20)"
            ),
        )
        log("[SS dump] DUMP_ACCUMULATOR set for reader 'v41'")
    try:
        run()
    except Exception as e:
        log(f"FATAL: {e!r}")
        traceback.print_exc()
        raise
    if DUMP_ACCUMULATOR is not None:
        out_path = DUMP_ACCUMULATOR.write()
        log(f"[SS dump] wrote per-slot predictions to {out_path}")
    log("=== Agent QQ (calibrate v20) DONE ===")


if __name__ == "__main__":
    main()
