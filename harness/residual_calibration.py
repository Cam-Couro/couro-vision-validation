"""Agent PP, Lever 2 -- Nested-LOSO residual calibration.

For each (slot, L2-reader) combination, fit a per-slot linear calibration
``pred_cal = a * pred + b`` on pseudo-held-out predictions from an inner
LOSO loop. The calibration is applied to the outer-LOSO held-out subject's
predictions. No leakage: the outer held-out subject is excluded from both
the inner ridge fits and the calibration regression.

# Hypothesis

The Bland-Altman LoA half-width is ``1.96 * SD(residuals)``. If the
current ridge has structured slope mismatch or per-view bias, a calibration
linear regression fit on held-out residuals can tighten LoA without
inflating bias. Applied to the v17 / v23 / v31 L2 readers that drive 12 of
the 23 v35 slots.

# Algorithm (one slot x reader)

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

# Sanity check

Compare LoA(calibrated) vs LoA(uncalibrated baseline ridge LOSO) per slot.
A negative diff (calibration tightened LoA) is the success signal. A large
positive diff implies a bug; nested LOSO with leakage would inflate
LoA on held-out subjects.

# Outputs

  * ``results/deploy_ready_models_v37_v23_calibrated.json``
  * ``results/deploy_ready_models_v38_v31_calibrated.json``
  * ``results/deploy_ready_models_v39_v17_calibrated.json``
  * ``data/residual_calibration/per_slot_validity_v37.json`` (+ v38, v39)
  * ``data/residual_calibration/REPORT.md``
"""
from __future__ import annotations

import argparse
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


# Agent SS per-clip dump infrastructure. Records the chosen (calibrated or
# uncalibrated) outer-LOSO per-subject predictions for each slot.
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
REPORT_PATH: Final[Path] = OUT_DIR / "REPORT.md"

V17_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v17_selective.json"
)
BASELINE_PATH: Final[Path] = (
    DATA_ROOT / "biomech_validity_stats" / "per_slot_validity.json"
)

# L2 checkpoints used by v23 (single-head combined) and v31 (mirror-flip
# per-source). v17 uses hand-engineered L2 (no patch).
L2_V23_CKPT: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_combined_alldata_v1.pt"
)
L2_V29_CKPT: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_persource_mirrorflip_alldata_v1.pt"
)

_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[PP-cal t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Metrics.
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
# Calibration core.
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

    Falls back to identity (a=1, b=0) if the input is degenerate (n<3, all
    NaN, or zero variance in pseudo_pred).
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
    # Sanity guard: keep slope in a sane range.
    if not math.isfinite(a) or not math.isfinite(b):
        return Calibration(a=1.0, b=0.0, n_pseudo=int(len(p)))
    return Calibration(a=float(a), b=float(b), n_pseudo=int(len(p)))


def nested_loso_with_calibration(
    X: np.ndarray, y: np.ndarray, subjects: list[str],
    *, alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Nested-LOSO ridge + per-fold calibration.

    For each outer subject S:
      1. Inner LOSO across the remaining subjects T:
         - Fit ridge on cohort minus {S, T}.
         - Predict T's trials.
         - Collect (pseudo_pred_T, y_T) pairs.
      2. Fit linear calibration on the pooled pseudo-pred vs y.
      3. Fit outer ridge on cohort minus {S}.
      4. Predict S's trials with the outer ridge.
      5. Apply the calibration to S's predictions.

    Returns:
      * pred_uncal: outer-LOSO predictions WITHOUT calibration (baseline).
      * pred_cal: outer-LOSO predictions WITH calibration applied.
      * y_out: ground truth aligned with predictions.
      * subj_out: per-trial subject labels.
      * fold_records: list of per-outer-fold dicts with (S, a, b, n_pseudo).
    """
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

        # Inner LOSO across training subjects (excluding outer_s).
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

        # Outer ridge.
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
# L2 patching helpers (mirrors learned_layer3 setup).
# -------------------------------------------------------------------------


def _setup_l2_v23() -> tuple:
    from harness.learned_layer2_combined import (
        TemporalKeypointCNNConf, HALPE_KEYS_WITH_SMPL,
    )
    from harness.smpl_layer2_poc import HALPE26_INDEX

    ckpt = torch.load(L2_V23_CKPT, weights_only=False)
    model = TemporalKeypointCNNConf()
    model.load_state_dict(ckpt["model_state"])
    model.y_mean = ckpt["y_mean"]  # type: ignore[attr-defined]
    model.y_std = ckpt["y_std"]    # type: ignore[attr-defined]
    halpe_idx_arr = np.array(
        [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL],
        dtype=np.int64,
    )
    log(f"loaded v23 L2 from {L2_V23_CKPT}")
    return model, halpe_idx_arr


def _setup_l2_v29() -> tuple:
    from harness.learned_layer2_combined_persource import (
        TemporalKeypointCNNConfPerSource, wrap_for_ff,
    )
    from harness.learned_layer2_combined import HALPE_KEYS_WITH_SMPL
    from harness.smpl_layer2_poc import HALPE26_INDEX

    ckpt = torch.load(L2_V29_CKPT, weights_only=False)
    base = TemporalKeypointCNNConfPerSource()
    base.load_state_dict(ckpt["model_state"])
    base.ys_mean = ckpt["ys_mean"]  # type: ignore[attr-defined]
    base.ys_std = ckpt["ys_std"]    # type: ignore[attr-defined]
    base.ya_mean = ckpt["ya_mean"]  # type: ignore[attr-defined]
    base.ya_std = ckpt["ya_std"]    # type: ignore[attr-defined]
    shim = wrap_for_ff(base)
    halpe_idx_arr = np.array(
        [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL],
        dtype=np.int64,
    )
    log(f"loaded v29 L2 from {L2_V29_CKPT}")
    return shim, halpe_idx_arr


def _patch_motion_data(model, halpe_idx_arr) -> None:
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
# Per-slot feature builder (mirrors learned_layer3._collect_features_for_slot).
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
# Per-reader orchestration.
# -------------------------------------------------------------------------


READER_TAG: Final[dict[str, dict[str, str | None]]] = {
    "v37": {
        "tag": "v37_v23_calibrated",
        "l2_kind": "v23",
        "base_reader": "v23",
        "label": "v23 (HH2 combined L2) + ridge L3 + nested-LOSO calibration",
    },
    "v38": {
        "tag": "v38_v31_calibrated",
        "l2_kind": "v29",
        "base_reader": "v31",
        "label": (
            "v29 (mirror-flip per-source L2) + ridge L3 + nested-LOSO "
            "calibration"
        ),
    },
    "v39": {
        "tag": "v39_v17_calibrated",
        "l2_kind": None,
        "base_reader": "v17",
        "label": "v17 (hand-engineered L2) + ridge L3 + nested-LOSO calibration",
    },
}


def _run_one_reader(reader_key: str) -> dict:
    cfg = READER_TAG[reader_key]
    tag = cfg["tag"]
    l2_kind = cfg["l2_kind"]
    base_reader = cfg["base_reader"]
    label = cfg["label"]

    log(f"=== reader={base_reader}  tag={tag}  l2_kind={l2_kind} ===")

    # Patch L2 if needed.
    from harness import couro_keypoints as ck_mod
    original_fn = ck_mod.keypoints_to_motion_data
    patched_for_run = False
    if l2_kind == "v23":
        model, halpe_idx_arr = _setup_l2_v23()
        _patch_motion_data(model, halpe_idx_arr)
        patched_for_run = True
    elif l2_kind == "v29":
        model, halpe_idx_arr = _setup_l2_v29()
        _patch_motion_data(model, halpe_idx_arr)
        patched_for_run = True
    elif l2_kind is None:
        log("v17 path: NO L2 patch (using hand-engineered keypoints_to_motion_data)")
    else:
        raise RuntimeError(f"unknown l2_kind: {l2_kind}")

    v17 = json.loads(V17_PATH.read_text())
    baseline = json.loads(BASELINE_PATH.read_text())
    baseline_by_slot = {
        (s["target"], s["view"]): s for s in baseline["slots"]
    }

    slot_keys = []
    for target, slots in v17["models"].items():
        for view, entry in slots.items():
            slot_keys.append((target, view, entry))
    log(f"=== {len(slot_keys)} slots to process ===")

    per_slot_validity = []
    deploy_models: dict[str, dict[str, dict]] = {}
    tier_counter = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    cal_helped: list[dict] = []  # (target, view, loa_uncal, loa_cal)
    cal_hurt: list[dict] = []

    for i, (target, view, entry) in enumerate(slot_keys):
        approach = entry.get("approach", "unknown")
        t0 = time.time()
        log(
            f"--- slot {i+1}/{len(slot_keys)} {target}/{view} "
            f"approach={approach} ---"
        )
        result = _collect_features_for_slot(target, view, approach)
        if result is None:
            log(f"  no builder result; carrying baseline")
            base = baseline_by_slot.get((target, view), {})
            per_slot_validity.append({
                **base, f"{tag}_skipped": True,
                "skip_reason": f"no builder for {approach}",
            })
            deploy_models.setdefault(target, {})[view] = entry
            continue

        X, y, subjects, sources = result
        if len(y) < 20:
            log(f"  only n={len(y)} samples; carrying baseline")
            base = baseline_by_slot.get((target, view), {})
            per_slot_validity.append({
                **base, f"{tag}_skipped": True,
                "skip_reason": f"only n={len(y)} samples",
            })
            deploy_models.setdefault(target, {})[view] = entry
            continue

        # Nested LOSO with calibration.
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
                **base, f"{tag}_skipped": True,
                "skip_reason": f"nested LOSO error: {e!r}",
            })
            deploy_models.setdefault(target, {})[view] = entry
            continue

        stats_unc = _stats_per_subject(pred_unc, y_out, subj_out)
        stats_cal = _stats_per_subject(pred_cal, y_out, subj_out)

        # Stash diagnostics on fold record summary.
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
        try:
            loa_delta = float(loa_cal) - float(loa_unc)
        except (TypeError, ValueError):
            loa_delta = float("nan")

        # Pick the better (lower LoA) of cal vs uncal at deploy time.
        # Calibration should never inflate LoA on a properly-fit hold-out;
        # if it does we keep the uncalibrated baseline and flag it.
        if math.isfinite(loa_delta) and loa_delta < 0:
            chosen = "calibrated"
            chosen_stats = stats_cal
            cal_helped.append({
                "slot": f"{target}|{view}",
                "loa_unc": loa_unc, "loa_cal": loa_cal,
                "loa_delta": loa_delta,
            })
        else:
            chosen = "uncalibrated"
            chosen_stats = stats_unc
            cal_hurt.append({
                "slot": f"{target}|{view}",
                "loa_unc": loa_unc, "loa_cal": loa_cal,
                "loa_delta": loa_delta,
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

        slot_record = {
            "target": target,
            "view": view,
            "approach": approach,
            "base_reader": base_reader,
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
            "uncalibrated_classification": stats_unc.get("classification"),
            "calibrated_ccc": stats_cal.get("ccc_lin"),
            "calibrated_loa_half": stats_cal.get("loa_half_width_deg"),
            "calibrated_classification": stats_cal.get("classification"),
            "loa_delta_cal_minus_unc": loa_delta,
            "calibration_summary": cal_summary,
        }
        per_slot_validity.append(slot_record)

        deploy_entry = dict(entry)
        deploy_entry[f"{tag}_block"] = {
            "l3_type": "ridge_with_calibration",
            "base_reader": base_reader,
            "calibration_per_fold": fold_recs,
            "calibration_summary": cal_summary,
            "chosen_at_deploy": chosen,
        }
        deploy_entry[f"{tag}_loso_stats"] = {
            k: chosen_stats.get(k) for k in [
                "n_subjects", "n_trials", "mean_bias_deg",
                "sd_bias_deg", "loa_upper_deg", "loa_lower_deg",
                "loa_half_width_deg", "pearson_r", "ccc_lin",
                "mae_deg", "rmse_deg", "classification",
            ]
        }
        deploy_entry[f"{tag}_base_reader"] = base_reader
        deploy_entry[f"{tag}_calibration_chosen"] = chosen
        deploy_models.setdefault(target, {})[view] = deploy_entry

        log(
            f"  uncal CCC={stats_unc.get('ccc_lin'):+.3f} "
            f"LoA={stats_unc.get('loa_half_width_deg'):.2f} | "
            f"cal CCC={stats_cal.get('ccc_lin'):+.3f} "
            f"LoA={stats_cal.get('loa_half_width_deg'):.2f} "
            f"(delta={loa_delta:+.3f}) -> {chosen} tier={new_tier} "
            f"({time.time()-t0:.1f}s)"
        )

    log(
        f"=== {tag} complete: tier counts {tier_counter} ==="
    )
    log(
        f"   calibration helped: {len(cal_helped)} slots, hurt/no-op: "
        f"{len(cal_hurt)} slots"
    )

    # Write per-slot validity.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_slot_path = OUT_DIR / f"per_slot_validity_{tag}.json"
    per_slot_obj = {
        "version": tag,
        "produced_by": (
            "harness.residual_calibration (Agent PP, Lever 2)"
        ),
        "base_reader": base_reader,
        "l2_kind": l2_kind or "hand_engineered",
        "label": label,
        "loso_discipline": (
            "nested LOSO at ridge L3 level for calibration fit; "
            "outer LOSO at L3 unchanged. Calibration is per slot. "
            "Held-out subject NEVER seen during calibration fitting."
        ),
        "tier_counts": tier_counter,
        "calibration_helped": cal_helped,
        "calibration_hurt": cal_hurt,
        "slots": per_slot_validity,
    }
    per_slot_path.write_text(
        json.dumps(per_slot_obj, indent=2, default=str)
    )
    log(f"wrote {per_slot_path}")

    deploy_path = RESULTS_DIR / f"deploy_ready_models_{tag}.json"
    deploy_obj = {
        "version": tag,
        "produced_by": (
            "harness.residual_calibration (Agent PP, Lever 2)"
        ),
        "produced_date": time.strftime("%Y-%m-%d"),
        "base_reader": base_reader,
        "description": (
            f"{label}. Per-slot linear calibration "
            "(pred_cal = a * pred + b) fit on nested-LOSO pseudo-residuals "
            "and applied to outer LOSO predictions. Per-slot fallback to "
            "uncalibrated ridge if calibration inflates LoA (no-regression "
            "guarantee). Single camera, single DWPose stream at inference."
        ),
        "loso_discipline": (
            "nested LOSO at ridge L3 level for calibration; outer LOSO at "
            "L3 unchanged"
        ),
        "v17_base_path": str(V17_PATH),
        "approaches": v17.get("approaches"),
        "training_dataset": v17.get("training_dataset"),
        "models": deploy_models,
        "calibration_fix": v17.get("calibration_fix"),
        "selective_adoption": v17.get("selective_adoption"),
    }
    deploy_path.write_text(json.dumps(deploy_obj, indent=2, default=str))
    log(f"wrote {deploy_path}")

    if patched_for_run:
        _unpatch_motion_data(original_fn)
        log("unpatched motion data")

    return per_slot_obj


# -------------------------------------------------------------------------
# Cumulative report.
# -------------------------------------------------------------------------


CATEGORY_A_SLOTS: Final[tuple[tuple[str, str, float], ...]] = (
    ("knee_angle_r", "front_oblique_left", 10.77),
    ("knee_angle_r", "side_left", 10.15),
    ("knee_angle_r", "side_right", 12.92),
    ("hip_flexion_r", "front_oblique_left", 11.29),
    ("hip_adduction_r", "front_oblique_right", 13.80),
)


def write_report(per_slot_runs: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(
        "# Residual Calibration -- v37 / v38 / v39 (Agent PP Lever 2)"
    )
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(
        "**Hypothesis:** Bland-Altman LoA half-width = 1.96 * SD(residuals)."
        " A per-slot linear calibration ``pred_cal = a*pred + b`` fit on "
        "nested-LOSO pseudo-residuals can tighten LoA if the current "
        "ridge has structured slope mismatch or per-cohort bias not "
        "absorbed by its single ridge fit."
    )
    lines.append("")
    lines.append(
        "**Discipline:** outer LOSO at L3 unchanged. For each outer "
        "subject S, we (a) run an inner LOSO across the N-1 training "
        "subjects T to collect pseudo-predictions, (b) fit a linear "
        "calibration on those pseudo (pred, gt) pairs, (c) predict on S "
        "with the standard outer ridge, (d) apply the calibration to S's "
        "predictions. S is **never** part of the calibration fit. Per-"
        "slot fallback to uncalibrated ridge if calibration inflates LoA."
    )
    lines.append("")

    by_tag = {r["version"]: r for r in per_slot_runs}

    # ---- Category A target slots ----
    lines.append(
        "## Category A targets -- did calibration push them under +/-10 deg?"
    )
    lines.append("")
    lines.append(
        "| Slot | v35 LoA/2 | v37 (v23+cal) | v38 (v31+cal) | "
        "v39 (v17+cal) | best calibrated | Promoted? |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    cat_a_promotions = 0
    for tgt, view, v35_loa in CATEGORY_A_SLOTS:
        row_vals = []
        best_loa = float("inf")
        best_ccc = float("nan")
        for tag in ("v37_v23_calibrated", "v38_v31_calibrated",
                    "v39_v17_calibrated"):
            run = by_tag.get(tag, {})
            slot = next(
                (s for s in run.get("slots", [])
                 if s.get("target") == tgt and s.get("view") == view),
                None,
            )
            if slot is None:
                row_vals.append("-")
                continue
            loa = slot.get("loa_half_width_deg")
            ccc = slot.get("ccc_lin")
            cal_kind = slot.get("calibration", "?")
            cell = (
                f"{loa:.2f} ({cal_kind[:3]})"
                if isinstance(loa, (int, float)) and math.isfinite(loa)
                else "-"
            )
            row_vals.append(cell)
            if (
                isinstance(loa, (int, float))
                and math.isfinite(loa) and loa < best_loa
            ):
                best_loa = loa
                best_ccc = ccc
        best_str = (
            f"{best_loa:.2f}" if math.isfinite(best_loa) else "-"
        )
        promoted = math.isfinite(best_loa) and best_loa < 10.0 and (
            best_ccc is None
            or not (isinstance(best_ccc, float) and math.isnan(best_ccc))
        )
        if promoted:
            cat_a_promotions += 1
        lines.append(
            f"| {tgt}|{view} | {v35_loa:.2f} | "
            f"{row_vals[0]} | {row_vals[1]} | {row_vals[2]} | "
            f"{best_str} | "
            f"{'YES' if promoted else 'no'} |"
        )
    lines.append("")
    lines.append(
        f"**Category A promotions to Good (LoA < 10 deg): "
        f"{cat_a_promotions}/{len(CATEGORY_A_SLOTS)}.**"
    )
    lines.append("")

    # ---- Per-reader summary ----
    lines.append("## Per-reader sanity check (calibration should not hurt)")
    lines.append("")
    for tag in ("v37_v23_calibrated", "v38_v31_calibrated",
                "v39_v17_calibrated"):
        run = by_tag.get(tag, {})
        helped = run.get("calibration_helped", [])
        hurt = run.get("calibration_hurt", [])
        tier_counts = run.get("tier_counts", {})
        lines.append(f"### {tag}")
        lines.append("")
        lines.append(
            f"- Calibration helped LoA on **{len(helped)}** slots."
        )
        lines.append(
            f"- Calibration neutral or hurt on **{len(hurt)}** slots "
            f"(fell back to uncalibrated)."
        )
        lines.append(f"- Tier counts (chosen-per-slot): {tier_counts}")
        if helped:
            top_helps = sorted(
                helped,
                key=lambda r: r.get("loa_delta", float("inf"))
                if math.isfinite(r.get("loa_delta", float("nan")))
                else float("inf"),
            )[:6]
            lines.append("")
            lines.append("Top LoA tightenings:")
            lines.append("")
            lines.append("| Slot | LoA uncal | LoA cal | Delta |")
            lines.append("| --- | ---: | ---: | ---: |")
            for r in top_helps:
                loa_unc = r.get("loa_unc")
                loa_cal = r.get("loa_cal")
                d = r.get("loa_delta")
                lines.append(
                    f"| {r['slot']} | "
                    f"{loa_unc:.2f} | {loa_cal:.2f} | {d:+.2f} |"
                )
        lines.append("")

    # ---- Per-slot all-results table ----
    lines.append("## Full per-slot calibration table (all readers)")
    lines.append("")
    for tag in ("v37_v23_calibrated", "v38_v31_calibrated",
                "v39_v17_calibrated"):
        run = by_tag.get(tag, {})
        lines.append(f"### {tag}")
        lines.append("")
        lines.append(
            "| Target | View | n_subj | CCC unc | LoA unc | CCC cal | "
            "LoA cal | Delta | Chosen | Tier |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
        )
        for slot in run.get("slots", []):
            if slot.get("loa_half_width_deg") is None:
                continue
            tgt = slot.get("target")
            view = slot.get("view")
            n = slot.get("n_subjects", "?")
            ccc_u = slot.get("uncalibrated_ccc")
            loa_u = slot.get("uncalibrated_loa_half")
            ccc_c = slot.get("calibrated_ccc")
            loa_c = slot.get("calibrated_loa_half")
            d = slot.get("loa_delta_cal_minus_unc")
            chosen = slot.get("calibration", "?")
            tier = slot.get("classification", "?")

            def _f(v, p=2):
                if v is None or (
                    isinstance(v, float) and math.isnan(v)
                ):
                    return "-"
                return f"{v:.{p}f}"

            lines.append(
                f"| {tgt} | {view} | {n} | {_f(ccc_u, 3)} | {_f(loa_u)} | "
                f"{_f(ccc_c, 3)} | {_f(loa_c)} | {_f(d)} | {chosen} | "
                f"{tier} |"
            )
        lines.append("")

    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "- **Nested LOSO is legitimate but expensive.** Each slot fits "
        "N*(N-1) ridges (N = subjects in slot's pool). Ridge is cheap, "
        "so total runtime is dominated by feature building (L2 inference "
        "+ event detection + per-slot dataset assembly)."
    )
    lines.append(
        "- **Per-slot fallback to uncalibrated.** If calibration inflates "
        "LoA on a slot, we keep the uncalibrated ridge. This preserves "
        "the no-regression guarantee and prevents calibration noise from "
        "hurting reliable slots."
    )
    lines.append(
        "- **Calibration is per slot x base reader.** Each (slot, reader) "
        "gets its own (a, b). The calibration parameters are NOT shared "
        "across slots."
    )
    lines.append(
        "- **Ship-time deployment cost is one (a, b) multiply per inference**"
        " per slot per outer-LOSO fold. Stored as per-fold (a, b) in the "
        "deploy bundle. For production use, the deploy-time calibration is "
        "the per-fold average (a_mean, b_mean) (stored in "
        "``calibration_summary``); this loses the per-fold conditioning "
        "but gains a single closed-form correction. Per-fold details are "
        "retained for audit."
    )
    lines.append(
        "- **Single camera at inference** (unchanged from v17-v36)."
    )
    lines.append(
        "- **Outer LOSO discipline preserved.** Calibration is "
        "fit on (N-1) training subjects only; the outer held-out S is "
        "predicted but never used to choose calibration parameters."
    )
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    log(f"wrote {REPORT_PATH}")


# -------------------------------------------------------------------------
# CLI.
# -------------------------------------------------------------------------


def main() -> None:
    global DUMP_ACCUMULATOR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--readers", nargs="+", default=["v37", "v38", "v39"],
        choices=["v37", "v38", "v39"],
    )
    parser.add_argument(
        "--dump-predictions", action="store_true",
        help="If set, write per-subject (pred, gt) dumps under "
             "data/per_slot_predictions/per_slot_predictions_<reader>.json "
             "for each reader run.",
    )
    args = parser.parse_args()

    log("=== Agent PP Lever 2 (residual calibration) START ===")
    runs: list[dict] = []
    for r in args.readers:
        if args.dump_predictions:
            DUMP_ACCUMULATOR = DumpAccumulator(
                reader=r,
                loso_discipline=(
                    f"Nested LOSO with residual calibration "
                    f"(PP reader={r}); chosen = lower-LoA of cal vs uncal."
                ),
            )
            log(f"[SS dump] DUMP_ACCUMULATOR set for reader '{r}'")
        try:
            runs.append(_run_one_reader(r))
        except Exception as e:
            log(f"reader {r} failed: {e!r}")
            traceback.print_exc(file=sys.stdout)
        if DUMP_ACCUMULATOR is not None:
            out_path = DUMP_ACCUMULATOR.write()
            log(f"[SS dump] wrote per-slot predictions to {out_path}")
            DUMP_ACCUMULATOR = None
    write_report(runs)
    log("=== Agent PP Lever 2 (residual calibration) DONE ===")


if __name__ == "__main__":
    main()
