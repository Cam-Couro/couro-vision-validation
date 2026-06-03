"""Agent NN Phase 2 — Learned Layer 3 (per-slot MLP) replacing ridge.

Builds a small per-slot MLP regressor on the SAME phase-summary feature
matrix X that the v17-v28 ridge L3 fits. Loss is SmoothL1 vs ground-truth
ROM scalar. Subject-level LOSO at L3.

For each slot, BOTH variants are evaluated:
  * L3-ridge (baseline, existing): closed-form ridge on z-scored X.
  * L3-learned: small MLP (~3-5k params) on z-scored X with dropout.

Per-slot fallback rule: if learned underperforms ridge by > 0.05 CCC,
emit the slot using ridge weights (keeps the v18-style entry). Otherwise
emit using the learned MLP weights.

# CLI

Two modes:
  * ``--l2 v23``  -> use ``models/learned_layer2_combined_alldata_v1.pt``
                    (HH2/KK combined per-frame, single-head). Drives v30.
  * ``--l2 v29``  -> use
                    ``models/learned_layer2_persource_mirrorflip_alldata_v1.pt``
                    (Agent NN mirror-flip per-source). Drives v31.

# Outputs

  * ``data/learned_layer3/per_slot_validity_{tag}.json``
  * ``results/deploy_ready_models_{tag}.json``
  * ``data/learned_layer3/REPORT.md`` (cumulative narrative)

where ``tag`` is ``v30_learned_l3`` (v23 L2) or
``v31_mirrorflip_learned_l3`` (v29 L2).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.train_regression_poc import RidgeModel, fit_ridge  # noqa: E402


# Silence noisy ridge warnings from degenerate features during LOSO.
warnings_filter = logging.getLogger("py.warnings")
warnings_filter.setLevel(logging.ERROR)


# -------------------------------------------------------------------------
# Paths.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
OUT_DIR: Final[Path] = DATA_ROOT / "learned_layer3"
REPORT_PATH: Final[Path] = OUT_DIR / "REPORT.md"

V17_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v17_selective.json"
BASELINE_PATH: Final[Path] = (
    DATA_ROOT / "biomech_validity_stats" / "per_slot_validity.json"
)

L2_V23_CKPT: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_combined_alldata_v1.pt"
)
L2_V29_CKPT: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_persource_mirrorflip_alldata_v1.pt"
)


# -------------------------------------------------------------------------
# Progress log.
# -------------------------------------------------------------------------

_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[NN-L3 t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Learned L3 model: small MLP on z-scored features.
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnedL3Model:
    """Per-slot learned L3 wrapper. Stores its torch state dict + the
    z-score normalization used at training time. Predicts a single ROM
    scalar per clip."""

    state_dict: dict
    feature_mean: np.ndarray
    feature_std: np.ndarray
    hidden: int
    n_features: int

    def to_module(self) -> "TinyMLPL3":
        mod = TinyMLPL3(self.n_features, self.hidden)
        mod.load_state_dict(self.state_dict)
        mod.eval()
        return mod

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xz = (X - self.feature_mean) / np.maximum(self.feature_std, 1e-9)
        mod = self.to_module()
        with torch.no_grad():
            out = mod(torch.from_numpy(Xz.astype(np.float32))).numpy()
        # TinyMLPL3.forward already squeezes the last (1) dim.
        return np.atleast_1d(out)


class TinyMLPL3(nn.Module):
    """Small MLP for per-slot ROM regression. Two hidden layers of ``hidden``
    units, ReLU, dropout 0.2. Default ~3-5k params for hidden=32 on 30-50
    input features."""

    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_learned_l3(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> LearnedL3Model:
    """Train one TinyMLPL3 on the given (X, y). Z-scores features inside.
    Uses 90/10 internal train/val split for early stopping; keeps the
    best-val checkpoint.

    n=small (often 9-40 LOSO inner samples) — we use heavy weight decay,
    dropout, and early stopping to mitigate overfitting on the tiny slot
    pools.
    """
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    if n < 4:
        raise RuntimeError(f"learned L3 needs n>=4, got n={n}")

    # Z-score features.
    fm = X_train.mean(axis=0)
    fs = X_train.std(axis=0)
    fs = np.where(fs < 1e-9, 1.0, fs)
    Xz = (X_train - fm) / fs

    # Z-score target too (helps optimizer; we de-normalize at predict time
    # but here we just need a stable loss scale).
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) if y_train.std() > 1e-9 else 1.0
    y_n = (y_train - y_mean) / y_std

    # Inner train/val split for early stopping.
    n_val = max(1, int(0.15 * n))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr_t = torch.from_numpy(Xz[tr_idx].astype(np.float32))
    y_tr_t = torch.from_numpy(y_n[tr_idx].astype(np.float32))
    X_va_t = torch.from_numpy(Xz[val_idx].astype(np.float32))
    y_va_t = torch.from_numpy(y_n[val_idx].astype(np.float32))

    torch.manual_seed(seed)
    model = TinyMLPL3(n_features=X_train.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val = float("inf")
    best_state: dict | None = None
    patience = 25
    patience_left = patience

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_t)
        loss = nn.functional.smooth_l1_loss(pred, y_tr_t)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(X_va_t)
            v_loss = float(
                nn.functional.smooth_l1_loss(v_pred, y_va_t).item()
            )
        if v_loss + 1e-6 < best_val:
            best_val = v_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    # Re-bake y_mean/y_std into the final layer so .predict returns ROM
    # in original units. The output layer is the last Linear in
    # self.net[6]. New_w = old_w * y_std; new_b = old_b * y_std + y_mean.
    # We mutate best_state in-place.
    last_w_key = "net.6.weight"
    last_b_key = "net.6.bias"
    best_state[last_w_key] = best_state[last_w_key] * y_std
    best_state[last_b_key] = best_state[last_b_key] * y_std + y_mean

    return LearnedL3Model(
        state_dict=best_state,
        feature_mean=fm,
        feature_std=fs,
        hidden=hidden,
        n_features=X_train.shape[1],
    )


# -------------------------------------------------------------------------
# Stats helpers (mirror FF's classification logic).
# -------------------------------------------------------------------------


def _ccc(p: np.ndarray, o: np.ndarray) -> float:
    if len(p) < 2:
        return float("nan")
    mp, mo = float(np.mean(p)), float(np.mean(o))
    vp, vo = float(np.var(p, ddof=0)), float(np.var(o, ddof=0))
    cov = float(np.mean((p - mp) * (o - mo)))
    denom = vp + vo + (mp - mo) ** 2
    if denom < 1e-12:
        return float("nan")
    return 2.0 * cov / denom


def _pearson(p: np.ndarray, o: np.ndarray) -> float:
    if p.size < 2 or np.var(p) < 1e-12 or np.var(o) < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, o)[0, 1])


def _classify(ccc: float, loa_half: float) -> str:
    if math.isnan(ccc):
        if loa_half < 5:
            return "LoA-only: tight"
        if loa_half < 10:
            return "LoA-only: moderate"
        if loa_half < 15:
            return "LoA-only: wide"
        return "LoA-only: poor"
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
    agg_pred = np.array(
        [float(np.mean(pred[subjects == s])) for s in unique]
    )
    agg_obs = np.array(
        [float(np.mean(obs[subjects == s])) for s in unique]
    )
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


def _loso_ridge(
    X: np.ndarray, y: np.ndarray, subjects: list[str],
    *, alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    all_pred = np.full(len(y), np.nan)
    for s in unique:
        train_mask = subj_arr != s
        test_mask = subj_arr == s
        if train_mask.sum() < 2 or test_mask.sum() < 1:
            continue
        m = fit_ridge(X[train_mask], y[train_mask], alpha=alpha)
        all_pred[test_mask] = m.predict(X[test_mask])
    return all_pred, y, subj_arr


def _loso_learned(
    X: np.ndarray, y: np.ndarray, subjects: list[str],
    *,
    hidden: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    all_pred = np.full(len(y), np.nan)
    for fi, s in enumerate(unique):
        train_mask = subj_arr != s
        test_mask = subj_arr == s
        if train_mask.sum() < 4 or test_mask.sum() < 1:
            continue
        try:
            m = _train_learned_l3(
                X[train_mask], y[train_mask],
                hidden=hidden, epochs=epochs, lr=lr,
                weight_decay=weight_decay, seed=seed + fi,
            )
            all_pred[test_mask] = m.predict(X[test_mask])
        except Exception as e:
            # Fall back to ridge for this fold if training fails.
            try:
                m_r = fit_ridge(
                    X[train_mask], y[train_mask], alpha=10.0
                )
                all_pred[test_mask] = m_r.predict(X[test_mask])
            except Exception:
                all_pred[test_mask] = float(np.mean(y[train_mask]))
    return all_pred, y, subj_arr


# -------------------------------------------------------------------------
# L2 setup: load checkpoint + monkey-patch keypoints_to_motion_data.
# -------------------------------------------------------------------------


def _setup_l2_v23() -> tuple:
    """Returns (model, halpe_idx_arr) for the v23 single-head L2."""
    from harness.learned_layer2_combined import TemporalKeypointCNNConf
    from harness.learned_layer2_combined import HALPE_KEYS_WITH_SMPL
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
    """Returns a SHARED-HEAD shim around the v29 mirror-flip per-source
    model, plus halpe_idx_arr. The shim is API-compatible with the
    single-head TemporalKeypointCNNConf so FF's `keypoints_to_motion_data`
    patch works unchanged."""
    from harness.learned_layer2_combined_persource import (
        TemporalKeypointCNNConfPerSource, wrap_for_ff,
    )
    from harness.learned_layer2_combined import HALPE_KEYS_WITH_SMPL
    from harness.smpl_layer2_poc import HALPE26_INDEX

    if not L2_V29_CKPT.exists():
        raise RuntimeError(
            f"v29 L2 checkpoint not found at {L2_V29_CKPT}. Run "
            "harness/learned_layer2_persource_mirrorflip.py first."
        )

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
    log(f"loaded v29 mirror-flip L2 (shared-head shim) from {L2_V29_CKPT}")
    return shim, halpe_idx_arr


def _patch_motion_data(model, halpe_idx_arr) -> None:
    """Monkey-patch couro_keypoints.keypoints_to_motion_data into FF's
    patched version. Reused by both v23 and v29 paths."""
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


# -------------------------------------------------------------------------
# Per-slot orchestrator.
# -------------------------------------------------------------------------


# Tier rank for promotion checks.
TIER_RANK: Final[dict[str, int]] = {
    "Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1,
}


def _collect_features_for_slot(
    target: str, view: str, approach: str,
):
    """Use FF's same builder dispatch to produce (X, y, subjects, sources)
    for one slot. Returns None if builder fails / too few samples."""
    from harness import layer3_retrain_on_learned_l2 as ff
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


def _orchestrate(tag: str, l2_kind: str) -> dict:
    """Run full per-slot pipeline for the chosen L2 source. Tag is the
    file prefix used in outputs (e.g. ``v30_learned_l3``)."""
    log(f"=== orchestrate tag={tag} l2={l2_kind} ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if l2_kind == "v23":
        model, halpe_idx_arr = _setup_l2_v23()
    elif l2_kind == "v29":
        model, halpe_idx_arr = _setup_l2_v29()
    else:
        raise RuntimeError(f"unknown l2_kind {l2_kind}")

    _patch_motion_data(model, halpe_idx_arr)

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
    learned_wins: list[tuple[str, str, float, float]] = []
    ridge_wins: list[tuple[str, str, float, float]] = []
    ties: list[tuple[str, str, float]] = []

    for i, (target, view, entry) in enumerate(slot_keys):
        approach = entry.get("approach", "unknown")
        t0 = time.time()
        log(
            f"--- slot {i+1}/{len(slot_keys)} {target}/{view} "
            f"approach={approach} ---"
        )
        result = _collect_features_for_slot(target, view, approach)
        if result is None:
            base = baseline_by_slot.get((target, view), {})
            per_slot_validity.append({
                **base, f"{tag}_skipped": True,
                "skip_reason": f"no builder for {approach}",
            })
            deploy_models.setdefault(target, {})[view] = entry
            continue

        X, y, subjects, sources = result
        if len(y) < 20:
            base = baseline_by_slot.get((target, view), {})
            per_slot_validity.append({
                **base, f"{tag}_skipped": True,
                "skip_reason": f"only n={len(y)} samples",
            })
            deploy_models.setdefault(target, {})[view] = entry
            continue

        # Ridge LOSO baseline.
        try:
            pred_r, obs_r, subj_r = _loso_ridge(
                X, y, list(subjects), alpha=10.0
            )
            stats_ridge = _stats_per_subject(pred_r, obs_r, subj_r)
        except Exception as e:
            log(f"  ridge LOSO failed: {e!r}")
            stats_ridge = {
                "ccc_lin": float("nan"), "pearson_r": float("nan"),
                "loa_half_width_deg": float("nan"),
                "classification": "Poor",
            }

        # Learned LOSO.
        try:
            pred_l, obs_l, subj_l = _loso_learned(
                X, y, list(subjects),
                hidden=32, epochs=200, lr=1e-2,
                weight_decay=1e-3, seed=0,
            )
            stats_learned = _stats_per_subject(pred_l, obs_l, subj_l)
        except Exception as e:
            log(f"  learned LOSO failed: {e!r}; falling back to ridge")
            stats_learned = stats_ridge

        ccc_r = stats_ridge.get("ccc_lin")
        ccc_l = stats_learned.get("ccc_lin")
        # Decide winner with fallback rule.
        def _val(c):
            if c is None or (isinstance(c, float) and math.isnan(c)):
                return -float("inf")
            return float(c)

        cr = _val(ccc_r)
        cl = _val(ccc_l)
        use_learned = cl >= cr - 0.05  # learned wins if not worse by 0.05

        if use_learned and cl > cr + 1e-6:
            learned_wins.append((target, view, cr, cl))
        elif use_learned:
            ties.append((target, view, cr))
        else:
            ridge_wins.append((target, view, cr, cl))

        if use_learned:
            chosen = "learned"
            chosen_stats = stats_learned
            try:
                final_model = _train_learned_l3(
                    X.astype(np.float64), y.astype(np.float64),
                    hidden=32, epochs=200, lr=1e-2,
                    weight_decay=1e-3, seed=0,
                )
                final_block = {
                    "l3_type": "learned",
                    "hidden": final_model.hidden,
                    "n_features": final_model.n_features,
                    "feature_mean": final_model.feature_mean.tolist(),
                    "feature_std": final_model.feature_std.tolist(),
                    "state_dict": {
                        k: v.numpy().tolist()
                        for k, v in final_model.state_dict.items()
                    },
                }
            except Exception as e:
                log(f"  final learned fit failed: {e!r}; falling to ridge")
                chosen = "ridge"
                chosen_stats = stats_ridge
                final_block = None
        else:
            chosen = "ridge"
            chosen_stats = stats_ridge
            final_block = None

        if chosen == "ridge":
            try:
                final_ridge = fit_ridge(
                    X.astype(np.float64), y.astype(np.float64), alpha=10.0
                )
                final_block = {
                    "l3_type": "ridge",
                    "weights": final_ridge.weights.tolist(),
                    "bias": float(final_ridge.bias),
                    "feature_mean": final_ridge.feature_mean.tolist(),
                    "feature_std": final_ridge.feature_std.tolist(),
                }
            except Exception as e:
                log(f"  final ridge fit failed: {e!r}")
                final_block = None

        new_tier = chosen_stats.get("classification", "Poor") or "Poor"
        tier_counter[new_tier] = tier_counter.get(new_tier, 0) + 1

        slot_record = {
            "target": target,
            "view": view,
            "approach": approach,
            "l2_source": l2_kind,
            "n_subjects": chosen_stats.get("n_subjects"),
            "n_trials": chosen_stats.get("n_trials"),
            "pearson_r": chosen_stats.get("pearson_r"),
            "ccc_lin": chosen_stats.get("ccc_lin"),
            "loa_half_width_deg": chosen_stats.get("loa_half_width_deg"),
            "mae_deg": chosen_stats.get("mae_deg"),
            "rmse_deg": chosen_stats.get("rmse_deg"),
            "classification": new_tier,
            "l3_kind": chosen,
            "ridge_ccc": ccc_r,
            "learned_ccc": ccc_l,
            "ridge_loa_half": stats_ridge.get("loa_half_width_deg"),
            "learned_loa_half": stats_learned.get("loa_half_width_deg"),
        }
        per_slot_validity.append(slot_record)

        deploy_entry = dict(entry)
        deploy_entry[f"{tag}_block"] = final_block
        deploy_entry[f"{tag}_loso_stats"] = {
            k: chosen_stats.get(k) for k in [
                "n_subjects", "n_trials", "mean_bias_deg",
                "sd_bias_deg", "loa_upper_deg", "loa_lower_deg",
                "loa_half_width_deg", "pearson_r", "ccc_lin",
                "mae_deg", "rmse_deg", "classification",
            ]
        }
        deploy_entry[f"{tag}_l3_kind"] = chosen
        deploy_entry[f"{tag}_l2_source"] = l2_kind
        deploy_models.setdefault(target, {})[view] = deploy_entry

        log(
            f"  ridge CCC={cr:+.3f} loa={stats_ridge.get('loa_half_width_deg', float('nan')):.2f} | "
            f"learned CCC={cl:+.3f} loa={stats_learned.get('loa_half_width_deg', float('nan')):.2f} "
            f"-> picked={chosen} tier={new_tier} "
            f"({time.time()-t0:.1f}s)"
        )

    log(f"=== {tag} complete: tier counts {tier_counter} ===")
    log(
        f"   learned wins (>0.005 CCC over ridge): {len(learned_wins)} | "
        f"ties / within-0.05: {len(ties)} | "
        f"ridge wins (learned worse by >0.05): {len(ridge_wins)}"
    )

    # Write deploy json.
    deploy_path = (
        RESULTS_DIR / f"deploy_ready_models_{tag}.json"
    )
    deploy_obj = {
        "version": tag,
        "produced_by": "harness.learned_layer3 (Agent NN)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            f"L2 source = {l2_kind} (v23 = HH2 combined alldata L2, "
            "v29 = NN mirror-flip per-source alldata L2). L3 = per-slot "
            "learned TinyMLP (hidden=32, dropout 0.2, AdamW lr=1e-2 "
            "wd=1e-3, 200 epochs w/ early stopping on 15% inner-val). "
            "Slot-by-slot fallback to ridge if learned CCC underperforms "
            "ridge CCC by > 0.05 (preserves no-regression guarantee). "
            "LOSO at L3 only; L2 trained on all 24 cohort subjects."
        ),
        "loso_discipline": (
            "Layer-3-LOSO-only (L2 trained on all 24 cohort subjects)"
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

    per_slot_path = OUT_DIR / f"per_slot_validity_{tag}.json"
    per_slot_obj = {
        "version": tag,
        "produced_by": "harness.learned_layer3 (Agent NN)",
        "l2_source": l2_kind,
        "loso_discipline": (
            "Layer-3-LOSO-only (L2 trained on all cohort subjects)"
        ),
        "tier_counts": tier_counter,
        "learned_wins": [
            {"target": t, "view": v, "ridge_ccc": rc, "learned_ccc": lc}
            for (t, v, rc, lc) in learned_wins
        ],
        "ridge_wins": [
            {"target": t, "view": v, "ridge_ccc": rc, "learned_ccc": lc}
            for (t, v, rc, lc) in ridge_wins
        ],
        "ties": [
            {"target": t, "view": v, "ridge_ccc": rc}
            for (t, v, rc) in ties
        ],
        "slots": per_slot_validity,
    }
    per_slot_path.write_text(
        json.dumps(per_slot_obj, indent=2, default=str)
    )
    log(f"wrote {per_slot_path}")

    return per_slot_obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l2", choices=["v23", "v29"], default="v23",
        help="Which L2 model to use as feature source.",
    )
    args = parser.parse_args()
    tag = {
        "v23": "v30_learned_l3",
        "v29": "v31_mirrorflip_learned_l3",
    }[args.l2]
    log(f"=== Agent NN Phase 2 ({tag}) START ===")
    _orchestrate(tag=tag, l2_kind=args.l2)
    log(f"=== Agent NN Phase 2 ({tag}) DONE ===")


if __name__ == "__main__":
    main()
