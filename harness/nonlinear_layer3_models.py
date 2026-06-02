"""Nonlinear Layer 3 models for the Couro deploy slots.

Tests whether Random Forest, Gradient Boosting, and a small MLP outperform
the current ridge regression baseline on the priority "Moderate, close to
Good" slots in `data/biomech_validity_stats/per_slot_validity.json`.

All models are evaluated with subject-level leave-one-subject-out CV
exactly the same way the ridge baseline was — predictions are stitched
across LOSO folds, then aggregated to one (pred, obs) point per subject
before computing Lin's CCC and Bland-Altman 95 percent LoA.

Ridge is re-run inside this script under the SAME pipeline so the
comparison is apples-to-apples; we do not trust the deploy_ready_models
ridge numbers blindly even though they use the same dataset builders.

Run:
    cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
    python -m harness.nonlinear_layer3_models

Outputs:
    data/nonlinear_models/per_slot_nonlinear_results.json
    data/nonlinear_models/REPORT.md
    (and recommended_v18_updates.json IFF any slot crosses the Good gate
     under LOSO-rigorous evaluation)
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# --- path setup so this runs both as `python -m` and as a script -----------
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Silence noisy convergence warnings from MLPRegressor on tiny LOSO folds.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

import harness.train_v9_phased as v9
import harness.train_v12_combined as v12
import harness.train_event_anchored_all as ea
import harness.train_v14_full_dwpose as v14
from harness.train_multi_metric import VIEW_BUCKET_NAMES

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation")
DATA_ROOT = ROOT / "data"
OUT_DIR = DATA_ROOT / "nonlinear_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_PATH = DATA_ROOT / "biomech_validity_stats" / "per_slot_validity.json"

VIEW_TO_CAM = {v: k for k, v in VIEW_BUCKET_NAMES.items()}


# ----------------------------------------------------------------------
# Priority target slots (from the task brief)
# ----------------------------------------------------------------------
PRIORITY_SLOTS: list[tuple[str, str, str]] = [
    ("knee_angle_r",      "front_oblique_right", "v9_phased"),
    ("hip_flexion_r",     "front_oblique_left",  "v13_dwpose_hybrid"),
    ("knee_angle_r",      "side_left",           "v14_full_dwpose"),
    ("knee_angle_r",      "side_right",          "v12_combined"),
    ("lumbar_extension",  "front_center",        "event_anchored"),
    ("lumbar_extension",  "front_oblique_left",  "event_anchored"),
    ("ankle_angle_r",     "front_oblique_left",  "v14_full_dwpose"),
    ("lumbar_extension",  "side_right",          "v14_full_dwpose"),
]


# ----------------------------------------------------------------------
# Dataset builders — mirror harness.biomech_validity_stats, single source
# of truth here so the comparison is apples-to-apples.
# ----------------------------------------------------------------------

def _build_v12(target: str, view: str):
    v9.KP_DIR = DATA_ROOT / "couro_keypoints"
    v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_couro_keypoints"
    return v12.build_combined_dataset(target, view)


def _build_v13(target: str, view: str):
    v9.KP_DIR = DATA_ROOT / "couro_keypoints"
    v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
    return v12.build_combined_dataset(target, view)


def _build_v14(target: str, view: str):
    v9.KP_DIR = DATA_ROOT / "opencap_dwpose_keypoints"
    v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
    if target == "ankle_angle_r":
        return v14.build_opencap_only_dataset(target, view)
    return v12.build_combined_dataset(target, view)


def _build_v9_phased(target: str, view: str):
    v9.KP_DIR = DATA_ROOT / "couro_keypoints"
    cam = VIEW_TO_CAM.get(view)
    if cam is None:
        return None
    X, y, subjects, _, _ = v9.build_dataset_v9(target, cam)
    if len(y) == 0:
        return None
    return X.astype(np.float64), np.asarray(y, dtype=np.float64), subjects, ["opencap"] * len(y)


def _build_event_anchored(target: str, view: str):
    ea.KP_DIR = DATA_ROOT / "couro_keypoints"
    cam = VIEW_TO_CAM.get(view)
    if cam is None:
        return None
    X, y, subjects, _, _ = ea.build_dataset(target, cam)
    if len(y) == 0:
        return None
    return X.astype(np.float64), np.asarray(y, dtype=np.float64), subjects, ["opencap"] * len(y)


BUILDERS: dict[str, Callable[[str, str], Any]] = {
    "v12_combined":      _build_v12,
    "v13_dwpose_hybrid": _build_v13,
    "v14_full_dwpose":   _build_v14,
    "v9_phased":         _build_v9_phased,
    "event_anchored":    _build_event_anchored,
}


# ----------------------------------------------------------------------
# Model factories. Hyperparameters are intentionally fixed at the values
# specified in the task brief — we are testing model class, not tuning.
# ----------------------------------------------------------------------

def _make_ridge() -> Any:
    return Ridge(alpha=10.0)


def _make_rf() -> Any:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=1,  # keep deterministic and small-job friendly
    )


def _make_gb() -> Any:
    return GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=0,
    )


def _make_mlp() -> Any:
    return MLPRegressor(
        hidden_layer_sizes=(32, 16),
        early_stopping=True,
        max_iter=300,
        random_state=0,
    )


MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "ridge":             _make_ridge,
    "random_forest":     _make_rf,
    "gradient_boosting": _make_gb,
    "mlp":               _make_mlp,
}

# Models that benefit from standardisation. Trees do not.
SCALE_MODELS = {"ridge", "mlp"}

# Random seeds used for seed-stability sweep. Ridge is deterministic so it is
# excluded from this sweep.
STABILITY_SEEDS: list[int] = [0, 1, 2, 3, 7, 17, 42]


def _make_rf_seed(seed: int) -> Any:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=1,
    )


def _make_gb_seed(seed: int) -> Any:
    return GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=seed,
    )


def _make_mlp_seed(seed: int) -> Any:
    return MLPRegressor(
        hidden_layer_sizes=(32, 16),
        early_stopping=True,
        max_iter=300,
        random_state=seed,
    )


SEED_FACTORIES: dict[str, Callable[[int], Any]] = {
    "random_forest":     _make_rf_seed,
    "gradient_boosting": _make_gb_seed,
    "mlp":               _make_mlp_seed,
}


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SlotResult:
    target: str
    view: str
    approach: str
    model: str
    n_subjects: int
    n_trials: int
    train_fit_r2: float           # honest overfit indicator
    loso_pearson_r: float
    loso_ccc: float
    loso_mean_bias: float
    loso_sd_bias: float
    loso_loa_lower: float
    loso_loa_upper: float
    loso_loa_half_width: float
    loso_mae: float
    loso_rmse: float
    tier: str
    overfit_gap: float            # train_fit_r2 - loso_pearson_r**2


def _ccc(p: np.ndarray, o: np.ndarray) -> float:
    mp, mo = float(np.mean(p)), float(np.mean(o))
    vp, vo = float(np.var(p, ddof=0)), float(np.var(o, ddof=0))
    cov = float(np.mean((p - mp) * (o - mo)))
    denom = vp + vo + (mp - mo) ** 2
    if denom == 0:
        return float("nan")
    return 2.0 * cov / denom


def _pearson(p: np.ndarray, o: np.ndarray) -> float:
    if p.std(ddof=1) == 0 or o.std(ddof=1) == 0:
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


# ----------------------------------------------------------------------
# LOSO evaluation
# ----------------------------------------------------------------------

def _loso_predict(
    X: np.ndarray,
    y: np.ndarray,
    subjects: list[str],
    model_name: str,
    *,
    heartbeat_label: str,
    heartbeat_start: float,
) -> tuple[np.ndarray, list[str], float]:
    """Return (preds_per_trial_stitched, subjects_aligned, full_data_train_r2)."""
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    preds = np.full(len(y), np.nan)

    for i, s in enumerate(unique):
        fold_t0 = time.time()
        train_mask = subj_arr != s
        test_mask = subj_arr == s
        if train_mask.sum() < 2 or test_mask.sum() < 1:
            print(
                f"[CC2 heartbeat] {heartbeat_label} fold {i+1}/{len(unique)} "
                f"SKIPPED (insufficient data)",
                flush=True,
            )
            continue
        Xtr, ytr = X[train_mask], y[train_mask]
        Xte = X[test_mask]
        scaler = None
        if model_name in SCALE_MODELS:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)
        model = MODEL_FACTORIES[model_name]()
        model.fit(Xtr, ytr)
        preds[test_mask] = model.predict(Xte)
        dt = time.time() - fold_t0
        if dt > 30 or i % max(1, len(unique) // 4) == 0:
            print(
                f"[CC2 heartbeat] {heartbeat_label} fold {i+1}/{len(unique)} "
                f"({dt:.1f}s, total {time.time() - heartbeat_start:.0f}s)",
                flush=True,
            )

    # Full-data train fit (separate model trained on everything) so we get
    # an honest overfit indicator.
    train_r2 = float("nan")
    try:
        Xall = X
        if model_name in SCALE_MODELS:
            scaler_all = StandardScaler()
            Xall = scaler_all.fit_transform(Xall)
        full_model = MODEL_FACTORIES[model_name]()
        full_model.fit(Xall, y)
        yhat = full_model.predict(Xall)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot > 0:
            train_r2 = 1.0 - ss_res / ss_tot
    except Exception as e:
        print(f"[CC2 heartbeat] train-fit failed: {e!r}", flush=True)

    return preds, list(subj_arr), train_r2


def _aggregate_per_subject(
    preds: np.ndarray, obs: np.ndarray, subjects: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(preds) & np.isfinite(obs)
    p = preds[mask]
    o = obs[mask]
    s = np.asarray(subjects)[mask]
    unique = sorted(set(s.tolist()))
    agg_p = np.array([float(np.mean(p[s == u])) for u in unique])
    agg_o = np.array([float(np.mean(o[s == u])) for u in unique])
    return agg_p, agg_o


def _stats_per_subject(
    preds: np.ndarray, obs: np.ndarray, subjects: list[str],
) -> dict[str, float]:
    p, o = _aggregate_per_subject(preds, obs, subjects)
    if len(p) < 2:
        nan = float("nan")
        return {
            "n_subjects": int(len(p)),
            "pearson_r": nan, "ccc": nan,
            "mean_bias": nan, "sd_bias": nan,
            "loa_upper": nan, "loa_lower": nan, "loa_half_width": nan,
            "mae": nan, "rmse": nan,
        }
    diffs = p - o
    mean_bias = float(np.mean(diffs))
    sd_bias = float(np.std(diffs, ddof=1))
    return {
        "n_subjects": int(len(p)),
        "pearson_r": _pearson(p, o),
        "ccc": _ccc(p, o),
        "mean_bias": mean_bias,
        "sd_bias": sd_bias,
        "loa_upper": mean_bias + 1.96 * sd_bias,
        "loa_lower": mean_bias - 1.96 * sd_bias,
        "loa_half_width": max(
            abs(mean_bias + 1.96 * sd_bias - mean_bias),
            abs(mean_bias - (mean_bias - 1.96 * sd_bias)),
        ),
        "mae": float(np.mean(np.abs(diffs))),
        "rmse": float(np.sqrt(np.mean(diffs ** 2))),
    }


# ----------------------------------------------------------------------
# Per-slot driver
# ----------------------------------------------------------------------

def _loso_predict_seed(
    X: np.ndarray,
    y: np.ndarray,
    subjects: list[str],
    model_name: str,
    seed: int,
) -> np.ndarray:
    """Like _loso_predict but for a specific seed; quiet (no heartbeat) because
    this runs inside the seed sweep which prints its own progress."""
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    preds = np.full(len(y), np.nan)
    scale = model_name in SCALE_MODELS
    factory = SEED_FACTORIES[model_name]
    for s in unique:
        train_mask = subj_arr != s
        test_mask = subj_arr == s
        if train_mask.sum() < 2 or test_mask.sum() < 1:
            continue
        Xtr, ytr = X[train_mask], y[train_mask]
        Xte = X[test_mask]
        if scale:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)
        model = factory(seed)
        model.fit(Xtr, ytr)
        preds[test_mask] = model.predict(Xte)
    return preds


def seed_stability_sweep(
    X: np.ndarray,
    y: np.ndarray,
    subjects: list[str],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    """For each stochastic model class, run LOSO under each STABILITY_SEED
    and return a {model_name: {ccc_seeds, loa_half_seeds, mean, sd,
    good_tier_seeds}} dict."""
    out: dict[str, dict[str, Any]] = {}
    for model_name in SEED_FACTORIES:
        cccs: list[float] = []
        loas: list[float] = []
        for seed in STABILITY_SEEDS:
            preds = _loso_predict_seed(X, y, subjects, model_name, seed)
            stats = _stats_per_subject(preds, y, list(subjects))
            cccs.append(stats["ccc"])
            loas.append(stats["loa_half_width"])
        cccs_arr = np.asarray(cccs)
        loas_arr = np.asarray(loas)
        out[model_name] = {
            "seeds": list(STABILITY_SEEDS),
            "ccc_seeds": cccs,
            "loa_half_seeds": loas,
            "ccc_mean": float(np.mean(cccs_arr)),
            "ccc_sd": float(np.std(cccs_arr, ddof=1)),
            "loa_mean": float(np.mean(loas_arr)),
            "loa_sd": float(np.std(loas_arr, ddof=1)),
            "good_tier_seeds": int(
                sum(
                    1 for c, l in zip(cccs, loas)
                    if (not math.isnan(c)) and (not math.isnan(l)) and c > 0.60 and l < 10
                )
            ),
        }
        print(
            f"[CC2 stability] {label} {model_name}: "
            f"CCC={out[model_name]['ccc_mean']:+.3f}+/-{out[model_name]['ccc_sd']:.3f}  "
            f"good_seeds={out[model_name]['good_tier_seeds']}/{len(STABILITY_SEEDS)}",
            flush=True,
        )
    return out


def evaluate_slot(
    target: str,
    view: str,
    approach: str,
    *,
    slot_idx: int,
    n_slots: int,
    intermediate_path: Path,
    accumulated: dict[str, Any],
) -> dict[str, Any] | None:
    label = f"slot {slot_idx}/{n_slots}: {target}/{view}/{approach}"
    print(f"\n[CC2 {label}] starting", flush=True)
    t_slot0 = time.time()

    builder = BUILDERS.get(approach)
    if builder is None:
        print(f"[CC2 {label}] NO BUILDER for approach={approach}", flush=True)
        return None

    try:
        result = builder(target, view)
    except Exception as e:
        print(f"[CC2 {label}] BUILDER FAILED: {e!r}", flush=True)
        return None
    if result is None:
        print(f"[CC2 {label}] builder returned None", flush=True)
        return None

    X, y, subjects, _sources = result
    n_subj = len(set(subjects))
    n_trials = len(y)
    print(
        f"[CC2 {label}] dataset built: n_subjects={n_subj} n_trials={n_trials} "
        f"X.shape={X.shape}",
        flush=True,
    )

    slot_results: dict[str, dict[str, Any]] = {}

    for model_name in ["ridge", "random_forest", "gradient_boosting", "mlp"]:
        m_t0 = time.time()
        hb_label = f"{label} model={model_name}"
        print(f"[CC2 {hb_label}] LOSO start", flush=True)
        try:
            preds, subj_aligned, train_r2 = _loso_predict(
                X, y, list(subjects), model_name,
                heartbeat_label=hb_label, heartbeat_start=m_t0,
            )
            stats = _stats_per_subject(preds, y, subj_aligned)
            tier = _classify(stats["ccc"], stats["loa_half_width"])
            overfit_gap = (
                train_r2 - stats["pearson_r"] ** 2
                if not (math.isnan(train_r2) or math.isnan(stats["pearson_r"]))
                else float("nan")
            )
            slot_results[model_name] = {
                "n_subjects": stats["n_subjects"],
                "n_trials": n_trials,
                "train_fit_r2": train_r2,
                "loso_pearson_r": stats["pearson_r"],
                "loso_ccc": stats["ccc"],
                "loso_mean_bias": stats["mean_bias"],
                "loso_sd_bias": stats["sd_bias"],
                "loso_loa_lower": stats["loa_lower"],
                "loso_loa_upper": stats["loa_upper"],
                "loso_loa_half_width": stats["loa_half_width"],
                "loso_mae": stats["mae"],
                "loso_rmse": stats["rmse"],
                "tier": tier,
                "overfit_gap": overfit_gap,
                "elapsed_s": time.time() - m_t0,
            }
            print(
                f"[CC2 {label} DONE: model={model_name}] "
                f"ccc={stats['ccc']:.3f} loa_half={stats['loa_half_width']:.2f} "
                f"tier={tier} train_r2={train_r2:.3f} "
                f"overfit_gap={overfit_gap:.3f} ({time.time() - m_t0:.1f}s)",
                flush=True,
            )
        except Exception as e:
            slot_results[model_name] = {"error": repr(e)}
            print(f"[CC2 {label} model={model_name}] FAILED: {e!r}", flush=True)

        # Save intermediate state after EVERY model — anti-stall protocol.
        accumulated.setdefault("slots", {})[f"{target}|{view}|{approach}"] = {
            "target": target, "view": view, "approach": approach,
            "n_subjects": n_subj, "n_trials": n_trials,
            "models": slot_results,
        }
        intermediate_path.write_text(json.dumps(accumulated, indent=2))

    # Seed-stability sweep for the stochastic model classes (RF, GB, MLP).
    print(f"[CC2 {label}] starting seed-stability sweep", flush=True)
    try:
        stab = seed_stability_sweep(X, y, list(subjects), label=label)
        for model_name, stab_row in stab.items():
            if model_name in slot_results and "error" not in slot_results[model_name]:
                slot_results[model_name]["seed_stability"] = stab_row
        accumulated["slots"][f"{target}|{view}|{approach}"]["models"] = slot_results
        intermediate_path.write_text(json.dumps(accumulated, indent=2))
    except Exception as e:
        print(f"[CC2 {label}] seed-stability sweep FAILED: {e!r}", flush=True)

    print(
        f"[CC2 {label}] FINISHED total_slot_elapsed={time.time() - t_slot0:.1f}s",
        flush=True,
    )
    return accumulated["slots"][f"{target}|{view}|{approach}"]


# ----------------------------------------------------------------------
# Report rendering
# ----------------------------------------------------------------------

def _load_baseline() -> dict[str, dict[str, Any]]:
    """Map (target|view|approach) -> baseline ridge stats from per_slot_validity.json."""
    payload = json.loads(BASELINE_PATH.read_text())
    out: dict[str, dict[str, Any]] = {}
    for s in payload.get("slots", []):
        key = f"{s['target']}|{s['view']}|{s['approach']}"
        out[key] = s
    return out


def render_report(accumulated: dict[str, Any]) -> str:
    baseline = _load_baseline()
    lines: list[str] = []
    lines.append("# Couro Layer 3 Nonlinear Model Sweep")
    lines.append("")
    lines.append(
        "Tests whether Random Forest, Gradient Boosting, and a small MLP "
        "improve over the deployed ridge regression on eight priority slots. "
        "All evaluations are subject-level leave-one-subject-out (LOSO). "
        "Ridge is re-run inside this script under the same pipeline so the "
        "comparison is apples-to-apples."
    )
    lines.append("")
    lines.append("## Models tested")
    lines.append("")
    lines.append("- **Ridge** (baseline replay) — alpha=10.0, StandardScaler.")
    lines.append("- **Random Forest** — n_estimators=100, max_depth=4, min_samples_leaf=2.")
    lines.append("- **Gradient Boosting** — n_estimators=100, max_depth=3, learning_rate=0.05.")
    lines.append("- **MLP** — hidden_layer_sizes=(32,16), early_stopping=True, max_iter=300, StandardScaler.")
    lines.append("")
    lines.append("## Tier gates")
    lines.append("")
    lines.append("| Tier | CCC | LoA half-width |")
    lines.append("| --- | --- | --- |")
    lines.append("| Excellent | > 0.75 | < +/- 5 deg |")
    lines.append("| Good | 0.60 - 0.75 | +/- 5 - 10 deg |")
    lines.append("| Moderate | 0.40 - 0.60 | +/- 10 - 15 deg |")
    lines.append("| Poor | <= 0.40 | > +/- 15 deg |")
    lines.append("")
    lines.append("## Headline result table (per slot, per model)")
    lines.append("")
    lines.append(
        "| Slot | Model | n | CCC | LoA half (deg) | bias (deg) | MAE (deg) | "
        "Tier | train R^2 | overfit gap |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"
    )

    def fmt(v, prec=3):
        if v is None:
            return "-"
        if isinstance(v, float) and math.isnan(v):
            return "-"
        try:
            return f"{v:.{prec}f}"
        except Exception:
            return str(v)

    slot_keys_order = [
        f"{t}|{v}|{a}" for (t, v, a) in PRIORITY_SLOTS
    ]
    for key in slot_keys_order:
        slot = accumulated.get("slots", {}).get(key)
        if not slot:
            lines.append(f"| {key} | (no data) | - | - | - | - | - | - | - | - |")
            continue
        baseline_row = baseline.get(key, {})
        baseline_ccc = baseline_row.get("ccc_lin")
        baseline_loa = baseline_row.get("loa_half_width_deg")
        baseline_tier = baseline_row.get("classification", "-")
        # First, the baseline row from deploy ridge
        lines.append(
            f"| {slot['target']}/{slot['view']}/{slot['approach']} | "
            f"ridge (deploy doc) | {slot['n_subjects']} | "
            f"{fmt(baseline_ccc)} | {fmt(baseline_loa, 2)} | - | - | "
            f"{baseline_tier} | - | - |"
        )
        for model in ["ridge", "random_forest", "gradient_boosting", "mlp"]:
            row = slot["models"].get(model, {})
            if "error" in row:
                lines.append(
                    f"| | {model} (FAILED) | - | - | - | - | - | - | - | - |"
                )
                continue
            lines.append(
                f"| | {model} | {row.get('n_subjects', '-')} | "
                f"{fmt(row.get('loso_ccc'))} | "
                f"{fmt(row.get('loso_loa_half_width'), 2)} | "
                f"{fmt(row.get('loso_mean_bias'), 2)} | "
                f"{fmt(row.get('loso_mae'), 2)} | "
                f"{row.get('tier', '-')} | "
                f"{fmt(row.get('train_fit_r2'))} | "
                f"{fmt(row.get('overfit_gap'))} |"
            )
    lines.append("")

    # Per-slot tier verdict
    lines.append("## Per-slot tier verdict")
    lines.append("")
    for key in slot_keys_order:
        slot = accumulated.get("slots", {}).get(key)
        if not slot:
            continue
        baseline_row = baseline.get(key, {})
        baseline_tier = baseline_row.get("classification", "-")
        baseline_ccc = baseline_row.get("ccc_lin")
        models = slot["models"]
        best_ccc = -1.0
        best_model = None
        for mname, row in models.items():
            if "error" in row:
                continue
            c = row.get("loso_ccc")
            if c is None or math.isnan(c):
                continue
            if c > best_ccc:
                best_ccc = c
                best_model = mname
        if best_model is None:
            lines.append(f"- **{slot['target']}/{slot['view']}/{slot['approach']}**: all models failed.")
            continue
        best_row = models[best_model]
        verdict = "no change"
        if best_row["tier"] in ("Good", "Excellent") and baseline_tier not in ("Good", "Excellent"):
            verdict = "**TIER CROSS to {0}**".format(best_row["tier"])
        elif (
            isinstance(baseline_ccc, (int, float))
            and best_ccc - float(baseline_ccc) > 0.03
        ):
            verdict = f"meaningful improvement (+{best_ccc - float(baseline_ccc):.2f} CCC, same tier)"
        lines.append(
            f"- **{slot['target']}/{slot['view']}/{slot['approach']}**: "
            f"baseline ridge={baseline_tier} (CCC={fmt(baseline_ccc)}), "
            f"best={best_model} CCC={fmt(best_ccc)} tier={best_row['tier']} -> {verdict}"
        )
    lines.append("")

    # Overfit honesty
    lines.append("## Overfit assessment")
    lines.append("")
    lines.append(
        "`overfit_gap` = (train-on-all R^2) - (LOSO Pearson r)^2. Trees in "
        "particular fit training data very well even when they can't "
        "generalise; a wide gap is the honest signal that the model class "
        "isn't learning generalisable structure at this n."
    )
    lines.append("")
    lines.append("| Slot | Model | train R^2 | LOSO r^2 | gap | flag |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for key in slot_keys_order:
        slot = accumulated.get("slots", {}).get(key)
        if not slot:
            continue
        for model in ["ridge", "random_forest", "gradient_boosting", "mlp"]:
            row = slot["models"].get(model, {})
            if "error" in row:
                continue
            r = row.get("loso_pearson_r")
            tr2 = row.get("train_fit_r2")
            if r is None or tr2 is None:
                continue
            loso_r2 = r * r if not (isinstance(r, float) and math.isnan(r)) else float("nan")
            gap = tr2 - loso_r2 if not (isinstance(loso_r2, float) and math.isnan(loso_r2)) else float("nan")
            flag = ""
            if not math.isnan(gap):
                if gap > 0.5:
                    flag = "SEVERE overfit"
                elif gap > 0.3:
                    flag = "moderate overfit"
                elif gap > 0.15:
                    flag = "mild overfit"
                else:
                    flag = "minimal"
            lines.append(
                f"| {slot['target']}/{slot['view']} | {model} | "
                f"{fmt(tr2)} | {fmt(loso_r2)} | {fmt(gap)} | {flag} |"
            )
    lines.append("")

    # Honest summary
    lines.append("## Honest summary")
    lines.append("")
    crossings = []
    for key in slot_keys_order:
        slot = accumulated.get("slots", {}).get(key)
        if not slot:
            continue
        baseline_row = baseline.get(key, {})
        baseline_tier = baseline_row.get("classification", "-")
        for mname, row in slot["models"].items():
            if "error" in row:
                continue
            if row["tier"] in ("Good", "Excellent") and baseline_tier not in ("Good", "Excellent"):
                crossings.append((key, mname, row))
    if crossings:
        lines.append(f"{len(crossings)} slot/model combinations cross the Good gate under rigorous LOSO:")
        for key, mname, row in crossings:
            lines.append(
                f"- **{key}** with **{mname}**: CCC={fmt(row['loso_ccc'])}, "
                f"LoA half-width={fmt(row['loso_loa_half_width'], 2)} deg, "
                f"tier={row['tier']} (overfit_gap={fmt(row['overfit_gap'])})"
            )
    else:
        lines.append(
            "**No slots crossed the Good gate under rigorous LOSO evaluation.** "
            "This means model class is NOT the bottleneck at the current n; the "
            "features (or the within-subject calibration) are. This motivates "
            "Agent EE2's learned Layer 2 direction as the right next step."
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def maybe_write_v18_updates(accumulated: dict[str, Any]) -> Path | None:
    """Only emit v18 recommendation if a nonlinear model crosses the Good gate
    on ALL stability seeds (7/7). Anything less is seed-luck and the deploy
    doc keeps the ridge baseline."""
    baseline = _load_baseline()
    updates: dict[str, Any] = {"slots": []}
    n_seeds = len(STABILITY_SEEDS)
    for key, slot in accumulated.get("slots", {}).items():
        baseline_row = baseline.get(key, {})
        baseline_tier = baseline_row.get("classification", "-")
        if baseline_tier in ("Good", "Excellent"):
            continue  # already validated; nothing to improve
        best = None
        for mname, row in slot["models"].items():
            if "error" in row:
                continue
            stab = row.get("seed_stability")
            if not stab:
                continue
            if stab.get("good_tier_seeds", 0) < n_seeds:
                continue  # not robust across seeds
            if best is None or stab["ccc_mean"] > best[1]["seed_stability"]["ccc_mean"]:
                best = (mname, row)
        if best is None:
            continue
        mname, row = best
        stab = row["seed_stability"]
        updates["slots"].append({
            "target": slot["target"],
            "view": slot["view"],
            "approach": slot["approach"],
            "model_class": mname,
            "hyperparameters": _hyperparams_for(mname),
            "seed_mean_ccc": stab["ccc_mean"],
            "seed_mean_loa_half_width": stab["loa_mean"],
            "seed_ccc_sd": stab["ccc_sd"],
            "good_tier_seeds": stab["good_tier_seeds"],
            "n_seeds_tested": n_seeds,
            "baseline_tier": baseline_tier,
            "n_subjects": row["n_subjects"],
        })
    if not updates["slots"]:
        return None
    p = OUT_DIR / "recommended_v18_updates.json"
    p.write_text(json.dumps(updates, indent=2))
    return p


def _hyperparams_for(model: str) -> dict[str, Any]:
    if model == "random_forest":
        return {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 2, "random_state": 0}
    if model == "gradient_boosting":
        return {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "random_state": 0}
    if model == "mlp":
        return {"hidden_layer_sizes": [32, 16], "early_stopping": True, "max_iter": 300, "random_state": 0, "scaling": "StandardScaler"}
    if model == "ridge":
        return {"alpha": 10.0, "scaling": "StandardScaler"}
    return {}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    run_t0 = time.time()
    intermediate = OUT_DIR / "per_slot_nonlinear_results.partial.json"
    accumulated: dict[str, Any] = {
        "version": "1.1",
        "produced_by": "harness.nonlinear_layer3_models",
        "started_at": time.time(),
        "seed_stability_methodology": (
            "For RF/GB/MLP each slot was re-run under 7 random seeds "
            f"{STABILITY_SEEDS}. ccc_seeds/loa_half_seeds give the per-seed "
            "LOSO subject-level CCC and Bland-Altman 95% LoA half-width. "
            "good_tier_seeds is the count of seeds where the model crossed "
            "the Good gate (CCC > 0.60 AND LoA half-width < 10 deg). Ridge "
            "is deterministic, so no seed sweep applies. A v18 recommendation "
            "is only emitted when a model crosses Good on ALL seeds."
        ),
        "slots": {},
    }

    print(
        f"[CC2 main] starting nonlinear sweep across {len(PRIORITY_SLOTS)} priority slots",
        flush=True,
    )

    for i, (target, view, approach) in enumerate(PRIORITY_SLOTS, start=1):
        elapsed = time.time() - run_t0
        # 2.5 hour time budget guard
        if elapsed > 2.25 * 3600:
            print(
                f"[CC2 main] TIME BUDGET (2.25h) reached after slot {i-1}, stopping early",
                flush=True,
            )
            break
        try:
            evaluate_slot(
                target, view, approach,
                slot_idx=i, n_slots=len(PRIORITY_SLOTS),
                intermediate_path=intermediate,
                accumulated=accumulated,
            )
        except Exception as e:
            print(f"[CC2 main] slot {i} FAILED OUTRIGHT: {e!r}", flush=True)

    # Finalise
    final_path = OUT_DIR / "per_slot_nonlinear_results.json"
    final_path.write_text(json.dumps(accumulated, indent=2))
    report_path = OUT_DIR / "REPORT.md"
    report_path.write_text(render_report(accumulated))
    v18 = maybe_write_v18_updates(accumulated)

    print(
        f"\n[CC2 main] DONE in {time.time() - run_t0:.0f}s. "
        f"Wrote {final_path}, {report_path}"
        + (f", {v18}" if v18 else " (no Good-gate crossings -> no v18 updates)"),
        flush=True,
    )


if __name__ == "__main__":
    main()
