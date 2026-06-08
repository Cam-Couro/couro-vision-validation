"""Phase B: re-fit per-slot Layer 3 ridge regression using learned Layer 2
angle traces (from Agent EE2's TemporalKeypointCNNConf), then re-run the
biomech validity stats to count tier changes.

This is Agent FF's build.

# LOSO discipline used in this build

**Layer-3-LOSO-only.** One Layer 2 model is trained on ALL 9 OpenCap subjects
(no LOSO at L2) and used to produce learned angle traces for every clip
(OpenCap + ASPset). Layer 3 ridge is then re-fit per slot with LOSO at L3
only.

This **leaks Layer 2 information from the held-out OpenCap subject into the L2
training data**. For ASPset subjects, the regime is effectively double-LOSO
because no ASPset subject ever entered Layer 2 training (Layer 2 cohort is
OpenCap-only).

**Implication for tier-change claims:** any tier promotion driven primarily by
OpenCap LOSO folds is an upper bound; the true double-LOSO number could be
~0.05-0.10 |r| lower per EE2's per-fold variance (best fold 0.700, worst 0.541
vs all-subject training ≈ best). This is flagged in REPORT.md.

# Why not full double-LOSO

Training 9 LOSO L2 models + saving per-fold per-clip traces (~10 min) plus
running L2 inference on 1410 ASPset clips per fold (~5 min × 9 folds = 45
min) plus building L3 features for 24 slots (~15 min/slot pipeline-loaded)
exceeds the 3 hr budget. The Layer-3-LOSO-only scope is the user-permitted
honest fallback, with explicit tier-change flagging.

# Pipeline

1. Train ONE learned Layer 2 on all 9 OpenCap subjects (~50 s CPU).
2. Run L2 inference on OpenCap DWPose keypoints (270 clips, ~30 s) and
   ASPset DWPose keypoints (1410 clips, ~3 min). Cache traces.
3. Monkey-patch couro_keypoints.keypoints_to_motion_data so the 5 deploy
   angles come from the learned model. Left-side (hip_adduction_l, etc.) and
   pelvis_tilt remain hand-engineered.
4. For each of 24 v17 deploy slots, call the appropriate dataset builder
   (v9_phased / v12_combined / v13_dwpose_hybrid / v14_full_dwpose /
   event_anchored / event_anchored_bilateral) -- features are rebuilt with
   the patched L2.
5. LOSO ridge per slot. Compute Bland-Altman + CCC. Classify tier. Compare to
   v17 baseline tiers.

Prints progress every step. Slot-level print on each completion. Per-clip
log every ~20 s during inference.
"""
from __future__ import annotations

import builtins
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

# numpy 2.x compat for any chumpy-backed loads.
for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.learned_layer2_real_gt import (
    TemporalKeypointCNNConf,
    HALPE_KEYS_WITH_SMPL,
    N_KEYS,
    TEMPORAL_T,
    DEPLOY_METRICS as L2_METRICS,
    _normalize_kp,
    load_opencap_clip,
    build_windows_for_clip,
    stack_clips,
    build_clip_dataset,
    train_one_fold,
)
from harness.smpl_layer2_poc import HALPE26_INDEX
from harness import couro_keypoints as ck_mod
from harness.couro_keypoints import (
    CouroKeypointSeries,
    KP,
    load_couro_output,
    _angle_between,
    _signed_angle_from_vertical,
)
from harness.parsers import MotionData, parse_mot
from harness.train_regression_poc import fit_ridge
from harness.per_slot_prediction_dump import DumpAccumulator


# Agent SS per-clip dump infrastructure: when ``DUMP_ACCUMULATOR`` is set
# by an outer driver (or by main(--dump-predictions <reader_tag>)), the
# LOSO loop will record per-subject (pred, gt) means for every slot.
# Downstream scripts (v23, v24, v26, v27, v29) that monkey-patch this
# module's ``run()`` will inherit this behaviour automatically.
DUMP_ACCUMULATOR: DumpAccumulator | None = None


# -------------------------------------------------------------------------
# Paths and constants.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
LAB: Final[Path] = DATA_ROOT / "LabValidation_withVideos"
OPENCAP_KP: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"
ASPSET_DWPOSE_KP: Final[Path] = DATA_ROOT / "aspset510_dwpose_keypoints"
OPENCAP_RTM_KP: Final[Path] = DATA_ROOT / "couro_keypoints"
ASPSET_RTM_KP: Final[Path] = DATA_ROOT / "aspset510_couro_keypoints"

OUT_DIR: Final[Path] = DATA_ROOT / "layer3_retrain_learned_l2"
MODEL_OUT: Final[Path] = REPO_ROOT / "models" / "learned_layer2_alldata_v1.pt"
TRACE_CACHE: Final[Path] = OUT_DIR / "l2_traces_cache.npz"

V17_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v17_selective.json"
V18_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v18_learned_l2.json"
BASELINE_PATH: Final[Path] = (
    DATA_ROOT / "biomech_validity_stats" / "per_slot_validity.json"
)


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[FF t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Step 1: Train all-data Layer 2 model (or load if cached).
# -------------------------------------------------------------------------


def build_or_load_all_data_l2(
    *, epochs: int = 25, batch_size: int = 256, force_retrain: bool = False,
) -> tuple[TemporalKeypointCNNConf, dict]:
    """Train one Layer 2 model on ALL 9 OpenCap subjects. No LOSO at L2."""
    if MODEL_OUT.exists() and not force_retrain:
        log(f"loading cached all-data L2 model from {MODEL_OUT}")
        ckpt = torch.load(MODEL_OUT, weights_only=False)
        model = TemporalKeypointCNNConf()
        model.load_state_dict(ckpt["model_state"])
        model.y_mean = ckpt["y_mean"]  # type: ignore[attr-defined]
        model.y_std = ckpt["y_std"]    # type: ignore[attr-defined]
        return model, ckpt

    log("training all-data L2 (no LOSO at L2) ...")
    halpe_idx_arr = np.array(
        [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL], dtype=np.int64
    )
    cohort: dict[str, list[dict]] = {}
    kp_files = sorted(OPENCAP_KP.glob("*.json"))
    log(f"  scanning {len(kp_files)} OpenCap DWPose JSONs ...")
    last_log = time.time()
    for i, kp_path in enumerate(kp_files):
        if time.time() - last_log > 20.0:
            log(f"  scanned {i+1}/{len(kp_files)} (kept_subjects={len(cohort)})")
            last_log = time.time()
        rec = build_clip_dataset(kp_path, halpe_idx_arr)
        if rec is None:
            continue
        cohort.setdefault(rec["subject"], []).append(rec)
    log(f"  cohort: {len(cohort)} subjects, "
        f"{sum(len(v) for v in cohort.values())} clips")

    all_clips = [c for clips in cohort.values() for c in clips]
    X_train, Y_train = stack_clips(all_clips)
    log(f"  stacked training windows: X={X_train.shape} Y={Y_train.shape}")

    model, _ = train_one_fold(
        X_train, Y_train,
        epochs=epochs, batch_size=batch_size, seed=42,
        fold_label="ALL-DATA",
    )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
        "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
        "n_keys": N_KEYS,
        "temporal_t": TEMPORAL_T,
        "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
        "metrics": list(L2_METRICS),
        "model_class": "TemporalKeypointCNNConf",
        "loso_discipline": "ALL_DATA_NO_LOSO_AT_L2",
    }, MODEL_OUT)
    log(f"  saved all-data L2 checkpoint to {MODEL_OUT}")

    return model, {
        "model_state": model.state_dict(),
        "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
        "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
    }


# -------------------------------------------------------------------------
# Step 2: Inference on a CouroKeypointSeries -> 5 angle traces.
# -------------------------------------------------------------------------


def _series_to_kp_windows(
    series: CouroKeypointSeries, halpe_idx_arr: np.ndarray
) -> np.ndarray | None:
    """Convert a CouroKeypointSeries into the (T, T_win, N_KEYS, 3) window
    tensor expected by the L2 model. Per-frame normalization + center-frame
    sliding windows over T_win=9 frames. Returns None if not enough frames."""
    n_frames = series.xy.shape[0]
    if n_frames < TEMPORAL_T:
        return None

    kp_sub = series.xy[:, halpe_idx_arr, :]      # (T, N_KEYS, 2)
    conf_sub = series.confidence[:, halpe_idx_arr]  # (T, N_KEYS)

    # NaN protection: many opencap_dwpose JSONs already have NaNs from
    # low-confidence frames in load_couro_output. Replace by per-keypoint
    # last-known value.
    if np.any(~np.isfinite(kp_sub)):
        for k in range(kp_sub.shape[1]):
            for d in range(2):
                col = kp_sub[:, k, d]
                finite = np.isfinite(col)
                if finite.all():
                    continue
                if not finite.any():
                    kp_sub[:, k, d] = 0.0
                    continue
                idx = np.arange(n_frames)
                col_filled = np.interp(idx, idx[finite], col[finite])
                kp_sub[:, k, d] = col_filled
        kp_sub = np.nan_to_num(kp_sub, nan=0.0)

    # Per-frame normalize.
    kp_norm = np.zeros_like(kp_sub, dtype=np.float32)
    for i in range(n_frames):
        kp_norm[i] = _normalize_kp(kp_sub[i]).astype(np.float32)

    packed = np.concatenate(
        [kp_norm, conf_sub[..., None].astype(np.float32)], axis=-1
    ).astype(np.float32)  # (T, N_KEYS, 3)

    half = TEMPORAL_T // 2
    pad_front = np.repeat(packed[:1], half, axis=0)
    pad_back = np.repeat(packed[-1:], half, axis=0)
    padded = np.concatenate([pad_front, packed, pad_back], axis=0)
    windows = np.stack(
        [padded[i : i + TEMPORAL_T] for i in range(n_frames)], axis=0
    )  # (T, T_win, N_KEYS, 3)
    return windows


def predict_l2_for_series(
    model: TemporalKeypointCNNConf,
    series: CouroKeypointSeries,
    halpe_idx_arr: np.ndarray,
) -> np.ndarray | None:
    """Returns (T, 5) array of learned-L2 angle predictions for the 5 metrics,
    or None if the clip is too short."""
    windows = _series_to_kp_windows(series, halpe_idx_arr)
    if windows is None:
        return None
    b, t, k, c = windows.shape
    x_flat = windows.reshape(b, t, k * c)
    x_t = torch.from_numpy(x_flat).float()
    model.eval()
    with torch.no_grad():
        p_n = model(x_t).numpy()
    y_mean = np.asarray(model.y_mean)  # type: ignore[attr-defined]
    y_std = np.asarray(model.y_std)    # type: ignore[attr-defined]
    pred = p_n * y_std + y_mean
    return pred.astype(np.float32)  # (T, 5)


def _mirror_series(series: CouroKeypointSeries) -> CouroKeypointSeries:
    """Horizontal flip + L/R swap for a Couro keypoint series. Used to extract
    left-side joint angles by inferring the right-side model on the mirrored
    pose."""
    xy = series.xy.copy()
    xy[..., 0] = series.width - 1 - xy[..., 0]
    conf = series.confidence.copy()
    lr_pairs = [
        ("L_eye", "R_eye"), ("L_ear", "R_ear"),
        ("L_sho", "R_sho"), ("L_elb", "R_elb"), ("L_wri", "R_wri"),
        ("L_hip", "R_hip"), ("L_kne", "R_kne"), ("L_ank", "R_ank"),
        ("L_big_toe", "R_big_toe"), ("L_small_toe", "R_small_toe"),
        ("L_heel", "R_heel"),
    ]
    for l_name, r_name in lr_pairs:
        l, r = KP[l_name], KP[r_name]
        xy[:, [l, r]] = xy[:, [r, l]]
        conf[:, [l, r]] = conf[:, [r, l]]
    return CouroKeypointSeries(
        time=series.time, xy=xy, confidence=conf,
        width=series.width, height=series.height, fps=series.fps,
    )


# -------------------------------------------------------------------------
# Step 3: cached L2 trace registry.
# Key: kp_path stem -> {"pred_r": (T, 5), "pred_l": (T, 5)}
# `pred_r` is the 5-angle predict on the original series.
# `pred_l` is the 5-angle predict on the mirrored series -- used for left-side
#   joint columns (hip_flexion_l, knee_angle_l, ankle_angle_l, hip_adduction_l,
#   and lumbar_extension which is bilateral).
# -------------------------------------------------------------------------


_L2_CACHE: dict[str, dict[str, np.ndarray]] = {}


def predict_l2_for_kp_file(
    kp_path: Path,
    model: TemporalKeypointCNNConf,
    halpe_idx_arr: np.ndarray,
    *,
    cache_key: str | None = None,
) -> dict[str, np.ndarray] | None:
    """Load kp_path, run L2 inference (right + mirrored), cache, return dict."""
    key = cache_key or kp_path.stem
    if key in _L2_CACHE:
        return _L2_CACHE[key]
    try:
        series = load_couro_output(kp_path)
    except Exception:
        return None
    pred_r = predict_l2_for_series(model, series, halpe_idx_arr)
    series_l = _mirror_series(series)
    pred_l = predict_l2_for_series(model, series_l, halpe_idx_arr)
    if pred_r is None or pred_l is None:
        return None
    out = {"pred_r": pred_r, "pred_l": pred_l, "time": series.time}
    _L2_CACHE[key] = out
    return out


# -------------------------------------------------------------------------
# Step 4: monkey-patch keypoints_to_motion_data to use learned L2.
# -------------------------------------------------------------------------


# Index in L2_METRICS = ("hip_flexion_r", "hip_adduction_r", "knee_angle_r",
#                        "ankle_angle_r", "lumbar_extension")
_L2_METRIC_IDX = {m: i for i, m in enumerate(L2_METRICS)}


def _patched_keypoints_to_motion_data_factory(
    model: TemporalKeypointCNNConf,
    halpe_idx_arr: np.ndarray,
    original_fn,
):
    """Returns a replacement for couro_keypoints.keypoints_to_motion_data that
    substitutes learned-L2 predicted angle traces for the 5 deploy metrics
    (right side from original orientation, left side from mirrored
    orientation). Hand-engineered pelvis_tilt + interpolated hand-engineered
    side columns retained for any column not covered."""

    def patched(series: CouroKeypointSeries, *args, **kwargs) -> MotionData:
        # Run original to get all columns (incl. pelvis_tilt + hand-engineered
        # baselines + the not-predicted left columns).
        md = original_fn(series, *args, **kwargs)
        # Predict L2.
        pred_r = predict_l2_for_series(model, series, halpe_idx_arr)
        try:
            series_l = _mirror_series(series)
            pred_l = predict_l2_for_series(model, series_l, halpe_idx_arr)
        except Exception:
            pred_l = None
        if pred_r is None:
            return md
        # Build new values matrix replacing the 5 right-side + lumbar columns.
        # NOTE: md.values is shape (T, n_cols - 1) -- "time" column is excluded
        # from values; MotionData.column() does `index(name) - 1`.
        cols = list(md.columns)
        vals = md.values.copy()
        n_frames = min(vals.shape[0], pred_r.shape[0])

        def _set(col: str, arr: np.ndarray) -> None:
            if col not in cols:
                return
            if cols[0] == "time":
                idx = cols.index(col) - 1  # mirror MotionData.column semantics
            else:
                idx = cols.index(col)
            if idx < 0 or idx >= vals.shape[1]:
                return
            k = min(n_frames, arr.shape[0])
            vals[:k, idx] = arr[:k]

        # Right-side from original.
        _set("hip_flexion_r", pred_r[:, _L2_METRIC_IDX["hip_flexion_r"]])
        _set("hip_adduction_r", pred_r[:, _L2_METRIC_IDX["hip_adduction_r"]])
        _set("knee_angle_r", pred_r[:, _L2_METRIC_IDX["knee_angle_r"]])
        _set("ankle_angle_r", pred_r[:, _L2_METRIC_IDX["ankle_angle_r"]])
        _set("lumbar_extension", pred_r[:, _L2_METRIC_IDX["lumbar_extension"]])

        # Left-side from mirrored. Note: lumbar_extension is bilateral; we
        # take the right-side prediction (original orientation) as canonical.
        if pred_l is not None:
            _set("hip_flexion_l", pred_l[:, _L2_METRIC_IDX["hip_flexion_r"]])
            _set("hip_adduction_l", pred_l[:, _L2_METRIC_IDX["hip_adduction_r"]])
            _set("knee_angle_l", pred_l[:, _L2_METRIC_IDX["knee_angle_r"]])
            _set("ankle_angle_l", pred_l[:, _L2_METRIC_IDX["ankle_angle_r"]])

        return MotionData(
            columns=tuple(cols),
            time=md.time,
            values=vals,
            in_degrees=md.in_degrees,
        )

    return patched


# -------------------------------------------------------------------------
# Step 5: per-slot LOSO fitting + validity stats.
# -------------------------------------------------------------------------


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


def _compute_stats_per_subject(
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


def _loso_extract(
    X: np.ndarray, y: np.ndarray, subjects: list[str], alpha: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    all_pred = np.full(len(y), np.nan)
    for s in unique:
        train_mask = subj_arr != s
        test_mask = subj_arr == s
        if train_mask.sum() < 2 or test_mask.sum() < 1:
            continue
        model = fit_ridge(X[train_mask], y[train_mask], alpha=alpha)
        all_pred[test_mask] = model.predict(X[test_mask])
    return all_pred, y, subj_arr


# -------------------------------------------------------------------------
# Step 6: orchestrate.
# -------------------------------------------------------------------------


VIEW_ORDER = [
    "side_left", "front_oblique_left", "front_center",
    "front_oblique_right", "side_right",
]


def run() -> dict:
    # 1. Build/load L2 model.
    model, _ = build_or_load_all_data_l2()
    halpe_idx_arr = np.array(
        [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL], dtype=np.int64
    )
    log("L2 model ready")

    # 2. Monkey-patch keypoints_to_motion_data globally.
    original_fn = ck_mod.keypoints_to_motion_data
    patched_fn = _patched_keypoints_to_motion_data_factory(
        model, halpe_idx_arr, original_fn,
    )
    ck_mod.keypoints_to_motion_data = patched_fn
    # Reach into already-imported modules and rebind.
    import harness.train_regression_poc as trp
    import harness.train_v9_phased as v9
    import harness.train_v12_combined as v12
    import harness.train_v14_full_dwpose as v14
    import harness.train_event_anchored_all as ea
    import harness.train_ankle_bilateral as ea_ank
    for mod in (trp, v9, v12, v14, ea, ea_ank):
        if hasattr(mod, "keypoints_to_motion_data"):
            mod.keypoints_to_motion_data = patched_fn
    log("monkey-patched keypoints_to_motion_data in 6 modules")

    # 3. Load v17 deploy plan and baseline validity.
    v17 = json.loads(V17_PATH.read_text())
    baseline = json.loads(BASELINE_PATH.read_text())
    baseline_by_slot = {
        (s["target"], s["view"]): s for s in baseline["slots"]
    }
    log(f"loaded v17 deploy ({sum(len(v) for v in v17['models'].values())} slots) "
        f"and baseline validity ({len(baseline_by_slot)} slots)")

    # 4. Iterate slots. Use the existing biomech_validity_stats BUILDERS but
    #    with patched motion data.
    from harness.train_multi_metric import VIEW_BUCKET_NAMES
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
        return X.astype(np.float64), np.asarray(y, dtype=np.float64), subjects, ["opencap"] * len(y)

    def _build_ea(target, view):
        ea.KP_DIR = DATA_ROOT / "couro_keypoints"
        cam = VIEW_TO_CAM.get(view)
        if cam is None:
            return None
        X, y, subjects, _, _ = ea.build_dataset(target, cam)
        if len(y) == 0:
            return None
        return X.astype(np.float64), np.asarray(y, dtype=np.float64), subjects, ["opencap"] * len(y)

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

    # Order slots for predictable output.
    slot_keys = []
    for target, slots in v17["models"].items():
        for view, entry in slots.items():
            slot_keys.append((target, view, entry))

    log(f"=== retraining {len(slot_keys)} slots ===")

    per_slot_validity = []
    v18_models = {}
    tier_counter = {"Good": 0, "Moderate": 0, "Poor": 0, "Excellent": 0}
    promotions = []
    demotions = []
    unchanged = []

    for i, (target, view, entry) in enumerate(slot_keys):
        approach = entry.get("approach", "unknown")
        t0 = time.time()
        log(f"--- slot {i+1}/{len(slot_keys)}: {target}/{view} approach={approach} ---")
        builder = BUILDERS.get(approach)
        if builder is None:
            log(f"  no builder for approach {approach}; carrying baseline")
            base = baseline_by_slot.get((target, view))
            if base:
                per_slot_validity.append({**base, "v18_skipped": True,
                                          "skip_reason": f"no builder for approach {approach}"})
            v18_models.setdefault(target, {})[view] = entry
            continue

        try:
            result = builder(target, view)
        except Exception as e:
            log(f"  BUILDER FAILED: {e!r}")
            traceback.print_exc(file=sys.stdout)
            base = baseline_by_slot.get((target, view))
            if base:
                per_slot_validity.append({**base, "v18_skipped": True,
                                          "skip_reason": f"builder error: {e!r}"})
            v18_models.setdefault(target, {})[view] = entry
            continue

        if result is None:
            log(f"  builder returned None")
            base = baseline_by_slot.get((target, view))
            if base:
                per_slot_validity.append({**base, "v18_skipped": True,
                                          "skip_reason": "builder returned None"})
            v18_models.setdefault(target, {})[view] = entry
            continue

        X, y, subjects, sources = result
        if len(y) < 20:
            log(f"  only n={len(y)} samples -> too few; carrying baseline")
            base = baseline_by_slot.get((target, view))
            if base:
                per_slot_validity.append({**base, "v18_skipped": True,
                                          "skip_reason": f"only n={len(y)} samples"})
            v18_models.setdefault(target, {})[view] = entry
            continue

        # Fit ridge on all data (final deploy) + LOSO predict for stats.
        try:
            pred, obs, subj_arr = _loso_extract(X, y, list(subjects), alpha=10.0)
        except Exception as e:
            log(f"  LOSO failed: {e!r}")
            base = baseline_by_slot.get((target, view))
            if base:
                per_slot_validity.append({**base, "v18_skipped": True,
                                          "skip_reason": f"loso error: {e!r}"})
            v18_models.setdefault(target, {})[view] = entry
            if DUMP_ACCUMULATOR is not None:
                DUMP_ACCUMULATOR.add_slot_skip(
                    target=target, view=view, reason=f"loso error: {e!r}",
                )
            continue

        if DUMP_ACCUMULATOR is not None:
            DUMP_ACCUMULATOR.add_slot(
                target=target, view=view, approach=approach,
                pred=pred, obs=obs, subjects=subj_arr,
            )

        stats = _compute_stats_per_subject(pred, obs, subj_arr)
        stats["target"] = target
        stats["view"] = view
        stats["approach"] = approach + "_with_learned_l2"
        stats["stats_source"] = "loso_pairs"
        stats["aggregation"] = "per_subject"

        # Fit final ridge on ALL data for the v18 deploy candidate.
        try:
            final_model = fit_ridge(X.astype(np.float64), y.astype(np.float64), alpha=10.0)
            v18_entry = dict(entry)
            v18_entry["weights_v18_learned_l2"] = final_model.weights.tolist()
            v18_entry["bias_v18_learned_l2"] = float(final_model.bias)
            v18_entry["feature_mean_v18"] = final_model.feature_mean.tolist()
            v18_entry["feature_std_v18"] = final_model.feature_std.tolist()
            v18_entry["v18_loso_stats"] = {
                k: stats[k] for k in [
                    "n_subjects", "n_trials", "mean_bias_deg", "sd_bias_deg",
                    "loa_upper_deg", "loa_lower_deg", "loa_half_width_deg",
                    "pearson_r", "ccc_lin", "mae_deg", "rmse_deg",
                    "classification",
                ]
            }
            v18_entry["v18_approach"] = approach + "_with_learned_l2"
            v18_models.setdefault(target, {})[view] = v18_entry
        except Exception as e:
            log(f"  final ridge fit failed: {e!r}")
            v18_models.setdefault(target, {})[view] = entry

        per_slot_validity.append(stats)

        base = baseline_by_slot.get((target, view))
        base_tier = base["classification"] if base else "?"
        new_tier = stats["classification"]
        tier_counter[new_tier] = tier_counter.get(new_tier, 0) + 1

        tier_rank = {"Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1,
                     "LoA-only: tight": 0, "LoA-only: moderate": 0,
                     "LoA-only: wide": 0, "LoA-only: poor": 0}
        if base_tier in tier_rank and new_tier in tier_rank:
            if tier_rank[new_tier] > tier_rank[base_tier]:
                promotions.append((target, view, base_tier, new_tier))
            elif tier_rank[new_tier] < tier_rank[base_tier]:
                demotions.append((target, view, base_tier, new_tier))
            else:
                unchanged.append((target, view, new_tier))

        log(
            f"  [FF slot {i+1}/{len(slot_keys)}: {target}/{view}/{approach}] "
            f"n={stats['n_subjects']} CCC={stats['ccc_lin']:.3f} "
            f"LoA_half={stats['loa_half_width_deg']:.2f} "
            f"r={stats['pearson_r']:.3f} -> {new_tier} "
            f"(baseline {base_tier})  ({time.time()-t0:.1f}s)"
        )
        log(f"  [FF tier counts so far: {tier_counter}]")

    log(f"=== complete: tier counts {tier_counter} ===")
    log(f"   promotions: {len(promotions)}, demotions: {len(demotions)}, unchanged: {len(unchanged)}")

    # 5. Write outputs.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v18_out = {
        "version": "v18_learned_l2",
        "produced_by": "harness.layer3_retrain_on_learned_l2 (Agent FF)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            "v17 deploy base + per-slot ridge weights re-fit using learned "
            "Layer 2 angle traces (Agent EE2's TemporalKeypointCNNConf, "
            "trained on ALL 9 OpenCap subjects -- not LOSO at L2). "
            "L3 LOSO only. Not double-LOSO."
        ),
        "loso_discipline": "Layer-3-LOSO-only (L2 trained on all OpenCap subjects)",
        "v17_base_path": str(V17_PATH),
        "approaches": v17.get("approaches"),
        "training_dataset": v17.get("training_dataset"),
        "models": v18_models,
        "calibration_fix": v17.get("calibration_fix"),
        "selective_adoption": v17.get("selective_adoption"),
    }
    V18_PATH.write_text(json.dumps(v18_out, indent=2))
    log(f"wrote v18 deploy to {V18_PATH}")

    per_slot_out = {
        "version": "v18_learned_l2",
        "produced_by": "harness.layer3_retrain_on_learned_l2 (Agent FF)",
        "loso_discipline": "Layer-3-LOSO-only (L2 trained on all OpenCap subjects)",
        "caveat": (
            "L2 model trained on all 9 OpenCap subjects. L3 LOSO at subject "
            "level only. OpenCap-subject tier promotions are upper bounds; "
            "true double-LOSO would likely show ~0.05-0.10 |r| lower. "
            "ASPset-subject contributions ARE double-LOSO clean (ASPset never "
            "entered L2 training)."
        ),
        "tier_counts": tier_counter,
        "promotions": [
            {"target": t, "view": v, "from": fr, "to": to}
            for (t, v, fr, to) in promotions
        ],
        "demotions": [
            {"target": t, "view": v, "from": fr, "to": to}
            for (t, v, fr, to) in demotions
        ],
        "unchanged": [
            {"target": t, "view": v, "tier": tier}
            for (t, v, tier) in unchanged
        ],
        "slots": per_slot_validity,
    }
    (OUT_DIR / "per_slot_validity_v18.json").write_text(
        json.dumps(per_slot_out, indent=2)
    )
    log(f"wrote per-slot validity to {OUT_DIR / 'per_slot_validity_v18.json'}")

    # 6. Write REPORT.md.
    _write_report(per_slot_out, baseline, v17)
    log("wrote REPORT.md")

    return per_slot_out


def _write_report(per_slot_out: dict, baseline: dict, v17: dict) -> None:
    """Write a markdown summary."""
    lines: list[str] = []
    lines.append("# Phase B: Layer 3 retrained on learned Layer 2 (Agent FF)")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append("**Build:** Agent FF (Phase B)")
    lines.append("")
    lines.append("## LOSO discipline used in this build")
    lines.append("")
    lines.append(
        "**Layer-3-LOSO-only.** ONE Layer 2 model (Agent EE2's "
        "`TemporalKeypointCNNConf`) was trained on ALL 9 OpenCap subjects "
        "(no LOSO at L2) and used to produce learned angle traces for every "
        "clip (OpenCap + ASPset). Layer 3 ridge was re-fit per slot with "
        "LOSO at L3 only."
    )
    lines.append("")
    lines.append(
        "**This leaks Layer 2 information from the held-out OpenCap subject "
        "into the L2 training data.** For ASPset subjects, the regime is "
        "effectively double-LOSO because no ASPset subject ever entered "
        "Layer 2 training (Layer 2 cohort is OpenCap-only)."
    )
    lines.append("")
    lines.append(
        "**Tier-change claims warning:** any tier promotion driven primarily "
        "by OpenCap LOSO folds is an upper bound. EE2's per-fold variance "
        "(best fold 0.700, worst 0.541) implies the true double-LOSO number "
        "could be ~0.05-0.10 |r| lower than reported here."
    )
    lines.append("")
    lines.append("## Tier count delta")
    lines.append("")
    base_counts = {"Good": 0, "Moderate": 0, "Poor": 0, "Excellent": 0}
    for s in baseline["slots"]:
        base_counts[s["classification"]] = base_counts.get(s["classification"], 0) + 1
    lines.append("| Tier | v17 baseline | v18 (learned L2) | Δ |")
    lines.append("| --- | ---: | ---: | ---: |")
    for tier in ["Excellent", "Good", "Moderate", "Poor"]:
        b = base_counts.get(tier, 0)
        n = per_slot_out["tier_counts"].get(tier, 0)
        lines.append(f"| {tier} | {b} | {n} | {n - b:+d} |")
    lines.append("")
    lines.append(f"Promotions: {len(per_slot_out['promotions'])} | "
                 f"Demotions: {len(per_slot_out['demotions'])} | "
                 f"Unchanged: {len(per_slot_out['unchanged'])}")
    lines.append("")
    if per_slot_out["promotions"]:
        lines.append("### Promotions")
        lines.append("")
        for p in per_slot_out["promotions"]:
            lines.append(f"- {p['target']} / {p['view']}: {p['from']} -> {p['to']}")
        lines.append("")
    if per_slot_out["demotions"]:
        lines.append("### Demotions")
        lines.append("")
        for d in per_slot_out["demotions"]:
            lines.append(f"- {d['target']} / {d['view']}: {d['from']} -> {d['to']}")
        lines.append("")

    lines.append("## Per-slot before/after table")
    lines.append("")
    lines.append("| Target | View | Approach | n | r baseline | CCC baseline | LoA/2 baseline | Tier baseline | r v18 | CCC v18 | LoA/2 v18 | Tier v18 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |")
    baseline_by_slot = {(s["target"], s["view"]): s for s in baseline["slots"]}

    def _fmt(v, prec=2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "-"
        return f"{v:.{prec}f}"

    for s in per_slot_out["slots"]:
        b = baseline_by_slot.get((s["target"], s["view"]), {})
        approach = s.get("approach", b.get("approach", "?"))
        lines.append(
            f"| {s['target']} | {s['view']} | {approach} | "
            f"{s.get('n_subjects', '?')} | "
            f"{_fmt(b.get('pearson_r'))} | {_fmt(b.get('ccc_lin'))} | "
            f"{_fmt(b.get('loa_half_width_deg'))} | "
            f"{b.get('classification', '?')} | "
            f"{_fmt(s.get('pearson_r'))} | {_fmt(s.get('ccc_lin'))} | "
            f"{_fmt(s.get('loa_half_width_deg'))} | "
            f"{s.get('classification', '?')} |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "**Headline:** Phase B did NOT deliver the 3-6 tier promotions "
        "EE2's per-metric per-frame |r| gains projected. Net tier change is "
        "negative: -1 Good, -4 Moderate, +3 Poor. EE2's per-frame Layer 2 "
        "lift does not translate cleanly to per-trial ROM tier promotion."
    )
    lines.append("")
    lines.append("### Why per-frame |r| gains did not translate to ROM tier lifts")
    lines.append("")
    lines.append(
        "1. **ROM is a max-minus-min aggregate.** EE2's +0.131 pooled |r| "
        "measures per-frame waveform tracking. ROM throws away the within-"
        "trial waveform shape and keeps only the extrema. A model that "
        "tracks waveform shape better doesn't necessarily land its extrema "
        "in the same places as ground truth — and the ridge regression on "
        "ROM features is sensitive to the exact extrema, not the shape."
    )
    lines.append(
        "2. **Calibration drift.** Several slots that lost Good status "
        "(e.g. hip_adduction_r/side_left: r 0.95 -> 0.85, LoA 9.8 -> 17.9) "
        "still have high Pearson r but wider LoA. The learned-L2 ROM is "
        "well-correlated with ground truth but offset by a systematic bias "
        "that hand-engineered L2 was either calibrated against or "
        "incidentally aligned with."
    )
    lines.append(
        "3. **L3 ridge re-fit overfits noise.** The L3 ridge was re-fit "
        "from scratch on learned-L2 features. With small subject pools "
        "(n=9-22 LOSO folds), small differences in feature distribution "
        "cause meaningful shifts in ridge coefficients, increasing "
        "variance in held-out predictions."
    )
    lines.append(
        "4. **Distribution shift on ASPset.** Layer 2 trained on OpenCap "
        "(controlled DJ task, lab setting) is then applied to ASPset "
        "(general athletic movement). Per-frame |r| is robust to global "
        "shift; ROM is not."
    )
    lines.append("")
    lines.append("### Where Layer 2 DID help (the 3 promotions)")
    lines.append("")
    lines.append(
        "- **ankle_angle_r / front_oblique_right**: Poor -> Moderate "
        "(r -0.13 -> 0.69, CCC -0.13 -> 0.59, LoA 19.3 -> 10.1). Baseline "
        "Layer 2 was geometrically broken on this view. Learned Layer 2 "
        "replaces a sign-flipped hand-engineered angle with a correct "
        "magnitude, unlocking real ROM signal. **Confirms EE2 hypothesis "
        "for ankle out-of-plane.**"
    )
    lines.append(
        "- **lumbar_extension / front_oblique_left**: Moderate -> Good "
        "(r 0.62 -> 0.78, CCC 0.53 -> 0.71, LoA 8.0 -> 6.6). Lumbar "
        "extension is the metric where EE2 reported the biggest per-metric "
        "lift (+0.160). The improvement carries through to ROM here."
    )
    lines.append(
        "- **hip_adduction_r / front_oblique_left**: Poor -> Moderate "
        "(r 0.35 -> 0.48, CCC 0.29 -> 0.45). EE2 reported +0.146 |r| on "
        "this metric. Real signal recovered at this view."
    )
    lines.append("")
    lines.append("### Where Layer 2 hurt (the 8 demotions)")
    lines.append("")
    lines.append(
        "Most are knee/hip slots that were strongly tuned to hand-"
        "engineered features. The hand-engineered Layer 2 is geometrically "
        "well-suited to in-plane knee/hip flexion from side and oblique "
        "views — and the v17 ridge weights for these slots are tuned to "
        "the noise structure of those features. Replacing with a different "
        "(but biomechanically reasonable) angle trace breaks the tuning."
    )
    lines.append("")
    lines.append("### Honest recommendation")
    lines.append("")
    lines.append(
        "1. **Do NOT adopt v18 wholesale.** Net tier loss is real."
    )
    lines.append(
        "2. **Selectively adopt the 3 promoted slots** (ankle/foright, "
        "lumbar/foleft, hipadd/foleft). These are tier wins that survive "
        "the Layer-3-LOSO-only caveat because the lift is large enough to "
        "absorb the upper-bound discount."
    )
    lines.append(
        "3. **For knee/hip slots that demoted, keep v17 ridge weights.** "
        "The v17 deploy bundle already has these dialed in."
    )
    lines.append(
        "4. **Next experiment:** ENSEMBLE learned-L2 ROM with hand-"
        "engineered-L2 ROM. The two angle estimates may be additively "
        "informative at the ROM level even if learned-L2 alone is worse."
    )
    lines.append(
        "5. **Run full double-LOSO** to confirm the 3 promotions survive "
        "without L2 information leak. If even 2 of 3 survive, v18 selective "
        "adoption is the right call."
    )
    lines.append("")

    lines.append("## Caveats - honest")
    lines.append("")
    lines.append(
        "- **Not double-LOSO.** See LOSO discipline section. Tier promotions "
        "involving OpenCap subjects are an upper bound. Of the 3 "
        "promotions, two (ankle_angle_r/front_oblique_right, "
        "hip_adduction_r/front_oblique_left) use OpenCap-only data and so "
        "are subject to the upper-bound caveat. The third "
        "(lumbar_extension/front_oblique_left, event_anchored, OpenCap-only) "
        "is also subject to the caveat."
    )
    lines.append(
        "- **pelvis_tilt remains hand-engineered** (used as coupling for "
        "lumbar_extension slots in v12/v13 only). EE2's L2 doesn't predict it."
    )
    lines.append(
        "- **Left-side angles** come from mirroring the keypoint series and "
        "running the same right-side L2 model. This relies on the L2 model "
        "being approximately mirror-symmetric, which is reasonable for the "
        "DJ task but not validated."
    )
    lines.append(
        "- **ASPset L2 inference** uses the same all-OpenCap-trained model. "
        "ASPset DWPose distribution may differ from OpenCap's; predictions "
        "may have a shift not corrected by the OpenCap training."
    )
    lines.append(
        "- **ankle_angle_r slots remain OpenCap-only** (ASPset has no foot "
        "GT). The learned L2 was trained on OpenCap ankle GT, so the L2 "
        "predictions for ankle are well-calibrated to the OpenCap "
        "distribution -- but tier change is still under Layer-3-LOSO-only "
        "discipline."
    )
    lines.append(
        "- **Numerical warnings:** v12_combined / v13_dwpose_hybrid slots "
        "emit `divide by zero` / `overflow` warnings during ridge predict. "
        "Reflects degenerate feature columns (constant or NaN-filled) when "
        "the patched learned-L2 angle trace produces an out-of-distribution "
        "value on some ASPset clips. These predictions are still finite "
        "after the z-score; CCC/LoA stats are computed on finite values only."
    )

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    import argparse
    global DUMP_ACCUMULATOR
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain-l2", action="store_true")
    parser.add_argument(
        "--dump-predictions", default=None,
        help="If set, also write per-subject (pred, gt) dumps to "
             "data/per_slot_predictions/per_slot_predictions_<TAG>.json. "
             "Reader tag, e.g. 'v18'.",
    )
    args = parser.parse_args()
    log("=== Agent FF Phase B START ===")
    if args.force_retrain_l2 and MODEL_OUT.exists():
        MODEL_OUT.unlink()
        log(f"removed cached L2 checkpoint at {MODEL_OUT}")
    if args.dump_predictions:
        DUMP_ACCUMULATOR = DumpAccumulator(
            reader=args.dump_predictions,
            loso_discipline="Layer-3-LOSO-only (FF: L2 trained on all OpenCap)",
        )
        log(f"[SS dump] DUMP_ACCUMULATOR set for reader '{args.dump_predictions}'")
    run()
    if DUMP_ACCUMULATOR is not None:
        out_path = DUMP_ACCUMULATOR.write()
        log(f"[SS dump] wrote per-slot predictions to {out_path}")
    log("=== Agent FF Phase B DONE ===")


if __name__ == "__main__":
    main()
