"""Layer 3 retrain on VideoPose3D-derived Layer 2 angles -> v44 (Agent TT).

Sister script to ``layer3_retrain_on_combined_l2`` (KK -> v23) and
``layer3_retrain_on_persource_perframe_l2`` (MM -> v26): same Phase B
pipeline, but the Layer 2 angle stream comes from VideoPose3D 2D->3D
lifting (Pavllo et al. 2019, FAIR, Apache 2.0) instead of from a
keypoint CNN trained on OpenCap.

Pipeline
--------
1. Run ``learned_layer2_videopose3d.run_inference_all_clips`` to populate
   per-clip 3D-derived angles (cached on disk under
   ``data/videopose3d_layer2/per_clip_angles.json``).
2. Monkey-patch ``couro_keypoints.keypoints_to_motion_data`` so the 5
   deploy angles come from the VideoPose3D-derived 3D angles for the
   matching clip. Hand-engineered pelvis_tilt + interpolated left/right
   side columns retained for any column not covered. ASPset clips are
   not in our cache and fall through to the original 2D function.
3. For each v17 deploy slot, call the matching dataset builder (v9/v12/
   v13/v14/event_anchored/event_anchored_bilateral) -- features are
   rebuilt with the patched L2.
4. LOSO ridge per slot. Compute Bland-Altman + CCC. Classify tier.

LOSO discipline
---------------
VideoPose3D was pretrained on Human3.6M (Ionescu et al. 2014), whose 11
subjects do not overlap with OpenCap or ASPset. The L2 lifter is therefore
disjoint from the L3 LOSO pool, so this is a clean Layer-3-LOSO setup --
no double-LOSO violation as in v18/v20/v23.

Outputs
-------
- ``models/`` -- no new checkpoint; VideoPose3D model is loaded from
  pretrained checkpoint at inference time
- ``data/layer3_retrain_videopose3d/per_slot_validity_v44.json``
- ``results/deploy_ready_models_v44_videopose3d.json``
- ``data/layer3_retrain_videopose3d/REPORT.md``
"""
from __future__ import annotations

import builtins
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Final

import numpy as np

# numpy 2.x compat (matches the other retrain scripts).
for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness import couro_keypoints as ck_mod
from harness.couro_keypoints import CouroKeypointSeries
from harness.parsers import MotionData
from harness.train_regression_poc import fit_ridge
from harness.per_slot_prediction_dump import DumpAccumulator
from harness.learned_layer2_videopose3d import (
    DEPLOY_METRICS as L2_METRICS,
    angles_for_clip,
    angles_from_h36m_3d,
    load_opencap_clip_kp,
    run_inference_all_clips,
)
from harness.videopose3d import lift_sequence, load_pretrained


# Same hook as FF: SS dump driver can override this.
DUMP_ACCUMULATOR: DumpAccumulator | None = None


DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
LAB: Final[Path] = DATA_ROOT / "LabValidation_withVideos"
OPENCAP_KP: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"
ASPSET_DWPOSE_KP: Final[Path] = DATA_ROOT / "aspset510_dwpose_keypoints"

OUT_DIR: Final[Path] = DATA_ROOT / "layer3_retrain_videopose3d"
V17_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v17_selective.json"
V44_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v44_videopose3d.json"
PER_SLOT_V44_OUT: Final[Path] = OUT_DIR / "per_slot_validity_v44.json"
BASELINE_PATH: Final[Path] = (
    DATA_ROOT / "biomech_validity_stats" / "per_slot_validity.json"
)


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[TT t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Build/load 3D-angle cache.
# -------------------------------------------------------------------------


_ANGLE_CACHE: dict[str, dict] = {}
_VIDEOPOSE3D_MODEL = None
_CACHE_LOADED = False


def _ensure_cache() -> dict[str, dict]:
    """Lazily build (or load) the cached per-clip 3D angles."""
    global _ANGLE_CACHE, _CACHE_LOADED
    if not _CACHE_LOADED:
        log("loading VideoPose3D cache via run_inference_all_clips ...")
        _ANGLE_CACHE = run_inference_all_clips(device="cpu")
        _CACHE_LOADED = True
    return _ANGLE_CACHE


def _ensure_model():
    """Lazily load the VideoPose3D pretrained model for live inference."""
    global _VIDEOPOSE3D_MODEL
    if _VIDEOPOSE3D_MODEL is None:
        ck = REPO_ROOT / "models" / "videopose3d_h36m.bin"
        _VIDEOPOSE3D_MODEL = load_pretrained(ck, device="cpu")
    return _VIDEOPOSE3D_MODEL


def _angles_from_series(
    series: CouroKeypointSeries, mirror: bool,
) -> np.ndarray | None:
    """Lift one CouroKeypointSeries to (T, 5) 3D-derived angles.

    Used as a fallback for ASPset clips not in our pre-built cache.
    Reuses VideoPose3D pretrained checkpoint, runs lift_sequence on the
    series' 2D Halpe-26 keypoints, then derives the 5 deploy angles.
    """
    if series.xy.shape[0] < 3 or series.width <= 0 or series.height <= 0:
        return None
    model = _ensure_model()
    kp_2d = np.nan_to_num(series.xy, nan=0.0).astype(np.float32)
    try:
        pos_3d = lift_sequence(
            model, kp_2d, width=series.width, height=series.height,
            device="cpu",
        )
        return angles_from_h36m_3d(pos_3d, mirror=mirror)
    except Exception:
        return None


# -------------------------------------------------------------------------
# Clip-id resolution for the monkey-patched motion-data function.
# -------------------------------------------------------------------------
#
# FF's monkey-patch sees only a CouroKeypointSeries, not the path. We bridge
# by tracking the "current clip id" via a module-level setter that the
# builders' kp_path iteration calls right before loading the series. To
# avoid intrusively changing every builder, we instead monkey-patch
# load_couro_output: it captures the kp_path stem in a thread-local before
# returning the parsed series. The patched keypoints_to_motion_data then
# reads the most recent stem.
# -------------------------------------------------------------------------


_CURRENT_CLIP_ID: str | None = None


def _patched_load_couro_output_factory(original_fn):
    def patched(path):
        global _CURRENT_CLIP_ID
        _CURRENT_CLIP_ID = Path(path).stem
        return original_fn(path)
    return patched


# -------------------------------------------------------------------------
# Patched keypoints_to_motion_data: inject VideoPose3D-derived angles.
# -------------------------------------------------------------------------


_L2_METRIC_IDX = {m: i for i, m in enumerate(L2_METRICS)}


def _patched_motion_data_factory(original_fn):
    cache = _ensure_cache()

    def patched(series: CouroKeypointSeries, *args, **kwargs) -> MotionData:
        md = original_fn(series, *args, **kwargs)
        # Lookup 3D angles by clip id.
        clip_id = _CURRENT_CLIP_ID
        pred_r = pred_l = None
        if clip_id is not None and clip_id in cache:
            rec = cache[clip_id]
            pred_r = np.asarray(rec["pred_r"], dtype=np.float32)
            pred_l = np.asarray(rec["pred_l"], dtype=np.float32)
        else:
            # ASPset (or any cache miss) -- live inference.
            pred_r = _angles_from_series(series, mirror=False)
            pred_l = _angles_from_series(series, mirror=True)
        if pred_r is None:
            return md

        cols = list(md.columns)
        vals = md.values.copy()
        n_frames = min(vals.shape[0], pred_r.shape[0])

        def _set(col: str, arr: np.ndarray) -> None:
            if col not in cols:
                return
            if cols[0] == "time":
                idx = cols.index(col) - 1
            else:
                idx = cols.index(col)
            if idx < 0 or idx >= vals.shape[1]:
                return
            k = min(n_frames, arr.shape[0])
            vals[:k, idx] = arr[:k]

        _set("hip_flexion_r", pred_r[:, _L2_METRIC_IDX["hip_flexion_r"]])
        _set("hip_adduction_r", pred_r[:, _L2_METRIC_IDX["hip_adduction_r"]])
        _set("knee_angle_r", pred_r[:, _L2_METRIC_IDX["knee_angle_r"]])
        _set("ankle_angle_r", pred_r[:, _L2_METRIC_IDX["ankle_angle_r"]])
        _set("lumbar_extension", pred_r[:, _L2_METRIC_IDX["lumbar_extension"]])

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
# Stats helpers (copied from FF for self-containment).
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
    X: np.ndarray, y: np.ndarray, subjects: list[str], alpha: float = 10.0,
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
# Slot iteration -- mirrors FF.run().
# -------------------------------------------------------------------------


def run() -> dict:
    """Drive the FF-style L3 retrain pipeline with VideoPose3D L2."""
    _ensure_cache()
    log("VideoPose3D cache loaded")

    # Monkey-patch load_couro_output to capture clip_id, and
    # keypoints_to_motion_data to inject 3D-derived angles.
    original_load = ck_mod.load_couro_output
    original_kd = ck_mod.keypoints_to_motion_data
    ck_mod.load_couro_output = _patched_load_couro_output_factory(original_load)
    ck_mod.keypoints_to_motion_data = _patched_motion_data_factory(original_kd)

    # Rebind into already-imported builder modules.
    import harness.train_regression_poc as trp
    import harness.train_v9_phased as v9
    import harness.train_v12_combined as v12
    import harness.train_v14_full_dwpose as v14
    import harness.train_event_anchored_all as ea
    import harness.train_ankle_bilateral as ea_ank
    for mod in (trp, v9, v12, v14, ea, ea_ank):
        if hasattr(mod, "keypoints_to_motion_data"):
            mod.keypoints_to_motion_data = ck_mod.keypoints_to_motion_data
        if hasattr(mod, "load_couro_output"):
            mod.load_couro_output = ck_mod.load_couro_output
    log("monkey-patched keypoints_to_motion_data + load_couro_output in 6 modules")

    v17 = json.loads(V17_PATH.read_text())
    baseline = json.loads(BASELINE_PATH.read_text())
    baseline_by_slot = {(s["target"], s["view"]): s for s in baseline["slots"]}
    log(
        f"loaded v17 deploy ({sum(len(v) for v in v17['models'].values())} slots) "
        f"and baseline ({len(baseline_by_slot)} slots)"
    )

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
        return (X.astype(np.float64), np.asarray(y, dtype=np.float64),
                subjects, ["opencap"] * len(y))

    def _build_ea(target, view):
        ea.KP_DIR = DATA_ROOT / "couro_keypoints"
        cam = VIEW_TO_CAM.get(view)
        if cam is None:
            return None
        X, y, subjects, _, _ = ea.build_dataset(target, cam)
        if len(y) == 0:
            return None
        return (X.astype(np.float64), np.asarray(y, dtype=np.float64),
                subjects, ["opencap"] * len(y))

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

    slot_keys = []
    for target, slots in v17["models"].items():
        for view, entry in slots.items():
            slot_keys.append((target, view, entry))

    log(f"=== retraining {len(slot_keys)} slots with VideoPose3D L2 ===")

    per_slot_validity: list[dict] = []
    v44_models: dict[str, dict] = {}
    tier_counter = {"Good": 0, "Moderate": 0, "Poor": 0, "Excellent": 0}
    promotions: list[tuple] = []
    demotions: list[tuple] = []
    unchanged: list[tuple] = []

    for i, (target, view, entry) in enumerate(slot_keys):
        approach = entry.get("approach", "unknown")
        t0 = time.time()
        log(
            f"--- slot {i+1}/{len(slot_keys)}: {target}/{view} "
            f"approach={approach} ---"
        )
        builder = BUILDERS.get(approach)
        if builder is None:
            log(f"  no builder for approach {approach}; carrying baseline")
            base = baseline_by_slot.get((target, view))
            if base:
                per_slot_validity.append({**base, "v44_skipped": True,
                                          "skip_reason": f"no builder for approach {approach}"})
            v44_models.setdefault(target, {})[view] = entry
            continue

        try:
            result = builder(target, view)
        except Exception as e:
            log(f"  BUILDER FAILED: {e!r}")
            traceback.print_exc(file=sys.stdout)
            base = baseline_by_slot.get((target, view))
            if base:
                per_slot_validity.append({**base, "v44_skipped": True,
                                          "skip_reason": f"builder error: {e!r}"})
            v44_models.setdefault(target, {})[view] = entry
            continue

        if result is None:
            log("  builder returned None")
            base = baseline_by_slot.get((target, view))
            if base:
                per_slot_validity.append({**base, "v44_skipped": True,
                                          "skip_reason": "builder returned None"})
            v44_models.setdefault(target, {})[view] = entry
            continue

        X, y, subjects, sources = result
        if len(y) < 20:
            log(f"  only n={len(y)} samples -> too few; carrying baseline")
            base = baseline_by_slot.get((target, view))
            if base:
                per_slot_validity.append({**base, "v44_skipped": True,
                                          "skip_reason": f"only n={len(y)} samples"})
            v44_models.setdefault(target, {})[view] = entry
            continue

        try:
            pred, obs, subj_arr = _loso_extract(
                X, y, list(subjects), alpha=10.0,
            )
        except Exception as e:
            log(f"  LOSO failed: {e!r}")
            base = baseline_by_slot.get((target, view))
            if base:
                per_slot_validity.append({**base, "v44_skipped": True,
                                          "skip_reason": f"loso error: {e!r}"})
            v44_models.setdefault(target, {})[view] = entry
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
        stats["approach"] = approach + "_with_videopose3d_l2"
        stats["stats_source"] = "loso_pairs"
        stats["aggregation"] = "per_subject"

        try:
            final_model = fit_ridge(
                X.astype(np.float64), y.astype(np.float64), alpha=10.0,
            )
            v44_entry = dict(entry)
            v44_entry["weights_v44_videopose3d_l2"] = final_model.weights.tolist()
            v44_entry["bias_v44_videopose3d_l2"] = float(final_model.bias)
            v44_entry["feature_mean_v44"] = final_model.feature_mean.tolist()
            v44_entry["feature_std_v44"] = final_model.feature_std.tolist()
            v44_entry["v44_loso_stats"] = {
                k: stats[k] for k in [
                    "n_subjects", "n_trials", "mean_bias_deg", "sd_bias_deg",
                    "loa_upper_deg", "loa_lower_deg", "loa_half_width_deg",
                    "pearson_r", "ccc_lin", "mae_deg", "rmse_deg",
                    "classification",
                ]
            }
            v44_entry["v44_approach"] = approach + "_with_videopose3d_l2"
            v44_models.setdefault(target, {})[view] = v44_entry
        except Exception as e:
            log(f"  final ridge fit failed: {e!r}")
            v44_models.setdefault(target, {})[view] = entry

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
            f"  [TT slot {i+1}/{len(slot_keys)}: {target}/{view}/{approach}] "
            f"n={stats['n_subjects']} CCC={stats['ccc_lin']:.3f} "
            f"LoA_half={stats['loa_half_width_deg']:.2f} "
            f"r={stats['pearson_r']:.3f} -> {new_tier} "
            f"(baseline {base_tier})  ({time.time()-t0:.1f}s)"
        )

    log(f"=== complete: tier counts {tier_counter} ===")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v44_out = {
        "version": "v44_videopose3d_l2",
        "produced_by": "harness.layer3_retrain_on_videopose3d_l2 (Agent TT)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            "v17 deploy base + per-slot ridge weights re-fit using "
            "VideoPose3D-lifted 3D-derived Layer 2 angle traces "
            "(Pavllo et al. 2019, FAIR, Apache 2.0). VideoPose3D is "
            "pretrained on Human3.6M (Ionescu et al. 2014) -- 11 subjects "
            "disjoint from the L3 LOSO pool, so L3 LOSO is clean."
        ),
        "loso_discipline": (
            "Layer-3-LOSO (clean -- VideoPose3D L2 pretraining cohort "
            "Human3.6M does not overlap with OpenCap or ASPset)"
        ),
        "v17_base_path": str(V17_PATH),
        "approaches": v17.get("approaches"),
        "training_dataset": v17.get("training_dataset"),
        "l2_model": "videopose3d_h36m.bin",
        "l2_model_license": "Apache-2.0",
        "models": v44_models,
        "calibration_fix": v17.get("calibration_fix"),
        "selective_adoption": v17.get("selective_adoption"),
    }
    V44_PATH.write_text(json.dumps(v44_out, indent=2))
    log(f"wrote v44 deploy to {V44_PATH}")

    per_slot_out = {
        "version": "v44_videopose3d_l2",
        "produced_by": "harness.layer3_retrain_on_videopose3d_l2 (Agent TT)",
        "loso_discipline": (
            "Layer-3-LOSO (clean) -- VideoPose3D pretrained on Human3.6M "
            "(disjoint from OpenCap and ASPset)"
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
    PER_SLOT_V44_OUT.write_text(json.dumps(per_slot_out, indent=2))
    log(f"wrote per-slot validity to {PER_SLOT_V44_OUT}")

    return per_slot_out


def main() -> None:
    log("=== Agent TT Phase B START (v44 VideoPose3D L2) ===")
    run()
    log("=== Agent TT Phase B DONE ===")


if __name__ == "__main__":
    main()
