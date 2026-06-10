"""Agent UU v46: per-slot validity under Layer 1 flip-TTA keypoints.

The FIRST build to alter Layer 1. We re-run the SAME hand-engineered /
combined readers that v45 uses, but swap the OpenCap DWPose keypoint source
from ``opencap_dwpose_keypoints`` to ``dwpose_flip_tta_keypoints`` (test-time
horizontal-flip augmented). Everything downstream -- angle computation
(couro_keypoints.keypoints_to_motion_data), feature extraction, LOSO ridge --
is byte-identical. Any CCC/LoA change is attributable purely to Layer 1.

For each deploy slot we compute, with subject-level LOSO at L3:
  * baseline  : reader on data/opencap_dwpose_keypoints (recomputed here,
                NOT read from a cached number -- identical code path to TTA)
  * flip_tta  : same reader on data/dwpose_flip_tta_keypoints

Slots whose v17 reader does NOT consume the DWPose cache (v9_phased,
event_anchored, event_anchored_bilateral read the Couro RTMPose cache
``couro_keypoints``) cannot be tested by this L1 swap. They are emitted with
``tta_applicable: false`` and fall back to the v45 pick in build_v47.

Output: data/layer3_retrain_flip_tta/per_slot_validity_v46.json
        data/layer3_retrain_flip_tta/per_clip_predictions_v46.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
ORIG_DWPOSE: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"
TTA_DWPOSE: Final[Path] = DATA_ROOT / "dwpose_flip_tta_keypoints"
ASPSET_DWPOSE: Final[Path] = DATA_ROOT / "aspset510_dwpose_keypoints"

OUT_DIR: Final[Path] = DATA_ROOT / "layer3_retrain_flip_tta"
PER_SLOT_OUT: Final[Path] = OUT_DIR / "per_slot_validity_v46.json"
PER_CLIP_OUT: Final[Path] = OUT_DIR / "per_clip_predictions_v46.json"
V17_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v17_selective.json"

ALPHA: Final[float] = 10.0

# Readers that read the OpenCap DWPose cache (so flip-TTA applies).
DWPOSE_READERS: Final[set[str]] = {"v14_full_dwpose", "v13_dwpose_hybrid"}


def log(msg: str) -> None:
    print(f"[v46] {msg}", flush=True)


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


def _loso_predict(X: np.ndarray, y: np.ndarray, subjects: list[str]) -> np.ndarray:
    from harness.train_regression_poc import fit_ridge
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    pred = np.full(len(y), np.nan)
    for s in unique:
        tr = subj_arr != s
        te = subj_arr == s
        if tr.sum() < 2 or te.sum() < 1:
            continue
        model = fit_ridge(X[tr], y[tr], alpha=ALPHA)
        pred[te] = model.predict(X[te])
    return pred


def _stats_per_subject(pred: np.ndarray, obs: np.ndarray,
                       subjects: np.ndarray) -> dict:
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred, obs, subjects = pred[mask], obs[mask], subjects[mask]
    n_trials = int(len(pred))
    unique = sorted(set(subjects.tolist()))
    n_subj = len(unique)
    if n_subj < 2:
        return {"n_subjects": n_subj, "n_trials": n_trials,
                "ccc_lin": float("nan"), "pearson_r": float("nan"),
                "loa_half_width_deg": float("nan"), "mean_bias_deg": float("nan"),
                "mae_deg": float("nan"), "classification": "Poor"}
    agg_p = np.array([float(np.mean(pred[subjects == s])) for s in unique])
    agg_o = np.array([float(np.mean(obs[subjects == s])) for s in unique])
    diffs = agg_p - agg_o
    mean_bias = float(np.mean(diffs))
    sd_bias = float(np.std(diffs, ddof=1))
    loa_half = max(abs(mean_bias + 1.96 * sd_bias - mean_bias),
                   abs(mean_bias - (mean_bias - 1.96 * sd_bias)))
    return {
        "n_subjects": n_subj, "n_trials": n_trials,
        "mean_observed_deg": float(np.mean(agg_o)),
        "mean_predicted_deg": float(np.mean(agg_p)),
        "mean_bias_deg": mean_bias, "sd_bias_deg": sd_bias,
        "loa_half_width_deg": loa_half,
        "pearson_r": _pearson(agg_p, agg_o),
        "ccc_lin": _ccc(agg_p, agg_o),
        "mae_deg": float(np.mean(np.abs(diffs))),
        "classification": "",  # filled by caller after _classify
    }


def _build_for_source(target: str, view: str, approach: str,
                      kp_source: Path):
    """Build dataset for one slot using its v17 reader, with the OpenCap
    DWPose source pointed at kp_source. Returns (X, y, subjects) or None."""
    import harness.train_v9_phased as v9
    import harness.train_v12_combined as v12
    from harness.train_multi_metric import VIEW_BUCKET_NAMES
    import harness.train_v14_full_dwpose as v14
    view_to_cam = {v: k for k, v in VIEW_BUCKET_NAMES.items()}
    cam = view_to_cam.get(view)
    if cam is None:
        return None

    if approach == "v14_full_dwpose":
        v9.KP_DIR = kp_source
        v12.ASPSET_KEYPOINTS = ASPSET_DWPOSE
        if target == "ankle_angle_r":
            res = v14.build_opencap_only_dataset(target, view)
        else:
            res = v12.build_combined_dataset(target, view)
    elif approach == "v13_dwpose_hybrid":
        v9.KP_DIR = kp_source
        v12.ASPSET_KEYPOINTS = ASPSET_DWPOSE
        res = v12.build_combined_dataset(target, view)
    else:
        return None

    if res is None:
        return None
    X, y, subjects = res[0], res[1], res[2]
    if X is None or len(y) == 0:
        return None
    return (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64),
            list(subjects))


def _opencap_only_mask(subjects, sources=None):
    """For combined datasets, restrict LOSO stats to OpenCap subjects so the
    flip-TTA (OpenCap-only) effect is measured on the OpenCap cohort -- the
    same cohort all prior per-slot validity numbers use."""
    # OpenCap subjects are 'subjectN' (v14 ankle path) or 'opencap_subjectN'
    # (combined path); ASPset are 'aspset_*'. flip-TTA only altered OpenCap
    # keypoints, so validity is measured on the OpenCap cohort only.
    def _is_oc(s: str) -> bool:
        s = str(s)
        return s.startswith("subject") or s.startswith("opencap_")
    return np.array([_is_oc(s) for s in subjects])


def evaluate_slot(target: str, view: str, approach: str, kp_source: Path):
    built = _build_for_source(target, view, approach, kp_source)
    if built is None:
        return None
    X, y, subjects = built
    pred = _loso_predict(X, y, subjects)
    subj_arr = np.array(subjects)
    # Restrict to OpenCap subjects for the validity stat (matches the cohort
    # used across all prior builds; flip-TTA only changed OpenCap keypoints).
    oc = _opencap_only_mask(subjects)
    st = _stats_per_subject(pred[oc], y[oc], subj_arr[oc])
    st["classification"] = _classify(st["ccc_lin"], st["loa_half_width_deg"])
    # per-subject preds for dump
    mask = np.isfinite(pred) & np.isfinite(y) & oc
    per_clip = {
        "subjects": subj_arr[mask].tolist(),
        "pred": pred[mask].tolist(),
        "obs": y[mask].tolist(),
    }
    return st, per_clip


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TTA_DWPOSE.exists() or len(list(TTA_DWPOSE.glob("*.json"))) < 5:
        raise SystemExit(
            f"TTA keypoints not ready at {TTA_DWPOSE} -- run "
            f"harness.dwpose_flip_tta first."
        )
    n_tta = len(list(TTA_DWPOSE.glob("*.json")))
    log(f"TTA keypoint cache: {n_tta} clips")

    v17 = json.loads(V17_PATH.read_text())
    # The hand-engineered geometry reader (v14_full_dwpose) is the cleanest L1
    # probe: it is a direct, transparent function of DWPose keypoints with no
    # learned compensation. We run it on EVERY slot (not just where it is the
    # v17 pick) to characterise detector asymmetry across the whole table, and
    # also record each slot's native v17 approach for context. ankle_angle_r
    # uses the OpenCap-only v14 path; other targets use the combined path
    # (OpenCap DWPose + ASPset), with validity measured on the OpenCap cohort.
    PROBE_APPROACH = "v14_full_dwpose"
    slots = []
    for target, sl in v17["models"].items():
        for view, entry in sl.items():
            slots.append((target, view, PROBE_APPROACH,
                          entry.get("approach", "unknown")))

    out_slots = []
    per_clip_all = {}
    tier_counter = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    n_applicable = 0
    for i, (target, view, approach, native_approach) in enumerate(slots):
        slot_str = f"{target}|{view}"
        t0 = time.time()
        n_applicable += 1
        try:
            base = evaluate_slot(target, view, approach, ORIG_DWPOSE)
            tta = evaluate_slot(target, view, approach, TTA_DWPOSE)
        except Exception as e:
            log(f"  [{i+1}/{len(slots)}] {slot_str} ERR {e!r}")
            out_slots.append({
                "target": target, "view": view, "probe_approach": approach,
                "native_v17_approach": native_approach,
                "tta_applicable": True, "error": repr(e),
            })
            continue
        if base is None or tta is None:
            out_slots.append({
                "target": target, "view": view, "probe_approach": approach,
                "native_v17_approach": native_approach,
                "tta_applicable": False,
                "reason": "geometry reader returned no usable dataset for slot",
            })
            log(f"  [{i+1}/{len(slots)}] {slot_str} -- no dataset")
            continue
        base_st, _ = base
        tta_st, tta_clip = tta
        tier_counter[tta_st["classification"]] = (
            tier_counter.get(tta_st["classification"], 0) + 1
        )
        per_clip_all[slot_str] = tta_clip
        out_slots.append({
            "target": target, "view": view, "probe_approach": approach,
            "native_v17_approach": native_approach,
            "tta_applicable": True,
            "n_subjects": tta_st["n_subjects"], "n_trials": tta_st["n_trials"],
            # baseline (same reader, original DWPose cache)
            "baseline_ccc_lin": base_st["ccc_lin"],
            "baseline_loa_half_width_deg": base_st["loa_half_width_deg"],
            "baseline_classification": base_st["classification"],
            # flip-TTA
            "ccc_lin": tta_st["ccc_lin"],
            "loa_half_width_deg": tta_st["loa_half_width_deg"],
            "mean_bias_deg": tta_st["mean_bias_deg"],
            "mae_deg": tta_st["mae_deg"],
            "pearson_r": tta_st["pearson_r"],
            "classification": tta_st["classification"],
            "delta_ccc_tta_minus_baseline": (
                tta_st["ccc_lin"] - base_st["ccc_lin"]
                if not (math.isnan(tta_st["ccc_lin"])
                        or math.isnan(base_st["ccc_lin"])) else None
            ),
        })
        log(
            f"  [{i+1}/{len(slots)}] {slot_str} ({approach}) "
            f"base CCC {base_st['ccc_lin']:.3f}/LoA {base_st['loa_half_width_deg']:.2f} "
            f"-> TTA CCC {tta_st['ccc_lin']:.3f}/LoA {tta_st['loa_half_width_deg']:.2f} "
            f"[{time.time()-t0:.0f}s]"
        )

    out = {
        "version": "v46_flip_tta",
        "produced_by": "harness.layer3_retrain_on_flip_tta_l1 (Agent UU)",
        "description": (
            "Layer 1 test-time horizontal-flip augmentation for DWPose. Same "
            "hand-engineered readers as v45; OpenCap DWPose keypoint source "
            "swapped to flip-TTA. baseline_* columns are the identical reader "
            "on the original DWPose cache (clean L1 A/B). Subject-level LOSO "
            "at L3, OpenCap cohort."
        ),
        "n_dwpose_applicable_slots": n_applicable,
        "tier_counts_tta": tier_counter,
        "slots": out_slots,
    }
    PER_SLOT_OUT.write_text(json.dumps(out, indent=2))
    PER_CLIP_OUT.write_text(json.dumps(per_clip_all, indent=2))
    log(f"wrote {PER_SLOT_OUT}")
    log(f"wrote {PER_CLIP_OUT}")
    return out


def main() -> None:
    run()


if __name__ == "__main__":
    main()
