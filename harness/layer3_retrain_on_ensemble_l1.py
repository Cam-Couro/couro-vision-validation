"""Agent VV2 v48: per-slot validity under Layer 1 DWPose+RTMPose detector ensemble.

Mirrors harness.layer3_retrain_on_flip_tta_l1 (Agent UU's v46 producer) exactly,
but swaps the OpenCap DWPose keypoint source from ``opencap_dwpose_keypoints`` to
``dwpose_rtmpose_ensemble_keypoints`` (a confidence-weighted average of the
DWPose and RTMPose-Halpe26 detectors, run on the SAME single-camera frame --
two ARCHITECTURES, NOT two cameras). Everything downstream -- angle computation
(couro_keypoints), feature extraction, LOSO ridge -- is byte-identical. Any
CCC/LoA change is attributable purely to the Layer 1 detector ensemble.

For each deploy slot we compute, with subject-level LOSO at L3 (OpenCap cohort):
  * baseline : reader on data/opencap_dwpose_keypoints (recomputed here, NOT a
               cached number -- identical code path to the ensemble pass)
  * ensemble : same reader on data/dwpose_rtmpose_ensemble_keypoints

Output: data/layer3_retrain_detector_ensemble/per_slot_validity_v48.json
        data/layer3_retrain_detector_ensemble/per_clip_predictions_v48.json
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
ENSEMBLE_DWPOSE: Final[Path] = DATA_ROOT / "dwpose_rtmpose_ensemble_keypoints"
ASPSET_DWPOSE: Final[Path] = DATA_ROOT / "aspset510_dwpose_keypoints"

OUT_DIR: Final[Path] = DATA_ROOT / "layer3_retrain_detector_ensemble"
PER_SLOT_OUT: Final[Path] = OUT_DIR / "per_slot_validity_v48.json"
PER_CLIP_OUT: Final[Path] = OUT_DIR / "per_clip_predictions_v48.json"
V17_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v17_selective.json"

ALPHA: Final[float] = 10.0


def log(msg: str) -> None:
    print(f"[v48] {msg}", flush=True)


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
    """Restrict LOSO stats to OpenCap subjects so the ensemble (OpenCap-only)
    effect is measured on the OpenCap cohort -- same cohort all prior per-slot
    validity numbers use. The detector ensemble only altered OpenCap keypoints."""
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
    oc = _opencap_only_mask(subjects)
    st = _stats_per_subject(pred[oc], y[oc], subj_arr[oc])
    st["classification"] = _classify(st["ccc_lin"], st["loa_half_width_deg"])
    mask = np.isfinite(pred) & np.isfinite(y) & oc
    per_clip = {
        "subjects": subj_arr[mask].tolist(),
        "pred": pred[mask].tolist(),
        "obs": y[mask].tolist(),
    }
    return st, per_clip


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ENSEMBLE_DWPOSE.exists() or len(list(ENSEMBLE_DWPOSE.glob("*.json"))) < 5:
        raise SystemExit(
            f"Ensemble keypoints not ready at {ENSEMBLE_DWPOSE} -- run "
            f"harness.dwpose_rtmpose_ensemble first."
        )
    n_ens = len([p for p in ENSEMBLE_DWPOSE.glob("*.json") if p.name != "_timing.json"])
    log(f"detector-ensemble keypoint cache: {n_ens} clips")

    # Which cams actually have ensemble keypoints (scoped run = side cams).
    ens_cams = sorted({p.stem.split("_")[-1] for p in ENSEMBLE_DWPOSE.glob("*.json")
                       if p.name != "_timing.json"})
    log(f"ensemble cams present: {ens_cams}")

    v17 = json.loads(V17_PATH.read_text())
    # The hand-engineered geometry reader (v14_full_dwpose) is the cleanest L1
    # probe: a direct, transparent function of keypoints with no learned
    # compensation. We run it on every slot (probe), and record each slot's
    # native v17 approach for context. Same as the v46 flip-TTA producer.
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
            ens = evaluate_slot(target, view, approach, ENSEMBLE_DWPOSE)
        except Exception as e:
            log(f"  [{i+1}/{len(slots)}] {slot_str} ERR {e!r}")
            out_slots.append({
                "target": target, "view": view, "probe_approach": approach,
                "native_v17_approach": native_approach,
                "ensemble_applicable": True, "error": repr(e),
            })
            continue
        if base is None or ens is None:
            out_slots.append({
                "target": target, "view": view, "probe_approach": approach,
                "native_v17_approach": native_approach,
                "ensemble_applicable": False,
                "reason": "geometry reader returned no usable dataset for slot "
                          "(view's cam may be outside the scoped ensemble run)",
            })
            log(f"  [{i+1}/{len(slots)}] {slot_str} -- no dataset")
            continue
        base_st, _ = base
        ens_st, ens_clip = ens
        tier_counter[ens_st["classification"]] = (
            tier_counter.get(ens_st["classification"], 0) + 1
        )
        per_clip_all[slot_str] = ens_clip
        out_slots.append({
            "target": target, "view": view, "probe_approach": approach,
            "native_v17_approach": native_approach,
            "ensemble_applicable": True,
            "n_subjects": ens_st["n_subjects"], "n_trials": ens_st["n_trials"],
            # baseline (same reader, original DWPose cache)
            "baseline_ccc_lin": base_st["ccc_lin"],
            "baseline_loa_half_width_deg": base_st["loa_half_width_deg"],
            "baseline_classification": base_st["classification"],
            # detector ensemble
            "ccc_lin": ens_st["ccc_lin"],
            "loa_half_width_deg": ens_st["loa_half_width_deg"],
            "mean_bias_deg": ens_st["mean_bias_deg"],
            "mae_deg": ens_st["mae_deg"],
            "pearson_r": ens_st["pearson_r"],
            "classification": ens_st["classification"],
            "delta_ccc_ensemble_minus_baseline": (
                ens_st["ccc_lin"] - base_st["ccc_lin"]
                if not (math.isnan(ens_st["ccc_lin"])
                        or math.isnan(base_st["ccc_lin"])) else None
            ),
        })
        log(
            f"  [{i+1}/{len(slots)}] {slot_str} ({approach}) "
            f"base CCC {base_st['ccc_lin']:.3f}/LoA {base_st['loa_half_width_deg']:.2f} "
            f"-> ENS CCC {ens_st['ccc_lin']:.3f}/LoA {ens_st['loa_half_width_deg']:.2f} "
            f"[{time.time()-t0:.0f}s]"
        )

    out = {
        "version": "v48_detector_ensemble",
        "produced_by": "harness.layer3_retrain_on_ensemble_l1 (Agent VV2)",
        "description": (
            "Layer 1 DWPose+RTMPose two-architecture detector ensemble "
            "(confidence-weighted average of DWPose and RTMPose-Halpe26 on the "
            "SAME single-camera frame; NOT multi-camera fusion). Same "
            "hand-engineered readers as v45/v47; OpenCap DWPose keypoint source "
            "swapped to the detector ensemble. baseline_* columns are the "
            "identical reader on the original DWPose cache (clean L1 A/B). "
            "Subject-level LOSO at L3, OpenCap cohort."
        ),
        "n_dwpose_applicable_slots": n_applicable,
        "ensemble_cams": ens_cams,
        "tier_counts_ensemble": tier_counter,
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
