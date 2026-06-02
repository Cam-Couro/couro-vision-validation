"""Biomech-standard validity statistics for the v14 deploy models.

Computes per-slot Bland-Altman 95% Limits of Agreement, CCC (Lin 1989),
mean bias, MAE, and RMSE — the device-validation metrics that diligence
reviewers expect (Oura/Polar/force-plate papers report these, not just
Pearson r).

For each of the 25 (target metric x camera view) deploy slots:
  1. Reconstruct the LOSO CV that produced the deploy model
  2. Extract paired (predicted_ROM, observed_ROM) per held-out subject
  3. Compute biomech-standard stats:
       - Pearson r (carried over from existing doc)
       - CCC = 2*cov(p,o) / (var(p) + var(o) + (mean(p) - mean(o))^2)
       - mean_bias = mean(p - o)
       - SD(diff) = SD(p - o)
       - 95% LoA = mean_bias +- 1.96 * SD(diff)
       - MAE = mean(|p - o|)
       - RMSE = sqrt(mean((p - o)^2))

Run:
    cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
    python -m harness.biomech_validity_stats

Outputs:
    data/biomech_validity_stats/per_slot_validity.json
    data/biomech_validity_stats/REPORT.md
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.train_v9_phased as v9
import harness.train_v12_combined as v12
import harness.train_event_anchored_all as ea
import harness.train_ankle_bilateral as ea_ank
import harness.train_v14_full_dwpose as v14
from harness.train_multi_metric import VIEW_BUCKET_NAMES
from harness.train_regression_poc import fit_ridge


RESULTS_DIR = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results"
)
DATA_ROOT = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data"
)
OUT_DIR = DATA_ROOT / "biomech_validity_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIEW_TO_CAM = {v: k for k, v in VIEW_BUCKET_NAMES.items()}

DEPLOY_PATH = RESULTS_DIR / "deploy_ready_models.json"


# ---------------------------------------------------------------------------
# Per-approach paired-prediction builders.
# Each returns (all_pred, y, subjects_array) for one (target, view) slot
# using the same LOSO recipe as the original trainer that produced the slot
# in deploy_ready_models.json.
# ---------------------------------------------------------------------------


def _loso_extract(X: np.ndarray, y: np.ndarray, subjects: list[str], alpha: float = 10.0):
    """Leave-one-subject-out CV that returns paired pred/obs arrays."""
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


def _build_v12_dataset(target: str, view: str):
    """Combined OpenCap (RTMPose) + ASPset (RTMPose) — v12 trainer."""
    # v12 uses OpenCap RTMPose keypoints (default KP_DIR in train_v9_phased)
    # and ASPset RTMPose keypoints (default ASPSET_KEYPOINTS in v12).
    v9.KP_DIR = DATA_ROOT / "couro_keypoints"
    v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_couro_keypoints"
    return v12.build_combined_dataset(target, view)


def _build_v13_dataset(target: str, view: str):
    """Hybrid: OpenCap RTMPose + ASPset DWPose."""
    v9.KP_DIR = DATA_ROOT / "couro_keypoints"
    v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
    return v12.build_combined_dataset(target, view)


def _build_v14_dataset(target: str, view: str):
    """Full DWPose: OpenCap DWPose + ASPset DWPose. Ankle is OpenCap-only."""
    v9.KP_DIR = DATA_ROOT / "opencap_dwpose_keypoints"
    v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
    if target == "ankle_angle_r":
        return v14.build_opencap_only_dataset(target, view)
    return v12.build_combined_dataset(target, view)


def _build_v9_phased_dataset(target: str, view: str):
    """OpenCap-only v9 phased features."""
    v9.KP_DIR = DATA_ROOT / "couro_keypoints"
    cam = VIEW_TO_CAM.get(view)
    if cam is None:
        return None
    X, y, subjects, _, _ = v9.build_dataset_v9(target, cam)
    if len(y) == 0:
        return None
    return X.astype(np.float64), np.asarray(y, dtype=np.float64), subjects, ["opencap"] * len(y)


def _build_event_anchored_dataset(target: str, view: str):
    """OpenCap-only event-anchored features (train_event_anchored_all)."""
    ea.KP_DIR = DATA_ROOT / "couro_keypoints"
    cam = VIEW_TO_CAM.get(view)
    if cam is None:
        return None
    X, y, subjects, _, _ = ea.build_dataset(target, cam)
    if len(y) == 0:
        return None
    return X.astype(np.float64), np.asarray(y, dtype=np.float64), subjects, ["opencap"] * len(y)


def _build_event_anchored_bilateral_dataset(target: str, view: str):
    """OpenCap-only event-anchored bilateral ankle features (train_ankle_bilateral)."""
    if target != "ankle_angle_r":
        return None
    ea_ank.KP_DIR = DATA_ROOT / "couro_keypoints"
    cam = VIEW_TO_CAM.get(view)
    if cam is None:
        return None
    # train_ankle_bilateral has build_dataset taking just cam (target is fixed ankle_r)
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
    "v12_combined": _build_v12_dataset,
    "v13_dwpose_hybrid": _build_v13_dataset,
    "v14_full_dwpose": _build_v14_dataset,
    "v9_phased": _build_v9_phased_dataset,
    "event_anchored": _build_event_anchored_dataset,
    "event_anchored_bilateral": _build_event_anchored_bilateral_dataset,
}


# ---------------------------------------------------------------------------
# Biomech validity statistics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidityStats:
    target: str
    view: str
    approach: str
    n_subjects: int
    n_trials: int
    aggregation: str  # "per_subject" or "per_trial"
    mean_observed: float
    sd_observed: float
    mean_predicted: float
    sd_predicted: float
    mean_bias: float
    sd_bias: float
    upper_loa: float
    lower_loa: float
    pearson_r: float
    ccc: float
    mae: float
    rmse: float
    # Optional: one-sample t-test p-value for bias != 0
    p_value_bias: float | None = None
    # Provenance
    source: str = "loso_pairs"  # "loso_pairs" or "summary_only"


def _pearson(p: np.ndarray, o: np.ndarray) -> float:
    if p.std(ddof=1) == 0 or o.std(ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(p, o)[0, 1])


def _ccc(p: np.ndarray, o: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (1989)."""
    mp, mo = float(np.mean(p)), float(np.mean(o))
    vp, vo = float(np.var(p, ddof=0)), float(np.var(o, ddof=0))
    cov = float(np.mean((p - mp) * (o - mo)))
    denom = vp + vo + (mp - mo) ** 2
    if denom == 0:
        return float("nan")
    return 2.0 * cov / denom


def _t_test_p_value_bias(diffs: np.ndarray) -> float | None:
    """One-sample two-sided t-test that mean(diffs) != 0."""
    n = len(diffs)
    if n < 2:
        return None
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return None
    t = mean / (sd / math.sqrt(n))
    # Use Student-t two-sided p (df = n-1).
    # Tiny closed-form via numpy not available; do a Welch-style approximation
    # using survival of t-distribution via scipy if installed, else normal.
    try:
        from scipy.stats import t as t_dist  # type: ignore
        return float(2 * (1 - t_dist.cdf(abs(t), df=n - 1)))
    except Exception:
        # Fall back to normal approximation (conservative for small n)
        from math import erf, sqrt
        return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


def compute_stats_from_pairs(
    target: str, view: str, approach: str, pred: np.ndarray, obs: np.ndarray, subjects: np.ndarray,
    *, aggregate_per_subject: bool = True,
) -> ValidityStats:
    """Compute Bland-Altman + CCC stats from paired LOSO predictions.

    aggregate_per_subject=True: average predicted and observed across the
    trials of each subject first, then compute stats. This is the biomech
    device-validation standard (Atkinson & Nevill 1998) — keeps n_subjects
    independent samples without inflating df via within-subject replicates.

    aggregate_per_subject=False: compute stats over every trial. This
    matches the LOSO bias_deg / sd_of_difference_deg already stored in
    deploy_ready_models.json.
    """
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[mask]
    obs = obs[mask]
    subjects = subjects[mask] if isinstance(subjects, np.ndarray) else np.asarray(subjects)[mask]
    n_trials_pre = int(len(pred))
    n_subj_pre = int(len(set(subjects.tolist())))
    if aggregate_per_subject:
        unique = sorted(set(subjects.tolist()))
        agg_pred = np.array([
            float(np.mean(pred[subjects == s])) for s in unique
        ])
        agg_obs = np.array([
            float(np.mean(obs[subjects == s])) for s in unique
        ])
        pred = agg_pred
        obs = agg_obs
        subjects = np.array(unique)
    aggregation = "per_subject" if aggregate_per_subject else "per_trial"
    if len(pred) < 2:
        return ValidityStats(
            target=target, view=view, approach=approach,
            n_subjects=n_subj_pre, n_trials=n_trials_pre, aggregation=aggregation,
            mean_observed=float("nan"), sd_observed=float("nan"),
            mean_predicted=float("nan"), sd_predicted=float("nan"),
            mean_bias=float("nan"), sd_bias=float("nan"),
            upper_loa=float("nan"), lower_loa=float("nan"),
            pearson_r=float("nan"), ccc=float("nan"),
            mae=float("nan"), rmse=float("nan"),
            source="loso_pairs",
        )
    diffs = pred - obs
    mean_bias = float(np.mean(diffs))
    sd_bias = float(np.std(diffs, ddof=1))
    return ValidityStats(
        target=target, view=view, approach=approach,
        n_subjects=n_subj_pre, n_trials=n_trials_pre, aggregation=aggregation,
        mean_observed=float(np.mean(obs)),
        sd_observed=float(np.std(obs, ddof=1)),
        mean_predicted=float(np.mean(pred)),
        sd_predicted=float(np.std(pred, ddof=1)),
        mean_bias=mean_bias,
        sd_bias=sd_bias,
        upper_loa=mean_bias + 1.96 * sd_bias,
        lower_loa=mean_bias - 1.96 * sd_bias,
        pearson_r=_pearson(pred, obs),
        ccc=_ccc(pred, obs),
        mae=float(np.mean(np.abs(diffs))),
        rmse=float(np.sqrt(np.mean(diffs**2))),
        p_value_bias=_t_test_p_value_bias(diffs),
        source="loso_pairs",
    )


def stats_from_summary(target: str, view: str, model_entry: dict) -> ValidityStats:
    """Fallback when re-running LOSO is too costly: derive what we can
    from the saved summary stats. CCC and MAE are left NaN here."""
    s = model_entry["loso_cv_stats"]
    bias = float(s["bias_deg"])
    sd_diff = float(s["sd_of_difference_deg"])
    return ValidityStats(
        target=target, view=view, approach=model_entry.get("approach", "unknown"),
        n_subjects=int(model_entry.get("n_subjects", 0)),
        n_trials=int(model_entry.get("n_total", 0)),
        aggregation="per_trial",
        mean_observed=float("nan"), sd_observed=float("nan"),
        mean_predicted=float("nan"), sd_predicted=float("nan"),
        mean_bias=bias, sd_bias=sd_diff,
        upper_loa=bias + 1.96 * sd_diff,
        lower_loa=bias - 1.96 * sd_diff,
        pearson_r=float(s["pearson_r"]),
        ccc=float("nan"),
        mae=float("nan"),
        rmse=float(s["rmse_deg"]),
        source="summary_only",
    )


def classify(stats: ValidityStats) -> str:
    """Bland-Altman interpretation per biomech device-validation conventions."""
    ccc = stats.ccc
    half_loa = max(abs(stats.upper_loa - stats.mean_bias), abs(stats.mean_bias - stats.lower_loa))
    if not math.isnan(ccc):
        if ccc > 0.75 and half_loa < 5:
            return "Excellent"
        if ccc > 0.60 and half_loa < 10:
            return "Good"
        if ccc > 0.40 and half_loa < 15:
            return "Moderate"
        return "Poor"
    # Fall back to LoA-only classification
    if half_loa < 5:
        return "LoA-only: tight"
    if half_loa < 10:
        return "LoA-only: moderate"
    if half_loa < 15:
        return "LoA-only: wide"
    return "LoA-only: poor"


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------


def stats_to_dict(s: ValidityStats) -> dict:
    return {
        "target": s.target,
        "view": s.view,
        "approach": s.approach,
        "n_subjects": s.n_subjects,
        "n_trials": s.n_trials,
        "aggregation": s.aggregation,
        "mean_observed_deg": s.mean_observed,
        "sd_observed_deg": s.sd_observed,
        "mean_predicted_deg": s.mean_predicted,
        "sd_predicted_deg": s.sd_predicted,
        "mean_bias_deg": s.mean_bias,
        "sd_bias_deg": s.sd_bias,
        "loa_upper_deg": s.upper_loa,
        "loa_lower_deg": s.lower_loa,
        "loa_half_width_deg": max(
            abs(s.upper_loa - s.mean_bias), abs(s.mean_bias - s.lower_loa),
        ),
        "pearson_r": s.pearson_r,
        "ccc_lin": s.ccc,
        "mae_deg": s.mae,
        "rmse_deg": s.rmse,
        "p_value_bias_neq_0": s.p_value_bias,
        "classification": classify(s),
        "stats_source": s.source,
    }


def merged_slot_dict(
    per_subj: ValidityStats, per_trial: ValidityStats | None,
) -> dict:
    """One row per slot, with subject-level stats as the primary report and
    trial-level numbers carried alongside (matches deploy_ready_models.json)."""
    d = stats_to_dict(per_subj)
    if per_trial is not None:
        d["per_trial"] = {
            "n_trials": per_trial.n_trials,
            "mean_bias_deg": per_trial.mean_bias,
            "sd_bias_deg": per_trial.sd_bias,
            "loa_lower_deg": per_trial.lower_loa,
            "loa_upper_deg": per_trial.upper_loa,
            "pearson_r": per_trial.pearson_r,
            "ccc_lin": per_trial.ccc,
            "mae_deg": per_trial.mae,
            "rmse_deg": per_trial.rmse,
        }
    return d


def run(skip_combined: bool = False) -> dict:
    """Recompute validity stats for every deploy slot.

    Args:
        skip_combined: if True, skip the combined v12/v13/v14 slots (which
            require loading ~1410 ASPset DWPose JSONs and are slow). Useful
            for a fast smoke run that only covers OpenCap-only slots.
    """
    deploy = json.loads(DEPLOY_PATH.read_text())
    out_rows: list[dict] = []

    # Group by approach so we set up the data root only once per approach.
    for target, slots in deploy["models"].items():
        for view, entry in slots.items():
            approach = entry.get("approach", "unknown")
            print(f"[{target}|{view}] approach={approach} ... ", end="", flush=True)
            t0 = time.time()
            builder = BUILDERS.get(approach)
            if builder is None or (skip_combined and approach in (
                "v12_combined", "v13_dwpose_hybrid", "v14_full_dwpose",
            )):
                stats = stats_from_summary(target, view, entry)
                print(f"summary-only ({time.time() - t0:.1f}s)")
                out_rows.append(stats_to_dict(stats))
                continue
            try:
                result = builder(target, view)
            except Exception as e:
                print(f"BUILDER FAILED: {e!r}")
                stats = stats_from_summary(target, view, entry)
                out_rows.append(stats_to_dict(stats))
                continue
            if result is None:
                print("builder returned None -> summary-only")
                stats = stats_from_summary(target, view, entry)
                out_rows.append(stats_to_dict(stats))
                continue
            X, y, subjects, _ = result
            if len(y) < 20:
                print(f"only n={len(y)} -> summary-only")
                stats = stats_from_summary(target, view, entry)
                out_rows.append(merged_slot_dict(stats, None))
                continue
            pred, obs, subj = _loso_extract(X, y, list(subjects), alpha=10.0)
            per_subj = compute_stats_from_pairs(
                target, view, approach, pred, obs, subj, aggregate_per_subject=True,
            )
            per_trial = compute_stats_from_pairs(
                target, view, approach, pred, obs, subj, aggregate_per_subject=False,
            )
            print(
                f"r={per_subj.pearson_r:+.3f} CCC={per_subj.ccc:+.3f} "
                f"bias={per_subj.mean_bias:+.2f} "
                f"LoA=[{per_subj.lower_loa:+.1f},{per_subj.upper_loa:+.1f}] "
                f"(per-subj n={per_subj.n_subjects}, per-trial n={per_trial.n_trials}, "
                f"{time.time() - t0:.1f}s)"
            )
            out_rows.append(merged_slot_dict(per_subj, per_trial))

    out = {
        "version": "1.0",
        "produced_by": "harness.biomech_validity_stats",
        "source_deploy": str(DEPLOY_PATH),
        "stats_definitions": {
            "pearson_r": "Pearson correlation of paired (predicted_ROM, observed_ROM).",
            "ccc_lin": (
                "Lin's 1989 concordance correlation coefficient: "
                "2*cov(p,o) / (var(p) + var(o) + (mean(p) - mean(o))^2). "
                "Penalises systematic bias and scale mismatch, unlike Pearson r."
            ),
            "loa": (
                "Bland-Altman 95% Limits of Agreement = mean_bias +- 1.96 * SD(differences). "
                "Range within which 95% of (pred - obs) differences fall."
            ),
            "classification": (
                "Excellent: CCC>0.75 and LoA<+/-5deg. Good: CCC>0.60 and LoA<+/-10deg. "
                "Moderate: CCC>0.40 and LoA<+/-15deg. Poor: otherwise."
            ),
        },
        "slots": out_rows,
        "mpi_inf_3dhp_rear_cam7": _mpi_rear_stats(),
    }
    return out


def write_outputs(out: dict) -> None:
    (OUT_DIR / "per_slot_validity.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "REPORT.md").write_text(render_markdown(out))


def render_markdown(out: dict) -> str:
    rows = out["slots"]
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["target"],
            ["side_left", "front_oblique_left", "front_center",
             "front_oblique_right", "side_right"].index(r["view"]),
        ),
    )

    # Pick the best slot per metric from actual data (lowest LoA half-width
    # among slots where CCC > 0.4), plus one rear/oblique highlight and one
    # weak slot for transparency.
    by_metric: dict[str, list[dict]] = {}
    for r in rows:
        if math.isnan(r["ccc_lin"]):
            continue
        by_metric.setdefault(r["target"], []).append(r)
    headline = []
    seen = set()
    for tgt, slot_list in by_metric.items():
        # Best slot per metric by CCC (per-subject)
        best = max(slot_list, key=lambda r: (r["ccc_lin"], -r["loa_half_width_deg"]))
        if (best["target"], best["view"]) not in seen:
            headline.append(best)
            seen.add((best["target"], best["view"]))
    # Add explicit rear-view best (hip_flexion_l from MPI cohort) — handled in
    # a separate report section below since MPI uses different methodology.
    # Add one weak slot for transparency: lowest CCC slot that still has data.
    weak_candidates = [r for r in rows if not math.isnan(r["ccc_lin"])]
    if weak_candidates:
        weak = min(weak_candidates, key=lambda r: r["ccc_lin"])
        if (weak["target"], weak["view"]) not in seen:
            headline.append(weak)
            seen.add((weak["target"], weak["view"]))
    # Add one extra strong slot for hip_flexion_r (the most-used metric)
    hip_slots = [r for r in rows if r["target"] == "hip_flexion_r" and not math.isnan(r["ccc_lin"])]
    hip_slots.sort(key=lambda r: r["ccc_lin"], reverse=True)
    for r in hip_slots[:2]:
        if (r["target"], r["view"]) not in seen:
            headline.append(r)
            seen.add((r["target"], r["view"]))

    def fmt(v: float | None, prec: int = 2) -> str:
        if v is None or (isinstance(v, float) and (math.isnan(v))):
            return "—"
        return f"{v:.{prec}f}"

    lines = []
    lines.append("# Couro v14 Deploy: Biomech-Standard Validity Statistics")
    lines.append("")
    lines.append(
        "Bland-Altman 95% Limits of Agreement (LoA), Lin's CCC, and mean bias "
        "for every (metric x camera view) slot in the v14 deploy bundle. These "
        "are the device-validation stats Oura/Polar/force-plate papers report "
        "and that diligence reviewers expect alongside Pearson r."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("Per slot we reconstruct the leave-one-subject-out (LOSO) cross-validation that produced the deploy ridge regression (using the same dataset builder as the original trainer — v12/v13/v14/v9_phased/event_anchored/event_anchored_bilateral). Each held-out subject's trials yield paired (predicted_ROM, observed_ROM) values.")
    lines.append("")
    lines.append("**Two aggregation levels are reported per slot:**")
    lines.append("")
    lines.append("- **Per-subject** (the headline, matches Atkinson & Nevill 1998 device-validation convention): trials of each subject are averaged into one (pred, obs) point before Bland-Altman; n = unique subjects.")
    lines.append("- **Per-trial** (matches `deploy_ready_models.json` LOSO summary): every trial is a separate point; n = total LOSO test trials. Wider LoA because within-subject trial variability adds in.")
    lines.append("")
    lines.append("Subject-level LoA is the appropriate accuracy claim for an athlete-level metric (one ROM number per session). Trial-level LoA is the lower bound for per-rep accuracy.")
    lines.append("")
    lines.append("Statistics computed per slot:")
    lines.append("")
    lines.append("- **Pearson r** — same as existing doc; measures linear association only.")
    lines.append("- **CCC** (Lin 1989) = `2·cov(p, o) / (var(p) + var(o) + (mean(p) − mean(o))²)`. Penalises systematic bias and scale mismatch — what Pearson r misses.")
    lines.append("- **Mean bias** = mean(predicted − observed). Sign tells direction of systematic offset.")
    lines.append("- **SD(differences)** = SD(predicted − observed). The random-error component.")
    lines.append("- **95% LoA** = mean_bias ± 1.96 × SD(differences). 95% of paired differences fall in this range.")
    lines.append("- **MAE** = mean(|predicted − observed|).")
    lines.append("- **RMSE** = sqrt(mean((predicted − observed)²)).")
    lines.append("- **p-value (bias ≠ 0)** — one-sample t-test on the differences.")
    lines.append("")
    lines.append("**Classification (biomech device-validation conventions):**")
    lines.append("")
    lines.append("| Tier | CCC | LoA half-width |")
    lines.append("| --- | --- | --- |")
    lines.append("| Excellent | > 0.75 | < ±5° |")
    lines.append("| Good | 0.60–0.75 | ±5–10° |")
    lines.append("| Moderate | 0.40–0.60 | ±10–15° |")
    lines.append("| Poor | ≤ 0.40 | > ±15° |")
    lines.append("")

    lines.append("## Headline slots (recommended for validation doc)")
    lines.append("")
    lines.append("Per-subject Bland-Altman + Lin's CCC. n = unique LOSO-held-out subjects.")
    lines.append("")
    lines.append(
        "| Slot | n | r | CCC | bias (°) | 95% LoA (°) | MAE (°) | RMSE (°) | Tier |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |"
    )
    for r in headline:
        loa = f"[{fmt(r['loa_lower_deg'], 1)}, {fmt(r['loa_upper_deg'], 1)}]"
        lines.append(
            f"| {r['target']} / {r['view']} | {r['n_subjects']} | "
            f"{fmt(r['pearson_r'])} | {fmt(r['ccc_lin'])} | "
            f"{fmt(r['mean_bias_deg'])} | {loa} | "
            f"{fmt(r['mae_deg'])} | {fmt(r['rmse_deg'])} | "
            f"{r['classification']} |"
        )
    lines.append("")
    lines.append("### Same headline slots, per-trial (matches existing deploy doc)")
    lines.append("")
    lines.append(
        "| Slot | n trials | r | CCC | bias (°) | 95% LoA (°) | MAE (°) | RMSE (°) |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"
    )
    for r in headline:
        pt = r.get("per_trial") or {}
        if not pt:
            continue
        loa = f"[{fmt(pt.get('loa_lower_deg'), 1)}, {fmt(pt.get('loa_upper_deg'), 1)}]"
        lines.append(
            f"| {r['target']} / {r['view']} | {pt.get('n_trials')} | "
            f"{fmt(pt.get('pearson_r'))} | {fmt(pt.get('ccc_lin'))} | "
            f"{fmt(pt.get('mean_bias_deg'))} | {loa} | "
            f"{fmt(pt.get('mae_deg'))} | {fmt(pt.get('rmse_deg'))} |"
        )
    lines.append("")

    lines.append("## All 25 deploy slots — per-subject Bland-Altman summary")
    lines.append("")
    lines.append("Per-subject aggregation: trials of each subject averaged into one paired point before computing Bland-Altman + CCC.")
    lines.append("")
    lines.append(
        "| Target | View | Approach | n subj | mean obs | mean pred | bias (°) | SD diff (°) | "
        "lower LoA (°) | upper LoA (°) | r | CCC | MAE (°) | RMSE (°) | Tier |"
    )
    lines.append(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | --- |"
    )
    for r in rows_sorted:
        lines.append(
            f"| {r['target']} | {r['view']} | {r['approach']} | "
            f"{r['n_subjects']} | "
            f"{fmt(r['mean_observed_deg'])} | {fmt(r['mean_predicted_deg'])} | "
            f"{fmt(r['mean_bias_deg'])} | {fmt(r['sd_bias_deg'])} | "
            f"{fmt(r['loa_lower_deg'])} | {fmt(r['loa_upper_deg'])} | "
            f"{fmt(r['pearson_r'])} | {fmt(r['ccc_lin'])} | "
            f"{fmt(r['mae_deg'])} | {fmt(r['rmse_deg'])} | "
            f"{r['classification']} |"
        )
    lines.append("")

    lines.append("## All 25 deploy slots — per-trial summary (matches deploy_ready_models.json)")
    lines.append("")
    lines.append("Every LOSO test trial is one point. Wider LoA than per-subject because within-subject variance is included.")
    lines.append("")
    lines.append(
        "| Target | View | n trials | bias (°) | SD diff (°) | lower LoA (°) | upper LoA (°) | r | CCC | MAE (°) | RMSE (°) |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in rows_sorted:
        pt = r.get("per_trial")
        if pt:
            lines.append(
                f"| {r['target']} | {r['view']} | {pt['n_trials']} | "
                f"{fmt(pt['mean_bias_deg'])} | {fmt(pt['sd_bias_deg'])} | "
                f"{fmt(pt['loa_lower_deg'])} | {fmt(pt['loa_upper_deg'])} | "
                f"{fmt(pt['pearson_r'])} | {fmt(pt['ccc_lin'])} | "
                f"{fmt(pt['mae_deg'])} | {fmt(pt['rmse_deg'])} |"
            )
        else:
            # summary-only fallback (no per-subject pairs available)
            lines.append(
                f"| {r['target']} | {r['view']} | {r['n_trials']} | "
                f"{fmt(r['mean_bias_deg'])} | {fmt(r['sd_bias_deg'])} | "
                f"{fmt(r['loa_lower_deg'])} | {fmt(r['loa_upper_deg'])} | "
                f"{fmt(r['pearson_r'])} | — | — | {fmt(r['rmse_deg'])} |"
            )
    lines.append("")

    # Compare CCC vs Pearson r (per-subject)
    lines.append("## Where CCC changes the story (per-subject)")
    lines.append("")
    lines.append(
        "Pearson r ignores systematic bias. CCC penalises it. Slots with large "
        "(r − CCC) gaps are correlated-but-biased — the model is tracking the "
        "shape of the ROM across subjects but offsetting it systematically."
    )
    lines.append("")
    gaps = [
        r for r in rows
        if not math.isnan(r["ccc_lin"]) and not math.isnan(r["pearson_r"])
    ]
    gaps.sort(key=lambda r: (r["pearson_r"] - r["ccc_lin"]), reverse=True)
    lines.append("| Slot | r | CCC | r − CCC | bias (°) | Interpretation |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for r in gaps[:8]:
        gap = r["pearson_r"] - r["ccc_lin"]
        if abs(r["mean_bias_deg"]) > 5:
            interp = "Large systematic bias — calibration needed"
        elif gap > 0.10:
            interp = "Scale mismatch (slope ≠ 1)"
        else:
            interp = "Concordance matches correlation"
        lines.append(
            f"| {r['target']} / {r['view']} | "
            f"{fmt(r['pearson_r'])} | {fmt(r['ccc_lin'])} | {fmt(gap)} | "
            f"{fmt(r['mean_bias_deg'])} | {interp} |"
        )
    lines.append("")

    # Compare per-subject vs per-trial — the most striking divergence story
    lines.append("## Where the per-subject view changes the story")
    lines.append("")
    lines.append(
        "Per-trial Pearson r mixes between-subject and within-subject variance. "
        "Per-subject Pearson r asks: does this model differentiate athletes? "
        "Slots where per-trial r is good but per-subject r collapses are the "
        "model tracking exercise shape (e.g. DJ trial-to-trial variation) but "
        "failing to rank athletes — a common failure mode for ridge regression "
        "on small subject-pool datasets."
    )
    lines.append("")
    lines.append("| Slot | per-trial r | per-subject r | Δ (trial − subj) | per-subject CCC | bias (°) | Interpretation |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    div_rows = []
    for r in rows:
        pt = r.get("per_trial")
        if not pt:
            continue
        if math.isnan(pt.get("pearson_r", float("nan"))) or math.isnan(r.get("pearson_r", float("nan"))):
            continue
        delta = pt["pearson_r"] - r["pearson_r"]
        if delta > 0.15:
            div_rows.append((delta, r, pt))
    div_rows.sort(key=lambda x: x[0], reverse=True)
    for delta, r, pt in div_rows[:8]:
        if r["pearson_r"] < 0:
            interp = "Model ranks athletes wrong — between-subject signal is absent"
        elif r["pearson_r"] < 0.3:
            interp = "Tracks reps but not athletes — limited discriminative validity"
        else:
            interp = "Some between-subject signal, mostly within-subject"
        lines.append(
            f"| {r['target']} / {r['view']} | "
            f"{fmt(pt['pearson_r'])} | {fmt(r['pearson_r'])} | {fmt(delta)} | "
            f"{fmt(r['ccc_lin'])} | {fmt(r['mean_bias_deg'])} | {interp} |"
        )
    lines.append("")

    # Validated label section
    lines.append("## Slots warranting the 'validated' label")
    lines.append("")
    lines.append(
        "By biomech device-validation conventions (CCC > 0.60 AND 95% LoA "
        "half-width < ±10°), the following slots are validated:"
    )
    lines.append("")
    validated = [
        r for r in rows
        if r["classification"] in ("Excellent", "Good")
    ]
    if validated:
        for r in validated:
            lines.append(
                f"- **{r['target']} / {r['view']}** "
                f"(n={r['n_subjects']}, r={fmt(r['pearson_r'])}, "
                f"CCC={fmt(r['ccc_lin'])}, "
                f"95% LoA = [{fmt(r['loa_lower_deg'], 1)}°, {fmt(r['loa_upper_deg'], 1)}°], "
                f"tier={r['classification']})"
            )
    else:
        lines.append("- None at the Good/Excellent threshold — see Moderate tier.")
    lines.append("")
    moderate = [r for r in rows if r["classification"] == "Moderate"]
    if moderate:
        lines.append("Moderate tier (CCC 0.40–0.60, LoA ±10–15°) — usable with caveats:")
        lines.append("")
        for r in moderate:
            lines.append(
                f"- {r['target']} / {r['view']} "
                f"(n={r['n_subjects']}, r={fmt(r['pearson_r'])}, "
                f"CCC={fmt(r['ccc_lin'])}, "
                f"95% LoA = [{fmt(r['loa_lower_deg'], 1)}°, {fmt(r['loa_upper_deg'], 1)}°])"
            )
        lines.append("")

    # MPI rear cohort
    mpi_section = render_mpi_rear_section()
    if mpi_section:
        lines.append(mpi_section)

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- All stats computed on LOSO-CV held-out subjects, n = number of unique subjects in that slot."
    )
    lines.append(
        "- For ankle_angle_r slots, training is OpenCap-only (ASPset has no foot markers)."
    )
    lines.append(
        "- For slots where the LOSO rerun was skipped (e.g. cost-prohibitive combined "
        "datasets), CCC and MAE are reported as `—` and Pearson r / LoA come from the "
        "deploy_ready_models.json summary stats."
    )
    return "\n".join(lines) + "\n"


def _mpi_rear_stats() -> list[dict]:
    """Per-metric per-subject Pearson r distribution across the 8-subject MPI
    rear-view cohort. Uses the pre-computed per-subject r files at
    data/mpi_inf_3dhp_rear_validation/r_S*_cam7.json (and r_cam7.json for S1).

    Raw frame-aligned recomputation isn't done here because pred and GT
    streams have different sampling rates and the alignment pipeline lives
    in mpi3dhp_correlate.py / mpi3dhp_pred_angles.py. The existing per-subject
    r values are the canonical numbers reported by the rear-view validation.

    MPI rear is a time-series correlation validation (waveform-shape), not
    a per-trial ROM regression — Bland-Altman in degrees on absolute angle
    is not meaningful because the angle streams aren't calibrated to MPI's
    mocap zero pose.
    """
    rear_dir = DATA_ROOT / "mpi_inf_3dhp_rear_validation"
    subjects = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    metrics = [
        "hip_flexion_r", "hip_flexion_l",
        "knee_angle_r", "knee_angle_l",
        "ankle_angle_r", "ankle_angle_l",
        "hip_adduction_r", "hip_adduction_l",
        "lumbar_extension",
    ]
    rows = []
    for metric in metrics:
        per_subj_rows = []
        for subj in subjects:
            # S1 lives in r_cam7.json (no subject prefix); rest in r_S{N}_cam7.json
            if subj == "S1":
                rpath = rear_dir / "r_cam7.json"
            else:
                rpath = rear_dir / f"r_{subj}_cam7.json"
            if not rpath.exists():
                continue
            try:
                rdata = json.loads(rpath.read_text())
            except Exception:
                continue
            r_entry = rdata.get("results", {}).get(metric, {})
            # Use |r| because some metrics flip sign depending on coordinate
            # convention (the existing aggregate uses abs).
            r_val = r_entry.get("pearson_r_abs_gt", r_entry.get("pearson_r"))
            if r_val is None:
                continue
            r_abs = abs(float(r_val))
            n_frames = int(r_entry.get("n_frames", 0))
            per_subj_rows.append({
                "subject": subj,
                "n_frames": n_frames,
                "pearson_r_abs": r_abs,
            })
        if not per_subj_rows:
            continue
        r_arr = np.asarray([r["pearson_r_abs"] for r in per_subj_rows])
        rows.append({
            "metric": metric,
            "n_subjects": int(len(per_subj_rows)),
            "n_frames_total": int(sum(r["n_frames"] for r in per_subj_rows)),
            "per_subject_r": {
                "mean_abs_r": float(np.mean(r_arr)),
                "sd_abs_r": float(np.std(r_arr, ddof=1)) if r_arr.size > 1 else float("nan"),
                "min_abs_r": float(np.min(r_arr)),
                "max_abs_r": float(np.max(r_arr)),
                "values": [round(float(r), 3) for r in r_arr.tolist()],
            },
            "per_subject": per_subj_rows,
        })
    return rows


def render_mpi_rear_section() -> str:
    rows = _mpi_rear_stats()
    if not rows:
        return ""
    def fmt(v, prec=2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f"{v:.{prec}f}"
    lines = []
    lines.append("## MPI-INF-3DHP rear-view cohort (cam7, n=8 subjects)")
    lines.append("")
    lines.append(
        "Per-frame joint-angle validation against MPI-INF-3DHP mocap on the rear "
        "oblique camera (cam7, azimuth −62°). Methodology differs from the deploy "
        "ROM slots above: this is a per-frame waveform-shape correlation, not a "
        "per-trial ROM regression. Reported as per-subject Pearson r summarised "
        "across the 8 subjects."
    )
    lines.append("")
    lines.append("### Per-subject Pearson r (waveform shape agreement, |r|)")
    lines.append("")
    lines.append("| Metric | n subj | n frames | mean abs(r) ± SD | range |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for r in rows:
        psr = r["per_subject_r"]
        lines.append(
            f"| {r['metric']} | {r['n_subjects']} | {r['n_frames_total']} | "
            f"{fmt(psr['mean_abs_r'])} ± {fmt(psr['sd_abs_r'])} | "
            f"[{fmt(psr['min_abs_r'])}, {fmt(psr['max_abs_r'])}] |"
        )
    lines.append("")
    lines.append("Notes: |r| is reported because some metrics flip sign with coordinate convention (an absolute correlation of 0.9 means the waveform is tracked, just inverted). The MPI rear validation is a waveform-shape claim — it does not warrant absolute-angle Bland-Altman because the predicted angle streams are not calibrated to MPI's mocap zero pose. A Bland-Altman accuracy claim on MPI rear would require a per-subject calibration step that the current pipeline does not perform.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-combined", action="store_true",
        help="Skip v12/v13/v14 combined datasets (use summary stats only for them)",
    )
    args = parser.parse_args()
    out = run(skip_combined=args.skip_combined)
    write_outputs(out)
    print(f"\nWrote {OUT_DIR / 'per_slot_validity.json'}")
    print(f"Wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
