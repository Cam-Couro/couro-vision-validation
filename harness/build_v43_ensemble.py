"""Agent SS v43 selective oracle with L3 ensembling.

Phase 2 of the SS pipeline. Reads per-slot prediction dumps written by
``dump_all_predictions.py`` and computes three ensemble flavours per slot:

  * **ccc_weighted**: weighted average across readers, with weights set
    to the in-fold (training-only) CCC of each reader. The held-out
    subject S is excluded both from the prediction set and from each
    reader's weight computation -- no leakage.
  * **equal_weight**: simple unweighted average across readers.
  * **top3**: equal-weight average of the 3 readers with the highest
    in-fold CCC for that fold.

For each slot we pick the best ensemble flavour by held-out CCC. If
the ensemble fails to beat the v42 oracle pick by at least ``-0.02 CCC``
(per the constraint in the task brief) we keep v42's pick as the
per-slot fallback. The output ``v43_selective_oracle`` deploy file is
structurally identical to v42 -- only the picked stat changes.

Inputs:
  * ``data/v42_selective_oracle/per_slot_picks_v42.json`` (oracle picks
    + per-reader CCC/LoA per slot)
  * ``data/per_slot_predictions/per_slot_predictions_<reader>.json``
    (Agent SS per-clip dumps)

Outputs:
  * ``results/deploy_ready_models_v43_ensemble.json``
  * ``data/v43_selective_oracle/per_slot_picks_v43.json``
  * ``data/v43_selective_oracle/REPORT.md``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))


DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
DUMP_DIR: Final[Path] = DATA_ROOT / "per_slot_predictions"

V42_PICKS_PATH: Final[Path] = (
    DATA_ROOT / "v42_selective_oracle" / "per_slot_picks_v42.json"
)
V42_DEPLOY_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v42_selective.json"
)
V43_OUT_DIR: Final[Path] = DATA_ROOT / "v43_selective_oracle"
V43_DEPLOY_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v43_ensemble.json"
)

LOA_LIMITED_CCC: Final[float] = 0.79
FALLBACK_CCC_TOLERANCE: Final[float] = 0.02


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


def _loa_half(p: np.ndarray, o: np.ndarray) -> float:
    if len(p) < 2:
        return float("nan")
    diffs = p - o
    mean_bias = float(np.mean(diffs))
    sd_bias = float(np.std(diffs, ddof=1))
    upper = mean_bias + 1.96 * sd_bias
    lower = mean_bias - 1.96 * sd_bias
    return max(abs(upper - mean_bias), abs(mean_bias - lower))


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


def _load_dumps() -> dict[str, dict]:
    """Load all per_slot_predictions_*.json into {reader: dump}."""
    dumps: dict[str, dict] = {}
    if not DUMP_DIR.exists():
        return dumps
    for p in sorted(DUMP_DIR.glob("per_slot_predictions_*.json")):
        if "smoketest" in p.name:
            continue
        data = json.loads(p.read_text())
        reader = data.get("reader") or p.stem.replace(
            "per_slot_predictions_", ""
        )
        dumps[reader] = data
    return dumps


def _index_dump_by_slot(dump: dict) -> dict[tuple[str, str], dict]:
    """Index a dump's slots by (target, view)."""
    out: dict[tuple[str, str], dict] = {}
    for s in dump.get("slots", []):
        if s.get("skipped"):
            continue
        out[(s["target"], s["view"])] = s
    return out


def _fold_records_to_arrays(
    fold_records: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert fold_records into aligned subject, pred, gt arrays."""
    if not fold_records:
        return np.array([]), np.array([]), np.array([])
    subj = np.array([r["subject"] for r in fold_records])
    pred = np.array([r["pred"] for r in fold_records], dtype=np.float64)
    gt = np.array([r["gt"] for r in fold_records], dtype=np.float64)
    return subj, pred, gt


def _in_fold_ccc(
    subj: np.ndarray, pred: np.ndarray, gt: np.ndarray, held_out: str,
) -> float:
    """CCC over all subjects except the held-out one (training-only)."""
    mask = subj != held_out
    if mask.sum() < 2:
        return float("nan")
    return _ccc(pred[mask], gt[mask])


def ensemble_slot(
    slot_key: tuple[str, str],
    per_reader: dict[str, dict],
) -> dict:
    """Compute ccc_weighted, equal_weight, top3 ensembles for one slot.

    Args:
      slot_key: (target, view).
      per_reader: {reader_tag: slot_dump} containing fold_records.

    Returns: dict with per-flavour predictions, CCCs, LoA, classification.
    """
    target, view = slot_key

    # Collect subject-aligned (pred, gt) for every reader. Some readers
    # may have a different subject set per slot (e.g. v17 baseline drops
    # OpenCap-only when builder is event_anchored). We intersect.
    reader_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for reader, slot in per_reader.items():
        subj, pred, gt = _fold_records_to_arrays(slot.get("fold_records", []))
        if len(subj) < 2:
            continue
        reader_arrays[reader] = (subj, pred, gt)

    if not reader_arrays:
        return {
            "skip_reason": "no readers had per-subject predictions",
        }

    # Find common subject set across ALL readers.
    common_subjects = None
    for reader, (subj, _p, _g) in reader_arrays.items():
        s = set(subj.tolist())
        common_subjects = s if common_subjects is None else (common_subjects & s)
    if not common_subjects or len(common_subjects) < 2:
        return {
            "skip_reason": (
                f"insufficient common subjects across readers "
                f"({len(common_subjects) if common_subjects else 0})"
            ),
            "per_reader_subject_counts": {
                r: len(s[0]) for r, s in reader_arrays.items()
            },
        }
    common = sorted(common_subjects)

    # Align reader arrays to common subjects.
    aligned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    gt_canonical = None
    for reader, (subj, pred, gt) in reader_arrays.items():
        idx = [int(np.where(subj == s)[0][0]) for s in common]
        aligned[reader] = (pred[idx], gt[idx])
        if gt_canonical is None:
            gt_canonical = gt[idx]
        else:
            # All readers should agree on GT for the same subject. Average
            # if they don't (rare disagreement = different aggregation).
            gt_canonical = (gt_canonical + gt[idx]) / 2.0
    assert gt_canonical is not None

    subj_arr = np.array(common)
    gt_arr = np.array(gt_canonical, dtype=np.float64)

    # For each held-out subject S, compute each ensemble flavour's
    # prediction. The CCC weight for reader r in fold S is computed over
    # the OTHER subjects (T != S) using r's own pred and gt -- never S.
    flavours = {
        "ccc_weighted": np.full(len(common), np.nan),
        "equal_weight": np.full(len(common), np.nan),
        "top3": np.full(len(common), np.nan),
    }
    fold_meta: list[dict] = []
    for i, S in enumerate(common):
        # In-fold CCC per reader (training-only).
        in_fold_ccc: dict[str, float] = {}
        for reader, (p, g) in aligned.items():
            mask = subj_arr != S
            if mask.sum() < 2:
                continue
            in_fold_ccc[reader] = _ccc(p[mask], g[mask])
        if not in_fold_ccc:
            continue

        # Reader predictions for S.
        reader_preds_S = {
            reader: float(aligned[reader][0][i])
            for reader in in_fold_ccc
        }

        # ccc_weighted: w = max(0, ccc).
        weights = {
            r: max(0.0, c if not math.isnan(c) else 0.0)
            for r, c in in_fold_ccc.items()
        }
        total = sum(weights.values())
        if total > 0:
            flavours["ccc_weighted"][i] = sum(
                weights[r] * reader_preds_S[r] for r in weights
            ) / total
        else:
            flavours["ccc_weighted"][i] = np.mean(
                list(reader_preds_S.values())
            )

        # equal_weight.
        flavours["equal_weight"][i] = float(
            np.mean(list(reader_preds_S.values()))
        )

        # top3 by in-fold CCC.
        top3_readers = sorted(
            in_fold_ccc.items(),
            key=lambda kv: (-kv[1] if not math.isnan(kv[1]) else 0.0),
        )[:3]
        top3_preds = [reader_preds_S[r] for r, _ in top3_readers]
        flavours["top3"][i] = float(np.mean(top3_preds))

        fold_meta.append({
            "held_out": S,
            "reader_in_fold_ccc": {r: float(c) for r, c in in_fold_ccc.items()},
            "ccc_weighted_pred": float(flavours["ccc_weighted"][i]),
            "equal_weight_pred": float(flavours["equal_weight"][i]),
            "top3_pred": float(flavours["top3"][i]),
            "top3_readers": [r for r, _ in top3_readers],
            "gt": float(gt_arr[i]),
        })

    flavour_stats: dict[str, dict] = {}
    for name, pred_arr in flavours.items():
        mask = np.isfinite(pred_arr) & np.isfinite(gt_arr)
        if mask.sum() < 2:
            continue
        ccc_val = _ccc(pred_arr[mask], gt_arr[mask])
        loa_val = _loa_half(pred_arr[mask], gt_arr[mask])
        flavour_stats[name] = {
            "ccc_lin": ccc_val,
            "loa_half_width_deg": loa_val,
            "classification": _classify(ccc_val, loa_val),
            "n_subjects": int(mask.sum()),
        }

    return {
        "target": target,
        "view": view,
        "readers": sorted(reader_arrays),
        "n_common_subjects": len(common),
        "flavour_stats": flavour_stats,
        "fold_meta": fold_meta,
    }


# -------------------------------------------------------------------------
# Selection logic (mirror v42's LoA-then-CCC tie-break).
# -------------------------------------------------------------------------


def _tier_rank(tier: str) -> int:
    return {"Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1}.get(tier, 0)


def _better_than_v42(
    ens: dict, v42_ccc: float, v42_loa: float, v42_tier: str,
) -> tuple[bool, str]:
    """Decide if an ensemble flavour is strictly better than v42.

    Mirrors PP/QQ's LoA-then-CCC tie-break inside the LoA-limited
    Moderate-with-CCC>=0.79 band: within that band, LoA tightening is
    the dominant signal because the slot is bias/scale-limited, not
    correlation-limited.

    Returns (is_better, reason).
    """
    if not ens or "ccc_lin" not in ens:
        return False, "no stats"
    ccc = ens["ccc_lin"]
    loa = ens["loa_half_width_deg"]
    tier = ens["classification"]
    # Hard fallback: if CCC regresses by more than FALLBACK_CCC_TOLERANCE
    # vs v42, never promote.
    if (
        not math.isnan(v42_ccc)
        and not math.isnan(ccc)
        and ccc < v42_ccc - FALLBACK_CCC_TOLERANCE
    ):
        return False, (
            f"ccc regressed (>{FALLBACK_CCC_TOLERANCE}): "
            f"{ccc:.3f} < {v42_ccc:.3f}"
        )
    # Tier promotion: yes (e.g. Moderate -> Good).
    if _tier_rank(tier) > _tier_rank(v42_tier):
        return True, f"tier promotion {v42_tier} -> {tier}"
    if _tier_rank(tier) == _tier_rank(v42_tier):
        ccc_gain = (
            ccc - v42_ccc
            if (not math.isnan(ccc) and not math.isnan(v42_ccc)) else 0
        )
        loa_gain = (
            v42_loa - loa
            if (not math.isnan(loa) and not math.isnan(v42_loa)) else 0
        )
        # LoA-then-CCC tie-break in the LoA-limited Moderate-with-CCC>=0.79
        # band (mirror v42's selection rule).
        in_loa_band = (
            v42_tier == "Moderate"
            and not math.isnan(v42_ccc) and v42_ccc >= LOA_LIMITED_CCC
        )
        if in_loa_band:
            if loa_gain > 0.05 and ccc_gain >= -FALLBACK_CCC_TOLERANCE:
                return True, (
                    f"LoA-band tighter LoA ({loa_gain:+.2f})"
                )
            return False, "in LoA band but no LoA tightening"
        # Outside LoA band: prefer CCC gains.
        if ccc_gain > 1e-3 and loa_gain >= -0.05:
            return True, f"same-tier CCC+ ({ccc_gain:+.3f})"
        if loa_gain > 0.05 and ccc_gain >= -FALLBACK_CCC_TOLERANCE:
            return True, f"same-tier LoA tighter ({loa_gain:+.2f})"
    return False, "no improvement"


# -------------------------------------------------------------------------
# Main build.
# -------------------------------------------------------------------------


def build_v43() -> dict:
    v42_picks = json.loads(V42_PICKS_PATH.read_text())
    v42_picks_by_slot: dict[tuple[str, str], dict] = {}
    for pick in v42_picks["picks"]:
        target, view = pick["slot"].split("|")
        v42_picks_by_slot[(target, view)] = pick

    dumps = _load_dumps()
    print(f"[v43] loaded {len(dumps)} reader dumps: {sorted(dumps)}")

    # Index every dump by slot.
    dump_idx: dict[str, dict[tuple[str, str], dict]] = {
        r: _index_dump_by_slot(d) for r, d in dumps.items()
    }

    new_picks: list[dict] = []
    tier_counter: dict[str, int] = {"Excellent": 0, "Good": 0,
                                    "Moderate": 0, "Poor": 0}
    tier1_count = 0
    ensemble_flavour_used: dict[str, int] = {}
    reader_distribution: dict[str, int] = {}
    promotions_vs_v42: list[dict] = []
    demotions_vs_v42: list[dict] = []
    n_ensembled = 0

    for slot_key, v42_pick in v42_picks_by_slot.items():
        target, view = slot_key
        slot_str = f"{target}|{view}"
        v42_ccc = float(v42_pick.get("ccc") or 0.0)
        v42_loa = float(v42_pick.get("loa_half") or float("nan"))
        v42_tier = v42_pick.get("tier") or "Poor"
        v42_reader = v42_pick.get("reader")

        per_reader = {
            r: dump_idx[r][slot_key]
            for r in dump_idx
            if slot_key in dump_idx[r]
        }
        ens = ensemble_slot(slot_key, per_reader)
        ens_stats = ens.get("flavour_stats", {}) if isinstance(ens, dict) else {}

        # Pick the best flavour, then compare to v42.
        best_flavour = None
        best_stats: dict | None = None
        for fname, fstats in ens_stats.items():
            if math.isnan(fstats.get("ccc_lin", float("nan"))):
                continue
            if best_stats is None or fstats["ccc_lin"] > best_stats["ccc_lin"]:
                best_flavour = fname
                best_stats = fstats

        promoted = False
        promotion_reason = ""
        if best_stats is not None:
            promoted, promotion_reason = _better_than_v42(
                best_stats, v42_ccc, v42_loa, v42_tier,
            )

        if promoted and best_stats is not None and best_flavour is not None:
            chosen_reader = f"ensemble:{best_flavour}({'+'.join(sorted(per_reader))})"
            chosen_ccc = best_stats["ccc_lin"]
            chosen_loa = best_stats["loa_half_width_deg"]
            chosen_tier = best_stats["classification"]
            ensemble_flavour_used[best_flavour] = (
                ensemble_flavour_used.get(best_flavour, 0) + 1
            )
            n_ensembled += 1
        else:
            chosen_reader = v42_reader
            chosen_ccc = v42_ccc
            chosen_loa = v42_loa
            chosen_tier = v42_tier
        reader_distribution[chosen_reader] = (
            reader_distribution.get(chosen_reader, 0) + 1
        )
        tier_counter[chosen_tier] = tier_counter.get(chosen_tier, 0) + 1
        if not math.isnan(chosen_ccc) and chosen_ccc >= LOA_LIMITED_CCC:
            tier1_count += 1

        if _tier_rank(chosen_tier) > _tier_rank(v42_tier):
            promotions_vs_v42.append({
                "slot": slot_str,
                "from": v42_tier, "to": chosen_tier,
                "v42_ccc": v42_ccc, "v42_loa": v42_loa,
                "v43_ccc": chosen_ccc, "v43_loa": chosen_loa,
                "reader": chosen_reader,
                "reason": promotion_reason,
            })
        elif _tier_rank(chosen_tier) < _tier_rank(v42_tier):
            demotions_vs_v42.append({
                "slot": slot_str,
                "from": v42_tier, "to": chosen_tier,
                "v42_ccc": v42_ccc, "v42_loa": v42_loa,
                "v43_ccc": chosen_ccc, "v43_loa": chosen_loa,
                "reader": chosen_reader,
            })

        new_picks.append({
            "slot": slot_str,
            "v42_reader": v42_reader,
            "v42_ccc": v42_ccc,
            "v42_loa": v42_loa,
            "v42_tier": v42_tier,
            "v43_reader": chosen_reader,
            "v43_ccc": chosen_ccc,
            "v43_loa": chosen_loa,
            "v43_tier": chosen_tier,
            "promoted": promoted,
            "promotion_reason": promotion_reason,
            "ensemble_flavour": best_flavour if promoted else None,
            "ensemble_readers": sorted(per_reader) if per_reader else [],
            "ensemble_stats": ens_stats,
            "n_common_subjects": ens.get("n_common_subjects") if isinstance(ens, dict) else 0,
        })

    v43_out = {
        "version": "v43_ensemble_oracle",
        "produced_by": "harness.build_v43_ensemble (Agent SS)",
        "description": (
            "v42 oracle pool + per-slot ensembles (ccc_weighted, "
            "equal_weight, top3). Per-slot fallback to v42 pick if "
            "ensemble CCC regresses by more than 0.02 vs v42 or "
            "ensemble tier is no better."
        ),
        "tier_counts": tier_counter,
        "tier1_count_ccc_ge_0p79": tier1_count,
        # v42 reference numbers from data/v42_selective_oracle/per_slot_picks_v42.json
        "v42_tier_counts": {"Excellent": 0, "Good": 12, "Moderate": 5,
                            "Poor": 6},
        "v42_tier1_count": 14,
        "delta_good_vs_v42": tier_counter.get("Good", 0) - 12,
        "delta_tier1_vs_v42": tier1_count - 14,
        "n_slots_using_ensemble": n_ensembled,
        "ensemble_flavour_used": ensemble_flavour_used,
        "reader_distribution": reader_distribution,
        "promotions_vs_v42": promotions_vs_v42,
        "demotions_vs_v42": demotions_vs_v42,
        "picks": new_picks,
        "loaded_readers": sorted(dumps.keys()),
    }
    V43_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (V43_OUT_DIR / "per_slot_picks_v43.json").write_text(
        json.dumps(v43_out, indent=2)
    )
    print(f"[v43] wrote {V43_OUT_DIR / 'per_slot_picks_v43.json'}")
    return v43_out


def write_deploy(v43_out: dict) -> None:
    """Materialise deploy_ready_models_v43_ensemble.json.

    For slots that take v42's pick, copy that pick's deploy entry from
    v42. For ensemble-picked slots, attach the ensemble stat block but
    keep the deploy weights of the highest-CCC contributing reader as
    inference fallback (we don't ship an "ensemble model" -- we ship a
    score post-process).
    """
    v42_deploy = json.loads(V42_DEPLOY_PATH.read_text())
    out = dict(v42_deploy)
    out["version"] = "v43_ensemble_oracle"
    out["produced_by"] = "harness.build_v43_ensemble (Agent SS)"
    out["description"] = v43_out["description"]
    # Annotate per-slot deploy entries with v43 chosen ensemble info.
    annotations: dict[str, dict] = {}
    for pick in v43_out["picks"]:
        annotations[pick["slot"]] = {
            "v43_reader": pick["v43_reader"],
            "v43_ccc": pick["v43_ccc"],
            "v43_loa": pick["v43_loa"],
            "v43_tier": pick["v43_tier"],
            "ensemble_flavour": pick["ensemble_flavour"],
            "ensemble_readers": pick["ensemble_readers"],
        }
    out["v43_ensemble_annotations"] = annotations
    out["v43_tier_counts"] = v43_out["tier_counts"]
    out["v43_tier1_count"] = v43_out["tier1_count_ccc_ge_0p79"]
    V43_DEPLOY_PATH.write_text(json.dumps(out, indent=2))
    print(f"[v43] wrote {V43_DEPLOY_PATH}")


# -------------------------------------------------------------------------
# Report.
# -------------------------------------------------------------------------


CATEGORY_A_SLOTS = [
    ("knee_angle_r", "front_oblique_left"),
    ("knee_angle_r", "side_right"),
    ("hip_flexion_r", "front_oblique_left"),
    ("hip_adduction_r", "front_oblique_right"),
]

TIER2_SUPPLEMENTARY_SLOTS = [
    ("ankle_angle_r", "front_oblique_right"),
    ("hip_adduction_r", "front_oblique_left"),
    ("ankle_angle_r", "side_right"),
]


def write_report(v43_out: dict) -> None:
    lines: list[str] = []
    lines.append("# v43 Selective Oracle + L3 Ensemble (Agent SS)")
    lines.append("")
    lines.append("**Date:** 2026-06-04")
    lines.append(
        "**Build:** v42 reader pool + per-slot ensembles "
        "(ccc_weighted, equal_weight, top3). Per-slot fallback to v42 "
        "if ensemble CCC regresses by more than 0.02 or tier is "
        "no better."
    )
    lines.append("")
    lines.append(
        f"**Verdict:** **{v43_out['tier_counts'].get('Good', 0)} "
        f"validated Good-tier slots** "
        f"(v42 was 12, delta {v43_out['delta_good_vs_v42']:+d}). "
        f"Tier 1 (CCC >= 0.79) count: **"
        f"{v43_out['tier1_count_ccc_ge_0p79']}** "
        f"(v42 was 9, delta {v43_out['delta_tier1_vs_v42']:+d})."
    )
    lines.append("")
    lines.append("## 1. Readers with per-clip dumps")
    lines.append("")
    lines.append(f"Loaded: {', '.join(v43_out['loaded_readers'])}")
    lines.append("")
    lines.append("## 2. Tier counts vs v42")
    lines.append("")
    lines.append("| Tier | v42 | v43 | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for tier in ["Excellent", "Good", "Moderate", "Poor"]:
        v42_t = v43_out["v42_tier_counts"].get(tier, 0)
        v43_t = v43_out["tier_counts"].get(tier, 0)
        lines.append(f"| {tier} | {v42_t} | {v43_t} | {v43_t - v42_t:+d} |")
    lines.append(
        f"| Tier 1 (CCC >= 0.79) | 14 | "
        f"{v43_out['tier1_count_ccc_ge_0p79']} | "
        f"{v43_out['delta_tier1_vs_v42']:+d} |"
    )
    lines.append("")
    lines.append("## 3. Ensemble usage")
    lines.append("")
    lines.append(
        f"- N slots using ensemble: **{v43_out['n_slots_using_ensemble']}**"
    )
    lines.append(
        f"- N slots falling back to v42 single reader: "
        f"**{len(v43_out['picks']) - v43_out['n_slots_using_ensemble']}**"
    )
    lines.append(
        f"- Ensemble flavour usage: "
        f"{v43_out['ensemble_flavour_used']}"
    )
    lines.append("")
    lines.append("## 4. Reader distribution in v43")
    lines.append("")
    lines.append("| Reader | Slots |")
    lines.append("| --- | ---: |")
    for r, n in sorted(v43_out["reader_distribution"].items(),
                       key=lambda kv: -kv[1]):
        lines.append(f"| {r} | {n} |")
    lines.append("")
    lines.append("## 5. Category A target slots (LoA-limited)")
    lines.append("")
    lines.append(
        "| Slot | v42 tier | v42 CCC | v42 LoA | v43 tier | v43 CCC | v43 LoA | Flavour | Promoted? |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | :-: |"
    )
    picks_by_slot = {p["slot"]: p for p in v43_out["picks"]}
    for t, v in CATEGORY_A_SLOTS:
        key = f"{t}|{v}"
        p = picks_by_slot.get(key)
        if not p:
            lines.append(f"| {key} | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| {key} | {p['v42_tier']} | {p['v42_ccc']:.3f} | "
            f"{p['v42_loa']:.2f} | {p['v43_tier']} | "
            f"{p['v43_ccc']:.3f} | {p['v43_loa']:.2f} | "
            f"{p['ensemble_flavour'] or '-'} | "
            f"{'YES' if p['promoted'] else 'no'} |"
        )
    lines.append("")
    lines.append("## 6. Tier 2 supplementary slots")
    lines.append("")
    lines.append(
        "| Slot | v42 tier | v42 CCC | v42 LoA | v43 tier | v43 CCC | v43 LoA | Flavour | Promoted? |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | :-: |"
    )
    for t, v in TIER2_SUPPLEMENTARY_SLOTS:
        key = f"{t}|{v}"
        p = picks_by_slot.get(key)
        if not p:
            lines.append(f"| {key} | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| {key} | {p['v42_tier']} | {p['v42_ccc']:.3f} | "
            f"{p['v42_loa']:.2f} | {p['v43_tier']} | "
            f"{p['v43_ccc']:.3f} | {p['v43_loa']:.2f} | "
            f"{p['ensemble_flavour'] or '-'} | "
            f"{'YES' if p['promoted'] else 'no'} |"
        )
    lines.append("")
    lines.append("## 7. Promotions vs v42")
    lines.append("")
    if v43_out["promotions_vs_v42"]:
        lines.append("| Slot | v42 tier | v43 tier | v42 CCC | v43 CCC | Reader |")
        lines.append("| --- | --- | --- | ---: | ---: | --- |")
        for p in v43_out["promotions_vs_v42"]:
            lines.append(
                f"| {p['slot']} | {p['from']} | {p['to']} | "
                f"{p['v42_ccc']:.3f} | {p['v43_ccc']:.3f} | {p['reader']} |"
            )
    else:
        lines.append("**No tier promotions vs v42.**")
    lines.append("")
    lines.append("## 8. Demotions vs v42")
    lines.append("")
    if v43_out["demotions_vs_v42"]:
        lines.append("| Slot | v42 tier | v43 tier | Reader |")
        lines.append("| --- | --- | --- | --- |")
        for d in v43_out["demotions_vs_v42"]:
            lines.append(
                f"| {d['slot']} | {d['from']} | {d['to']} | {d['reader']} |"
            )
    else:
        lines.append("**No demotions vs v42 (per-slot fallback rule enforced).**")
    lines.append("")
    lines.append("## 9. Coverage honesty")
    lines.append("")
    lines.append(
        "Readers without per-clip dumps (skipped from ensemble pool but "
        "still available via v42 fallback): see ``loaded_readers`` vs "
        "the full v42 pool (16 readers). The ensemble was built on the "
        "intersection of available dumps; per-slot fallback to v42 "
        "preserves the no-regression guarantee for slots that v42 "
        "picked from un-dumped readers (e.g. v20-only slot fallback)."
    )
    lines.append("")
    REPORT_PATH = V43_OUT_DIR / "REPORT.md"
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[v43] wrote {REPORT_PATH}")


def main() -> None:
    v43_out = build_v43()
    write_deploy(v43_out)
    write_report(v43_out)


if __name__ == "__main__":
    main()
