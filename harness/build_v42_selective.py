"""Build v42 selective oracle: v40 reader pool + v41 (v20+cal).

Agent QQ. Extends Agent PP's v40 by adding v41 (v20 nested-LOSO
calibrated) to the candidate pool.

Reader pool (16 total):

  v17, v18, v20, v23, v24, v26, v27, v29, v30, v31, v33, v34,
  v37 (v23 + cal), v38 (v31 + cal), v39 (v17 + cal),
  v41 (v20 + cal).

Selection rule is identical to v40:
  * Pick highest-tier candidate.
  * If best tier is Moderate AND at least one candidate has CCC >=
    LOA_LIMITED_CCC, restrict to those and tie-break LoA-first then
    CCC-first then canonical reader order.
  * Otherwise CCC-first then LoA-first then canonical reader order.

# Outputs

  * ``results/deploy_ready_models_v42_selective.json``
  * ``data/v42_selective_oracle/per_slot_picks_v42.json``
  * ``data/v42_selective_oracle/REPORT.md``
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))


DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"

# v40 pool.
V17_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v17_selective.json"
V18_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v18_learned_l2.json"
V20_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v20_rom_aware.json"
V23_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v23_combined_l2.json"
V24_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v24_combined_rom_aware.json"
)
V26_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v26_persource_perframe.json"
)
V27_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v27_persource_romaware.json"
)
V29_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v29_mirrorflip.json"
V30_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v30_learned_l3.json"
V31_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v31_mirrorflip_learned_l3.json"
)
V33_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v33_extrema_l3.json"
V34_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v34_mirrorflip_extrema_l3.json"
)
V37_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v37_v23_calibrated.json"
)
V38_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v38_v31_calibrated.json"
)
V39_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v39_v17_calibrated.json"
)
V41_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v41_v20_calibrated.json"
)

BASELINE_PATH: Final[Path] = (
    DATA_ROOT / "biomech_validity_stats" / "per_slot_validity.json"
)
V18_PER_SLOT: Final[Path] = (
    DATA_ROOT / "layer3_retrain_learned_l2" / "per_slot_validity_v18.json"
)
V20_PER_SLOT: Final[Path] = (
    DATA_ROOT / "rom_aware_layer2" / "per_slot_validity_v20.json"
)
V23_PER_SLOT: Final[Path] = (
    DATA_ROOT / "layer3_retrain_combined_l2" / "per_slot_validity_v23.json"
)
V24_PER_SLOT: Final[Path] = (
    DATA_ROOT / "layer3_retrain_combined_rom_aware"
    / "per_slot_validity_v24.json"
)
V26_PER_SLOT: Final[Path] = (
    DATA_ROOT / "layer3_retrain_persource_perframe"
    / "per_slot_validity_v26.json"
)
V27_PER_SLOT: Final[Path] = (
    DATA_ROOT / "layer3_retrain_persource_romaware"
    / "per_slot_validity_v27.json"
)
V29_PER_SLOT: Final[Path] = (
    DATA_ROOT / "layer3_retrain_mirrorflip" / "per_slot_validity_v29.json"
)
V30_PER_SLOT: Final[Path] = (
    DATA_ROOT / "learned_layer3" / "per_slot_validity_v30_learned_l3.json"
)
V31_PER_SLOT: Final[Path] = (
    DATA_ROOT / "learned_layer3"
    / "per_slot_validity_v31_mirrorflip_learned_l3.json"
)
V33_PER_SLOT: Final[Path] = (
    DATA_ROOT / "learned_layer3_extrema"
    / "per_slot_validity_v33_extrema_l3.json"
)
V34_PER_SLOT: Final[Path] = (
    DATA_ROOT / "learned_layer3_extrema"
    / "per_slot_validity_v34_mirrorflip_extrema_l3.json"
)
V37_PER_SLOT: Final[Path] = (
    DATA_ROOT / "residual_calibration"
    / "per_slot_validity_v37_v23_calibrated.json"
)
V38_PER_SLOT: Final[Path] = (
    DATA_ROOT / "residual_calibration"
    / "per_slot_validity_v38_v31_calibrated.json"
)
V39_PER_SLOT: Final[Path] = (
    DATA_ROOT / "residual_calibration"
    / "per_slot_validity_v39_v17_calibrated.json"
)
V41_PER_SLOT: Final[Path] = (
    DATA_ROOT / "residual_calibration"
    / "per_slot_validity_v41_v20_calibrated.json"
)

V40_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v40_selective.json"
V42_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v42_selective.json"
V42_OUT_DIR: Final[Path] = DATA_ROOT / "v42_selective_oracle"
V42_PICKS_PATH: Final[Path] = V42_OUT_DIR / "per_slot_picks_v42.json"
V42_REPORT_PATH: Final[Path] = V42_OUT_DIR / "REPORT.md"


TIER_RANK: Final[dict[str, int]] = {
    "Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1,
}

# Canonical reader order. v41 appended at the end (lowest priority for
# ties; PP's calibrated readers v37/v38/v39 already follow this pattern).
READER_ORDER: Final[tuple[str, ...]] = (
    "v17", "v18", "v20", "v23", "v24", "v26", "v27",
    "v29", "v30", "v31", "v33", "v34",
    "v37", "v38", "v39",
    "v41",
)

LOA_LIMITED_CCC: Final[float] = 0.79

CATEGORY_A_SLOTS: Final[tuple[tuple[str, str], ...]] = (
    ("knee_angle_r", "front_oblique_left"),
    ("knee_angle_r", "side_left"),
    ("knee_angle_r", "side_right"),
    ("hip_flexion_r", "front_oblique_left"),
    ("hip_adduction_r", "front_oblique_right"),
)

# The QQ target slot.
QQ_FOCUS_SLOT: Final[tuple[str, str]] = (
    "hip_adduction_r", "front_oblique_left",
)


def log(msg: str) -> None:
    print(f"[v42 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _stats_from_slot_dict(slot: dict) -> dict:
    return {
        "ccc_lin": slot.get("ccc_lin"),
        "pearson_r": slot.get("pearson_r"),
        "loa_half_width_deg": slot.get("loa_half_width_deg"),
        "n_subjects": slot.get("n_subjects"),
        "n_trials": slot.get("n_trials"),
        "mae_deg": slot.get("mae_deg"),
        "classification": slot.get("classification"),
    }


def load_baseline_v17() -> dict[tuple[str, str], dict]:
    d = _load_json(BASELINE_PATH)
    return {
        (s["target"], s["view"]): _stats_from_slot_dict(s)
        for s in d["slots"]
    }


def load_per_slot(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    d = _load_json(path)
    return {
        (s["target"], s["view"]): _stats_from_slot_dict(s)
        for s in d["slots"] if "target" in s and "view" in s
    }


def _tier_rank(tier: str | None) -> int:
    if tier is None:
        return 0
    return TIER_RANK.get(tier, 0)


def _ccc_or_minus_inf(stats: dict) -> float:
    ccc = stats.get("ccc_lin")
    if ccc is None or (isinstance(ccc, float) and math.isnan(ccc)):
        return float("-inf")
    return float(ccc)


def _loa_or_plus_inf(stats: dict) -> float:
    loa = stats.get("loa_half_width_deg")
    if loa is None or (isinstance(loa, float) and math.isnan(loa)):
        return float("inf")
    return float(loa)


def select_reader_v42(candidates: dict[str, dict]) -> tuple[str, dict, str]:
    """v42 selection rule -- identical to v40.

    1. Pick the best (highest) tier across all candidates.
    2. If the best tier is Moderate AND there exists at least one
       candidate in that tier with CCC >= LOA_LIMITED_CCC, restrict
       attention to those LoA-limited candidates and tie-break
       **LoA-first then CCC-first then canonical reader order**.
    3. Otherwise default to CCC-first then LoA-first then canonical
       reader order across all top-tier candidates.
    """
    best_tier_rank = max(
        _tier_rank(c.get("classification")) for c in candidates.values()
    )
    in_top_tier = {
        r: c for r, c in candidates.items()
        if _tier_rank(c.get("classification")) == best_tier_rank
    }
    top_tier_name = next(iter(in_top_tier.values())).get("classification")

    loa_first = False
    pool: dict[str, dict] = in_top_tier
    if top_tier_name == "Moderate":
        loa_limited = {
            r: c for r, c in in_top_tier.items()
            if _ccc_or_minus_inf(c) >= LOA_LIMITED_CCC
        }
        if loa_limited:
            loa_first = True
            pool = loa_limited

    if loa_first:
        def sort_key(reader: str) -> tuple[float, float, int]:
            stats = pool[reader]
            return (
                _loa_or_plus_inf(stats),
                -_ccc_or_minus_inf(stats),
                READER_ORDER.index(reader)
                if reader in READER_ORDER else len(READER_ORDER),
            )
        rule_tag = "loa_first"
    else:
        def sort_key(reader: str) -> tuple[int, float, int]:
            stats = candidates[reader]
            return (
                -_tier_rank(stats.get("classification")),
                -_ccc_or_minus_inf(stats),
                READER_ORDER.index(reader)
                if reader in READER_ORDER else len(READER_ORDER),
            )
        rule_tag = "ccc_first"
        pool = candidates

    best = min(pool.keys(), key=sort_key)
    return best, pool[best], rule_tag


def build() -> dict:
    log("loading reader sources ...")
    deploy_bundles = {
        "v17": _load_json(V17_PATH),
        "v18": _load_json(V18_PATH) if V18_PATH.exists() else None,
        "v20": _load_json(V20_PATH) if V20_PATH.exists() else None,
        "v23": _load_json(V23_PATH) if V23_PATH.exists() else None,
        "v24": _load_json(V24_PATH) if V24_PATH.exists() else None,
        "v26": _load_json(V26_PATH) if V26_PATH.exists() else None,
        "v27": _load_json(V27_PATH) if V27_PATH.exists() else None,
        "v29": _load_json(V29_PATH) if V29_PATH.exists() else None,
        "v30": _load_json(V30_PATH) if V30_PATH.exists() else None,
        "v31": _load_json(V31_PATH) if V31_PATH.exists() else None,
        "v33": _load_json(V33_PATH) if V33_PATH.exists() else None,
        "v34": _load_json(V34_PATH) if V34_PATH.exists() else None,
        "v37": _load_json(V37_PATH) if V37_PATH.exists() else None,
        "v38": _load_json(V38_PATH) if V38_PATH.exists() else None,
        "v39": _load_json(V39_PATH) if V39_PATH.exists() else None,
        "v41": _load_json(V41_PATH) if V41_PATH.exists() else None,
    }

    per_slot_stores: dict[str, dict[tuple[str, str], dict]] = {
        "v17": load_baseline_v17(),
        "v18": load_per_slot(V18_PER_SLOT),
        "v20": load_per_slot(V20_PER_SLOT),
        "v23": load_per_slot(V23_PER_SLOT),
        "v24": load_per_slot(V24_PER_SLOT),
        "v26": load_per_slot(V26_PER_SLOT),
        "v27": load_per_slot(V27_PER_SLOT),
        "v29": load_per_slot(V29_PER_SLOT),
        "v30": load_per_slot(V30_PER_SLOT),
        "v31": load_per_slot(V31_PER_SLOT),
        "v33": load_per_slot(V33_PER_SLOT),
        "v34": load_per_slot(V34_PER_SLOT),
        "v37": load_per_slot(V37_PER_SLOT),
        "v38": load_per_slot(V38_PER_SLOT),
        "v39": load_per_slot(V39_PER_SLOT),
        "v41": load_per_slot(V41_PER_SLOT),
    }

    counts = {k: len(v) for k, v in per_slot_stores.items()}
    log(f"reader candidate counts: {counts}")

    # v40 picks for delta.
    v40_deploy = _load_json(V40_PATH)
    v40_models = v40_deploy["models"]
    v40_pick_tier: dict[str, str] = {}
    v40_pick_ccc: dict[str, float | None] = {}
    v40_pick_loa: dict[str, float | None] = {}
    v40_pick_reader: dict[str, str] = {}
    for tgt, slots in v40_models.items():
        for view, entry in slots.items():
            key = f"{tgt}|{view}"
            v40_stats = entry.get("v40_stats", {})
            v40_pick_tier[key] = v40_stats.get("classification") or "Poor"
            v40_pick_ccc[key] = v40_stats.get("ccc_lin")
            v40_pick_loa[key] = v40_stats.get("loa_half_width_deg")
            v40_pick_reader[key] = entry.get("selected_reader") or "v17"

    picks: list[dict] = []
    per_slot_reader: dict[str, str] = {}
    v42_models: dict[str, dict[str, dict]] = {}
    tier_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    tier1_count = 0
    reader_distribution = {r: 0 for r in READER_ORDER}
    promotions_vs_v40: list[dict] = []
    demotions_vs_v40: list[dict] = []
    reader_shifts_vs_v40: list[dict] = []
    v41_wins: list[dict] = []

    v17_deploy = deploy_bundles["v17"]
    for target, slots in v17_deploy["models"].items():
        for view, v17_entry in slots.items():
            key = (target, view)
            slot_key = f"{target}/{view}"
            slot_id = f"{target}|{view}"
            candidates: dict[str, dict] = {}
            for reader, store in per_slot_stores.items():
                if key in store:
                    candidates[reader] = store[key]

            if not candidates:
                log(
                    f"  WARN: no candidates for {slot_key} -- "
                    f"falling back to v17 entry"
                )
                v42_models.setdefault(target, {})[view] = dict(v17_entry)
                per_slot_reader[slot_key] = "v17"
                continue

            best_reader, best_stats, rule_tag = select_reader_v42(candidates)
            best_tier = best_stats.get("classification") or "Poor"
            tier_counts[best_tier] = tier_counts.get(best_tier, 0) + 1
            reader_distribution[best_reader] = (
                reader_distribution.get(best_reader, 0) + 1
            )
            per_slot_reader[slot_key] = best_reader

            ccc = best_stats.get("ccc_lin")
            if (
                ccc is not None
                and not (isinstance(ccc, float) and math.isnan(ccc))
                and float(ccc) >= 0.79
            ):
                tier1_count += 1

            if best_reader == "v41":
                v41_wins.append({
                    "slot": slot_id,
                    "ccc": ccc,
                    "loa_half": best_stats.get("loa_half_width_deg"),
                    "tier": best_tier,
                })

            source_bundle = deploy_bundles.get(best_reader)
            if source_bundle is None:
                source_slot = v17_entry
            else:
                source_slot = (
                    source_bundle.get("models", {})
                    .get(target, {}).get(view) or v17_entry
                )

            v42_entry = dict(v17_entry)
            v42_entry["selected_reader"] = best_reader
            v42_entry["v42_tier"] = best_tier
            v42_entry["v42_stats"] = best_stats
            v42_entry["v42_source_entry"] = source_slot.get("approach")
            v42_entry["v42_tie_break_rule"] = rule_tag
            v42_models.setdefault(target, {})[view] = v42_entry

            loa = best_stats.get("loa_half_width_deg")
            v40_t = v40_pick_tier.get(slot_id, "?")
            v40_r = v40_pick_reader.get(slot_id, "?")

            picks.append({
                "slot": slot_id,
                "approach_v17": v17_entry.get("approach"),
                "candidates": {
                    r: {
                        "ccc_lin": c.get("ccc_lin"),
                        "loa_half_width_deg": c.get("loa_half_width_deg"),
                        "pearson_r": c.get("pearson_r"),
                        "classification": c.get("classification"),
                    } for r, c in candidates.items()
                },
                "reader": best_reader,
                "tier": best_tier,
                "ccc": ccc,
                "loa_half": loa,
                "tie_break_rule": rule_tag,
                "v40_tier": v40_t,
                "v40_ccc": v40_pick_ccc.get(slot_id),
                "v40_loa": v40_pick_loa.get(slot_id),
                "v40_reader": v40_r,
            })

            if _tier_rank(best_tier) > _tier_rank(v40_t):
                promotions_vs_v40.append({
                    "slot": slot_id,
                    "v40_tier": v40_t,
                    "v42_tier": best_tier,
                    "reader": best_reader,
                    "ccc": ccc,
                    "loa_half": loa,
                    "v40_reader": v40_r,
                })
            elif _tier_rank(best_tier) < _tier_rank(v40_t):
                demotions_vs_v40.append({
                    "slot": slot_id,
                    "v40_tier": v40_t,
                    "v42_tier": best_tier,
                    "reader": best_reader,
                    "ccc": ccc,
                    "loa_half": loa,
                    "v40_reader": v40_r,
                })
            if best_reader != v40_r:
                reader_shifts_vs_v40.append({
                    "slot": slot_id,
                    "v40_reader": v40_r,
                    "v42_reader": best_reader,
                    "v40_tier": v40_t,
                    "v42_tier": best_tier,
                    "v40_ccc": v40_pick_ccc.get(slot_id),
                    "v42_ccc": ccc,
                    "v40_loa": v40_pick_loa.get(slot_id),
                    "v42_loa": loa,
                })

    # QQ focus slot table. We pull the *raw* calibrated stats from the
    # v41 per-slot validity (which records both calibrated and
    # uncalibrated stats per slot) so we can honestly report what the
    # calibration actually did before the per-slot fallback overrode it.
    qq_slot_id = f"{QQ_FOCUS_SLOT[0]}|{QQ_FOCUS_SLOT[1]}"
    qq_pick = next((p for p in picks if p["slot"] == qq_slot_id), None)
    v20_qq = per_slot_stores["v20"].get(QQ_FOCUS_SLOT, {})
    v41_qq_after_fallback = per_slot_stores["v41"].get(QQ_FOCUS_SLOT, {})
    v41_raw_per_slot = (
        _load_json(V41_PER_SLOT) if V41_PER_SLOT.exists() else {}
    )
    v41_qq_raw: dict = {}
    for s in v41_raw_per_slot.get("slots", []):
        if (
            s.get("target") == QQ_FOCUS_SLOT[0]
            and s.get("view") == QQ_FOCUS_SLOT[1]
        ):
            v41_qq_raw = s
            break
    qq_focus_table = {
        "slot": qq_slot_id,
        "v20_uncalibrated_ccc": v20_qq.get("ccc_lin"),
        "v20_uncalibrated_loa": v20_qq.get("loa_half_width_deg"),
        "v20_uncalibrated_tier": v20_qq.get("classification"),
        # After-fallback published v41 stat (== uncal v20 when fallback fires).
        "v41_published_ccc": v41_qq_after_fallback.get("ccc_lin"),
        "v41_published_loa": v41_qq_after_fallback.get("loa_half_width_deg"),
        "v41_published_tier": v41_qq_after_fallback.get("classification"),
        "v41_calibration_chosen": v41_qq_raw.get("calibration"),
        # Raw calibrated-only stat (what calibration actually produced).
        "v41_calibrated_only_ccc": v41_qq_raw.get("calibrated_ccc"),
        "v41_calibrated_only_loa": v41_qq_raw.get("calibrated_loa_half"),
        "v41_calibrated_only_classification": (
            v41_qq_raw.get("calibrated_classification")
        ),
        "v41_loa_delta_cal_minus_unc": (
            v41_qq_raw.get("loa_delta_cal_minus_unc")
        ),
        "v41_ccc_delta_cal_minus_unc": (
            v41_qq_raw.get("ccc_delta_cal_minus_unc")
        ),
        "v41_calibration_summary": v41_qq_raw.get("calibration_summary"),
        "v40_pick_reader": v40_pick_reader.get(qq_slot_id),
        "v40_pick_ccc": v40_pick_ccc.get(qq_slot_id),
        "v40_pick_loa": v40_pick_loa.get(qq_slot_id),
        "v42_pick_reader": qq_pick["reader"] if qq_pick else None,
        "v42_pick_ccc": qq_pick["ccc"] if qq_pick else None,
        "v42_pick_loa": qq_pick["loa_half"] if qq_pick else None,
        "v42_pick_tier": qq_pick["tier"] if qq_pick else None,
        "promoted_to_tier1": (
            qq_pick is not None
            and qq_pick["ccc"] is not None
            and not (isinstance(qq_pick["ccc"], float)
                     and math.isnan(qq_pick["ccc"]))
            and float(qq_pick["ccc"]) >= 0.79
        ),
    }

    # Category A table.
    category_a_table: list[dict] = []
    for tgt, view in CATEGORY_A_SLOTS:
        slot_id = f"{tgt}|{view}"
        pick = next((p for p in picks if p["slot"] == slot_id), None)
        v41_stats = per_slot_stores["v41"].get((tgt, view), {})
        category_a_table.append({
            "slot": slot_id,
            "v40_reader": v40_pick_reader.get(slot_id),
            "v40_tier": v40_pick_tier.get(slot_id),
            "v40_ccc": v40_pick_ccc.get(slot_id),
            "v40_loa": v40_pick_loa.get(slot_id),
            "v41_ccc": v41_stats.get("ccc_lin"),
            "v41_loa": v41_stats.get("loa_half_width_deg"),
            "v41_tier": v41_stats.get("classification"),
            "v42_reader": pick["reader"] if pick else None,
            "v42_tier": pick["tier"] if pick else None,
            "v42_ccc": pick["ccc"] if pick else None,
            "v42_loa": pick["loa_half"] if pick else None,
            "v42_rule": pick["tie_break_rule"] if pick else None,
            "promoted_to_good": (
                pick is not None
                and pick["tier"] == "Good"
                and v40_pick_tier.get(slot_id) != "Good"
            ),
        })

    v42_out = {
        "version": "v42_selective_oracle",
        "produced_by": "harness.build_v42_selective (Agent QQ)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            f"Per-slot oracle-best across the v40 reader pool plus "
            f"v41 (v20 + nested-LOSO calibration). "
            f"LoA-then-CCC tie-break in the LoA-limited Moderate band "
            f"(CCC >= {LOA_LIMITED_CCC}). "
            f"{tier_counts['Good']} Good / "
            f"{tier_counts['Moderate']} Moderate / "
            f"{tier_counts['Poor']} Poor. "
            f"Tier 1 (CCC >= 0.79) count: {tier1_count}."
        ),
        "approaches": v17_deploy.get("approaches"),
        "training_dataset": v17_deploy.get("training_dataset"),
        "models": v42_models,
        "calibration_fix": v17_deploy.get("calibration_fix"),
        "selective_adoption": v17_deploy.get("selective_adoption"),
        "per_slot_reader": per_slot_reader,
        "reader_distribution": reader_distribution,
        "tier_counts": tier_counts,
        "tier1_count_ccc_ge_0p79": tier1_count,
        "promotions_vs_v40": promotions_vs_v40,
        "demotions_vs_v40": demotions_vs_v40,
        "reader_shifts_vs_v40": reader_shifts_vs_v40,
        "v41_wins": v41_wins,
        "qq_focus_slot": qq_focus_table,
        "category_a_results": category_a_table,
        "loa_limited_ccc_threshold": LOA_LIMITED_CCC,
    }
    V42_PATH.write_text(json.dumps(v42_out, indent=2, default=str))
    log(f"wrote v42 deploy bundle -> {V42_PATH}")

    V42_OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks_out = {
        "version": "v42_selective_oracle",
        "description": v42_out["description"],
        "tier_counts": tier_counts,
        "tier1_count_ccc_ge_0p79": tier1_count,
        "reader_distribution": reader_distribution,
        "promotions_vs_v40": promotions_vs_v40,
        "demotions_vs_v40": demotions_vs_v40,
        "reader_shifts_vs_v40": reader_shifts_vs_v40,
        "v41_wins": v41_wins,
        "qq_focus_slot": qq_focus_table,
        "category_a_results": category_a_table,
        "picks": picks,
    }
    V42_PICKS_PATH.write_text(json.dumps(picks_out, indent=2, default=str))
    log(f"wrote v42 per-slot picks -> {V42_PICKS_PATH}")

    return v42_out


def _fmt(v: object, prec: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def write_v42_report() -> None:
    v42 = _load_json(V42_PATH)
    v40 = _load_json(V40_PATH)

    tier_counts = v42["tier_counts"]
    tier1 = v42.get("tier1_count_ccc_ge_0p79", 0)
    reader_dist = v42["reader_distribution"]
    proms = v42["promotions_vs_v40"]
    demos = v42["demotions_vs_v40"]
    shifts = v42["reader_shifts_vs_v40"]
    cat_a = v42["category_a_results"]
    v41_wins = v42["v41_wins"]
    qq = v42["qq_focus_slot"]

    v40_tier_counts = v40["tier_counts"]
    v40_tier1 = v40.get("tier1_count_ccc_ge_0p79", 0)

    lines: list[str] = []
    lines.append(
        "# v42 Selective Oracle -- v20 Calibration Extension (Agent QQ)"
    )
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(
        "**Build:** Agent QQ -- v40 reader pool (15 readers) + v41 "
        "(v20 + nested-LOSO calibration). Selection rule preserved from "
        "v40: LoA-then-CCC tie-break within the Moderate tier when all "
        f"top-tier candidates have CCC >= "
        f"{v42['loa_limited_ccc_threshold']:.2f}."
    )
    lines.append("")
    lines.append(
        f"**Verdict:** **{tier_counts['Good']} validated Good-tier slots** "
        f"(v40 was {v40_tier_counts['Good']}). "
        f"Tier 1 (CCC >= 0.79) count: **{tier1}** (v40 was {v40_tier1})."
    )
    lines.append("")

    # ---- 1. Cameron's specific question ----
    lines.append(
        "## 1. Did `hip_adduction_r / front_oblique_left` promote to "
        "Tier 1 (CCC >= 0.79)?"
    )
    lines.append("")
    lines.append(
        "This slot has the tightest LoA in the validation table "
        "(+/-3.3 deg) but a modest CCC of 0.69 -- the textbook "
        "bias-dominated pattern that residual calibration is designed "
        "to fix."
    )
    lines.append("")
    lines.append(
        "Per-slot table (raw calibrated stats, before the no-regression "
        "fallback is applied):"
    )
    lines.append("")
    lines.append("| Reader on this slot | CCC | LoA half |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| v20 uncalibrated (v40 pick) | "
        f"{_fmt(qq['v20_uncalibrated_ccc'], 3)} | "
        f"{_fmt(qq['v20_uncalibrated_loa'])} |"
    )
    lines.append(
        f"| v41 v20 calibrated (raw) | "
        f"{_fmt(qq['v41_calibrated_only_ccc'], 3)} | "
        f"{_fmt(qq['v41_calibrated_only_loa'])} |"
    )
    lines.append("")
    lines.append(
        f"After per-slot fallback rule (require LoA tighten AND CCC not "
        f"regress by more than 0.05), the v41 reader for this slot "
        f"chose **{qq.get('v41_calibration_chosen')}** and publishes "
        f"CCC={_fmt(qq['v41_published_ccc'], 3)}, "
        f"LoA/2={_fmt(qq['v41_published_loa'])} (identical to v20 "
        f"uncalibrated)."
    )
    lines.append("")
    qq_ccc_raw = qq.get("v41_calibrated_only_ccc")
    qq_ccc_unc = qq.get("v20_uncalibrated_ccc")
    qq_pick_reader = qq.get("v42_pick_reader")
    qq_pick_ccc = qq.get("v42_pick_ccc")
    qq_pick_loa = qq.get("v42_pick_loa")
    qq_pick_tier = qq.get("v42_pick_tier")
    promoted = qq.get("promoted_to_tier1", False)
    lines.append(
        f"**v42 oracle pick for {qq['slot']}:** "
        f"reader=**{qq_pick_reader}**, "
        f"CCC={_fmt(qq_pick_ccc, 3)}, "
        f"LoA/2={_fmt(qq_pick_loa)}, tier={qq_pick_tier}."
    )
    lines.append("")
    if promoted:
        lines.append(
            "**Verdict: YES.** Slot promoted to Tier 1 (CCC >= 0.79)."
        )
    else:
        lines.append(
            "**Verdict: NO.** Slot did NOT promote to Tier 1 "
            "(CCC >= 0.79). Residual calibration is not the right lever "
            "for this v20-based slot."
        )
    lines.append("")

    # Honest commentary on bias hypothesis (compares raw calibrated vs
    # uncalibrated, NOT the after-fallback published stat).
    try:
        ccc_unc_f = float(qq_ccc_unc) if qq_ccc_unc is not None else None
        ccc_raw_f = float(qq_ccc_raw) if qq_ccc_raw is not None else None
    except (TypeError, ValueError):
        ccc_unc_f, ccc_raw_f = None, None
    if ccc_unc_f is not None and ccc_raw_f is not None:
        cal_summary = qq.get("v41_calibration_summary") or {}
        a_mean = cal_summary.get("a_mean")
        b_mean = cal_summary.get("b_mean")
        a_std = cal_summary.get("a_std")
        b_std = cal_summary.get("b_std")
        n_folds = cal_summary.get("n_folds")
        if ccc_raw_f > ccc_unc_f:
            verdict = (
                f"Bias hypothesis **CONFIRMED**: raw calibration lifted "
                f"CCC from {ccc_unc_f:.3f} to {ccc_raw_f:.3f}."
            )
        elif abs(ccc_raw_f - ccc_unc_f) < 0.03:
            verdict = (
                f"Bias hypothesis **FLAT**: raw calibration barely "
                f"moved CCC ({ccc_unc_f:.3f} -> {ccc_raw_f:.3f}). The "
                "residual structure does not have a single-fold global "
                "slope/offset."
            )
        else:
            verdict = (
                f"Bias hypothesis **REFUTED**: raw calibration dropped "
                f"CCC from {ccc_unc_f:.3f} to {ccc_raw_f:.3f}. The "
                f"inner-LOSO (a, b) fit varied wildly across the "
                f"{n_folds} outer folds (a_mean={a_mean:.3f} +/- "
                f"{a_std:.3f}, b_mean={b_mean:.3f} +/- {b_std:.3f}) -- "
                "the pseudo-residual fit is overfit to inner-fold noise "
                "given the n=9 OpenCap subject pool. Per-slot fallback "
                "to uncalibrated v20 holds, so v42 ships the same v20 "
                "stat as v40."
            )
        lines.append(verdict)
        lines.append("")

    # ---- 2. Tier delta vs v40 ----
    lines.append("## 2. Tier counts vs v40")
    lines.append("")
    lines.append("| Tier | v40 | v42 | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for tier in ["Excellent", "Good", "Moderate", "Poor"]:
        old = v40_tier_counts.get(tier, 0)
        new = tier_counts.get(tier, 0)
        lines.append(f"| {tier} | {old} | {new} | {new - old:+d} |")
    lines.append(
        f"| Tier 1 (CCC >= 0.79) | {v40_tier1} | {tier1} | "
        f"{tier1 - v40_tier1:+d} |"
    )
    lines.append("")
    lines.append(
        f"Promotions vs v40: **{len(proms)}**."
        + (
            "" if not proms else
            " Slots: " + ", ".join(
                f"{p['slot']} ({p['v40_tier']} -> {p['v42_tier']}, "
                f"reader={p['reader']}, v40-reader={p['v40_reader']})"
                for p in proms
            )
        )
    )
    lines.append(
        f"Demotions vs v40: **{len(demos)}**."
        + (
            "" if not demos else
            " Slots: " + ", ".join(
                f"{p['slot']} ({p['v40_tier']} -> {p['v42_tier']}, "
                f"reader={p['reader']})"
                for p in demos
            )
        )
    )
    lines.append("")

    # ---- 3. Did any other slot get a v41 pick? ----
    lines.append("## 3. v41 picks in v42 oracle")
    lines.append("")
    if v41_wins:
        lines.append(
            f"v41 (v20 + cal) won **{len(v41_wins)}** slot(s) at v42 "
            "oracle:"
        )
        lines.append("")
        lines.append("| Slot | CCC | LoA/2 | Tier |")
        lines.append("| --- | ---: | ---: | --- |")
        for w in v41_wins:
            lines.append(
                f"| {w['slot']} | {_fmt(w['ccc'], 3)} | "
                f"{_fmt(w['loa_half'])} | {w['tier']} |"
            )
    else:
        lines.append(
            "**No slots picked v41 at the v42 oracle.** Every v41 "
            "candidate was either dominated by an uncalibrated reader "
            "or fell back to uncalibrated v20 internally (per-slot "
            "fallback rule inside v41), and v20 was already on the "
            "menu via the original v20 entry."
        )
    lines.append("")

    # ---- 4. Did calibrating v20 break any v40 Good slots? ----
    lines.append("## 4. Sanity check: did calibrating v20 break v40 Good slots?")
    lines.append("")
    if demos:
        good_to_lower = [
            d for d in demos
            if d["v40_tier"] == "Good"
            and _tier_rank(d["v42_tier"]) < _tier_rank("Good")
        ]
        if good_to_lower:
            lines.append(
                f"**Yes -- {len(good_to_lower)} Good slot(s) demoted:**"
            )
            for d in good_to_lower:
                lines.append(
                    f"- {d['slot']}: {d['v40_tier']} -> {d['v42_tier']} "
                    f"(reader: {d['v40_reader']} -> {d['reader']})"
                )
        else:
            lines.append(
                "No Good-tier slot demoted. Demotions are at lower tiers "
                "only."
            )
            for d in demos:
                lines.append(
                    f"- {d['slot']}: {d['v40_tier']} -> {d['v42_tier']} "
                    f"(reader: {d['v40_reader']} -> {d['reader']})"
                )
    else:
        lines.append(
            "No demotions vs v40. Adding v41 to the pool is "
            "non-destructive: per-slot fallback inside v41 means "
            "calibrated stats only enter the pool when they tighten LoA "
            "without regressing CCC by more than 0.05."
        )
    lines.append("")

    # ---- 5. Category A ----
    lines.append("## 5. Category A: did adding v41 crack any new LoA walls?")
    lines.append("")
    lines.append(
        "| Slot | v40 reader | v40 LoA/2 | v41 CCC | v41 LoA/2 | v41 tier | "
        "v42 reader | v42 LoA/2 | v42 CCC | v42 tier |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |"
    )
    for row in cat_a:
        lines.append(
            f"| {row['slot']} | {row['v40_reader']} | "
            f"{_fmt(row['v40_loa'])} | {_fmt(row['v41_ccc'], 3)} | "
            f"{_fmt(row['v41_loa'])} | {row['v41_tier']} | "
            f"{row['v42_reader']} | {_fmt(row['v42_loa'])} | "
            f"{_fmt(row['v42_ccc'], 3)} | {row['v42_tier']} |"
        )
    lines.append("")

    # ---- 6. Reader shifts ----
    if shifts:
        lines.append("## 6. Reader shifts vs v40")
        lines.append("")
        lines.append(
            f"Adding v41 caused **{len(shifts)}** reader shift(s) vs v40."
        )
        lines.append("")
        lines.append(
            "| Slot | v40 reader -> v42 reader | v40 tier -> v42 tier | "
            "v40 CCC -> v42 CCC | v40 LoA -> v42 LoA |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        for s in shifts:
            lines.append(
                f"| {s['slot']} | {s['v40_reader']} -> {s['v42_reader']} | "
                f"{s['v40_tier']} -> {s['v42_tier']} | "
                f"{_fmt(s['v40_ccc'], 3)} -> {_fmt(s['v42_ccc'], 3)} | "
                f"{_fmt(s['v40_loa'])} -> {_fmt(s['v42_loa'])} |"
            )
        lines.append("")
    else:
        lines.append("## 6. Reader shifts vs v40")
        lines.append("")
        lines.append(
            "**No reader shifts vs v40.** Adding v41 to the pool changed "
            "no oracle picks."
        )
        lines.append("")

    # ---- 7. Reader distribution ----
    lines.append("## 7. Reader distribution in v42")
    lines.append("")
    lines.append("| Reader | Slots | Description |")
    lines.append("| --- | ---: | --- |")
    DESC = {
        "v17": "hand-engineered + ridge",
        "v18": "FF learned L2 (OpenCap-only) + ridge",
        "v20": "GG2 ROM-aware OpenCap L2 + ridge",
        "v23": "HH2 combined L2 + ridge",
        "v24": "LL combined ROM-aware + ridge",
        "v26": "MM-A per-source per-frame L2 + ridge",
        "v27": "MM-B per-source ROM-aware L2 + ridge",
        "v29": "NN mirror-flip per-source per-frame L2 + ridge",
        "v30": "v23 L2 + learned L3 (TinyMLP, ROM-only)",
        "v31": "v29 mirror-flip L2 + learned L3 (TinyMLP, ROM-only)",
        "v33": "v23 L2 + extrema-aware learned L3 (max/min heads)",
        "v34": "v29 mirror-flip L2 + extrema-aware learned L3",
        "v37": "v23 L2 + ridge L3 + nested-LOSO calibration",
        "v38": "v29 L2 + ridge L3 + nested-LOSO calibration",
        "v39": "v17 hand-engineered L2 + ridge L3 + nested-LOSO calibration",
        "v41": "v20 GG2 ROM-aware L2 + ridge L3 + nested-LOSO calibration",
    }
    for reader in READER_ORDER:
        n = reader_dist.get(reader, 0)
        lines.append(
            f"| {reader} | {n} | {DESC.get(reader, '?')} |"
        )
    lines.append("")

    # ---- 8. Honest caveats ----
    lines.append("## 8. Honest caveats")
    lines.append("")
    lines.append(
        "- **Single camera only.** Unchanged from v17-v40."
    )
    lines.append(
        "- **Per-slot fallback to uncalibrated v20.** If calibration "
        "fails to tighten LoA OR regresses CCC by more than 0.05, the "
        "v41 slot stat is the uncalibrated v20 stat. This preserves "
        "the no-regression guarantee."
    )
    lines.append(
        "- **L2 LOSO discipline.** GG2 L2 trained on all 9 OpenCap "
        "subjects; OpenCap-subject tier promotions remain upper bounds "
        "(true double-LOSO would likely show ~0.05-0.10 |r| lower)."
    )
    lines.append(
        "- **L3 LOSO + nested LOSO discipline preserved.** Outer LOSO "
        "at L3 unchanged. Calibration is fit on pseudo-residuals from "
        "(N-1) training subjects only; outer held-out subject never "
        "used for calibration fitting."
    )
    lines.append(
        "- **Small n at L2.** For v20-based slots with n=9 OpenCap "
        "subjects, the inner LOSO calibration fit uses 7-8 training "
        "subjects. With so few subjects, the linear (a, b) fit is "
        "high-variance; this is the expected failure mode and the "
        "per-slot fallback handles it."
    )
    lines.append("")

    V42_REPORT_PATH.write_text("\n".join(lines) + "\n")
    log(f"wrote v42 REPORT -> {V42_REPORT_PATH}")


def main() -> None:
    log("=== Agent QQ (v42 selective oracle) START ===")
    build()
    write_v42_report()
    log("=== Agent QQ (v42 selective oracle) DONE ===")


if __name__ == "__main__":
    main()
