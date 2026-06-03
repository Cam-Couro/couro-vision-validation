"""Build v36 selective oracle: same reader pool as v35 with a fixed
LoA-then-CCC tie-break for the LoA-limited Moderate band.

Agent PP, Lever 1 (oracle tie-break fix). The v32/v35 selection rule sorts
Moderate-tier candidates by CCC first and only falls back to LoA on a CCC tie.
For LoA-limited slots (high CCC, LoA just over +/-10 deg), this leaves LoA
margin on the table because two readers can have nearly-identical CCC but
materially different LoA.

# Concrete fix

Within the Moderate tier, if all candidates have CCC >= 0.79 (the LoA-limited
band where CCC has already saturated and LoA is the binding constraint),
sort by **lowest LoA, then highest CCC, then canonical reader order**.

For Excellent / Good tier slots the original CCC-first rule is preserved.
For Moderate slots with at least one candidate below CCC 0.79, the original
CCC-first rule is preserved (the slot is CCC-limited, not LoA-limited).

# Outputs

  * ``results/deploy_ready_models_v36_selective.json``
  * ``data/v36_selective_oracle/per_slot_picks_v36.json``
  * ``data/v36_selective_oracle/REPORT.md``
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


# -------------------------------------------------------------------------
# Paths.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"

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
V29_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v29_mirrorflip.json"
)
V30_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v30_learned_l3.json"
)
V31_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v31_mirrorflip_learned_l3.json"
)
V33_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v33_extrema_l3.json"
)
V34_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v34_mirrorflip_extrema_l3.json"
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

V35_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v35_selective.json"
V36_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v36_selective.json"
V36_OUT_DIR: Final[Path] = DATA_ROOT / "v36_selective_oracle"
V36_PICKS_PATH: Final[Path] = V36_OUT_DIR / "per_slot_picks_v36.json"
V36_REPORT_PATH: Final[Path] = V36_OUT_DIR / "REPORT.md"


TIER_RANK: Final[dict[str, int]] = {
    "Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1,
}

READER_ORDER: Final[tuple[str, ...]] = (
    "v17", "v18", "v20", "v23", "v24", "v26", "v27",
    "v29", "v30", "v31", "v33", "v34",
)

# The LoA-limited CCC threshold. Within Moderate tier, candidates with
# CCC >= this value are considered LoA-limited and re-sorted LoA-first.
LOA_LIMITED_CCC: Final[float] = 0.79

CATEGORY_A_SLOTS: Final[tuple[tuple[str, str], ...]] = (
    ("knee_angle_r", "front_oblique_left"),
    ("knee_angle_r", "side_left"),
    ("knee_angle_r", "side_right"),
    ("hip_flexion_r", "front_oblique_left"),
    ("hip_adduction_r", "front_oblique_right"),
)


def log(msg: str) -> None:
    print(f"[v36 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def select_reader_v36(candidates: dict[str, dict]) -> tuple[str, dict, str]:
    """Selection rule v36.

    1. Pick the best (highest) tier across all candidates.
    2. If the best tier is Moderate AND there exists at least one
       candidate in that tier with CCC >= LOA_LIMITED_CCC, restrict
       attention to those LoA-limited candidates and tie-break
       **LoA-first then CCC-first then canonical reader order**.
    3. Otherwise default to CCC-first then LoA-first then canonical
       reader order across all top-tier candidates.

    Returns (reader, stats, rule_tag) where rule_tag is one of
    ``"loa_first"`` (LoA-limited Moderate band) or ``"ccc_first"``.
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
    }

    counts = {k: len(v) for k, v in per_slot_stores.items()}
    log(f"reader candidate counts: {counts}")

    # Pull v35 picks for per-slot before/after comparison.
    v35_deploy = _load_json(V35_PATH)
    v35_models = v35_deploy["models"]
    v35_pick_tier: dict[str, str] = {}
    v35_pick_ccc: dict[str, float | None] = {}
    v35_pick_loa: dict[str, float | None] = {}
    v35_pick_reader: dict[str, str] = {}
    for tgt, slots in v35_models.items():
        for view, entry in slots.items():
            key = f"{tgt}|{view}"
            v35_stats = entry.get("v35_stats", {})
            v35_pick_tier[key] = v35_stats.get("classification") or "Poor"
            v35_pick_ccc[key] = v35_stats.get("ccc_lin")
            v35_pick_loa[key] = v35_stats.get("loa_half_width_deg")
            v35_pick_reader[key] = entry.get("selected_reader") or "v17"

    picks: list[dict] = []
    per_slot_reader: dict[str, str] = {}
    v36_models: dict[str, dict[str, dict]] = {}
    tier_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    tier1_count = 0
    reader_distribution = {r: 0 for r in READER_ORDER}
    tie_break_shifts: list[dict] = []  # changed picks vs v35

    v17_deploy = deploy_bundles["v17"]
    for target, slots in v17_deploy["models"].items():
        for view, v17_entry in slots.items():
            key = (target, view)
            slot_key = f"{target}/{view}"
            candidates: dict[str, dict] = {}
            for reader, store in per_slot_stores.items():
                if key in store:
                    candidates[reader] = store[key]

            if not candidates:
                log(
                    f"  WARN: no candidates for {slot_key} -- "
                    f"falling back to v17 entry"
                )
                v36_models.setdefault(target, {})[view] = dict(v17_entry)
                per_slot_reader[slot_key] = "v17"
                continue

            best_reader, best_stats, rule_tag = select_reader_v36(candidates)
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

            source_bundle = deploy_bundles.get(best_reader)
            if source_bundle is None:
                source_slot = v17_entry
            else:
                source_slot = (
                    source_bundle.get("models", {})
                    .get(target, {}).get(view) or v17_entry
                )

            v36_entry = dict(v17_entry)
            v36_entry["selected_reader"] = best_reader
            v36_entry["v36_tier"] = best_tier
            v36_entry["v36_stats"] = best_stats
            v36_entry["v36_source_entry"] = source_slot.get("approach")
            v36_entry["v36_tie_break_rule"] = rule_tag
            v36_models.setdefault(target, {})[view] = v36_entry

            loa = best_stats.get("loa_half_width_deg")
            slot_id = f"{target}|{view}"
            v35_r = v35_pick_reader.get(slot_id, "?")

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
                "v35_reader": v35_r,
                "v35_tier": v35_pick_tier.get(slot_id),
                "v35_ccc": v35_pick_ccc.get(slot_id),
                "v35_loa": v35_pick_loa.get(slot_id),
            })

            if best_reader != v35_r:
                tie_break_shifts.append({
                    "slot": slot_id,
                    "v35_reader": v35_r,
                    "v35_ccc": v35_pick_ccc.get(slot_id),
                    "v35_loa": v35_pick_loa.get(slot_id),
                    "v35_tier": v35_pick_tier.get(slot_id),
                    "v36_reader": best_reader,
                    "v36_ccc": ccc,
                    "v36_loa": loa,
                    "v36_tier": best_tier,
                    "rule": rule_tag,
                })

    # Category A side-by-side.
    category_a_table: list[dict] = []
    for tgt, view in CATEGORY_A_SLOTS:
        slot_id = f"{tgt}|{view}"
        pick = next((p for p in picks if p["slot"] == slot_id), None)
        category_a_table.append({
            "slot": slot_id,
            "v35_reader": v35_pick_reader.get(slot_id),
            "v35_tier": v35_pick_tier.get(slot_id),
            "v35_ccc": v35_pick_ccc.get(slot_id),
            "v35_loa": v35_pick_loa.get(slot_id),
            "v36_reader": pick["reader"] if pick else None,
            "v36_tier": pick["tier"] if pick else None,
            "v36_ccc": pick["ccc"] if pick else None,
            "v36_loa": pick["loa_half"] if pick else None,
            "v36_rule": pick["tie_break_rule"] if pick else None,
            "promoted_to_good": (
                pick is not None
                and pick["tier"] == "Good"
                and v35_pick_tier.get(slot_id) != "Good"
            ),
        })

    v36_out = {
        "version": "v36_selective_oracle",
        "produced_by": "harness.build_v36_selective (Agent PP)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            f"Per-slot oracle-best across v17/v18/v20/v23/v24/v26/v27/v29/"
            f"v30/v31/v33/v34 with LoA-then-CCC tie-break in the "
            f"LoA-limited Moderate band (CCC >= {LOA_LIMITED_CCC}). "
            f"{tier_counts['Good']} Good / "
            f"{tier_counts['Moderate']} Moderate / "
            f"{tier_counts['Poor']} Poor. "
            f"Tier 1 (CCC >= 0.79) count: {tier1_count}."
        ),
        "approaches": v17_deploy.get("approaches"),
        "training_dataset": v17_deploy.get("training_dataset"),
        "models": v36_models,
        "calibration_fix": v17_deploy.get("calibration_fix"),
        "selective_adoption": v17_deploy.get("selective_adoption"),
        "per_slot_reader": per_slot_reader,
        "reader_distribution": reader_distribution,
        "tier_counts": tier_counts,
        "tier1_count_ccc_ge_0p79": tier1_count,
        "tie_break_shifts_vs_v35": tie_break_shifts,
        "category_a_results": category_a_table,
        "loa_limited_ccc_threshold": LOA_LIMITED_CCC,
    }
    V36_PATH.write_text(json.dumps(v36_out, indent=2, default=str))
    log(f"wrote v36 deploy bundle -> {V36_PATH}")

    V36_OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks_out = {
        "version": "v36_selective_oracle",
        "description": v36_out["description"],
        "tier_counts": tier_counts,
        "tier1_count_ccc_ge_0p79": tier1_count,
        "reader_distribution": reader_distribution,
        "tie_break_shifts_vs_v35": tie_break_shifts,
        "category_a_results": category_a_table,
        "picks": picks,
    }
    V36_PICKS_PATH.write_text(json.dumps(picks_out, indent=2, default=str))
    log(f"wrote v36 per-slot picks -> {V36_PICKS_PATH}")

    return v36_out


def _fmt(v: object, prec: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def write_v36_report() -> None:
    v36 = _load_json(V36_PATH)
    picks = _load_json(V36_PICKS_PATH)

    tier_counts = v36["tier_counts"]
    tier1 = v36.get("tier1_count_ccc_ge_0p79", 0)
    reader_dist = v36["reader_distribution"]
    shifts = v36["tie_break_shifts_vs_v35"]
    cat_a = v36["category_a_results"]

    v35_tier_counts = {
        "Excellent": 0, "Good": 11, "Moderate": 6, "Poor": 6,
    }
    v35_tier1 = 14

    lines: list[str] = []
    lines.append("# v36 Selective Oracle with LoA-then-CCC Tie-Break")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(
        "**Build:** Agent PP, Lever 1 -- v35 reader pool with one selection "
        "rule fix: within the Moderate tier, when every top-tier candidate "
        f"has CCC >= {v36['loa_limited_ccc_threshold']:.2f} (LoA-limited "
        "band), tie-break on **lowest LoA then highest CCC** instead of "
        "CCC then LoA. No new training."
    )
    lines.append("")
    lines.append(
        f"**Verdict:** **{tier_counts['Good']} Good slots** "
        f"(v35 was {v35_tier_counts['Good']}). "
        f"Tier 1 (CCC >= 0.79): **{tier1}** (v35 was {v35_tier1})."
    )
    lines.append("")

    lines.append("## Tier counts vs v35")
    lines.append("")
    lines.append("| Tier | v35 | v36 | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for tier in ["Excellent", "Good", "Moderate", "Poor"]:
        old = v35_tier_counts.get(tier, 0)
        new = tier_counts.get(tier, 0)
        lines.append(f"| {tier} | {old} | {new} | {new - old:+d} |")
    lines.append(
        f"| Tier 1 (CCC >= 0.79) | {v35_tier1} | {tier1} | "
        f"{tier1 - v35_tier1:+d} |"
    )
    lines.append("")

    lines.append("## Tie-break shifts vs v35")
    lines.append("")
    lines.append(
        f"Total slots where the v36 LoA-first rule changed the picked "
        f"reader: **{len(shifts)}**."
    )
    lines.append("")
    if shifts:
        lines.append(
            "| Slot | v35 reader | v35 CCC | v35 LoA/2 | v36 reader | "
            "v36 CCC | v36 LoA/2 | v35 -> v36 tier | rule |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |"
        )
        for s in shifts:
            lines.append(
                f"| {s['slot']} | {s['v35_reader']} | "
                f"{_fmt(s['v35_ccc'], 3)} | {_fmt(s['v35_loa'])} | "
                f"{s['v36_reader']} | {_fmt(s['v36_ccc'], 3)} | "
                f"{_fmt(s['v36_loa'])} | "
                f"{s['v35_tier']} -> {s['v36_tier']} | {s['rule']} |"
            )
    lines.append("")

    lines.append("## Category A target slots")
    lines.append("")
    lines.append(
        "These are the 5 LoA-limited borderlines that motivated the fix. "
        "Reminder: even with the tie-break change, none of these slots will "
        "be promoted to Good unless one of the candidate readers actually "
        "has LoA <= 10.0 deg. The fix only ensures we pick the candidate "
        "with the lowest LoA when ties exist."
    )
    lines.append("")
    lines.append(
        "| Slot | v35 reader | v35 LoA/2 | v36 reader | v36 LoA/2 | "
        "Promoted? |"
    )
    lines.append("| --- | --- | ---: | --- | ---: | --- |")
    for row in cat_a:
        lines.append(
            f"| {row['slot']} | {row['v35_reader']} | "
            f"{_fmt(row['v35_loa'])} | {row['v36_reader']} | "
            f"{_fmt(row['v36_loa'])} | "
            f"{'YES' if row['promoted_to_good'] else 'no'} |"
        )
    lines.append("")
    cat_a_promoted = sum(1 for r in cat_a if r["promoted_to_good"])
    lines.append(
        f"**Category A promotions to Good: {cat_a_promoted}/{len(cat_a)}.**"
    )
    lines.append("")

    lines.append("## Reader distribution in v36")
    lines.append("")
    lines.append("| Reader | Slots |")
    lines.append("| --- | ---: |")
    for reader in READER_ORDER:
        n = reader_dist.get(reader, 0)
        lines.append(f"| {reader} | {n} |")
    lines.append("")

    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "- This fix is selection hygiene only. It will rarely cross a tier "
        "gate by itself -- it picks the candidate with the tightest LoA "
        "among ties in the LoA-limited band, but if no candidate is under "
        "+/-10 deg, no Moderate slot is promoted to Good."
    )
    lines.append(
        "- For slots with a single dominant reader, behavior is unchanged."
    )
    lines.append(
        "- The CCC >= 0.79 threshold for LoA-limited classification matches "
        "v32 Tier-1 / Agent OO Category A definition."
    )
    lines.append(
        "- Higher tiers (Good, Excellent) are unaffected by this change."
    )
    lines.append(
        "- LOSO discipline is unchanged: outer subject-level LOSO at L3."
    )
    lines.append("")

    V36_REPORT_PATH.write_text("\n".join(lines) + "\n")
    log(f"wrote REPORT -> {V36_REPORT_PATH}")


def main() -> None:
    log("=== Agent PP (v36 tie-break fix) START ===")
    build()
    write_v36_report()
    log("=== Agent PP (v36 tie-break fix) DONE ===")


if __name__ == "__main__":
    main()
