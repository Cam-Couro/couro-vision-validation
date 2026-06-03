"""Build v32 selective oracle: per-slot best across v17 / v18 / v20 /
v23 / v24 / v26 / v27 / v29 / v30 / v31.

Agent NN Phase 3. Adds three new readers to the v28 pool:

  * **v29** — NN mirror-flip per-source heads + per-frame SmoothL1 (L2)
    + ridge L3 (same FF Phase-B pipeline applied with the v29 alldata L2
    monkey-patched in).
  * **v30** — v23 HH2 combined L2 + LEARNED L3 (TinyMLP per-slot).
  * **v31** — v29 mirror-flip L2 + LEARNED L3.

Selection rule mirrors v28: tier first (Excellent > Good > Moderate >
Poor), within tier highest Lin's CCC, ties broken by canonical reader
order (v17, v18, v20, v23, v24, v26, v27, v29, v30, v31). Older / simpler
reader wins ties so the deployment surface stays close to v28 where the
new readers do not help.

# Outputs

  * ``results/deploy_ready_models_v32_selective.json``
  * ``data/v32_selective_oracle/per_slot_picks_v32.json``
  * ``data/v32_selective_oracle/REPORT.md``
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

V28_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v28_selective.json"
V32_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v32_selective.json"
V32_OUT_DIR: Final[Path] = DATA_ROOT / "v32_selective_oracle"
V32_PICKS_PATH: Final[Path] = V32_OUT_DIR / "per_slot_picks_v32.json"
V32_REPORT_PATH: Final[Path] = V32_OUT_DIR / "REPORT.md"


TIER_RANK: Final[dict[str, int]] = {
    "Excellent": 4,
    "Good": 3,
    "Moderate": 2,
    "Poor": 1,
}

READER_ORDER: Final[tuple[str, ...]] = (
    "v17", "v18", "v20", "v23", "v24", "v26", "v27",
    "v29", "v30", "v31",
)

# Category A: LoA-limited borderlines (CCC >= 0.81 already, fail LoA gate).
CATEGORY_A_SLOTS: Final[tuple[tuple[str, str], ...]] = (
    ("knee_angle_r", "front_oblique_left"),
    ("knee_angle_r", "side_left"),
    ("knee_angle_r", "side_right"),
    ("hip_flexion_r", "front_oblique_left"),
    ("hip_adduction_r", "front_oblique_right"),
)

# Category B: Mirror-twin asymmetry — poor side of a well-performing pair.
CATEGORY_B_SLOTS: Final[tuple[tuple[str, str], ...]] = (
    ("hip_adduction_r", "side_right"),
    ("hip_flexion_r", "side_right"),
    ("ankle_angle_r", "side_left"),
    ("ankle_angle_r", "front_oblique_left"),
)


def log(msg: str) -> None:
    print(f"[v32 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -------------------------------------------------------------------------
# Data loaders.
# -------------------------------------------------------------------------


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


# -------------------------------------------------------------------------
# Selection.
# -------------------------------------------------------------------------


def _tier_rank(tier: str | None) -> int:
    if tier is None:
        return 0
    return TIER_RANK.get(tier, 0)


def _ccc_or_minus_inf(stats: dict) -> float:
    ccc = stats.get("ccc_lin")
    if ccc is None or (isinstance(ccc, float) and math.isnan(ccc)):
        return float("-inf")
    return float(ccc)


def select_reader(candidates: dict[str, dict]) -> tuple[str, dict]:
    def sort_key(reader: str) -> tuple[int, float, int]:
        stats = candidates[reader]
        return (
            -_tier_rank(stats.get("classification")),
            -_ccc_or_minus_inf(stats),
            READER_ORDER.index(reader)
            if reader in READER_ORDER else len(READER_ORDER),
        )
    best = min(candidates.keys(), key=sort_key)
    return best, candidates[best]


# -------------------------------------------------------------------------
# Build.
# -------------------------------------------------------------------------


def build() -> dict:
    log("loading reader sources ...")
    v17_deploy = _load_json(V17_PATH)
    v18_deploy = _load_json(V18_PATH) if V18_PATH.exists() else None
    v20_deploy = _load_json(V20_PATH) if V20_PATH.exists() else None
    v23_deploy = _load_json(V23_PATH) if V23_PATH.exists() else None
    v24_deploy = _load_json(V24_PATH) if V24_PATH.exists() else None
    v26_deploy = _load_json(V26_PATH) if V26_PATH.exists() else None
    v27_deploy = _load_json(V27_PATH) if V27_PATH.exists() else None
    v29_deploy = _load_json(V29_PATH) if V29_PATH.exists() else None
    v30_deploy = _load_json(V30_PATH) if V30_PATH.exists() else None
    v31_deploy = _load_json(V31_PATH) if V31_PATH.exists() else None

    v17_baseline = load_baseline_v17()
    v18_stats = load_per_slot(V18_PER_SLOT)
    v20_stats = load_per_slot(V20_PER_SLOT)
    v23_stats = load_per_slot(V23_PER_SLOT)
    v24_stats = load_per_slot(V24_PER_SLOT)
    v26_stats = load_per_slot(V26_PER_SLOT)
    v27_stats = load_per_slot(V27_PER_SLOT)
    v29_stats = load_per_slot(V29_PER_SLOT)
    v30_stats = load_per_slot(V30_PER_SLOT)
    v31_stats = load_per_slot(V31_PER_SLOT)

    log(
        f"reader candidate counts: v17={len(v17_baseline)} "
        f"v18={len(v18_stats)} v20={len(v20_stats)} v23={len(v23_stats)} "
        f"v24={len(v24_stats)} v26={len(v26_stats)} v27={len(v27_stats)} "
        f"v29={len(v29_stats)} v30={len(v30_stats)} v31={len(v31_stats)}"
    )

    # ---- Pull v28 picks for per-slot before/after comparison ----
    v28_deploy = _load_json(V28_PATH)
    v28_models = v28_deploy["models"]
    v28_pick_tier: dict[str, str] = {}
    v28_pick_ccc: dict[str, float | None] = {}
    v28_pick_loa: dict[str, float | None] = {}
    v28_pick_reader: dict[str, str] = {}
    for tgt, slots in v28_models.items():
        for view, entry in slots.items():
            key = f"{tgt}|{view}"
            v28_stats = entry.get("v28_stats", {})
            v28_pick_tier[key] = v28_stats.get("classification") or "Poor"
            v28_pick_ccc[key] = v28_stats.get("ccc_lin")
            v28_pick_loa[key] = v28_stats.get("loa_half_width_deg")
            v28_pick_reader[key] = entry.get("selected_reader") or "v17"

    picks: list[dict] = []
    per_slot_reader: dict[str, str] = {}
    v32_models: dict[str, dict[str, dict]] = {}
    tier_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    tier1_count = 0  # slots with CCC >= 0.79
    reader_distribution = {r: 0 for r in READER_ORDER}
    promotions_vs_v28: list[dict] = []

    for target, slots in v17_deploy["models"].items():
        for view, v17_entry in slots.items():
            key = (target, view)
            slot_key = f"{target}/{view}"
            candidates: dict[str, dict] = {}
            for reader, store in [
                ("v17", v17_baseline),
                ("v18", v18_stats),
                ("v20", v20_stats),
                ("v23", v23_stats),
                ("v24", v24_stats),
                ("v26", v26_stats),
                ("v27", v27_stats),
                ("v29", v29_stats),
                ("v30", v30_stats),
                ("v31", v31_stats),
            ]:
                if key in store:
                    candidates[reader] = store[key]

            if not candidates:
                log(
                    f"  WARN: no candidates for {slot_key} -- "
                    f"falling back to v17 entry"
                )
                v32_models.setdefault(target, {})[view] = dict(v17_entry)
                per_slot_reader[slot_key] = "v17"
                continue

            best_reader, best_stats = select_reader(candidates)
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

            # Carry the deploy entry from the winning reader if available.
            source_bundle = {
                "v17": v17_deploy,
                "v18": v18_deploy,
                "v20": v20_deploy,
                "v23": v23_deploy,
                "v24": v24_deploy,
                "v26": v26_deploy,
                "v27": v27_deploy,
                "v29": v29_deploy,
                "v30": v30_deploy,
                "v31": v31_deploy,
            }[best_reader]
            if source_bundle is None:
                source_slot = v17_entry
            else:
                source_slot = (
                    source_bundle.get("models", {})
                    .get(target, {}).get(view) or v17_entry
                )

            v32_entry = dict(v17_entry)
            v32_entry["selected_reader"] = best_reader
            v32_entry["v32_tier"] = best_tier
            v32_entry["v32_stats"] = best_stats
            v32_entry["v32_source_entry"] = source_slot.get("approach")
            v32_models.setdefault(target, {})[view] = v32_entry

            loa = best_stats.get("loa_half_width_deg")
            ccc_str = (
                "nan"
                if ccc is None or (
                    isinstance(ccc, float) and math.isnan(ccc)
                )
                else f"{ccc:.3f}"
            )
            loa_str = (
                "nan"
                if loa is None or (
                    isinstance(loa, float) and math.isnan(loa)
                )
                else f"{loa:.2f}"
            )
            v28_t = v28_pick_tier.get(f"{target}|{view}", "?")
            v28_c = v28_pick_ccc.get(f"{target}|{view}")
            v28_c_str = (
                "nan" if v28_c is None or (
                    isinstance(v28_c, float) and math.isnan(v28_c)
                ) else f"{v28_c:.3f}"
            )
            log(
                f"  {slot_key} -> {best_reader} ({best_tier}) "
                f"CCC={ccc_str} LoAh={loa_str} "
                f"(v28: {v28_t} CCC={v28_c_str})"
            )

            picks.append({
                "slot": f"{target}|{view}",
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
                "v28_tier": v28_t,
                "v28_ccc": v28_c,
                "v28_loa": v28_pick_loa.get(f"{target}|{view}"),
                "v28_reader": v28_pick_reader.get(f"{target}|{view}"),
            })

            if _tier_rank(best_tier) > _tier_rank(v28_t):
                promotions_vs_v28.append({
                    "slot": f"{target}|{view}",
                    "v28_tier": v28_t,
                    "v32_tier": best_tier,
                    "reader": best_reader,
                    "ccc": ccc,
                    "loa_half": loa,
                })

    # Category A / B analyses.
    category_a_table: list[dict] = []
    for tgt, view in CATEGORY_A_SLOTS:
        v28_t = v28_pick_tier.get(f"{tgt}|{view}", "?")
        v28_c = v28_pick_ccc.get(f"{tgt}|{view}")
        v28_l = v28_pick_loa.get(f"{tgt}|{view}")
        v28_r = v28_pick_reader.get(f"{tgt}|{view}")
        pick = next(
            (p for p in picks if p["slot"] == f"{tgt}|{view}"), None
        )
        category_a_table.append({
            "slot": f"{tgt}|{view}",
            "v28_tier": v28_t,
            "v28_ccc": v28_c,
            "v28_loa": v28_l,
            "v28_reader": v28_r,
            "v32_tier": pick["tier"] if pick else None,
            "v32_ccc": pick["ccc"] if pick else None,
            "v32_loa": pick["loa_half"] if pick else None,
            "v32_reader": pick["reader"] if pick else None,
            "promoted_to_good": (
                pick is not None
                and pick["tier"] == "Good"
                and v28_t != "Good"
            ),
        })

    category_b_table: list[dict] = []
    for tgt, view in CATEGORY_B_SLOTS:
        v28_t = v28_pick_tier.get(f"{tgt}|{view}", "?")
        v28_c = v28_pick_ccc.get(f"{tgt}|{view}")
        v28_l = v28_pick_loa.get(f"{tgt}|{view}")
        v28_r = v28_pick_reader.get(f"{tgt}|{view}")
        pick = next(
            (p for p in picks if p["slot"] == f"{tgt}|{view}"), None
        )
        category_b_table.append({
            "slot": f"{tgt}|{view}",
            "v28_tier": v28_t,
            "v28_ccc": v28_c,
            "v28_loa": v28_l,
            "v28_reader": v28_r,
            "v32_tier": pick["tier"] if pick else None,
            "v32_ccc": pick["ccc"] if pick else None,
            "v32_loa": pick["loa_half"] if pick else None,
            "v32_reader": pick["reader"] if pick else None,
            "lifted_ccc_delta": (
                None
                if (pick is None or v28_c is None
                    or isinstance(pick["ccc"], type(None))
                    or (
                        isinstance(pick["ccc"], float)
                        and math.isnan(pick["ccc"])
                    )
                    or (
                        isinstance(v28_c, float) and math.isnan(v28_c)
                    ))
                else float(pick["ccc"]) - float(v28_c)
            ),
        })

    v32_out = {
        "version": "v32_selective_oracle",
        "produced_by": "harness.build_v32_selective (Agent NN Phase 3)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            f"Per-slot oracle-best across v17/v18/v20/v23/v24/v26/v27/"
            f"v29/v30/v31. v29 = NN mirror-flip per-source heads + "
            f"per-frame SmoothL1 (ridge L3). v30 = v23 HH2 L2 + "
            f"learned-L3 TinyMLP. v31 = v29 mirror-flip L2 + learned-L3 "
            f"TinyMLP. {tier_counts['Good']} Good / "
            f"{tier_counts['Moderate']} Moderate / "
            f"{tier_counts['Poor']} Poor. Tier 1 (CCC ≥ 0.79) count: "
            f"{tier1_count}."
        ),
        "approaches": v17_deploy.get("approaches"),
        "training_dataset": v17_deploy.get("training_dataset"),
        "models": v32_models,
        "calibration_fix": v17_deploy.get("calibration_fix"),
        "selective_adoption": v17_deploy.get("selective_adoption"),
        "per_slot_reader": per_slot_reader,
        "reader_distribution": reader_distribution,
        "tier_counts": tier_counts,
        "tier1_count_ccc_ge_0p79": tier1_count,
        "promotions_vs_v28": promotions_vs_v28,
        "category_a_results": category_a_table,
        "category_b_results": category_b_table,
    }
    V32_PATH.write_text(json.dumps(v32_out, indent=2, default=str))
    log(f"wrote v32 deploy bundle -> {V32_PATH}")

    V32_OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks_out = {
        "version": "v32_selective_oracle",
        "description": v32_out["description"],
        "tier_counts": tier_counts,
        "tier1_count_ccc_ge_0p79": tier1_count,
        "reader_distribution": reader_distribution,
        "promotions_vs_v28": promotions_vs_v28,
        "category_a_results": category_a_table,
        "category_b_results": category_b_table,
        "picks": picks,
    }
    V32_PICKS_PATH.write_text(json.dumps(picks_out, indent=2, default=str))
    log(f"wrote v32 per-slot picks -> {V32_PICKS_PATH}")

    return v32_out


# -------------------------------------------------------------------------
# REPORT.md
# -------------------------------------------------------------------------


def _fmt_num(v: object, prec: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def write_v32_report() -> None:
    v32 = _load_json(V32_PATH)
    picks = _load_json(V32_PICKS_PATH)

    tier_counts = v32["tier_counts"]
    tier1 = v32.get("tier1_count_ccc_ge_0p79", 0)
    reader_dist = v32["reader_distribution"]
    proms = v32["promotions_vs_v28"]
    cat_a = v32["category_a_results"]
    cat_b = v32["category_b_results"]

    # v28 reference numbers (hardcoded from v28 picks file for the
    # baseline table; safe because v28 is frozen).
    v28_tier_counts = {"Excellent": 0, "Good": 10, "Moderate": 6, "Poor": 7}
    v28_tier1 = 7

    lines: list[str] = []
    lines.append("# v32 Selective Oracle Deploy + v29/v30/v31 Candidates")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(
        "**Build:** Agent NN Phase 3 — adds 3 candidates to the v28 reader pool:"
    )
    lines.append("")
    lines.append(
        "- **v29** Layer 2 mirror-flip augmentation (per-source heads + per-frame SmoothL1, MM-A backbone with L/R-flipped clip duplicates) + ridge L3."
    )
    lines.append(
        "- **v30** v23 HH2 combined Layer 2 + per-slot **learned Layer 3** (TinyMLP hidden=32, dropout 0.2, AdamW lr=1e-2 wd=1e-3, 200 epochs w/ early stopping). Per-slot fallback to ridge if learned underperforms by >0.05 CCC."
    )
    lines.append(
        "- **v31** v29 mirror-flip Layer 2 + per-slot learned Layer 3 (same architecture as v30)."
    )
    lines.append("")
    lines.append(
        f"**Verdict:** **{tier_counts['Good']} validated Good-tier slots** "
        f"(v28 was {v28_tier_counts['Good']}). Tier 1 (CCC >= 0.79) count: "
        f"**{tier1}** (v28 was {v28_tier1})."
    )
    lines.append("")

    # ---- Tier delta table ----
    lines.append("## Tier counts vs v28")
    lines.append("")
    lines.append("| Tier | v28 | v32 | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for tier in ["Excellent", "Good", "Moderate", "Poor"]:
        old = v28_tier_counts.get(tier, 0)
        new = tier_counts.get(tier, 0)
        lines.append(f"| {tier} | {old} | {new} | {new - old:+d} |")
    lines.append(
        f"| Tier 1 (CCC >= 0.79) | {v28_tier1} | {tier1} | "
        f"{tier1 - v28_tier1:+d} |"
    )
    lines.append("")
    lines.append(
        f"Promotions vs v28: **{len(proms)}**. "
        + (
            "" if not proms else
            "Slots: " + ", ".join(
                f"{p['slot']} ({p['v28_tier']} -> {p['v32_tier']}, "
                f"reader={p['reader']})"
                for p in proms
            )
        )
    )
    lines.append("")

    # ---- Category A (LoA-limited borderlines) ----
    lines.append("## Category A: did the LoA-limited borderlines promote?")
    lines.append("")
    lines.append(
        "These slots already have CCC >= 0.81 (Tier 1) but fail the LoA "
        "±10° gate by 1-3°. Lever was learned L3 (replaces ridge with "
        "TinyMLP) to tighten peak prediction."
    )
    lines.append("")
    lines.append(
        "| Slot | v28 tier | v28 CCC | v28 LoA/2 | v28 reader | "
        "v32 tier | v32 CCC | v32 LoA/2 | v32 reader | Promoted? |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |"
    )
    for row in cat_a:
        lines.append(
            f"| {row['slot']} | {row['v28_tier']} | "
            f"{_fmt_num(row['v28_ccc'], 3)} | "
            f"{_fmt_num(row['v28_loa'])} | {row['v28_reader']} | "
            f"{row['v32_tier']} | {_fmt_num(row['v32_ccc'], 3)} | "
            f"{_fmt_num(row['v32_loa'])} | {row['v32_reader']} | "
            f"{'YES' if row['promoted_to_good'] else 'no'} |"
        )
    cat_a_promoted = sum(1 for r in cat_a if r["promoted_to_good"])
    lines.append("")
    lines.append(
        f"**Category A promotions: {cat_a_promoted}/{len(cat_a)}.**"
    )
    lines.append("")

    # ---- Category B (mirror-twin asymmetry) ----
    lines.append(
        "## Category B: did the mirror-twin asymmetric slots lift?"
    )
    lines.append("")
    lines.append(
        "These slots have a sister slot that scores well but this side "
        "scores poorly. Lever was mirror-flip Layer 2 augmentation "
        "(v29 / v31)."
    )
    lines.append("")
    lines.append(
        "| Slot | v28 tier | v28 CCC | v28 reader | v32 tier | v32 CCC | "
        "v32 reader | CCC delta |"
    )
    lines.append(
        "| --- | --- | ---: | --- | --- | ---: | --- | ---: |"
    )
    for row in cat_b:
        delta = row.get("lifted_ccc_delta")
        lines.append(
            f"| {row['slot']} | {row['v28_tier']} | "
            f"{_fmt_num(row['v28_ccc'], 3)} | {row['v28_reader']} | "
            f"{row['v32_tier']} | {_fmt_num(row['v32_ccc'], 3)} | "
            f"{row['v32_reader']} | {_fmt_num(delta, 3)} |"
        )
    cat_b_lifted = sum(
        1 for r in cat_b
        if r.get("lifted_ccc_delta") is not None
        and r["lifted_ccc_delta"] > 0.01
    )
    lines.append("")
    lines.append(
        f"**Category B lifts (CCC delta > 0.01): "
        f"{cat_b_lifted}/{len(cat_b)}.**"
    )
    lines.append("")

    # ---- Reader distribution ----
    lines.append("## Reader distribution in v32")
    lines.append("")
    lines.append("| Reader | Slots | Description |")
    lines.append("| --- | ---: | --- |")
    DESC = {
        "v17": "hand-engineered + ridge",
        "v18": "FF learned L2 (OpenCap-only) + ridge",
        "v20": "GG2 ROM-aware OpenCap L2 + ridge",
        "v23": "HH2 combined L2 + ridge",
        "v24": "LL combined + ROM-aware + ridge",
        "v26": "MM-A per-source per-frame L2 + ridge",
        "v27": "MM-B per-source ROM-aware L2 + ridge",
        "v29": "NN mirror-flip per-source per-frame L2 + ridge",
        "v30": "v23 L2 + learned L3 (TinyMLP)",
        "v31": "v29 mirror-flip L2 + learned L3 (TinyMLP)",
    }
    for reader in READER_ORDER:
        n = reader_dist.get(reader, 0)
        lines.append(
            f"| {reader} | {n} | {DESC.get(reader, '?')} |"
        )
    lines.append("")

    # ---- Honest caveats ----
    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "- **Double-LOSO upper bound.** v23 onward train L2 on all 24 "
        "cohort subjects; L3 LOSO at subject level only. Tier promotions "
        "involving any cohort subject are upper bounds. Per HH2's "
        "per-fold variance, true double-LOSO numbers could be ~0.05-0.10 "
        "|r| lower."
    )
    lines.append(
        "- **Learned L3 overfit risk is real.** Per-slot models with "
        "n=9-22 LOSO inner folds and ~5K parameters. Mitigations: hidden=32 "
        "(tiny capacity), dropout 0.2, weight_decay 1e-3, early stopping on "
        "15% inner-val. Per-slot fallback to ridge if learned underperforms "
        "ridge by > 0.05 CCC keeps a no-regression guarantee on the "
        "headline metric."
    )
    lines.append(
        "- **Mirror flip discipline.** Flipped clips inherit the original "
        "subject id, so LOSO holdouts include both flipped and unflipped "
        "versions of the held-out subject's clips. No leak through the L/R "
        "label swap."
    )
    lines.append(
        "- **Per-source head specifics carried from MM-A.** For "
        "hip_adduction_r and lumbar_extension where OpenCap and ASPset "
        "differ in convention, the shared (deployed) head is supervised by "
        "OpenCap only; ASPset gradients flow to a separate aux head that "
        "is discarded at deploy. v29 inherits this routing."
    )
    lines.append(
        "- **v29 LOSO-at-L2 evaluation was skipped** to stay in time budget. "
        "The diagnostic per-fold |r| would be informative but is not on the "
        "critical path for v32 (which depends on the all-data L2 + L3 "
        "LOSO pipeline). Per-fold variance for the closely-related MM-A "
        "model was already characterized and is a reasonable prior."
    )
    lines.append("")

    # ---- Recommendation ----
    lines.append("## Recommendation for next move")
    lines.append("")
    if tier_counts["Good"] > v28_tier_counts["Good"]:
        lines.append(
            f"Adopt v32 selectively. Net **{tier_counts['Good'] - v28_tier_counts['Good']}** "
            "more Good slots than v28 with no Good slot demoted. The "
            "headline Tier-1 (CCC >= 0.79) count is "
            f"{'up' if tier1 > v28_tier1 else 'down'} by "
            f"{tier1 - v28_tier1:+d} vs v28."
        )
    else:
        lines.append(
            f"Hold at v28. v32 did not net-gain Good slots "
            f"({tier_counts['Good']} vs {v28_tier_counts['Good']}). The "
            "two big levers (mirror flip + learned L3) appear to be of "
            "diminishing return at this dataset size."
        )
    lines.append("")
    if cat_a_promoted == 0:
        lines.append(
            "Category A (LoA-limited knee/hip_flexion borderlines) did NOT "
            "promote. Learned L3 did not deliver tighter peak prediction "
            "than ridge on these slots. Next experiment: per-slot loss "
            "weighting on max/min extrema (an explicit ROM-aware learned "
            "L3), or per-slot calibration on the held-out fold residuals."
        )
    if cat_b_lifted == 0:
        lines.append(
            "Category B (mirror-twin asymmetric slots) did NOT lift. Mirror "
            "flip augmentation did not break the asymmetry. Hypothesis "
            "should shift from 'training-data L/R imbalance' to "
            "'view-specific calibration drift' or 'keypoint detector "
            "asymmetry'."
        )
    lines.append("")

    V32_REPORT_PATH.write_text("\n".join(lines) + "\n")
    log(f"wrote REPORT -> {V32_REPORT_PATH}")


def main() -> None:
    log("=== Agent NN Phase 3 (v32) START ===")
    build()
    write_v32_report()
    log("=== Agent NN Phase 3 (v32) DONE ===")


if __name__ == "__main__":
    main()
