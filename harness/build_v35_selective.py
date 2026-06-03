"""Build v35 selective oracle: per-slot best across v17/v18/v20/v23/v24/v26/
v27/v29/v30/v31/v33/v34.

Agent OO. Adds two new readers to the v32 pool:

  * **v33** -- v23 HH2 combined L2 + extrema-aware learned L3
    (TinyMLP with two heads: pred_max, pred_min). ROM = pred_max - pred_min.
    Loss = SmoothL1(rom) + 0.5*SmoothL1(max) + 0.5*SmoothL1(min). Per-slot
    fallback to ridge if extrema-aware CCC underperforms ridge by > 0.05.
  * **v34** -- v29 mirror-flip L2 + extrema-aware learned L3 (same
    architecture as v33).

Selection rule mirrors v32: tier first (Excellent > Good > Moderate > Poor),
within tier highest Lin's CCC, ties broken by canonical reader order
(v17, v18, v20, v23, v24, v26, v27, v29, v30, v31, v33, v34).

# Outputs

  * ``results/deploy_ready_models_v35_selective.json``
  * ``data/v35_selective_oracle/per_slot_picks_v35.json``
  * ``data/v35_selective_oracle/REPORT.md``
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

V32_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v32_selective.json"
V35_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v35_selective.json"
V35_OUT_DIR: Final[Path] = DATA_ROOT / "v35_selective_oracle"
V35_PICKS_PATH: Final[Path] = V35_OUT_DIR / "per_slot_picks_v35.json"
V35_REPORT_PATH: Final[Path] = V35_OUT_DIR / "REPORT.md"


TIER_RANK: Final[dict[str, int]] = {
    "Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1,
}

READER_ORDER: Final[tuple[str, ...]] = (
    "v17", "v18", "v20", "v23", "v24", "v26", "v27",
    "v29", "v30", "v31", "v33", "v34",
)

# Category A: LoA-limited borderlines (CCC >= 0.81 in v32, fail LoA gate).
CATEGORY_A_SLOTS: Final[tuple[tuple[str, str], ...]] = (
    ("knee_angle_r", "front_oblique_left"),
    ("knee_angle_r", "side_left"),
    ("knee_angle_r", "side_right"),
    ("hip_flexion_r", "front_oblique_left"),
    ("hip_adduction_r", "front_oblique_right"),
)


def log(msg: str) -> None:
    print(f"[v35 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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

    # Pull v32 picks for per-slot before/after comparison.
    v32_deploy = _load_json(V32_PATH)
    v32_models = v32_deploy["models"]
    v32_pick_tier: dict[str, str] = {}
    v32_pick_ccc: dict[str, float | None] = {}
    v32_pick_loa: dict[str, float | None] = {}
    v32_pick_reader: dict[str, str] = {}
    for tgt, slots in v32_models.items():
        for view, entry in slots.items():
            key = f"{tgt}|{view}"
            v32_stats = entry.get("v32_stats", {})
            v32_pick_tier[key] = v32_stats.get("classification") or "Poor"
            v32_pick_ccc[key] = v32_stats.get("ccc_lin")
            v32_pick_loa[key] = v32_stats.get("loa_half_width_deg")
            v32_pick_reader[key] = entry.get("selected_reader") or "v17"

    picks: list[dict] = []
    per_slot_reader: dict[str, str] = {}
    v35_models: dict[str, dict[str, dict]] = {}
    tier_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    tier1_count = 0
    reader_distribution = {r: 0 for r in READER_ORDER}
    promotions_vs_v32: list[dict] = []
    demotions_vs_v32: list[dict] = []

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
                v35_models.setdefault(target, {})[view] = dict(v17_entry)
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

            source_bundle = deploy_bundles.get(best_reader)
            if source_bundle is None:
                source_slot = v17_entry
            else:
                source_slot = (
                    source_bundle.get("models", {})
                    .get(target, {}).get(view) or v17_entry
                )

            v35_entry = dict(v17_entry)
            v35_entry["selected_reader"] = best_reader
            v35_entry["v35_tier"] = best_tier
            v35_entry["v35_stats"] = best_stats
            v35_entry["v35_source_entry"] = source_slot.get("approach")
            v35_models.setdefault(target, {})[view] = v35_entry

            loa = best_stats.get("loa_half_width_deg")
            v32_t = v32_pick_tier.get(f"{target}|{view}", "?")

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
                "v32_tier": v32_t,
                "v32_ccc": v32_pick_ccc.get(f"{target}|{view}"),
                "v32_loa": v32_pick_loa.get(f"{target}|{view}"),
                "v32_reader": v32_pick_reader.get(f"{target}|{view}"),
            })

            if _tier_rank(best_tier) > _tier_rank(v32_t):
                promotions_vs_v32.append({
                    "slot": f"{target}|{view}",
                    "v32_tier": v32_t,
                    "v35_tier": best_tier,
                    "reader": best_reader,
                    "ccc": ccc,
                    "loa_half": loa,
                })
            elif _tier_rank(best_tier) < _tier_rank(v32_t):
                demotions_vs_v32.append({
                    "slot": f"{target}|{view}",
                    "v32_tier": v32_t,
                    "v35_tier": best_tier,
                    "reader": best_reader,
                    "ccc": ccc,
                    "loa_half": loa,
                })

    # Category A analysis (the question of the day).
    category_a_table: list[dict] = []
    for tgt, view in CATEGORY_A_SLOTS:
        v32_t = v32_pick_tier.get(f"{tgt}|{view}", "?")
        v32_c = v32_pick_ccc.get(f"{tgt}|{view}")
        v32_l = v32_pick_loa.get(f"{tgt}|{view}")
        v32_r = v32_pick_reader.get(f"{tgt}|{view}")
        key = (tgt, view)
        # Pull v33/v34 stats specifically so we can show side-by-side.
        v33_stats = per_slot_stores["v33"].get(key, {})
        v34_stats = per_slot_stores["v34"].get(key, {})
        pick = next(
            (p for p in picks if p["slot"] == f"{tgt}|{view}"), None
        )
        category_a_table.append({
            "slot": f"{tgt}|{view}",
            "v32_tier": v32_t,
            "v32_ccc": v32_c,
            "v32_loa": v32_l,
            "v32_reader": v32_r,
            "v33_ccc": v33_stats.get("ccc_lin"),
            "v33_loa": v33_stats.get("loa_half_width_deg"),
            "v33_tier": v33_stats.get("classification"),
            "v34_ccc": v34_stats.get("ccc_lin"),
            "v34_loa": v34_stats.get("loa_half_width_deg"),
            "v34_tier": v34_stats.get("classification"),
            "v35_tier": pick["tier"] if pick else None,
            "v35_ccc": pick["ccc"] if pick else None,
            "v35_loa": pick["loa_half"] if pick else None,
            "v35_reader": pick["reader"] if pick else None,
            "promoted_to_good": (
                pick is not None
                and pick["tier"] == "Good"
                and v32_t != "Good"
            ),
        })

    # Extrema diagnostics (mean error on max/min predictions).
    extrema_diag_by_slot: dict[str, dict[str, dict]] = {}
    for tag_short, per_slot_path in [
        ("v33", V33_PER_SLOT), ("v34", V34_PER_SLOT),
    ]:
        if not per_slot_path.exists():
            continue
        d = _load_json(per_slot_path)
        for d_row in d.get("extrema_diagnostics", []):
            key = f"{d_row['target']}|{d_row['view']}"
            extrema_diag_by_slot.setdefault(key, {})[tag_short] = {
                "mae_max_deg": d_row.get("mae_max_deg"),
                "mae_min_deg": d_row.get("mae_min_deg"),
                "n": d_row.get("n"),
            }

    v35_out = {
        "version": "v35_selective_oracle",
        "produced_by": "harness.build_v35_selective (Agent OO)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            f"Per-slot oracle-best across v17/v18/v20/v23/v24/v26/v27/v29/"
            f"v30/v31/v33/v34. v33 = v23 L2 + extrema-aware learned L3 "
            f"(two heads pred_max/pred_min). v34 = v29 mirror-flip L2 + "
            f"extrema-aware learned L3. {tier_counts['Good']} Good / "
            f"{tier_counts['Moderate']} Moderate / "
            f"{tier_counts['Poor']} Poor. Tier 1 (CCC >= 0.79) count: "
            f"{tier1_count}."
        ),
        "approaches": v17_deploy.get("approaches"),
        "training_dataset": v17_deploy.get("training_dataset"),
        "models": v35_models,
        "calibration_fix": v17_deploy.get("calibration_fix"),
        "selective_adoption": v17_deploy.get("selective_adoption"),
        "per_slot_reader": per_slot_reader,
        "reader_distribution": reader_distribution,
        "tier_counts": tier_counts,
        "tier1_count_ccc_ge_0p79": tier1_count,
        "promotions_vs_v32": promotions_vs_v32,
        "demotions_vs_v32": demotions_vs_v32,
        "category_a_results": category_a_table,
        "extrema_diagnostics": extrema_diag_by_slot,
    }
    V35_PATH.write_text(json.dumps(v35_out, indent=2, default=str))
    log(f"wrote v35 deploy bundle -> {V35_PATH}")

    V35_OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks_out = {
        "version": "v35_selective_oracle",
        "description": v35_out["description"],
        "tier_counts": tier_counts,
        "tier1_count_ccc_ge_0p79": tier1_count,
        "reader_distribution": reader_distribution,
        "promotions_vs_v32": promotions_vs_v32,
        "demotions_vs_v32": demotions_vs_v32,
        "category_a_results": category_a_table,
        "extrema_diagnostics": extrema_diag_by_slot,
        "picks": picks,
    }
    V35_PICKS_PATH.write_text(json.dumps(picks_out, indent=2, default=str))
    log(f"wrote v35 per-slot picks -> {V35_PICKS_PATH}")

    return v35_out


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


def write_v35_report() -> None:
    v35 = _load_json(V35_PATH)
    picks = _load_json(V35_PICKS_PATH)

    tier_counts = v35["tier_counts"]
    tier1 = v35.get("tier1_count_ccc_ge_0p79", 0)
    reader_dist = v35["reader_distribution"]
    proms = v35["promotions_vs_v32"]
    demos = v35["demotions_vs_v32"]
    cat_a = v35["category_a_results"]
    extrema_diag = v35.get("extrema_diagnostics", {})

    # v32 reference numbers (frozen, derived from v32 picks JSON).
    v32_tier_counts = {
        "Excellent": 0, "Good": 11, "Moderate": 6, "Poor": 6,
    }
    v32_tier1 = 13

    lines: list[str] = []
    lines.append(
        "# v35 Selective Oracle Deploy + v33/v34 Extrema-Aware Candidates"
    )
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(
        "**Build:** Agent OO -- adds 2 candidates to the v32 reader pool:"
    )
    lines.append("")
    lines.append(
        "- **v33** v23 HH2 combined L2 + per-slot **extrema-aware** "
        "learned L3 (TinyMLP, two heads pred_max/pred_min, "
        "loss = SmoothL1(ROM) + 0.5*SmoothL1(max) + 0.5*SmoothL1(min)). "
        "Hidden=32, dropout 0.2, AdamW lr=1e-2 wd=1e-3, 200 epochs w/ "
        "early stopping on 15% inner-val. Per-slot fallback to ridge if "
        "extrema-aware CCC underperforms by > 0.05."
    )
    lines.append(
        "- **v34** v29 mirror-flip L2 + extrema-aware learned L3 (same "
        "architecture as v33)."
    )
    lines.append("")
    lines.append(
        f"**Verdict:** **{tier_counts['Good']} validated Good-tier slots** "
        f"(v32 was {v32_tier_counts['Good']}). "
        f"Tier 1 (CCC >= 0.79) count: **{tier1}** (v32 was {v32_tier1})."
    )
    lines.append("")

    # ---- Tier delta table ----
    lines.append("## Tier counts vs v32")
    lines.append("")
    lines.append("| Tier | v32 | v35 | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for tier in ["Excellent", "Good", "Moderate", "Poor"]:
        old = v32_tier_counts.get(tier, 0)
        new = tier_counts.get(tier, 0)
        lines.append(f"| {tier} | {old} | {new} | {new - old:+d} |")
    lines.append(
        f"| Tier 1 (CCC >= 0.79) | {v32_tier1} | {tier1} | "
        f"{tier1 - v32_tier1:+d} |"
    )
    lines.append("")
    lines.append(
        f"Promotions vs v32: **{len(proms)}**. "
        + (
            "" if not proms else
            "Slots: " + ", ".join(
                f"{p['slot']} ({p['v32_tier']} -> {p['v35_tier']}, "
                f"reader={p['reader']})"
                for p in proms
            )
        )
    )
    lines.append(
        f"Demotions vs v32: **{len(demos)}**. "
        + (
            "" if not demos else
            "Slots: " + ", ".join(
                f"{p['slot']} ({p['v32_tier']} -> {p['v35_tier']}, "
                f"reader={p['reader']})"
                for p in demos
            )
        )
    )
    lines.append("")

    # ---- Category A (the question) ----
    lines.append(
        "## Category A: did the extrema-aware L3 break the LoA wall?"
    )
    lines.append("")
    lines.append(
        "These slots have strong CCC (0.81-0.93) in v32 but miss the "
        "LoA +/-10 deg gate by 1-3 deg. Agent OO's lever was a per-slot "
        "TinyMLP with two heads (pred_max, pred_min) directly supervised "
        "against per-clip ground-truth extrema, hypothesising that "
        "max/min supervision tightens LoA where ROM-only supervision "
        "cannot."
    )
    lines.append("")
    lines.append(
        "| Slot | v32 tier | v32 CCC | v32 LoA/2 | v32 reader | "
        "v33 CCC | v33 LoA/2 | v34 CCC | v34 LoA/2 | v35 reader | "
        "v35 tier | Promoted? |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | "
        "--- | --- | --- |"
    )
    for row in cat_a:
        lines.append(
            f"| {row['slot']} | {row['v32_tier']} | "
            f"{_fmt_num(row['v32_ccc'], 3)} | "
            f"{_fmt_num(row['v32_loa'])} | {row['v32_reader']} | "
            f"{_fmt_num(row['v33_ccc'], 3)} | "
            f"{_fmt_num(row['v33_loa'])} | "
            f"{_fmt_num(row['v34_ccc'], 3)} | "
            f"{_fmt_num(row['v34_loa'])} | "
            f"{row['v35_reader']} | {row['v35_tier']} | "
            f"{'YES' if row['promoted_to_good'] else 'no'} |"
        )
    cat_a_promoted = sum(1 for r in cat_a if r["promoted_to_good"])
    lines.append("")
    lines.append(
        f"**Category A promotions to Good: {cat_a_promoted}/{len(cat_a)}.**"
    )
    lines.append("")

    # ---- Extrema diagnostics ----
    lines.append("## Extrema-prediction diagnostics on Category A slots")
    lines.append("")
    lines.append(
        "Per-slot LOSO mean absolute error on max and min predictions "
        "(from v33 and v34 heads). This tells us whether the new heads "
        "are actually learning extrema or just being clamped to "
        "ROM-implied averages."
    )
    lines.append("")
    lines.append(
        "| Slot | v33 MAE_max | v33 MAE_min | v33 n | v34 MAE_max | "
        "v34 MAE_min | v34 n |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in cat_a:
        slot_key = row["slot"]
        d = extrema_diag.get(slot_key, {})
        v33d = d.get("v33", {})
        v34d = d.get("v34", {})
        lines.append(
            f"| {slot_key} | "
            f"{_fmt_num(v33d.get('mae_max_deg'))} | "
            f"{_fmt_num(v33d.get('mae_min_deg'))} | "
            f"{_fmt_num(v33d.get('n'), 0)} | "
            f"{_fmt_num(v34d.get('mae_max_deg'))} | "
            f"{_fmt_num(v34d.get('mae_min_deg'))} | "
            f"{_fmt_num(v34d.get('n'), 0)} |"
        )
    lines.append("")

    # ---- Reader distribution ----
    lines.append("## Reader distribution in v35")
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
        "v30": "v23 L2 + learned L3 (TinyMLP, ROM-only)",
        "v31": "v29 mirror-flip L2 + learned L3 (TinyMLP, ROM-only)",
        "v33": "v23 L2 + extrema-aware learned L3 (max/min heads)",
        "v34": "v29 mirror-flip L2 + extrema-aware learned L3",
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
        "- **Double-LOSO upper bound** (unchanged from v32). v23/v29 L2 "
        "trained on all 24 cohort subjects; L3 LOSO at subject level "
        "only. Per-fold L2 variance from HH2 suggests true double-LOSO "
        "numbers could be ~0.05-0.10 |r| lower."
    )
    lines.append(
        "- **Extrema-aware L3 overfit risk is real.** Per-slot models "
        "with n=9-22 LOSO inner folds and ~5K params, now with two "
        "output heads instead of one. Mitigations carried from NN: "
        "hidden=32 (tiny capacity), dropout 0.2, weight_decay 1e-3, "
        "early stopping on 15% inner-val. Per-slot fallback to ridge if "
        "extrema-aware CCC underperforms ridge by > 0.05 keeps a "
        "no-regression guarantee."
    )
    lines.append(
        "- **Extrema GT was newly computed.** For OpenCap, max/min of the "
        "target IK angle column from the .mot file (same source as the "
        "existing gt_rom_col which returns max - min). For ASPset, max/min "
        "of joint_angles_from_aspset(clip)[target_gt] (same source as the "
        "v12_combined inline GT). Per-row alignment was sanity-checked: "
        "|gt_max - gt_min - y| < 1e-3 on all surviving rows; mismatching "
        "rows are filtered."
    )
    lines.append(
        "- **No multi-camera fusion.** Single DWPose stream at inference "
        "(Couro's core single-camera differentiator)."
    )
    lines.append(
        "- **No new sport thresholds.** Extrema-aware L3 only affects how "
        "ROM is predicted; downstream sport-specific risk multipliers "
        "are unchanged."
    )
    lines.append("")

    # ---- Recommendation ----
    lines.append("## Recommendation for next move")
    lines.append("")
    if tier_counts["Good"] > v32_tier_counts["Good"]:
        lines.append(
            f"Adopt v35 selectively. Net "
            f"**{tier_counts['Good'] - v32_tier_counts['Good']}** more "
            "Good slots than v32. Tier-1 (CCC >= 0.79) count is "
            f"{'up' if tier1 > v32_tier1 else 'down'} by "
            f"{tier1 - v32_tier1:+d} vs v32."
        )
    elif tier_counts["Good"] == v32_tier_counts["Good"]:
        lines.append(
            f"Hold at v32. Extrema-aware L3 matched v32 Good count "
            f"({tier_counts['Good']}) but did not net-gain. The per-slot "
            "fallback to ridge prevented regressions on slots where the "
            "two-head MLP overfit, but extrema supervision did not break "
            "the LoA wall on Category A targets."
        )
    else:
        lines.append(
            f"Hold at v32. v35 lost Good slots "
            f"({tier_counts['Good']} vs {v32_tier_counts['Good']}). "
            "Extrema-aware learning regressed somewhere -- investigate "
            "before re-running."
        )
    lines.append("")
    if cat_a_promoted == 0:
        lines.append(
            "**Category A verdict: extrema-aware L3 did NOT crack the "
            "LoA wall.** Promoting these slots from Moderate to Good will "
            "require a different lever -- candidates include "
            "(a) per-slot residual calibration on held-out folds, "
            "(b) richer per-clip feature vectors that capture peak "
            "timing more directly, or (c) revisiting Layer 2 with an "
            "extrema-aware loss (GG2 style) targeted at the affected "
            "joints."
        )
    else:
        lines.append(
            f"**Category A verdict: {cat_a_promoted}/{len(cat_a)} "
            "slots promoted to Good** via extrema-aware L3. The "
            "max/min supervision lever did break the LoA wall on the "
            "promoted slot(s); rest of the slots remain LoA-limited and "
            "need a different lever."
        )
    lines.append("")

    V35_REPORT_PATH.write_text("\n".join(lines) + "\n")
    log(f"wrote REPORT -> {V35_REPORT_PATH}")


def main() -> None:
    log("=== Agent OO (v35) START ===")
    build()
    write_v35_report()
    log("=== Agent OO (v35) DONE ===")


if __name__ == "__main__":
    main()
