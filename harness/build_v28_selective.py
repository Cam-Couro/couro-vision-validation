"""Build v28 selective oracle: per-slot best across
v17 / v18 / v20 / v23 / v24 / v26 / v27.

Sister of ``harness.build_v25_selective`` (Agent LL Phase C). v28 adds the
two new MM (per-source heads) candidates to the reader pool:

  * **v26** — MM-A per-source heads + per-frame SmoothL1 loss
  * **v27** — MM-B per-source heads + ROM-aware loss

# Selection rule
Same as v25: tier first (Excellent > Good > Moderate > Poor), within tier
highest Lin's CCC, ties broken by canonical reader order
(v17, v18, v20, v23, v24, v26, v27) — i.e., older / simpler reader wins
ties.

# Outputs
- ``results/deploy_ready_models_v28_selective.json``
- ``data/v28_selective_oracle/per_slot_picks_v28.json``
- ``data/v28_selective_oracle/REPORT.md``

# LOSO discipline (inherited)
v17 slots are clean double-LOSO; v18/v20/v23/v24/v26/v27 carry the
Layer-3-LOSO-only caveat (L2 trained on all subjects, L3 LOSO only).
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

V25_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v25_selective.json"
V28_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v28_selective.json"
V28_OUT_DIR: Final[Path] = DATA_ROOT / "v28_selective_oracle"
V28_PICKS_PATH: Final[Path] = V28_OUT_DIR / "per_slot_picks_v28.json"
V28_REPORT_PATH: Final[Path] = V28_OUT_DIR / "REPORT.md"


TIER_RANK: Final[dict[str, int]] = {
    "Excellent": 4,
    "Good": 3,
    "Moderate": 2,
    "Poor": 1,
}

READER_ORDER: Final[tuple[str, ...]] = (
    "v17", "v18", "v20", "v23", "v24", "v26", "v27",
)


# The 4 supplementary slots that are addressable through modeling
# (the 2 MM targets explicitly + 2 sister lumbar slots to watch for regression).
FLOOR_LIFT_TARGETS: Final[tuple[tuple[str, str, float, str], ...]] = (
    ("hip_adduction_r", "front_oblique_left", 0.69, "v20 (LL/v25 pick)"),
    ("lumbar_extension", "front_oblique_left", 0.71, "v18 (LL/v25 pick)"),
    # Sister slots to monitor for regression.
    ("lumbar_extension", "front_oblique_right", 0.79, "v23 (LL/v25 pick)"),
    ("lumbar_extension", "side_left", 0.88, "v23 (LL/v25 pick)"),
    ("lumbar_extension", "side_right", 0.85, "v23 (LL/v25 pick)"),
)


def log(msg: str) -> None:
    print(f"[v28 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
        (s["target"], s["view"]): _stats_from_slot_dict(s) for s in d["slots"]
    }


def load_per_slot(path: Path) -> dict[tuple[str, str], dict]:
    d = _load_json(path)
    return {
        (s["target"], s["view"]): _stats_from_slot_dict(s)
        for s in d["slots"]
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
# Slot weights carrier.
# -------------------------------------------------------------------------


def _pick_weights_block(reader: str, slot_entry: dict) -> dict:
    if reader == "v17":
        return {
            "reader": "v17",
            "weights_zscored": slot_entry.get("weights_zscored"),
            "intercept": slot_entry.get("intercept"),
            "feature_mean": slot_entry.get("feature_mean"),
            "feature_std": slot_entry.get("feature_std"),
            "approach": slot_entry.get("approach"),
        }
    # v18, v20, v23, v24, v26, v27 share FF's "_v18_" suffixed keys.
    return {
        "reader": reader,
        "weights": slot_entry.get("weights_v18_learned_l2"),
        "bias": slot_entry.get("bias_v18_learned_l2"),
        "feature_mean": slot_entry.get("feature_mean_v18"),
        "feature_std": slot_entry.get("feature_std_v18"),
        "approach": (
            slot_entry.get("v18_approach") or slot_entry.get("approach")
        ),
    }


# -------------------------------------------------------------------------
# Build.
# -------------------------------------------------------------------------


def build() -> dict:
    log("loading reader sources ...")
    v17_deploy = _load_json(V17_PATH)
    v18_deploy = _load_json(V18_PATH)
    v20_deploy = _load_json(V20_PATH)
    v23_deploy = _load_json(V23_PATH)
    v24_deploy = _load_json(V24_PATH)
    v26_deploy = _load_json(V26_PATH) if V26_PATH.exists() else None
    v27_deploy = _load_json(V27_PATH) if V27_PATH.exists() else None

    v17_baseline = load_baseline_v17()
    v18_stats = load_per_slot(V18_PER_SLOT)
    v20_stats = load_per_slot(V20_PER_SLOT)
    v23_stats = load_per_slot(V23_PER_SLOT)
    v24_stats = load_per_slot(V24_PER_SLOT)
    v26_stats = (
        load_per_slot(V26_PER_SLOT) if V26_PER_SLOT.exists() else {}
    )
    v27_stats = (
        load_per_slot(V27_PER_SLOT) if V27_PER_SLOT.exists() else {}
    )

    if not v26_deploy or not v26_stats:
        log("  WARN: v26 inputs missing — v28 will degrade to v25 selection")
    if not v27_deploy or not v27_stats:
        log("  WARN: v27 inputs missing — v28 will degrade to v25 selection")

    picks: list[dict] = []
    per_slot_reader: dict[str, str] = {}
    v28_models: dict[str, dict[str, dict]] = {}
    tier_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    reader_distribution = {r: 0 for r in READER_ORDER}
    promotions_vs_v17: list[dict] = []
    promotions_vs_v25: list[dict] = []
    new_good_via_mm: list[dict] = []

    # Load v25 picks for promotions-vs-v25 comparison.
    v25_models = _load_json(V25_PATH)["models"]
    v25_pick_tier: dict[str, str] = {}
    v25_pick_ccc: dict[str, float | None] = {}
    v25_pick_reader: dict[str, str] = {}
    for tgt, slots in v25_models.items():
        for view, entry in slots.items():
            key = f"{tgt}|{view}"
            v25_pick_tier[key] = entry.get("v25_tier") or "Poor"
            v25_pick_ccc[key] = (entry.get("v25_stats") or {}).get("ccc_lin")
            v25_pick_reader[key] = entry.get("selected_reader") or "v17"

    for target, slots in v17_deploy["models"].items():
        for view, v17_entry in slots.items():
            key = (target, view)
            slot_key = f"{target}/{view}"
            candidates: dict[str, dict] = {}
            if key in v17_baseline:
                candidates["v17"] = v17_baseline[key]
            if key in v18_stats:
                candidates["v18"] = v18_stats[key]
            if key in v20_stats:
                candidates["v20"] = v20_stats[key]
            if key in v23_stats:
                candidates["v23"] = v23_stats[key]
            if key in v24_stats:
                candidates["v24"] = v24_stats[key]
            if key in v26_stats:
                candidates["v26"] = v26_stats[key]
            if key in v27_stats:
                candidates["v27"] = v27_stats[key]

            if not candidates:
                log(
                    f"  WARN: no candidates for {slot_key} -- "
                    f"falling back to v17 entry"
                )
                v28_models.setdefault(target, {})[view] = dict(v17_entry)
                per_slot_reader[slot_key] = "v17"
                continue

            best_reader, best_stats = select_reader(candidates)
            best_tier = best_stats.get("classification") or "Poor"
            tier_counts[best_tier] = tier_counts.get(best_tier, 0) + 1
            reader_distribution[best_reader] = (
                reader_distribution.get(best_reader, 0) + 1
            )
            per_slot_reader[slot_key] = best_reader

            source_bundle = {
                "v17": v17_deploy,
                "v18": v18_deploy,
                "v20": v20_deploy,
                "v23": v23_deploy,
                "v24": v24_deploy,
                "v26": v26_deploy,
                "v27": v27_deploy,
            }[best_reader]
            if source_bundle is None:
                # Missing v26/v27 input shouldn't happen since we filter by
                # candidates, but be defensive.
                source_slot = v17_entry
            else:
                source_slot = (
                    source_bundle["models"].get(target, {}).get(view)
                    or v17_entry
                )

            weight_block = _pick_weights_block(best_reader, source_slot)
            v28_entry = dict(v17_entry)
            v28_entry["selected_reader"] = best_reader
            v28_entry["v28_tier"] = best_tier
            v28_entry["v28_stats"] = best_stats
            v28_entry["v28_weights"] = weight_block
            v28_models.setdefault(target, {})[view] = v28_entry

            v17_tier = (
                v17_baseline.get(key, {}).get("classification") or "Poor"
            )
            ccc = best_stats.get("ccc_lin")
            loa = best_stats.get("loa_half_width_deg")
            ccc_str = (
                "nan"
                if ccc is None or (isinstance(ccc, float) and math.isnan(ccc))
                else f"{ccc:.3f}"
            )
            loa_str = (
                "nan"
                if loa is None or (isinstance(loa, float) and math.isnan(loa))
                else f"{loa:.2f}"
            )
            v25_t = v25_pick_tier.get(f"{target}|{view}", "?")
            v25_c = v25_pick_ccc.get(f"{target}|{view}")
            v25_c_str = (
                "nan" if v25_c is None or (
                    isinstance(v25_c, float) and math.isnan(v25_c)
                ) else f"{v25_c:.3f}"
            )
            log(
                f"  {slot_key} -> {best_reader} ({best_tier}) "
                f"CCC={ccc_str} LoAh={loa_str} (v17 tier={v17_tier}, "
                f"v25 tier={v25_t} CCC={v25_c_str})"
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
                "v17_tier": v17_tier,
                "v25_tier": v25_t,
                "v25_ccc": v25_c,
                "v25_reader": v25_pick_reader.get(f"{target}|{view}"),
            })

            if _tier_rank(best_tier) > _tier_rank(v17_tier):
                promotions_vs_v17.append({
                    "slot": f"{target}|{view}|{v17_entry.get('approach')}",
                    "v17_tier": v17_tier,
                    "v28_tier": best_tier,
                    "reader": best_reader,
                    "ccc": ccc,
                    "loa_half": loa,
                })
            if _tier_rank(best_tier) > _tier_rank(v25_t):
                promotions_vs_v25.append({
                    "slot": f"{target}|{view}",
                    "v25_tier": v25_t,
                    "v28_tier": best_tier,
                    "reader": best_reader,
                    "ccc": ccc,
                    "loa_half": loa,
                })
            if best_tier == "Good" and best_reader in ("v26", "v27") \
                    and v25_t != "Good":
                new_good_via_mm.append({
                    "slot": f"{target}|{view}",
                    "reader": best_reader,
                    "ccc": ccc,
                    "loa_half": loa,
                    "v25_tier": v25_t,
                    "v17_tier": v17_tier,
                })

    # Floor lift analysis.
    floor_lift: list[dict] = []
    for tgt, view, prev_ccc, prev_origin in FLOOR_LIFT_TARGETS:
        key = (tgt, view)
        v26_st = v26_stats.get(key, {})
        v27_st = v27_stats.get(key, {})
        v26_ccc = v26_st.get("ccc_lin")
        v27_ccc = v27_st.get("ccc_lin")
        v26_tier = v26_st.get("classification")
        v27_tier = v27_st.get("classification")
        pick = next(
            (p for p in picks if p["slot"] == f"{tgt}|{view}"), None,
        )
        v28_ccc = pick["ccc"] if pick else None
        v28_tier = pick["tier"] if pick else None
        v28_reader = pick["reader"] if pick else None
        floor_lift.append({
            "slot": f"{tgt}|{view}",
            "previous_ccc": prev_ccc,
            "previous_reader_origin": prev_origin,
            "v26_ccc": v26_ccc,
            "v26_tier": v26_tier,
            "v27_ccc": v27_ccc,
            "v27_tier": v27_tier,
            "v28_ccc": v28_ccc,
            "v28_tier": v28_tier,
            "v28_reader": v28_reader,
            "crossed_0_79": (
                v28_ccc is not None
                and not (
                    isinstance(v28_ccc, float) and math.isnan(v28_ccc)
                )
                and v28_ccc >= 0.79
            ),
            "crossed_0_80": (
                v28_ccc is not None
                and not (
                    isinstance(v28_ccc, float) and math.isnan(v28_ccc)
                )
                and v28_ccc >= 0.80
            ),
        })

    floor_lifted_any_79 = any(f["crossed_0_79"] for f in floor_lift[:2])
    floor_lifted_any_80 = any(f["crossed_0_80"] for f in floor_lift[:2])

    v28_out = {
        "version": "v28_selective_oracle",
        "produced_by": "harness.build_v28_selective (Agent MM Phase C)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            f"Per-slot oracle-best across v17 / v18 / v20 / v23 / v24 / v26 / "
            f"v27 readers. {tier_counts['Good']} Good / "
            f"{tier_counts['Moderate']} Moderate / "
            f"{tier_counts['Poor']} Poor tier counts (vs v25: 9/7/7). "
            f"v26 = MM-A per-source heads + per-frame SmoothL1; "
            f"v27 = MM-B per-source heads + ROM-aware."
        ),
        "approaches": v17_deploy.get("approaches"),
        "training_dataset": v17_deploy.get("training_dataset"),
        "models": v28_models,
        "calibration_fix": v17_deploy.get("calibration_fix"),
        "selective_adoption": v17_deploy.get("selective_adoption"),
        "per_slot_reader": per_slot_reader,
        "reader_distribution": reader_distribution,
        "tier_counts": tier_counts,
        "promotions_vs_v17": promotions_vs_v17,
        "promotions_vs_v25": promotions_vs_v25,
        "new_good_via_mm": new_good_via_mm,
        "floor_lift_analysis": floor_lift,
        "floor_lifted_to_0_79_target_slots": floor_lifted_any_79,
        "floor_lifted_to_0_80_target_slots": floor_lifted_any_80,
    }
    V28_PATH.write_text(json.dumps(v28_out, indent=2))
    log(f"wrote v28 deploy bundle -> {V28_PATH}")

    V28_OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks_out = {
        "version": "v28_selective_oracle",
        "description": v28_out["description"],
        "tier_counts": tier_counts,
        "reader_distribution": reader_distribution,
        "promotions_vs_v17": promotions_vs_v17,
        "promotions_vs_v25": promotions_vs_v25,
        "new_good_via_mm": new_good_via_mm,
        "floor_lift_analysis": floor_lift,
        "floor_lifted_to_0_79_target_slots": floor_lifted_any_79,
        "floor_lifted_to_0_80_target_slots": floor_lifted_any_80,
        "picks": picks,
    }
    V28_PICKS_PATH.write_text(json.dumps(picks_out, indent=2))
    log(f"wrote v28 per-slot picks -> {V28_PICKS_PATH}")

    return v28_out


# -------------------------------------------------------------------------
# REPORT.md writer.
# -------------------------------------------------------------------------


def _fmt_num(v: object, prec: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def write_v28_report() -> None:
    v28 = _load_json(V28_PATH)
    v17_baseline = load_baseline_v17()
    v18_stats = load_per_slot(V18_PER_SLOT)
    v20_stats = load_per_slot(V20_PER_SLOT)
    v23_stats = load_per_slot(V23_PER_SLOT)
    v24_stats = load_per_slot(V24_PER_SLOT)
    v26_stats = (
        load_per_slot(V26_PER_SLOT) if V26_PER_SLOT.exists() else {}
    )
    v27_stats = (
        load_per_slot(V27_PER_SLOT) if V27_PER_SLOT.exists() else {}
    )
    picks = _load_json(V28_PICKS_PATH)

    tier_counts = v28["tier_counts"]
    reader_dist = v28["reader_distribution"]
    promotions_v17 = v28["promotions_vs_v17"]
    promotions_v25 = v28["promotions_vs_v25"]
    new_good_via_mm = v28.get("new_good_via_mm", [])
    floor_lift = v28.get("floor_lift_analysis", [])
    lifted_79 = v28.get("floor_lifted_to_0_79_target_slots", False)
    lifted_80 = v28.get("floor_lifted_to_0_80_target_slots", False)

    lines: list[str] = []
    lines.append(
        "# v28 Selective Oracle Deploy + v26/v27 Per-Source Heads Layer 2"
    )
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(
        "**Build:** Agent MM Phase C — MM-A (v26, per-source heads + "
        "per-frame SmoothL1) and MM-B (v27, per-source heads + ROM-aware) "
        "join the v17/v18/v20/v23/v24 reader pool in a per-slot oracle "
        "selection."
    )
    lines.append(
        f"**Verdict:** **{tier_counts['Good']} validated Good-tier slots** "
        f"(v25 was 9; v22 was 8; v17 baseline was 3)."
    )

    if lifted_80:
        lifted_slots = [
            f["slot"].replace("|", " / ")
            for f in floor_lift[:2] if f["crossed_0_80"]
        ]
        lines.append(
            f"**Floor lift to ≥0.80 CCC:** YES — target slot(s) crossed "
            f"the 0.80 bar under MM: {', '.join(lifted_slots)}."
        )
    elif lifted_79:
        lifted_slots = [
            f["slot"].replace("|", " / ")
            for f in floor_lift[:2] if f["crossed_0_79"]
        ]
        lines.append(
            f"**Floor lift to ≥0.79 CCC (Tier 1 promotion):** YES — "
            f"target slot(s) crossed the 0.79 Tier 1 bar under MM: "
            f"{', '.join(lifted_slots)}."
        )
    else:
        lines.append(
            "**Floor lift:** NO — neither of the 2 target supplementary "
            "slots (hip_adduction_r/front_oblique_left, "
            "lumbar_extension/front_oblique_left) crossed 0.79 CCC under "
            "MM-A or MM-B. Per-source heads did not unlock the bias-"
            "limited supplementary slots."
        )
    lines.append("")

    # ----- Floor lift answer (the headline) -----
    lines.append("## Floor lift verdict — the 2 target slots + sister monitor")
    lines.append("")
    lines.append(
        "The MM brief asked: did per-source heads push the 2 "
        "convention-mismatched supplementary slots to Tier 1 (CCC ≥ 0.79), "
        "or all the way to ≥0.80? Plus: do per-source heads hurt the "
        "existing Tier 1 lumbar slots?"
    )
    lines.append("")
    lines.append(
        "| Slot | v25 CCC (reader) | v26 CCC (MM-A) | v27 CCC (MM-B) | "
        "v28 oracle CCC (reader) | Tier change vs v25 | "
        "Crossed 0.79? |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | --- | --- | :---: |"
    )
    for f in floor_lift:
        slot_disp = f["slot"].replace("|", " / ")
        prev = (
            f"{f['previous_ccc']:.2f} ({f['previous_reader_origin']})"
        )
        v26c = _fmt_num(f.get("v26_ccc"))
        v27c = _fmt_num(f.get("v27_ccc"))
        v28c = (
            "-"
            if f["v28_ccc"] is None
            or (
                isinstance(f["v28_ccc"], float)
                and math.isnan(f["v28_ccc"])
            )
            else f"{f['v28_ccc']:.2f} ({f['v28_reader']})"
        )
        v28t = f["v28_tier"] or "-"
        prev_tier = (
            "Good"
            if f["previous_ccc"] >= 0.60
            else ("Moderate" if f["previous_ccc"] >= 0.40 else "Poor")
        )
        tier_change = (
            f"{prev_tier} → {v28t}"
            if v28t != prev_tier else f"{v28t} (unchanged)"
        )
        crossed = "YES" if f["crossed_0_79"] else "no"
        lines.append(
            f"| {slot_disp} | {prev} | {v26c} | {v27c} | {v28c} | "
            f"{tier_change} | {crossed} |"
        )
    lines.append("")

    # ----- Tier count delta -----
    lines.append("## Tier count delta")
    lines.append("")
    lines.append(
        "| Tier | v17 baseline | v22 selective | v25 selective | "
        "**v28 selective (+ v26 + v27)** | Δ vs v25 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    v17_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    for stats in v17_baseline.values():
        tier = stats.get("classification") or "Poor"
        v17_counts[tier] = v17_counts.get(tier, 0) + 1
    v22_picks = _load_json(
        DATA_ROOT / "v22_selective_oracle" / "per_slot_picks_v22.json"
    )
    v22_counts = v22_picks["tier_counts"]
    v25 = _load_json(V25_PATH)
    v25_counts = v25["tier_counts"]
    for tier in ["Excellent", "Good", "Moderate", "Poor"]:
        b = v17_counts.get(tier, 0)
        p22 = v22_counts.get(tier, 0)
        p25 = v25_counts.get(tier, 0)
        n = tier_counts.get(tier, 0)
        lines.append(
            f"| {tier} | {b} | {p22} | {p25} | **{n}** | {n - p25:+d} |"
        )
    lines.append("")

    # Reader distribution.
    lines.append("## Reader distribution across 23 slots (v28)")
    lines.append("")
    for r in READER_ORDER:
        cnt = reader_dist.get(r, 0)
        desc = {
            "v17": "hand-engineered Layer 2 (canonical)",
            "v18": "EE2 OpenCap-only learned Layer 2",
            "v20": "GG2 ROM-aware OpenCap-only learned Layer 2",
            "v23": "HH2 combined-cohort learned Layer 2 (Agent KK)",
            "v24": "LL combined-cohort + ROM-aware learned Layer 2",
            "v26": (
                "MM-A per-source heads + per-frame SmoothL1 learned "
                "Layer 2 (Agent MM, this build)"
            ),
            "v27": (
                "MM-B per-source heads + ROM-aware learned Layer 2 "
                "(Agent MM, this build)"
            ),
        }[r]
        lines.append(f"- **{r}** — {desc}: **{cnt} slots**")
    lines.append("")

    # The Good slots.
    lines.append(f"## The {tier_counts['Good']} validated Good slots (v28)")
    lines.append("")
    lines.append("| Slot | Reader | CCC | LoA half | v17 tier | v25 tier |")
    lines.append("| --- | --- | ---: | ---: | --- | --- |")
    for p in picks["picks"]:
        if p["tier"] == "Good":
            slot_disp = p["slot"].replace("|", " / ")
            lines.append(
                f"| {slot_disp} | {p['reader']} | "
                f"{_fmt_num(p['ccc'])} | ±{_fmt_num(p['loa_half'])}° | "
                f"{p['v17_tier']} | {p['v25_tier']} |"
            )
    lines.append("")

    # New Good via MM specifically.
    if new_good_via_mm:
        lines.append("### New Good slots from v26/v27 specifically")
        lines.append("")
        for g in new_good_via_mm:
            slot_disp = g["slot"].replace("|", " / ")
            lines.append(
                f"- **{slot_disp}**: {g['reader']} CCC={_fmt_num(g['ccc'])}, "
                f"LoA half=±{_fmt_num(g['loa_half'])}° "
                f"(was {g['v25_tier']} in v25; "
                f"{g['v17_tier']} in v17 baseline)"
            )
        lines.append("")
    else:
        lines.append("### New Good slots from v26/v27 specifically")
        lines.append("")
        lines.append(
            "**None.** No slot promoted to Good *via v26 or v27 "
            "specifically* beyond what v25 already offered. The "
            "per-source heads architecture did not unlock any new Good "
            "slot at deploy."
        )
        lines.append("")

    # Promotions vs v25.
    if promotions_v25:
        lines.append("## Promotions vs v25")
        lines.append("")
        lines.append(
            "| Slot | v25 tier | v28 tier | Reader | CCC | LoA half |"
        )
        lines.append("| --- | --- | --- | --- | ---: | ---: |")
        for p in promotions_v25:
            slot_disp = p["slot"].replace("|", " / ")
            lines.append(
                f"| {slot_disp} | {p['v25_tier']} | {p['v28_tier']} | "
                f"{p['reader']} | {_fmt_num(p['ccc'])} | "
                f"±{_fmt_num(p['loa_half'])}° |"
            )
        lines.append("")
    else:
        lines.append("## Promotions vs v25")
        lines.append("")
        lines.append(
            "**None.** v28 matches v25 on tier counts. Per-source heads "
            "(v26/v27) compete with the existing v17/v18/v20/v23/v24 "
            "readers but did not beat them at the tier-promotion bar on "
            "any slot."
        )
        lines.append("")

    # Promotions vs v17.
    lines.append("## All promotions vs v17 baseline (v28)")
    lines.append("")
    lines.append(
        "| Slot | v17 tier | v28 tier | Reader | CCC | LoA half |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: |")
    for p in promotions_v17:
        slot_disp = p["slot"].replace("|", " / ")
        lines.append(
            f"| {slot_disp} | {p['v17_tier']} | {p['v28_tier']} | "
            f"{p['reader']} | {_fmt_num(p['ccc'])} | "
            f"±{_fmt_num(p['loa_half'])}° |"
        )
    lines.append("")

    # Per-slot CCC table across all 7 readers.
    lines.append("## Per-slot CCC / LoA across all 7 readers")
    lines.append("")
    lines.append(
        "| Slot | v17 | v18 | v20 | v23 | v24 | v26 | v27 | v28 pick |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for p in picks["picks"]:
        tgt, view = p["slot"].split("|")
        key = (tgt, view)
        cells = []
        src_map = {
            "v17": v17_baseline, "v18": v18_stats,
            "v20": v20_stats, "v23": v23_stats,
            "v24": v24_stats, "v26": v26_stats, "v27": v27_stats,
        }
        for r in ("v17", "v18", "v20", "v23", "v24", "v26", "v27"):
            st = src_map[r].get(key, {})
            ccc = st.get("ccc_lin")
            loa = st.get("loa_half_width_deg")
            cells.append(
                f"{_fmt_num(ccc)}/{_fmt_num(loa, prec=1)}°"
            )
        lines.append(
            f"| {tgt} / {view} | "
            + " | ".join(cells)
            + f" | **{p['reader']}** |"
        )
    lines.append("")

    # Cameron-question table.
    lines.append("## Cameron's MM-brief table (per-slot pick history)")
    lines.append("")
    lines.append(
        "| Slot | v25 CCC | MM-A (v26) CCC | MM-B (v27) CCC | v28 oracle CCC |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for tgt, view, prev_ccc, _ in FLOOR_LIFT_TARGETS[:2]:
        key = (tgt, view)
        slot_disp = f"{tgt} / {view}"
        v26c = _fmt_num(v26_stats.get(key, {}).get("ccc_lin"))
        v27c = _fmt_num(v27_stats.get(key, {}).get("ccc_lin"))
        pick = next(
            (p for p in picks["picks"] if p["slot"] == f"{tgt}|{view}"), None,
        )
        v28c = _fmt_num(pick["ccc"]) if pick else "-"
        lines.append(
            f"| {slot_disp} | {prev_ccc:.2f} | {v26c} | {v27c} | {v28c} |"
        )
    lines.append("")

    # Honest caveats.
    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "1. **Layer-3-LOSO-only caveat applies to v18/v20/v23/v24/v26/v27.** "
        "L2 trained on ALL 24 cohort subjects (9 OpenCap + 15 ASPset). L3 "
        "ridge LOSO at subject level only. Tier promotions involving any "
        "cohort subject are upper bounds; per HH2's per-fold variance the "
        "true double-LOSO number could be ~0.05-0.10 |r| lower. **Only v17 "
        "(hand-engineered) slots are clean double-LOSO.**"
    )
    lines.append(
        "2. **Per-source heads tested ONLY at L2 architecture level.** v26 "
        "and v27 are HH2's own recommended fix for the OpenCap/ASPset "
        "convention mismatch on hip_adduction_r and lumbar_extension. The "
        "shared 5-output head consumed by the L3 ridge is what Couro "
        "deploys; the ASPset head is discarded at inference."
    )
    lines.append(
        "3. **All-data L2, not 24-fold LOSO Phase A.** Like the LL build, "
        "this build cycle skipped the 24-fold LOSO Phase A eval to fit "
        "compute budget. The all-data L2 cached in "
        "`models/learned_layer2_persource_perframe_alldata_v1.pt` and "
        "`models/learned_layer2_persource_romaware_alldata_v1.pt` is what "
        "v26/v27 use. The harness supports Phase A LOSO via "
        "`python3 -m harness.learned_layer2_combined_persource --variant "
        "mm_a` (and `--variant mm_b`)."
    )
    lines.append(
        "4. **Two supplementary slots are bias-limited, not noise-limited.** "
        "Cameron's brief flagged hip_adduction_r/front_oblique_left as "
        "having the tightest LoA in the entire table (±3.3° under v20). "
        "If MM did not lift it, the residual is a true convention/calibration "
        "bias that even per-source heads can't fix without "
        "metric-redefinition or new ground-truth collection."
    )
    lines.append(
        "5. **ASPset has no foot keypoints (ankle_angle_r cohort-limited).** "
        "Two of the four supplementary slots (ankle_angle_r/front_oblique_right "
        "at v23 CCC 0.73 and ankle_angle_r/side_right at v17 CCC 0.64) are "
        "cohort-limited (n=9 OpenCap). MM does not address those — they "
        "require fresh paired ankle GT collection."
    )
    lines.append(
        "6. **No invented numbers.** All CCC / LoA / |r| values are computed "
        "from this build's all-data L2 + L3 ridge re-fit, or carried "
        "verbatim from prior per-slot validity files (v17/v18/v20/v23/v24)."
    )
    lines.append("")

    # Single-camera reaffirmation.
    lines.append("## Single-camera contract preserved")
    lines.append("")
    lines.append(
        "Every reader in the v28 pool (v17, v18, v20, v23, v24, v26, v27) "
        "consumes a single DWPose stream from one phone camera. No "
        "multi-camera fusion. Same input/output contract as Couro's "
        "deployed Layer 2."
    )
    lines.append("")

    # Files.
    lines.append("## Files")
    lines.append("")
    lines.append(
        "- `results/deploy_ready_models_v28_selective.json` — v28 deploy "
        "bundle with `per_slot_reader` dispatch map"
    )
    lines.append(
        "- `results/deploy_ready_models_v26_persource_perframe.json` — "
        "v26 deploy candidate (MM-A per-source heads + per-frame SmoothL1)"
    )
    lines.append(
        "- `results/deploy_ready_models_v27_persource_romaware.json` — "
        "v27 deploy candidate (MM-B per-source heads + ROM-aware)"
    )
    lines.append(
        "- `data/v28_selective_oracle/per_slot_picks_v28.json` — per-slot "
        "pick audit trail with CCC / LoA per candidate reader"
    )
    lines.append(
        "- `data/layer3_retrain_persource_perframe/per_slot_validity_v26.json`"
        " — per-slot v26 validity stats (LOSO at L3)"
    )
    lines.append(
        "- `data/layer3_retrain_persource_romaware/per_slot_validity_v27.json`"
        " — per-slot v27 validity stats (LOSO at L3)"
    )
    lines.append(
        "- `data/layer3_retrain_persource_perframe/REPORT.md` — v26 narrative"
    )
    lines.append(
        "- `data/layer3_retrain_persource_romaware/REPORT.md` — v27 narrative"
    )
    lines.append(
        "- `models/learned_layer2_persource_perframe_alldata_v1.pt` — "
        "all-data MM-A L2 checkpoint"
    )
    lines.append(
        "- `models/learned_layer2_persource_romaware_alldata_v1.pt` — "
        "all-data MM-B L2 checkpoint"
    )
    lines.append("")

    V28_REPORT_PATH.write_text("\n".join(lines) + "\n")
    log(f"wrote v28 REPORT -> {V28_REPORT_PATH}")


def _write_phaseA_report(out_path: Path, variant: str) -> None:
    """Write the Phase A narrative for either MM-A or MM-B.

    Phase A 24-fold LOSO was skipped this build cycle (same budget trade-off
    as the LL build); the narrative describes the architecture, loss
    routing, cohort, and the all-data L2 + Phase B preview.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_a = variant == "mm_a"
    name = "MM-A (per-source heads + per-frame SmoothL1)" \
        if is_a else "MM-B (per-source heads + ROM-aware)"
    loss = "masked per-frame SmoothL1 summed across the two heads " \
           "(HH2-style loss)" if is_a else \
           "masked per-frame SmoothL1 + lam=1.0 * extrema (peak + min) " \
           "summed across the two heads (LL-style ROM-aware loss)"
    batch_recipe = "batch_size=256 frames (HH2-style frame batches), " \
        "train_stride=2" if is_a else \
        "clips_per_step=4 (LL-style full-clip mini-batches), required " \
        "by the per-(clip, metric) extrema computation"
    deploy_tag = "v26" if is_a else "v27"
    next_phase_cmd = (
        "python3 -m harness.learned_layer2_combined_persource "
        f"--variant {'mm_a' if is_a else 'mm_b'}"
    )
    ckpt = f"learned_layer2_persource_{'perframe' if is_a else 'romaware'}" \
        "_alldata_v1.pt"

    lines: list[str] = []
    lines.append(
        f"# Per-Source Heads Learned Layer 2 — {name} (Agent MM, Phase A)"
    )
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(f"**Build:** Agent MM Phase A — {variant} variant")
    lines.append("")
    lines.append("## What this is")
    lines.append("")
    lines.append(
        "HH2's own recommended fix for the OpenCap/ASPset convention "
        "mismatch on `hip_adduction_r` (and, by extension, "
        "`lumbar_extension`). HH2's REPORT, verbatim:"
    )
    lines.append("")
    lines.append(
        "> hip_adduction_r regressed on OC-held (−0.051). Root cause: "
        "ASPset's asin-based frontal-plane definition is geometrically "
        "not the same as OpenSim's lumped-rotation hip_adduction. "
        "Future fix: drop hip_adduction_r from ASPset training, or use "
        "per-source target heads."
    )
    lines.append("")
    lines.append(
        "LL chose the 'drop' route. On the bias-limited slot "
        "`hip_adduction_r / front_oblique_left` (v20 CCC 0.69, "
        "LoA ±3.29° — the tightest LoA in the entire deploy table), "
        "v24 collapsed to CCC −0.77, confirming that dropping "
        "supervision alone is not the right medicine. MM tries the "
        "alternative: **separate output heads per source** so each "
        "head sees a clean target signal in its own convention."
    )
    lines.append("")
    lines.append("## Architecture (TemporalKeypointCNNConfPerSource)")
    lines.append("")
    lines.append(
        "  * **Backbone**: same 2-layer 1D conv over "
        "(T=9 frames × 22 keypoints × 3 channels = 66 input channels) "
        "used by HH2 / LL. Hidden=128. Identical to "
        "`TemporalKeypointCNNConf` from "
        "`harness.learned_layer2_combined`."
    )
    lines.append(
        "  * **Shared head** → 5 outputs (hip_flexion_r, "
        "hip_adduction_r, knee_angle_r, ankle_angle_r, "
        "lumbar_extension). This head is **deployed** — produces the "
        "OpenCap-convention angle vector consumed by Layer 3."
    )
    lines.append(
        "  * **ASPset head** → 2 outputs (hip_adduction_r_aspset, "
        "lumbar_extension_aspset). This head is **discarded at "
        "deploy**; it exists only to absorb the ASPset gradient on the "
        "two convention-mismatched metrics so it doesn't pollute the "
        "shared head."
    )
    lines.append("")
    lines.append(
        "Total params: 109,127 (+8.3K over HH2's 100,741 from the "
        "second head)."
    )
    lines.append("")
    lines.append("## Loss routing")
    lines.append("")
    lines.append(
        "Each training sample carries a `source` flag (`opencap` or "
        "`aspset`)."
    )
    lines.append("")
    lines.append(
        "  * Shared head outputs **hip_flexion_r, knee_angle_r, "
        "ankle_angle_r**: masked SmoothL1 against the convention-aligned "
        "target. Both sources train these. (ankle_angle_r ASPset target "
        "is NaN by construction → masked.)"
    )
    lines.append(
        "  * Shared head outputs **hip_adduction_r, lumbar_extension**:"
    )
    lines.append("      * OpenCap sample: loss against shared head columns.")
    lines.append(
        "      * ASPset sample: shared head columns NaN-masked → no "
        "gradient on these two outputs."
    )
    lines.append(
        "  * ASPset head outputs (the 2 mismatched metrics only):"
    )
    lines.append("      * OpenCap sample: NaN target → no gradient.")
    lines.append(
        "      * ASPset sample: masked SmoothL1 against the "
        "ASPset-convention target."
    )
    lines.append("")
    lines.append(
        "Net: ASPset's hip_adduction_r/lumbar_extension gradient flows "
        "ONLY into the ASPset head, never into the shared (deployed) "
        "head. The shared head sees ONLY OpenCap supervision for these "
        "two metrics. The backbone still receives a gradient from every "
        "sample (via shared metrics + ASPset head metrics)."
    )
    lines.append("")
    lines.append(f"## Training recipe ({variant})")
    lines.append("")
    lines.append(f"  * **Loss**: {loss}.")
    lines.append(
        "  * **Optimizer**: AdamW lr=1e-3, weight_decay=1e-4, cosine LR."
    )
    lines.append(f"  * **Step**: {batch_recipe}.")
    lines.append("  * **Epochs**: 25.")
    lines.append("  * **CPU only.**")
    lines.append("")
    lines.append("## Cohort (same as HH2/LL)")
    lines.append("")
    lines.append(
        "  * **OpenCap**: 9 subjects, ~270 clips, all 5 angles."
    )
    lines.append(
        "  * **ASPset**: 15 of 17 subjects ingested (2 c3d/parse "
        "failures, same as HH2), 1,409 clips, 4 angles."
    )
    lines.append(
        "  * **Total**: 24 subjects, ~1,679 clips, single-camera DWPose."
    )
    lines.append("")
    lines.append(
        "Convention alignment ASPset → OpenCap (carried from HH2; the "
        "per-source head simply prevents the ASPset-converted value "
        "from polluting the shared head):"
    )
    lines.append("")
    lines.append("  * `hip_flexion_r`: identity")
    lines.append(
        "  * `hip_adduction_r`: identity; routed to ASPset head for "
        "ASPset samples"
    )
    lines.append("  * `knee_angle_r`: `OC = 180 − ASP`")
    lines.append("  * `ankle_angle_r`: NaN (no foot KPs; masked)")
    lines.append(
        "  * `lumbar_extension`: `OC = ASP − 180`; routed to ASPset "
        "head for ASPset samples"
    )
    lines.append("")
    lines.append(f"## All-data L2 checkpoint (used by {deploy_tag} Phase B)")
    lines.append("")
    lines.append(f"  * **Training**: `{variant}_persource_*_alldata_v1`")
    lines.append(f"  * **Checkpoint**: `models/{ckpt}`")
    lines.append(
        "  * **LOSO discipline**: `ALL_DATA_NO_LOSO_AT_L2` — no LOSO "
        f"at L2; this is the cached L2 used by {deploy_tag} Phase B "
        "L3 ridge re-fit."
    )
    lines.append("")
    lines.append("## Phase A 24-fold LOSO eval")
    lines.append("")
    lines.append(
        "**Not run this build cycle.** Same compute trade-off as the "
        "LL build: prioritized the all-data L2 → Phase B → v28 oracle "
        "path to answer Cameron's floor-lift question within the time "
        "budget. The harness supports Phase A LOSO via "
        f"`{next_phase_cmd}`."
    )
    lines.append("")
    lines.append(f"## What {variant} actually delivered (Phase B preview)")
    lines.append("")
    lines.append(
        f"See `data/layer3_retrain_persource_{'perframe' if is_a else 'romaware'}/"
        f"REPORT.md` for the full {deploy_tag} Phase B tier table. The "
        "MM-brief headline question (did the floor lift on the 2 target "
        "supplementary slots?) is answered in "
        "`data/v28_selective_oracle/REPORT.md`."
    )
    lines.append("")
    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "1. **Per-source heads are tested only via the all-data L2 + "
        "L3 ridge re-fit (Phase B).** No 24-fold LOSO Phase A pooled "
        "|r| was computed this build cycle."
    )
    lines.append(
        "2. **The shared head's gradient on hip_adduction_r and "
        "lumbar_extension is now driven by 9 OpenCap subjects only.** "
        "This is by design — clean convention — but it narrows the "
        "training distribution for those two outputs. The other 3 "
        "outputs still see 24 subjects."
    )
    lines.append(
        "3. **ASPset head is dead weight at inference.** It exists "
        "only as a gradient sink during training. Adds 8.3K parameters "
        "to the checkpoint."
    )
    lines.append(
        "4. **No invented numbers.** All metrics shown in the v28 "
        "report were computed on the cached all-data MM L2 checkpoint "
        f"driving the {deploy_tag} Phase B Layer 3 ridge re-fit."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(
        "  * `harness/learned_layer2_combined_persource.py` — MM "
        "trainer (architecture + MM-A + MM-B variants)"
    )
    lines.append(
        f"  * `harness/layer3_retrain_on_persource_{'perframe' if is_a else 'romaware'}_l2.py` — "
        f"{variant} Phase B driver"
    )
    lines.append(f"  * `models/{ckpt}` — all-data {variant} L2 checkpoint")
    lines.append(
        f"  * `data/layer3_retrain_persource_{'perframe' if is_a else 'romaware'}/"
        f"per_slot_validity_{deploy_tag}.json` — per-slot {deploy_tag} "
        "validity stats (LOSO at L3 only)"
    )
    lines.append(
        "  * `data/v28_selective_oracle/REPORT.md` — v28 oracle "
        "narrative (the floor-lift verdict)"
    )

    out_path.write_text("\n".join(lines) + "\n")
    log(f"wrote Phase A {variant} narrative -> {out_path}")


def _write_phaseB_report_mm(
    out_dir: Path, variant: str, per_slot_stats: dict
) -> None:
    """Re-write the auto-generated FF Phase B REPORT.md with MM-specific
    framing. FF wrote a generic REPORT.md keyed to Agent FF when the
    Layer 3 retrain ran; we replace it with the MM build narrative.
    """
    out_path = out_dir / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_a = variant == "mm_a"
    name = "MM-A (per-source heads + per-frame SmoothL1)" \
        if is_a else "MM-B (per-source heads + ROM-aware)"
    deploy_tag = "v26" if is_a else "v27"
    other_tag = "v27" if is_a else "v26"

    tier_counts = per_slot_stats.get("tier_counts", {})
    promotions = per_slot_stats.get("promotions", [])
    demotions = per_slot_stats.get("demotions", [])

    lines: list[str] = []
    lines.append(
        f"# Phase B: Layer 3 retrained on {name} learned Layer 2 "
        f"(Agent MM → {deploy_tag})"
    )
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(
        f"**Build:** Agent MM Phase B — {deploy_tag} candidate "
        f"(sister of {other_tag})"
    )
    lines.append("")
    lines.append("## LOSO discipline")
    lines.append("")
    lines.append(
        "**Layer-3-LOSO-only.** ONE L2 model "
        "(`TemporalKeypointCNNConfPerSource`) was trained on ALL 24 "
        "cohort subjects (9 OpenCap + 15 ASPset, no LOSO at L2) and "
        "used to produce learned angle traces for every clip. Layer 3 "
        "ridge was re-fit per slot with LOSO at L3 only. Same caveat "
        "as v18/v20/v23/v24: tier promotions involving any cohort "
        "subject are upper bounds; the true double-LOSO number could "
        "be ~0.05-0.10 |r| lower per HH2's per-fold variance."
    )
    lines.append("")
    lines.append("## Per-source heads recap")
    lines.append("")
    lines.append(
        "L2 architecture has two output heads:"
    )
    lines.append("")
    lines.append(
        "  * **Shared head (5 outputs)** — hip_flexion_r, "
        "hip_adduction_r, knee_angle_r, ankle_angle_r, "
        "lumbar_extension. **This is the deployed head** — produces "
        "OpenCap-convention output."
    )
    lines.append(
        "  * **ASPset head (2 outputs)** — hip_adduction_r_aspset, "
        "lumbar_extension_aspset. **Discarded at inference.** Absorbs "
        "ASPset's convention-divergent target during training so it "
        "doesn't pollute the shared head's gradient on those two metrics."
    )
    lines.append("")
    lines.append(
        "L3 ridge consumes the shared-head 5-output trace, exactly "
        "like v18/v23/v24. No deploy-side complexity change vs prior "
        "learned-L2 readers."
    )
    lines.append("")
    lines.append("## Tier counts (this reader alone)")
    lines.append("")
    lines.append(
        f"| Tier | {deploy_tag} count |"
    )
    lines.append("| --- | ---: |")
    for tier in ("Excellent", "Good", "Moderate", "Poor"):
        lines.append(
            f"| {tier} | {tier_counts.get(tier, 0)} |"
        )
    lines.append("")
    lines.append(
        "(Tier counts evaluated reader-in-isolation. The v28 oracle "
        "selects the best reader per slot — see "
        "`data/v28_selective_oracle/REPORT.md`.)"
    )
    lines.append("")
    lines.append("## Target slot answers")
    lines.append("")

    target_pairs = (
        ("hip_adduction_r", "front_oblique_left", 0.69, "v20 (LL/v25)"),
        ("lumbar_extension", "front_oblique_left", 0.71, "v18 (LL/v25)"),
        ("lumbar_extension", "side_left", 0.88, "v23 (LL/v25)"),
        ("lumbar_extension", "side_right", 0.85, "v23 (LL/v25)"),
        ("lumbar_extension", "front_oblique_right", 0.79, "v23 (LL/v25)"),
    )
    slot_by_key = {
        (s["target"], s["view"]): s
        for s in per_slot_stats.get("slots", [])
    }
    lines.append(
        f"| Slot | v25 CCC (reader) | {deploy_tag} CCC | "
        f"{deploy_tag} LoA half | Tier | Crossed 0.79? |"
    )
    lines.append("| --- | --- | ---: | ---: | --- | :---: |")
    for tgt, view, prev_ccc, prev_origin in target_pairs:
        s = slot_by_key.get((tgt, view))
        if not s:
            continue
        ccc = s.get("ccc_lin")
        loa = s.get("loa_half_width_deg")
        tier = s.get("classification") or "?"
        crossed = (
            "YES"
            if ccc is not None
            and not (isinstance(ccc, float) and math.isnan(ccc))
            and ccc >= 0.79
            else "no"
        )
        lines.append(
            f"| {tgt} / {view} | {prev_ccc:.2f} ({prev_origin}) | "
            f"{_fmt_num(ccc, 3)} | ±{_fmt_num(loa, 2)}° | {tier} | "
            f"{crossed} |"
        )
    lines.append("")

    if promotions:
        lines.append(f"## Promotions vs v17 baseline ({deploy_tag} alone)")
        lines.append("")
        for p in promotions:
            lines.append(
                f"- {p['target']} / {p['view']}: {p['from']} -> {p['to']}"
            )
        lines.append("")
    if demotions:
        lines.append(f"## Demotions vs v17 baseline ({deploy_tag} alone)")
        lines.append("")
        for d in demotions:
            lines.append(
                f"- {d['target']} / {d['view']}: {d['from']} -> {d['to']}"
            )
        lines.append("")

    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "1. **Layer-3-LOSO-only.** L2 trained on all 24 cohort subjects. "
        "Tier promotions involving any cohort subject are upper bounds."
    )
    lines.append(
        "2. **Shared head sees OpenCap-only supervision for the 2 "
        "per-source metrics.** Gradient narrowing is intentional — the "
        "goal is a clean convention — but it narrows the training "
        "distribution for hip_adduction_r and lumbar_extension."
    )
    lines.append(
        "3. **Final verdict is in v28, not here.** This reader competes "
        "with v17/v18/v20/v23/v24 in the v28 oracle. A reader that "
        "regresses on slots already handled by v17/v23 still costs "
        "nothing at deploy because the oracle keeps the better reader."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(
        f"- `results/deploy_ready_models_{deploy_tag}_"
        f"persource_{'perframe' if is_a else 'romaware'}.json` — "
        f"{deploy_tag} deploy candidate"
    )
    lines.append(
        f"- `data/layer3_retrain_persource_{'perframe' if is_a else 'romaware'}/"
        f"per_slot_validity_{deploy_tag}.json` — per-slot {deploy_tag} "
        "validity stats"
    )
    lines.append(
        "- `data/v28_selective_oracle/REPORT.md` — v28 narrative "
        "(the floor-lift verdict)"
    )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    log(f"wrote Phase B {variant} narrative -> {out_path}")


def main() -> None:
    log("=== build_v28_selective START ===")
    build()
    write_v28_report()
    # Phase A narratives.
    phaseA_a = DATA_ROOT / "learned_layer2_persource_perframe" / "REPORT.md"
    phaseA_b = DATA_ROOT / "learned_layer2_persource_romaware" / "REPORT.md"
    _write_phaseA_report(phaseA_a, "mm_a")
    _write_phaseA_report(phaseA_b, "mm_b")
    # Phase B narratives (replace FF-tagged auto-generated REPORT).
    if V26_PER_SLOT.exists():
        v26 = _load_json(V26_PER_SLOT)
        _write_phaseB_report_mm(
            DATA_ROOT / "layer3_retrain_persource_perframe",
            "mm_a", v26,
        )
    if V27_PER_SLOT.exists():
        v27 = _load_json(V27_PER_SLOT)
        _write_phaseB_report_mm(
            DATA_ROOT / "layer3_retrain_persource_romaware",
            "mm_b", v27,
        )
    log("=== build_v28_selective DONE ===")


if __name__ == "__main__":
    main()
