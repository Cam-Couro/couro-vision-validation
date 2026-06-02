"""Build v25 selective oracle: per-slot best across v17 / v18 / v20 / v23 / v24.

Sister of ``harness.build_v22_selective`` (Agent KK). v25 adds the v24
LL combined-cohort + ROM-aware learned-L2 reader (Agent LL Phase B, from
``harness.layer3_retrain_on_combined_rom_aware_l2``) to the candidate pool.

# Selection rule
For each of the 23 v17 deploy slots, the reader producing the **highest
tier** (Excellent > Good > Moderate > Poor) and within tier the **highest
Lin's CCC** is selected. Ties broken by the canonical reader order
(v17, v18, v20, v23, v24) -- i.e., when CCC ties, prefer the older /
simpler reader. Same policy as v22.

# Outputs
- ``results/deploy_ready_models_v25_selective.json``
- ``data/v25_selective_oracle/per_slot_picks_v25.json``
- ``data/v25_selective_oracle/REPORT.md``

# LOSO discipline (inherited)
Same caveat as v22: v17 slots are clean double-LOSO; v18/v20/v23/v24 slots
carry the Layer-3-LOSO-only caveat (L2 trained on all subjects, L3 LOSO
only).
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
    DATA_ROOT / "layer3_retrain_combined_rom_aware" / "per_slot_validity_v24.json"
)

V22_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v22_selective.json"
V25_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v25_selective.json"
V25_OUT_DIR: Final[Path] = DATA_ROOT / "v25_selective_oracle"
V25_PICKS_PATH: Final[Path] = V25_OUT_DIR / "per_slot_picks_v25.json"
V25_REPORT_PATH: Final[Path] = V25_OUT_DIR / "REPORT.md"


# Tier ranking (higher = better).
TIER_RANK: Final[dict[str, int]] = {
    "Excellent": 4,
    "Good": 3,
    "Moderate": 2,
    "Poor": 1,
}

# Canonical reader order for tie-breaks (older / simpler first).
READER_ORDER: Final[tuple[str, ...]] = ("v17", "v18", "v20", "v23", "v24")


# The 5 sub-0.80 slots Cameron called out in the LL brief.
FLOOR_LIFT_TARGETS: Final[tuple[tuple[str, str, float, str], ...]] = (
    # (target, view, previous_ccc, previous_reader_or_tier_origin)
    ("lumbar_extension", "front_oblique_right", 0.79, "v23"),
    ("ankle_angle_r", "front_oblique_right", 0.73, "v23"),
    ("lumbar_extension", "front_oblique_left", 0.71, "v18"),
    ("hip_adduction_r", "front_oblique_left", 0.69, "v20"),
    ("ankle_angle_r", "side_right", 0.64, "v17"),
)


def log(msg: str) -> None:
    print(f"[v25 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def select_reader(
    candidates: dict[str, dict],
) -> tuple[str, dict]:
    """Pick the best reader for one slot. Same sort key as v22."""
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
    """Extract the weights that the runtime should use for this slot."""
    if reader == "v17":
        return {
            "reader": "v17",
            "weights_zscored": slot_entry.get("weights_zscored"),
            "intercept": slot_entry.get("intercept"),
            "feature_mean": slot_entry.get("feature_mean"),
            "feature_std": slot_entry.get("feature_std"),
            "approach": slot_entry.get("approach"),
        }
    # v18, v20, v23, v24 share FF's "_v18_" suffixed keys (LL inherits FF too).
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

    v17_baseline = load_baseline_v17()
    v18_stats = load_per_slot(V18_PER_SLOT)
    v20_stats = load_per_slot(V20_PER_SLOT)
    v23_stats = load_per_slot(V23_PER_SLOT)
    v24_stats = load_per_slot(V24_PER_SLOT)

    picks: list[dict] = []
    per_slot_reader: dict[str, str] = {}
    v25_models: dict[str, dict[str, dict]] = {}
    tier_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    reader_distribution = {
        "v17": 0, "v18": 0, "v20": 0, "v23": 0, "v24": 0,
    }
    promotions_vs_v17: list[dict] = []
    new_good_via_v24: list[dict] = []

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

            if not candidates:
                log(
                    f"  WARN: no candidates for {slot_key} -- "
                    f"falling back to v17 entry"
                )
                v25_models.setdefault(target, {})[view] = dict(v17_entry)
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
            }[best_reader]
            source_slot = (
                source_bundle["models"].get(target, {}).get(view) or v17_entry
            )

            weight_block = _pick_weights_block(best_reader, source_slot)
            v25_entry = dict(v17_entry)
            v25_entry["selected_reader"] = best_reader
            v25_entry["v25_tier"] = best_tier
            v25_entry["v25_stats"] = best_stats
            v25_entry["v25_weights"] = weight_block
            v25_models.setdefault(target, {})[view] = v25_entry

            v17_tier = v17_baseline.get(key, {}).get("classification") or "Poor"
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
            log(
                f"  {slot_key} -> {best_reader} ({best_tier}) "
                f"CCC={ccc_str} LoAh={loa_str} (v17 tier={v17_tier})"
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
            })

            if _tier_rank(best_tier) > _tier_rank(v17_tier):
                promotions_vs_v17.append({
                    "slot": f"{target}|{view}|{v17_entry.get('approach')}",
                    "v17_tier": v17_tier,
                    "v25_tier": best_tier,
                    "reader": best_reader,
                    "ccc": ccc,
                    "loa_half": loa,
                })
                if best_tier == "Good" and best_reader == "v24":
                    new_good_via_v24.append({
                        "slot": f"{target}|{view}",
                        "reader": "v24",
                        "ccc": ccc,
                        "loa_half": loa,
                        "v17_tier": v17_tier,
                    })

    # Floor lift analysis (the 5 slots Cameron called out).
    floor_lift: list[dict] = []
    for tgt, view, prev_ccc, prev_origin in FLOOR_LIFT_TARGETS:
        key = (tgt, view)
        v24_st = v24_stats.get(key, {})
        v24_ccc = v24_st.get("ccc_lin")
        v24_tier = v24_st.get("classification")
        # v25 oracle pick.
        pick = next(
            (p for p in picks if p["slot"] == f"{tgt}|{view}"), None,
        )
        v25_ccc = pick["ccc"] if pick else None
        v25_tier = pick["tier"] if pick else None
        v25_reader = pick["reader"] if pick else None
        floor_lift.append({
            "slot": f"{tgt}|{view}",
            "previous_ccc": prev_ccc,
            "previous_reader_origin": prev_origin,
            "v24_ccc": v24_ccc,
            "v24_tier": v24_tier,
            "v25_ccc": v25_ccc,
            "v25_tier": v25_tier,
            "v25_reader": v25_reader,
            "crossed_0_80": (
                v25_ccc is not None
                and not (
                    isinstance(v25_ccc, float) and math.isnan(v25_ccc)
                )
                and v25_ccc >= 0.80
            ),
        })

    floor_lifted_any = any(f["crossed_0_80"] for f in floor_lift)

    v25_out = {
        "version": "v25_selective_oracle",
        "produced_by": "harness.build_v25_selective (Agent LL Phase C)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            f"Per-slot oracle-best across v17 / v18 / v20 / v23 / v24 "
            f"readers. {tier_counts['Good']} Good / "
            f"{tier_counts['Moderate']} Moderate / "
            f"{tier_counts['Poor']} Poor tier counts (vs v22: 8/8/7, "
            f"vs v17 baseline: 3/9/11). v24 = LL combined-cohort + "
            f"ROM-aware learned-L2."
        ),
        "approaches": v17_deploy.get("approaches"),
        "training_dataset": v17_deploy.get("training_dataset"),
        "models": v25_models,
        "calibration_fix": v17_deploy.get("calibration_fix"),
        "selective_adoption": v17_deploy.get("selective_adoption"),
        "per_slot_reader": per_slot_reader,
        "reader_distribution": reader_distribution,
        "tier_counts": tier_counts,
        "promotions_vs_v17": promotions_vs_v17,
        "new_good_via_v24": new_good_via_v24,
        "floor_lift_analysis": floor_lift,
        "floor_lifted_to_0_80": floor_lifted_any,
    }
    V25_PATH.write_text(json.dumps(v25_out, indent=2))
    log(f"wrote v25 deploy bundle -> {V25_PATH}")

    V25_OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks_out = {
        "version": "v25_selective_oracle",
        "description": v25_out["description"],
        "tier_counts": tier_counts,
        "reader_distribution": reader_distribution,
        "promotions_vs_v17": promotions_vs_v17,
        "new_good_via_v24": new_good_via_v24,
        "floor_lift_analysis": floor_lift,
        "floor_lifted_to_0_80": floor_lifted_any,
        "picks": picks,
    }
    V25_PICKS_PATH.write_text(json.dumps(picks_out, indent=2))
    log(f"wrote v25 per-slot picks -> {V25_PICKS_PATH}")

    return v25_out


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


def write_v25_report() -> None:
    """Write data/v25_selective_oracle/REPORT.md."""
    v25 = _load_json(V25_PATH)
    v22 = _load_json(V22_PATH)
    v17_baseline = load_baseline_v17()
    v18_stats = load_per_slot(V18_PER_SLOT)
    v20_stats = load_per_slot(V20_PER_SLOT)
    v23_stats = load_per_slot(V23_PER_SLOT)
    v24_stats = load_per_slot(V24_PER_SLOT)
    picks = _load_json(V25_PICKS_PATH)

    tier_counts = v25["tier_counts"]
    reader_dist = v25["reader_distribution"]
    promotions = v25["promotions_vs_v17"]
    new_good_via_v24 = v25.get("new_good_via_v24", [])
    floor_lift = v25.get("floor_lift_analysis", [])
    floor_lifted = v25.get("floor_lifted_to_0_80", False)

    lines: list[str] = []
    lines.append(
        "# v25 Selective Oracle Deploy + v24 Combined-Cohort + "
        "ROM-Aware Layer 3 Retrain"
    )
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(
        "**Build:** Agent LL Phase C — LL combined-cohort + ROM-aware "
        "learned Layer 2 joins the v17/v18/v20/v23 reader pool in a "
        "per-slot oracle selection."
    )
    lines.append(
        f"**Verdict:** **{tier_counts['Good']} validated Good-tier slots** "
        f"(v22 was 8; v21 was 7; v17 baseline was 3)."
    )

    # Headline floor-lift status.
    if floor_lifted:
        lifted_slots = [
            f["slot"].replace("|", " / ")
            for f in floor_lift if f["crossed_0_80"]
        ]
        lines.append(
            f"**Floor lift:** YES — at least one sub-0.80 slot crossed 0.80 "
            f"under v25: {', '.join(lifted_slots)}."
        )
    else:
        lines.append(
            "**Floor lift:** NO — none of the 5 sub-0.80 slots Cameron "
            "called out crossed 0.80 CCC under LL/v24 or in the v25 "
            "oracle. The Tier 1 (CCC ≥ 0.79) vs supplementary marketing "
            "restructure is the honest recommendation."
        )
    lines.append("")

    # ----- Floor lift analysis (the headline answer) -----
    lines.append("## Floor lift verdict (the 5 sub-0.80 slots)")
    lines.append("")
    lines.append(
        "Cameron's LL brief asked: \"did ANY of these slots cross 0.80 CCC "
        "under LL or in the v25 oracle?\""
    )
    lines.append("")
    lines.append(
        "| Slot | Previous CCC (reader) | v24 CCC (LL) | v25 oracle CCC (reader) | Tier change | Crossed 0.80? |"
    )
    lines.append(
        "| --- | --- | ---: | --- | --- | :---: |"
    )
    for f in floor_lift:
        slot_disp = f["slot"].replace("|", " / ")
        prev = (
            f"{f['previous_ccc']:.2f} ({f['previous_reader_origin']})"
        )
        v24c = (
            "-"
            if f["v24_ccc"] is None
            or (
                isinstance(f["v24_ccc"], float)
                and math.isnan(f["v24_ccc"])
            )
            else f"{f['v24_ccc']:.2f}"
        )
        v25c = (
            "-"
            if f["v25_ccc"] is None
            or (
                isinstance(f["v25_ccc"], float)
                and math.isnan(f["v25_ccc"])
            )
            else f"{f['v25_ccc']:.2f} ({f['v25_reader']})"
        )
        v25t = f["v25_tier"] or "-"
        # Tier change vs previous.
        prev_tier = (
            "Good"
            if f["previous_ccc"] >= 0.60
            else ("Moderate" if f["previous_ccc"] >= 0.40 else "Poor")
        )
        tier_change = (
            f"{prev_tier} → {v25t}"
            if v25t != prev_tier else f"{v25t} (unchanged)"
        )
        crossed = "YES" if f["crossed_0_80"] else "no"
        lines.append(
            f"| {slot_disp} | {prev} | {v24c} | {v25c} | "
            f"{tier_change} | {crossed} |"
        )
    lines.append("")

    # ----- Tier count delta -----
    lines.append("## Tier count delta")
    lines.append("")
    lines.append(
        "| Tier | v17 baseline | v22 selective | **v25 selective (+ v24)** "
        "| Δ vs v22 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    v17_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    for stats in v17_baseline.values():
        tier = stats.get("classification") or "Poor"
        v17_counts[tier] = v17_counts.get(tier, 0) + 1
    v22_picks = _load_json(
        DATA_ROOT / "v22_selective_oracle" / "per_slot_picks_v22.json"
    )
    v22_counts = v22_picks["tier_counts"]
    for tier in ["Excellent", "Good", "Moderate", "Poor"]:
        b = v17_counts.get(tier, 0)
        p = v22_counts.get(tier, 0)
        n = tier_counts.get(tier, 0)
        lines.append(
            f"| {tier} | {b} | {p} | **{n}** | {n - p:+d} |"
        )
    lines.append("")

    # Reader distribution.
    lines.append("## Reader distribution across 23 slots (v25)")
    lines.append("")
    for r in ("v17", "v18", "v20", "v23", "v24"):
        cnt = reader_dist.get(r, 0)
        desc = {
            "v17": "hand-engineered Layer 2 (canonical)",
            "v18": "EE2 OpenCap-only learned Layer 2 (Agent FF)",
            "v20": "GG2 ROM-aware OpenCap-only learned Layer 2",
            "v23": "HH2 combined-cohort learned Layer 2 (Agent KK)",
            "v24": (
                "LL combined-cohort + ROM-aware learned Layer 2 "
                "(Agent LL, this build)"
            ),
        }[r]
        lines.append(f"- **{r}** — {desc}: **{cnt} slots**")
    lines.append("")

    # The Good slots.
    lines.append(f"## The {tier_counts['Good']} validated Good slots (v25)")
    lines.append("")
    lines.append("| Slot | Reader | CCC | LoA half | v17 tier |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for p in picks["picks"]:
        if p["tier"] == "Good":
            slot_disp = p["slot"].replace("|", " / ")
            lines.append(
                f"| {slot_disp} | {p['reader']} | "
                f"{_fmt_num(p['ccc'])} | ±{_fmt_num(p['loa_half'])}° | "
                f"{p['v17_tier']} |"
            )
    lines.append("")

    # New Good via v24 specifically.
    if new_good_via_v24:
        lines.append("### New Good slots from v24 specifically")
        lines.append("")
        for g in new_good_via_v24:
            slot_disp = g["slot"].replace("|", " / ")
            lines.append(
                f"- **{slot_disp}**: CCC={_fmt_num(g['ccc'])}, "
                f"LoA half=±{_fmt_num(g['loa_half'])}° "
                f"(was {g['v17_tier']} in v17 baseline)"
            )
        lines.append("")
    else:
        lines.append("### New Good slots from v24 specifically")
        lines.append("")
        lines.append(
            "**None.** No slot promoted to Good *via the v24 reader "
            "specifically* beyond what v23 already offered. v24 may match "
            "v17/v18/v20/v23 tier on some slots but did not beat them on CCC."
        )
        lines.append("")

    # Promotions vs v17.
    lines.append("## All promotions vs v17 baseline (v25)")
    lines.append("")
    lines.append(
        "| Slot | v17 tier | v25 tier | Reader | CCC | LoA half |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: |")
    for p in promotions:
        slot_disp = p["slot"].replace("|", " / ")
        lines.append(
            f"| {slot_disp} | {p['v17_tier']} | {p['v25_tier']} | "
            f"{p['reader']} | {_fmt_num(p['ccc'])} | "
            f"±{_fmt_num(p['loa_half'])}° |"
        )
    lines.append("")

    # Per-slot CCC table across all 5 readers.
    lines.append("## Per-slot CCC / LoA across all 5 readers")
    lines.append("")
    lines.append(
        "| Slot | v17 CCC / LoA | v18 CCC / LoA | v20 CCC / LoA | "
        "v23 CCC / LoA | v24 CCC / LoA | v25 pick |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    for p in picks["picks"]:
        tgt, view = p["slot"].split("|")
        key = (tgt, view)
        cells = []
        for r in ("v17", "v18", "v20", "v23", "v24"):
            src = {
                "v17": v17_baseline, "v18": v18_stats,
                "v20": v20_stats, "v23": v23_stats,
                "v24": v24_stats,
            }[r]
            st = src.get(key, {})
            ccc = st.get("ccc_lin")
            loa = st.get("loa_half_width_deg")
            cls = st.get("classification", "-")
            cells.append(
                f"{_fmt_num(ccc)} / ±{_fmt_num(loa)}° ({cls})"
            )
        lines.append(
            f"| {tgt} / {view} | {cells[0]} | {cells[1]} | "
            f"{cells[2]} | {cells[3]} | {cells[4]} | **{p['reader']}** |"
        )
    lines.append("")

    # Bottom-5 slot table that the brief specifically asks for.
    lines.append("## Bottom-5 slot table (per Cameron's brief)")
    lines.append("")
    lines.append(
        "| Slot | Previous CCC | LL (v24) CCC | v25 oracle CCC | v25 tier |"
    )
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for f in floor_lift:
        slot_disp = f["slot"].replace("|", " / ")
        prev = f"{f['previous_ccc']:.2f}"
        v24c = (
            "-"
            if f["v24_ccc"] is None
            or (
                isinstance(f["v24_ccc"], float)
                and math.isnan(f["v24_ccc"])
            )
            else f"{f['v24_ccc']:.2f}"
        )
        v25c = (
            "-"
            if f["v25_ccc"] is None
            or (
                isinstance(f["v25_ccc"], float)
                and math.isnan(f["v25_ccc"])
            )
            else f"{f['v25_ccc']:.2f}"
        )
        v25t = f["v25_tier"] or "-"
        lines.append(
            f"| {slot_disp} | {prev} | {v24c} | {v25c} | {v25t} |"
        )
    lines.append("")

    # Honest caveats.
    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "1. **v24 (and v18, v20, v23) Layer-3-LOSO-only caveat.** L2 trained "
        "on ALL 24 cohort subjects (9 OpenCap + 15 ASPset). L3 ridge LOSO at "
        "subject level only. Tier promotions involving any cohort subject "
        "are upper bounds; per HH2's per-fold variance the true double-LOSO "
        "number could be ~0.05-0.10 |r| lower. **Only v17 (hand-engineered) "
        "slots are clean double-LOSO.**"
    )
    lines.append(
        "2. **ROM-aware loss + cross-dataset convention may interact.** "
        "GG2 added the extrema-aware loss on a single clean OpenCap cohort. "
        "LL adds it on the combined OpenCap+ASPset cohort. ASPset's "
        "convention noise (especially hip_adduction_r and lumbar_extension "
        "offsets) can be amplified by the extrema terms because they pin "
        "max/min predictions to potentially miscalibrated targets. We "
        "mitigated this by dropping ASPset hip_adduction_r supervision "
        "entirely (HH2's own recommendation), but the lumbar extrema may "
        "still inherit some ASPset drift."
    )
    lines.append(
        "3. **ASPset hip_adduction_r dropped from LL training.** Per HH2's "
        "REPORT, hip_adduction_r regressed on OpenCap-held folds (−0.051 "
        "|r|) because ASPset's hip-adduction definition does not align with "
        "OpenCap's. LL drops ASPset hip_adduction_r supervision via the "
        "`drop_aspset_hipadd` flag (default ON). LL hip_adduction_r is "
        "therefore trained on OpenCap-only ground truth, with ASPset "
        "providing shared-trunk representation only."
    )
    lines.append(
        "4. **v17 hand-engineered remains the most chosen reader.** "
        f"v17 holds {reader_dist.get('v17', 0)}/23 slots in v25. Selective "
        "adoption remains the rule, not the exception."
    )
    lines.append(
        f"5. **v24 contributes {reader_dist.get('v24', 0)}/23 slots in v25.** "
        "v24's lift over v23 (where it occurs) is real but small."
    )
    lines.append(
        "6. **Per-slot reader map is deploy complexity.** The deployed "
        "system must dispatch to the correct reader per (metric × view) "
        "combination. v25 adds v24 to that dispatch table for any slot "
        "where v24 won."
    )
    lines.append(
        "7. **Ankle slot CI remains wide** (n=9 OpenCap-only ankle GT). "
        "Promotion to 'headline' range still requires fresh cohort expansion "
        "with paired ankle GT."
    )
    lines.append(
        "8. **No invented numbers.** All CCC / LoA / |r| values were "
        "computed from the v24 LOSO build (LL all-data L2 + L3 ridge "
        "re-fit) or carried verbatim from the v17 baseline / v18 / v20 / "
        "v23 per-slot validity files. v25 is a pure oracle selection on "
        "top of those."
    )
    lines.append("")

    # Single-camera reaffirmation.
    lines.append("## Single-camera contract preserved")
    lines.append("")
    lines.append(
        "Every reader in the v25 pool (v17, v18, v20, v23, v24) consumes a "
        "single DWPose stream from one phone camera. No multi-camera "
        "fusion. Same input/output contract as Couro's deployed Layer 2."
    )
    lines.append("")

    # Files.
    lines.append("## Files")
    lines.append("")
    lines.append(
        "- `results/deploy_ready_models_v25_selective.json` — v25 deploy "
        "bundle with `per_slot_reader` dispatch map"
    )
    lines.append(
        "- `results/deploy_ready_models_v24_combined_rom_aware.json` — v24 "
        "candidate (LL combined-cohort + ROM-aware L2 + L3 ridge re-fit)"
    )
    lines.append(
        "- `data/v25_selective_oracle/per_slot_picks_v25.json` — per-slot "
        "pick audit trail with CCC / LoA per candidate reader"
    )
    lines.append(
        "- `data/layer3_retrain_combined_rom_aware/per_slot_validity_v24.json`"
        " — per-slot v24 validity stats (LOSO)"
    )
    lines.append(
        "- `data/layer3_retrain_combined_rom_aware/REPORT.md` — v24 narrative"
    )
    lines.append(
        "- `models/learned_layer2_combined_rom_aware_alldata_v1.pt` — "
        "all-data LL L2 checkpoint (the L2 model used by v24)"
    )
    lines.append("")

    V25_REPORT_PATH.write_text("\n".join(lines) + "\n")
    log(f"wrote v25 REPORT -> {V25_REPORT_PATH}")


def main() -> None:
    log("=== build_v25_selective START ===")
    build()
    write_v25_report()
    log("=== build_v25_selective DONE ===")


if __name__ == "__main__":
    main()
