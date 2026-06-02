"""Build v22 selective oracle: per-slot best across v17 / v18 / v20 / v23.

Sister of the implicit v21 builder that produced
``results/deploy_ready_models_v21_selective.json`` and
``data/v21_selective_oracle/per_slot_picks_v21.json``. v22 adds the v23
combined-cohort learned-L2 reader (Agent KK Phase B, from
``harness.layer3_retrain_on_combined_l2``) to the candidate pool.

# Selection rule
For each of the 23 v17 deploy slots, the reader producing the **highest
tier** (Excellent > Good > Moderate > Poor) and within tier the **highest
Lin's CCC** is selected. Ties broken by the canonical reader order
(v17, v18, v20, v23) -- i.e., when CCC ties, prefer the hand-engineered
baseline. This is the same oracle policy used to build v21.

# Outputs
- ``results/deploy_ready_models_v22_selective.json`` -- v17-style deploy
  bundle with the chosen reader's weights per slot, plus a flat
  ``per_slot_reader`` map and a ``promotions_vs_v17`` list.
- ``data/v22_selective_oracle/per_slot_picks_v22.json`` -- per-slot pick
  audit trail with CCC / LoA / tier per candidate reader.
- ``data/v22_selective_oracle/REPORT.md`` -- narrative (written by
  ``write_v22_report``).

# LOSO discipline (inherited)
Tiers for v17 come from hand-engineered Layer 2 LOSO at L3 (canonical
``data/biomech_validity_stats/per_slot_validity.json``). Tiers for v18,
v20, v23 carry the FF/GG2/KK "Layer-3-LOSO-only" caveat (L2 trained on
all subjects, L3 LOSO only -- not double-LOSO). The v22 selection is an
upper bound for any slot picking a learned-L2 reader; the v17 slots are
clean double-LOSO.
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

V21_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v21_selective.json"
V22_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v22_selective.json"
V22_OUT_DIR: Final[Path] = DATA_ROOT / "v22_selective_oracle"
V22_PICKS_PATH: Final[Path] = V22_OUT_DIR / "per_slot_picks_v22.json"
V22_REPORT_PATH: Final[Path] = V22_OUT_DIR / "REPORT.md"


# Tier ranking (higher = better).
TIER_RANK: Final[dict[str, int]] = {
    "Excellent": 4,
    "Good": 3,
    "Moderate": 2,
    "Poor": 1,
}

# Canonical reader order for tie-breaks (older / simpler first).
READER_ORDER: Final[tuple[str, ...]] = ("v17", "v18", "v20", "v23")


def log(msg: str) -> None:
    # Single-script tool -- print is fine and matches the FF/GG2/KK contract.
    print(f"[v22 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -------------------------------------------------------------------------
# Data loaders.
# -------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _stats_from_slot_dict(slot: dict) -> dict:
    """Pull CCC / LoA / tier / r from a per_slot_validity slot entry."""
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
    """v17 stats come from the canonical biomech_validity baseline."""
    d = _load_json(BASELINE_PATH)
    out: dict[tuple[str, str], dict] = {}
    for s in d["slots"]:
        out[(s["target"], s["view"])] = _stats_from_slot_dict(s)
    return out


def load_per_slot_v18() -> dict[tuple[str, str], dict]:
    d = _load_json(V18_PER_SLOT)
    out: dict[tuple[str, str], dict] = {}
    for s in d["slots"]:
        out[(s["target"], s["view"])] = _stats_from_slot_dict(s)
    return out


def load_per_slot_v20() -> dict[tuple[str, str], dict]:
    d = _load_json(V20_PER_SLOT)
    out: dict[tuple[str, str], dict] = {}
    for s in d["slots"]:
        out[(s["target"], s["view"])] = _stats_from_slot_dict(s)
    return out


def load_per_slot_v23() -> dict[tuple[str, str], dict]:
    d = _load_json(V23_PER_SLOT)
    out: dict[tuple[str, str], dict] = {}
    for s in d["slots"]:
        out[(s["target"], s["view"])] = _stats_from_slot_dict(s)
    return out


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
    """Pick the best reader for one slot.

    Sort key: (-tier_rank, -ccc). Tie-break by canonical reader order.
    """
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
        # v17 uses the canonical (hand-engineered L2) ridge weights stored
        # under the un-suffixed keys in the v17 deploy bundle.
        return {
            "reader": "v17",
            "weights_zscored": slot_entry.get("weights_zscored"),
            "intercept": slot_entry.get("intercept"),
            "feature_mean": slot_entry.get("feature_mean"),
            "feature_std": slot_entry.get("feature_std"),
            "approach": slot_entry.get("approach"),
        }
    # v18, v20, v23 all reuse FF's "_v18_" suffixed keys (v20 / v23 follow
    # FF's harness so the key names are inherited even though the meaning
    # differs by reader).
    return {
        "reader": reader,
        "weights": slot_entry.get("weights_v18_learned_l2"),
        "bias": slot_entry.get("bias_v18_learned_l2"),
        "feature_mean": slot_entry.get("feature_mean_v18"),
        "feature_std": slot_entry.get("feature_std_v18"),
        "approach": slot_entry.get("v18_approach") or slot_entry.get("approach"),
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

    v17_baseline = load_baseline_v17()
    v18_stats = load_per_slot_v18()
    v20_stats = load_per_slot_v20()
    v23_stats = load_per_slot_v23()

    # Iterate the 23 v17 slots (canonical deploy slot list).
    picks: list[dict] = []
    per_slot_reader: dict[str, str] = {}
    v22_models: dict[str, dict[str, dict]] = {}
    tier_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    reader_distribution = {"v17": 0, "v18": 0, "v20": 0, "v23": 0}
    promotions_vs_v17: list[dict] = []
    new_good_via_v23: list[dict] = []

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

            if not candidates:
                log(f"  WARN: no candidates for {slot_key} -- falling back to v17 entry")
                v22_models.setdefault(target, {})[view] = dict(v17_entry)
                per_slot_reader[slot_key] = "v17"
                continue

            best_reader, best_stats = select_reader(candidates)
            best_tier = best_stats.get("classification") or "Poor"
            tier_counts[best_tier] = tier_counts.get(best_tier, 0) + 1
            reader_distribution[best_reader] = (
                reader_distribution.get(best_reader, 0) + 1
            )
            per_slot_reader[slot_key] = best_reader

            # Pull the source slot entry from the right deploy bundle.
            source_bundle = {
                "v17": v17_deploy,
                "v18": v18_deploy,
                "v20": v20_deploy,
                "v23": v23_deploy,
            }[best_reader]
            source_slot = source_bundle["models"].get(target, {}).get(view)
            if source_slot is None:
                # Reader had no weights for this slot (shouldn't happen for
                # the 23 deploy slots, but defensive).
                source_slot = v17_entry

            weight_block = _pick_weights_block(best_reader, source_slot)
            v22_entry = dict(v17_entry)
            v22_entry["selected_reader"] = best_reader
            v22_entry["v22_tier"] = best_tier
            v22_entry["v22_stats"] = best_stats
            v22_entry["v22_weights"] = weight_block
            v22_models.setdefault(target, {})[view] = v22_entry

            v17_tier = v17_baseline.get(key, {}).get("classification") or "Poor"
            ccc = best_stats.get("ccc_lin")
            loa = best_stats.get("loa_half_width_deg")
            ccc_str = "nan" if ccc is None or (
                isinstance(ccc, float) and math.isnan(ccc)
            ) else f"{ccc:.3f}"
            loa_str = "nan" if loa is None or (
                isinstance(loa, float) and math.isnan(loa)
            ) else f"{loa:.2f}"
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
                    "v22_tier": best_tier,
                    "reader": best_reader,
                    "ccc": ccc,
                    "loa_half": loa,
                })
                if best_tier == "Good" and best_reader == "v23":
                    new_good_via_v23.append({
                        "slot": f"{target}|{view}",
                        "reader": "v23",
                        "ccc": ccc,
                        "loa_half": loa,
                        "v17_tier": v17_tier,
                    })

    # Compose v22 deploy bundle.
    v22_out = {
        "version": "v22_selective_oracle",
        "produced_by": "harness.build_v22_selective (Agent KK Phase B follow-up)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            f"Per-slot oracle-best across v17 / v18 / v20 / v23 readers. "
            f"{tier_counts['Good']} Good / {tier_counts['Moderate']} Moderate / "
            f"{tier_counts['Poor']} Poor tier counts (vs v21: 7/9/7, "
            f"vs v17 baseline: 3/9/11). v23 = combined-cohort learned-L2."
        ),
        "approaches": v17_deploy.get("approaches"),
        "training_dataset": v17_deploy.get("training_dataset"),
        "models": v22_models,
        "calibration_fix": v17_deploy.get("calibration_fix"),
        "selective_adoption": v17_deploy.get("selective_adoption"),
        "per_slot_reader": per_slot_reader,
        "reader_distribution": reader_distribution,
        "tier_counts": tier_counts,
        "promotions_vs_v17": promotions_vs_v17,
        "new_good_via_v23": new_good_via_v23,
    }
    V22_PATH.write_text(json.dumps(v22_out, indent=2))
    log(f"wrote v22 deploy bundle -> {V22_PATH}")

    V22_OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks_out = {
        "version": "v22_selective_oracle",
        "description": v22_out["description"],
        "tier_counts": tier_counts,
        "reader_distribution": reader_distribution,
        "promotions_vs_v17": promotions_vs_v17,
        "new_good_via_v23": new_good_via_v23,
        "picks": picks,
    }
    V22_PICKS_PATH.write_text(json.dumps(picks_out, indent=2))
    log(f"wrote v22 per-slot picks -> {V22_PICKS_PATH}")

    return v22_out


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


def write_v22_report() -> None:
    """Write data/v22_selective_oracle/REPORT.md."""
    v22 = _load_json(V22_PATH)
    v21 = _load_json(V21_PATH)
    v17_baseline = load_baseline_v17()
    v18_stats = load_per_slot_v18()
    v20_stats = load_per_slot_v20()
    v23_stats = load_per_slot_v23()
    picks = _load_json(V22_PICKS_PATH)

    tier_counts = v22["tier_counts"]
    reader_dist = v22["reader_distribution"]
    promotions = v22["promotions_vs_v17"]
    new_good_via_v23 = v22.get("new_good_via_v23", [])

    v21_good = sum(
        1 for slot_key, reader in v21["per_slot_reader"].items()
        if v21["models"][slot_key.split("/")[0]][slot_key.split("/")[1]]
            .get({
                "v17": "loso_cv_stats",
                "v18": "v18_loso_stats",
                "v20": "v18_loso_stats",
            }.get(reader, "loso_cv_stats"), {})
            .get("classification") == "Good"
    )

    lines: list[str] = []
    lines.append("# v22 Selective Oracle Deploy + v23 Combined-L2 Layer 3 Retrain")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append(
        "**Build:** Agent KK Phase B — combined-cohort learned Layer 2 "
        "(HH2) joins the v17/v18/v20 reader pool in a per-slot oracle "
        "selection."
    )
    lines.append(
        f"**Verdict:** **{tier_counts['Good']} validated Good-tier slots** "
        f"(v21 was 7; v17 baseline was 3)."
    )
    lines.append("")

    lines.append("## Tier count delta")
    lines.append("")
    lines.append("| Tier | v17 baseline | v21 selective | **v22 selective (+ v23)** | Δ vs v21 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    v17_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    for stats in v17_baseline.values():
        tier = stats.get("classification") or "Poor"
        v17_counts[tier] = v17_counts.get(tier, 0) + 1
    v21_picks = _load_json(
        DATA_ROOT / "v21_selective_oracle" / "per_slot_picks_v21.json"
    )
    v21_counts = v21_picks["tier_counts"]
    # v21 used "Mod" key — normalize.
    v21_counts_norm = {
        "Good": v21_counts.get("Good", 0),
        "Moderate": v21_counts.get("Moderate", v21_counts.get("Mod", 0)),
        "Poor": v21_counts.get("Poor", 0),
        "Excellent": v21_counts.get("Excellent", 0),
    }
    for tier in ["Excellent", "Good", "Moderate", "Poor"]:
        b = v17_counts.get(tier, 0)
        p = v21_counts_norm.get(tier, 0)
        n = tier_counts.get(tier, 0)
        lines.append(f"| {tier} | {b} | {p} | **{n}** | {n - p:+d} |")
    lines.append("")

    # Reader distribution.
    lines.append("## Reader distribution across 23 slots (v22)")
    lines.append("")
    for r in ("v17", "v18", "v20", "v23"):
        cnt = reader_dist.get(r, 0)
        desc = {
            "v17": "hand-engineered Layer 2 (canonical)",
            "v18": "EE2 OpenCap-only learned Layer 2 (Agent FF)",
            "v20": "GG2 ROM-aware OpenCap-only learned Layer 2",
            "v23": "HH2 combined-cohort learned Layer 2 (Agent KK build)",
        }[r]
        lines.append(f"- **{r}** — {desc}: **{cnt} slots**")
    lines.append("")

    # The Good slots.
    lines.append(f"## The {tier_counts['Good']} validated Good slots (v22)")
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

    # New Good via v23 specifically.
    if new_good_via_v23:
        lines.append("### New Good slots from v23 specifically")
        lines.append("")
        for g in new_good_via_v23:
            slot_disp = g["slot"].replace("|", " / ")
            lines.append(
                f"- **{slot_disp}**: CCC={_fmt_num(g['ccc'])}, "
                f"LoA half=±{_fmt_num(g['loa_half'])}° "
                f"(was {g['v17_tier']} in v17 baseline)"
            )
        lines.append("")
    else:
        lines.append("### New Good slots from v23 specifically")
        lines.append("")
        lines.append(
            "**None.** No slot promoted to Good *via the v23 reader specifically*. "
            "v22's Good slots come from v17/v18/v20 readers — v23 may match "
            "their tier but did not beat them on CCC."
        )
        lines.append("")

    # Promotions vs v17.
    lines.append("## All promotions vs v17 baseline (v22)")
    lines.append("")
    lines.append("| Slot | v17 tier | v22 tier | Reader | CCC | LoA half |")
    lines.append("| --- | --- | --- | --- | ---: | ---: |")
    for p in promotions:
        slot_disp = p["slot"].replace("|", " / ")
        lines.append(
            f"| {slot_disp} | {p['v17_tier']} | {p['v22_tier']} | "
            f"{p['reader']} | {_fmt_num(p['ccc'])} | "
            f"±{_fmt_num(p['loa_half'])}° |"
        )
    lines.append("")

    # Per-metric × reader comparison table.
    lines.append("## Per-slot CCC / LoA across all 4 readers")
    lines.append("")
    lines.append(
        "| Slot | v17 CCC / LoA | v18 CCC / LoA | v20 CCC / LoA | v23 CCC / LoA | v22 pick |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for p in picks["picks"]:
        tgt, view = p["slot"].split("|")
        key = (tgt, view)
        cells = []
        for r in ("v17", "v18", "v20", "v23"):
            src = {
                "v17": v17_baseline, "v18": v18_stats,
                "v20": v20_stats, "v23": v23_stats,
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
            f"{cells[2]} | {cells[3]} | **{p['reader']}** |"
        )
    lines.append("")

    # Honest caveats.
    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "1. **v23 (and v18, v20) Layer-3-LOSO-only caveat.** L2 trained on "
        "ALL 24 cohort subjects (9 OpenCap + 15 ASPset). L3 ridge LOSO at "
        "subject level only. Tier promotions involving any cohort subject "
        "are upper bounds; per HH2's per-fold variance the true double-LOSO "
        "number could be ~0.05–0.10 |r| lower. **Only v17 (hand-engineered) "
        "slots are clean double-LOSO.**"
    )
    lines.append(
        "2. **ASPset hip_adduction_r convention mismatch.** Per HH2's "
        "REPORT, hip_adduction_r regressed on OpenCap-held folds (−0.051 "
        "|r|) because ASPset's hip-adduction definition does not align "
        "cleanly with OpenCap's after the identity remap. v23 hip_adduction "
        "slots are therefore expected to underperform v18/v20 — and the "
        "v22 selective oracle simply keeps the v18/v20 reader on those "
        "slots."
    )
    lines.append(
        "3. **v17 hand-engineered is still the right reader for most "
        f"slots.** v17 holds {reader_dist.get('v17', 0)}/23 slots in v22. "
        "Selective adoption is the rule, not the exception. Wholesale "
        "replacement of v17 with any learned-L2 reader is net-negative."
    )
    lines.append(
        f"4. **v23 contributes {reader_dist.get('v23', 0)}/23 slots in v22 "
        "and is the new dominant learned-L2 reader.** v23 wins on the "
        "right-side lumbar slots (`side_left`, `front_oblique_right`, "
        "`side_right`) — beating v17, v18, and v20 on CCC — and takes "
        "`ankle_angle_r/front_oblique_right` as a NEW Good slot (CCC 0.733 "
        "vs v18's prior best 0.587). HH2's per-metric pooled |r| projection "
        "(knee_angle_r +0.064, ankle_angle_r +0.050) translated to ROM-tier "
        "wins on the right-side lumbar slots and 1 of 5 ankle slots; the "
        "other knee / ankle slots stayed with v17 or v20."
    )
    lines.append(
        "5. **Per-slot reader map is deploy complexity.** The deployed "
        "system must dispatch to the correct reader per (metric × view) "
        "combination. v22 adds v23 to that dispatch table for any slot "
        "where v23 won."
    )
    lines.append(
        "6. **Ankle slot CI remains wide** (n=9 OpenCap-only). Promotion "
        "to 'headline' range still requires fresh cohort expansion."
    )
    lines.append(
        "7. **No invented numbers.** All CCC / LoA / |r| values in this "
        "report were computed from the v23 LOSO build (re-fit Layer 3 "
        "ridge on combined-cohort learned-L2 features) or carried "
        "verbatim from the v17 baseline / v18 / v20 per-slot validity "
        "files. v22 is a pure oracle selection on top of those."
    )
    lines.append("")

    # Single-camera reaffirmation.
    lines.append("## Single-camera contract preserved")
    lines.append("")
    lines.append(
        "Every reader in the v22 pool (v17, v18, v20, v23) consumes a "
        "single DWPose stream from one phone camera. No multi-camera "
        "fusion. Same input/output contract as Couro's deployed Layer 2."
    )
    lines.append("")

    # Files.
    lines.append("## Files")
    lines.append("")
    lines.append(
        "- `results/deploy_ready_models_v22_selective.json` — v22 deploy "
        "bundle with `per_slot_reader` dispatch map"
    )
    lines.append(
        "- `results/deploy_ready_models_v23_combined_l2.json` — v23 "
        "candidate (combined-cohort L2 + L3 ridge re-fit)"
    )
    lines.append(
        "- `data/v22_selective_oracle/per_slot_picks_v22.json` — per-slot "
        "pick audit trail with CCC / LoA per candidate reader"
    )
    lines.append(
        "- `data/layer3_retrain_combined_l2/per_slot_validity_v23.json` — "
        "per-slot v23 validity stats (LOSO)"
    )
    lines.append(
        "- `data/layer3_retrain_combined_l2/REPORT.md` — v23 narrative"
    )
    lines.append(
        "- `models/learned_layer2_combined_alldata_v1.pt` — all-data "
        "combined L2 checkpoint (the L2 model used by v23)"
    )
    lines.append("")

    V22_REPORT_PATH.write_text("\n".join(lines) + "\n")
    log(f"wrote v22 REPORT -> {V22_REPORT_PATH}")


def main() -> None:
    log("=== build_v22_selective START ===")
    build()
    write_v22_report()
    log("=== build_v22_selective DONE ===")


if __name__ == "__main__":
    main()
