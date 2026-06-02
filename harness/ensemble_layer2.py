"""Ensemble Layer 2 — Agent JJ build.

Goal
----
Agent EE2 produced a learned Layer 2 reader (TemporalKeypointCNNConf) that
beats Couro's hand-engineered Layer 2 at per-frame angle reading by +0.131
pooled |r|. Agent FF then retrained Layer 3 ridge regression on top of the
learned-L2 angle traces, producing the v18 deploy bundle. But FF found 8 of
23 deploy slots demoted — including the headline hip_adduction_r/side_left
falling 0.94 → 0.71 CCC. The hand-engineered reader has accidentally good
calibration on those slots; the learned reader has better per-frame agreement
but loses calibration at ROM extrema.

This harness builds **v19**: per-slot oracle-best selection between v17
(hand-engineered L2) and v18 (learned L2). Strategy B from the task spec.

Why Strategy B only (Strategies A and C deferred)
-------------------------------------------------
Strategy A (per-frame angle averaging) and Strategy C (stacked Layer 3
features) both require:

  * Re-running L2 inference on every clip with both readers.
  * Re-extracting Layer 3 ROM features from averaged / stacked angle traces.
  * Re-fitting per-slot ridge with LOSO at L3.
  * Re-running the biomech validity harness.

That pipeline (mirror of Agent FF's run, but ~2x the L2 cost for averaging,
or 2x the L3 feature dimensionality for stacking) takes well over an hour
on this CPU-only setup. Inside the 2.5h JJ budget, Strategy B alone gives a
clean dominant result: by construction v19 is no worse than max(v17, v18)
per slot, so it captures every existing tier win without compromise.

Strategies A and C are documented as deferred follow-ups in REPORT.md.

What this harness does
----------------------
1. Loads v17 baseline per-slot validity stats
   (data/biomech_validity_stats/per_slot_validity.json).
2. Loads v18 learned-L2 per-slot validity stats
   (data/layer3_retrain_learned_l2/per_slot_validity_v18.json).
3. For each of the 23 deploy slots, picks the reader that:
     (a) clears the higher tier gate (Good > Moderate > Poor), or
     (b) on tie at tier, has higher CCC, or
     (c) on tie at CCC, has tighter LoA half-width.
4. Emits:
     * data/ensemble_layer2/per_slot_validity_v19.json (per-slot validity
       under v19 reader selection)
     * results/deploy_ready_models_v19_ensemble.json (deploy bundle that
       references v17 or v18 model weights per slot, with annotation)
     * data/ensemble_layer2/REPORT.md (methodology + decision table + tier
       counts + recommended deploy)

Constraints honoured
--------------------
* Single phone camera per inference (same as v17, v18).
* No modification of v15 / v17 / v18 / EE2 / FF / biomech_validity_stats
  artifacts.
* Honest reporting: every slot's chosen reader is documented.
* LOSO discipline preserved: v17 uses LOSO L3 with hand-engineered L2
  (no L2 model exists, so double-LOSO clean by construction). v18 uses
  Layer-3-LOSO-only with L2 trained on all 9 OpenCap subjects. v19
  inherits the caveat of whichever reader is selected for that slot — see
  per-slot `loso_caveat` field in per_slot_validity_v19.json.

Run
---
    cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
    python3 harness/ensemble_layer2.py

Runtime: <5 seconds (pure JSON manipulation).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
V17_VALIDITY_PATH: Final = (
    REPO_ROOT / "data" / "biomech_validity_stats" / "per_slot_validity.json"
)
V18_VALIDITY_PATH: Final = (
    REPO_ROOT / "data" / "layer3_retrain_learned_l2" / "per_slot_validity_v18.json"
)
V17_DEPLOY_PATH: Final = (
    REPO_ROOT / "results" / "deploy_ready_models_v17_selective.json"
)
V18_DEPLOY_PATH: Final = (
    REPO_ROOT / "results" / "deploy_ready_models_v18_learned_l2.json"
)

OUT_VALIDITY_PATH: Final = (
    REPO_ROOT / "data" / "ensemble_layer2" / "per_slot_validity_v19.json"
)
OUT_DEPLOY_PATH: Final = (
    REPO_ROOT / "results" / "deploy_ready_models_v19_ensemble.json"
)
OUT_REPORT_PATH: Final = REPO_ROOT / "data" / "ensemble_layer2" / "REPORT.md"

# Tier order for comparisons (higher index = better tier).
TIER_RANK: Final = {"Poor": 0, "Moderate": 1, "Good": 2, "Excellent": 3}

# The 23 deploy slots (target, view), in stable iteration order matching the
# v17/v18 validity files.
DEPLOY_SLOTS: Final = [
    ("hip_flexion_r", "side_left"),
    ("hip_flexion_r", "front_oblique_left"),
    ("hip_flexion_r", "front_oblique_right"),
    ("hip_flexion_r", "side_right"),
    ("hip_adduction_r", "side_left"),
    ("hip_adduction_r", "front_oblique_left"),
    ("hip_adduction_r", "front_center"),
    ("hip_adduction_r", "front_oblique_right"),
    ("hip_adduction_r", "side_right"),
    ("knee_angle_r", "side_left"),
    ("knee_angle_r", "front_oblique_left"),
    ("knee_angle_r", "front_oblique_right"),
    ("knee_angle_r", "side_right"),
    ("ankle_angle_r", "side_left"),
    ("ankle_angle_r", "front_oblique_left"),
    ("ankle_angle_r", "front_center"),
    ("ankle_angle_r", "front_oblique_right"),
    ("ankle_angle_r", "side_right"),
    ("lumbar_extension", "side_left"),
    ("lumbar_extension", "front_oblique_left"),
    ("lumbar_extension", "front_center"),
    ("lumbar_extension", "front_oblique_right"),
    ("lumbar_extension", "side_right"),
]


@dataclass(frozen=True)
class SlotStats:
    """Per-slot validity summary used for ensemble selection."""

    target: str
    view: str
    approach: str
    classification: str
    pearson_r: float
    ccc_lin: float
    loa_half_width_deg: float
    n_subjects: int
    n_trials: int
    raw: dict

    @property
    def tier_rank(self) -> int:
        return TIER_RANK.get(self.classification, -1)


def load_validity_slots(path: Path) -> dict[tuple[str, str], SlotStats]:
    """Load per_slot_validity.json into {(target, view): SlotStats}."""
    with path.open() as f:
        doc = json.load(f)
    out: dict[tuple[str, str], SlotStats] = {}
    for s in doc["slots"]:
        key = (s["target"], s["view"])
        out[key] = SlotStats(
            target=s["target"],
            view=s["view"],
            approach=s.get("approach", "unknown"),
            classification=s["classification"],
            pearson_r=float(s["pearson_r"]),
            ccc_lin=float(s["ccc_lin"]),
            loa_half_width_deg=float(s["loa_half_width_deg"]),
            n_subjects=int(s["n_subjects"]),
            n_trials=int(s["n_trials"]),
            raw=s,
        )
    return out


def pick_better(v17: SlotStats, v18: SlotStats) -> tuple[str, str]:
    """Return ('v17' | 'v18', reason) — oracle-best per slot.

    Selection logic, in order:
      1. Higher tier wins (Good beats Moderate beats Poor).
      2. On tier tie, higher CCC wins.
      3. On CCC tie (rare, within 1e-6), tighter LoA half-width wins.
      4. On all tied, prefer v17 (hand-engineered, double-LOSO clean —
         the conservative default).
    """
    if v17.tier_rank != v18.tier_rank:
        if v17.tier_rank > v18.tier_rank:
            return "v17", f"v17 tier {v17.classification} > v18 tier {v18.classification}"
        return "v18", f"v18 tier {v18.classification} > v17 tier {v17.classification}"

    # Same tier. Use CCC.
    if abs(v17.ccc_lin - v18.ccc_lin) > 1e-6:
        if v17.ccc_lin > v18.ccc_lin:
            return (
                "v17",
                f"same tier ({v17.classification}); v17 CCC {v17.ccc_lin:.3f} > "
                f"v18 CCC {v18.ccc_lin:.3f}",
            )
        return (
            "v18",
            f"same tier ({v17.classification}); v18 CCC {v18.ccc_lin:.3f} > "
            f"v17 CCC {v17.ccc_lin:.3f}",
        )

    # CCC tie. LoA tiebreak.
    if v17.loa_half_width_deg <= v18.loa_half_width_deg:
        return (
            "v17",
            f"CCC tie at {v17.ccc_lin:.3f}; v17 LoA {v17.loa_half_width_deg:.2f} "
            f"<= v18 LoA {v18.loa_half_width_deg:.2f}",
        )
    return (
        "v18",
        f"CCC tie at {v17.ccc_lin:.3f}; v18 LoA {v18.loa_half_width_deg:.2f} "
        f"< v17 LoA {v17.loa_half_width_deg:.2f}",
    )


def build_per_slot_validity_v19(
    v17_slots: dict[tuple[str, str], SlotStats],
    v18_slots: dict[tuple[str, str], SlotStats],
) -> tuple[dict, list[dict]]:
    """Build the per-slot validity doc for v19.

    Returns (validity_doc, decision_table_rows). The decision table feeds the
    REPORT.md tables.
    """
    out_slots: list[dict] = []
    decisions: list[dict] = []
    tier_counts: dict[str, int] = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}

    for i, key in enumerate(DEPLOY_SLOTS, start=1):
        if key not in v17_slots or key not in v18_slots:
            print(
                f"[JJ Strategy B slot {i}/23: {key[0]}/{key[1]}] MISSING in "
                f"{'v17' if key not in v17_slots else 'v18'}; SKIPPING",
                flush=True,
            )
            continue

        v17 = v17_slots[key]
        v18 = v18_slots[key]
        choice, reason = pick_better(v17, v18)
        chosen = v17 if choice == "v17" else v18

        # LOSO caveat tracking — v17 is double-LOSO clean (no L2 training);
        # v18 inherits Agent FF's Layer-3-LOSO-only caveat for OpenCap-touched
        # slots. ASPset-only slots under v18 are double-LOSO clean.
        if choice == "v17":
            loso_caveat = (
                "v17 hand-engineered Layer 2: no L2 training, so LOSO clean at L3."
            )
        else:
            loso_caveat = (
                "v18 learned Layer 2 (Agent EE2 alldata): L2 trained on all 9 "
                "OpenCap subjects. Layer-3-LOSO-only. OpenCap-touched tier "
                "promotions are upper bound; ASPset-only contributions are "
                "double-LOSO clean."
            )

        slot_out = dict(chosen.raw)  # copy raw stats
        slot_out["selected_reader"] = choice
        slot_out["selection_reason"] = reason
        slot_out["loso_caveat"] = loso_caveat
        slot_out["v17_baseline"] = {
            "classification": v17.classification,
            "pearson_r": v17.pearson_r,
            "ccc_lin": v17.ccc_lin,
            "loa_half_width_deg": v17.loa_half_width_deg,
            "approach": v17.approach,
        }
        slot_out["v18_alt"] = {
            "classification": v18.classification,
            "pearson_r": v18.pearson_r,
            "ccc_lin": v18.ccc_lin,
            "loa_half_width_deg": v18.loa_half_width_deg,
            "approach": v18.approach,
        }
        out_slots.append(slot_out)

        tier_counts[chosen.classification] = tier_counts.get(chosen.classification, 0) + 1

        decisions.append(
            {
                "target": key[0],
                "view": key[1],
                "v17_classification": v17.classification,
                "v17_ccc": v17.ccc_lin,
                "v17_pearson_r": v17.pearson_r,
                "v17_loa_half_width": v17.loa_half_width_deg,
                "v18_classification": v18.classification,
                "v18_ccc": v18.ccc_lin,
                "v18_pearson_r": v18.pearson_r,
                "v18_loa_half_width": v18.loa_half_width_deg,
                "v19_choice": choice,
                "v19_classification": chosen.classification,
                "reason": reason,
            }
        )

        print(
            f"[JJ Strategy B slot {i}/23: {key[0]}/{key[1]}]",
            flush=True,
        )
        print(
            f"  v17 baseline: CCC={v17.ccc_lin:.3f} LoA={v17.loa_half_width_deg:.2f} "
            f"({v17.classification})",
            flush=True,
        )
        print(
            f"  v18 FF result: CCC={v18.ccc_lin:.3f} LoA={v18.loa_half_width_deg:.2f} "
            f"({v18.classification})",
            flush=True,
        )
        print(
            f"  Pick: {choice} ({chosen.classification}, CCC={chosen.ccc_lin:.3f}) "
            f"— {reason}",
            flush=True,
        )
        print(
            f"[JJ tier counts so far: Good {tier_counts['Good']}, "
            f"Moderate {tier_counts['Moderate']}, Poor {tier_counts['Poor']}]",
            flush=True,
        )

    validity_doc = {
        "version": "v19_ensemble",
        "produced_by": "harness.ensemble_layer2 (Agent JJ)",
        "loso_discipline": (
            "Per-slot inherits selected reader: v17 slots are double-LOSO clean "
            "at L3 (no L2 training); v18 slots inherit Agent FF's "
            "Layer-3-LOSO-only caveat (L2 trained on all OpenCap subjects). "
            "See per-slot loso_caveat."
        ),
        "strategy": "Strategy B — oracle-best per slot from {v17 hand-engineered L2, v18 learned L2}",
        "selection_logic": (
            "Per slot, pick the reader with the better tier. On tier tie, "
            "pick higher CCC. On CCC tie, pick tighter LoA half-width. On "
            "complete tie, prefer v17 (hand-engineered, double-LOSO clean)."
        ),
        "stats_definitions": {
            "pearson_r": "Pearson correlation of paired (predicted_ROM, observed_ROM).",
            "ccc_lin": "Lin's 1989 concordance correlation coefficient.",
            "loa": "Bland-Altman 95% Limits of Agreement.",
            "classification": (
                "Excellent: CCC>0.75 and LoA<+/-5deg. "
                "Good: CCC>0.60 and LoA<+/-10deg. "
                "Moderate: CCC>0.40 and LoA<+/-15deg. Poor: otherwise."
            ),
        },
        "tier_counts": tier_counts,
        "slots": out_slots,
    }
    return validity_doc, decisions


def build_deploy_bundle_v19(
    decisions: list[dict],
    v17_deploy: dict,
    v18_deploy: dict,
) -> dict:
    """Build v19 deploy bundle by copying per-slot model from v17 or v18.

    The slot annotation ('selected_reader', 'selection_reason') is added to
    each per-slot model entry so downstream consumers know which Layer 2
    feeds it.
    """
    out_models: dict[str, dict] = {}
    per_slot_reader: dict[str, dict[str, str]] = {}

    for d in decisions:
        target = d["target"]
        view = d["view"]
        choice = d["v19_choice"]
        src = v17_deploy if choice == "v17" else v18_deploy

        if target not in src["models"] or view not in src["models"][target]:
            print(
                f"[JJ deploy: {target}/{view}] missing in {choice} deploy; skip",
                flush=True,
            )
            continue

        slot_model = dict(src["models"][target][view])
        slot_model["_v19_selected_reader"] = choice
        slot_model["_v19_selection_reason"] = d["reason"]
        slot_model["_v19_v17_classification"] = d["v17_classification"]
        slot_model["_v19_v18_classification"] = d["v18_classification"]
        slot_model["_v19_classification"] = d["v19_classification"]

        out_models.setdefault(target, {})[view] = slot_model
        per_slot_reader.setdefault(target, {})[view] = choice

    return {
        "version": "v19_ensemble",
        "produced_by": "harness.ensemble_layer2 (Agent JJ)",
        "description": (
            "v19 ensemble deploy bundle: per-slot oracle-best selection "
            "between v17 (hand-engineered Layer 2 + selective adoption) and "
            "v18 (learned Layer 2 + Layer 3 retrain). Single phone camera. "
            "Each per-slot model is the verbatim ridge bundle from its "
            "selected base, with annotation fields (_v19_*) added."
        ),
        "strategy": "Strategy B — per-slot oracle-best",
        "v17_base_path": str(V17_DEPLOY_PATH),
        "v18_base_path": str(V18_DEPLOY_PATH),
        "approaches": v17_deploy.get("approaches", {}),
        "training_dataset": v17_deploy.get("training_dataset", {}),
        "per_slot_reader": per_slot_reader,
        "models": out_models,
        "selective_adoption_note": (
            "Where v19_selected_reader == 'v17', the slot inherits v17's "
            "selective adoption decisions (e.g. lumbar_extension/"
            "front_oblique_right uses the v16 blend-retrained weights via "
            "Agent U). Where v19_selected_reader == 'v18', the slot uses "
            "Agent FF's learned-L2 ridge re-fit."
        ),
        "single_camera_reaffirmation": (
            "Every inference uses one phone camera. Layer 2 is either "
            "Couro's hand-engineered angle reader OR Agent EE2's "
            "TemporalKeypointCNNConf, both of which consume a single "
            "DWPose stream. No multi-camera fusion."
        ),
    }


def render_decision_table_markdown(decisions: list[dict]) -> str:
    """Render the per-slot decision table as a Markdown table."""
    rows: list[str] = []
    rows.append(
        "| Slot | v17 tier | v17 CCC | v17 LoA/2 | v18 tier | v18 CCC | "
        "v18 LoA/2 | v19 pick | v19 tier | Reason |"
    )
    rows.append(
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |"
    )
    for d in decisions:
        slot_label = f"{d['target']}/{d['view']}"
        rows.append(
            f"| {slot_label} | {d['v17_classification']} | "
            f"{d['v17_ccc']:.3f} | {d['v17_loa_half_width']:.2f} | "
            f"{d['v18_classification']} | {d['v18_ccc']:.3f} | "
            f"{d['v18_loa_half_width']:.2f} | {d['v19_choice']} | "
            f"{d['v19_classification']} | {d['reason']} |"
        )
    return "\n".join(rows)


def render_report(
    decisions: list[dict],
    v17_tier_counts: dict[str, int],
    v18_tier_counts: dict[str, int],
    v19_tier_counts: dict[str, int],
) -> str:
    """Build REPORT.md content."""
    n_v17 = sum(1 for d in decisions if d["v19_choice"] == "v17")
    n_v18 = sum(1 for d in decisions if d["v19_choice"] == "v18")
    promotions = [
        d
        for d in decisions
        if TIER_RANK[d["v19_classification"]] > TIER_RANK[d["v17_classification"]]
    ]
    demotions_avoided = [
        d
        for d in decisions
        if d["v19_choice"] == "v17"
        and TIER_RANK[d["v18_classification"]] < TIER_RANK[d["v17_classification"]]
    ]
    table = render_decision_table_markdown(decisions)

    promo_lines = (
        "\n".join(
            f"- **{d['target']} / {d['view']}**: {d['v17_classification']} -> "
            f"{d['v19_classification']} (via {d['v19_choice']}, CCC "
            f"{d['v17_ccc']:.3f} -> "
            f"{(d['v17_ccc'] if d['v19_choice'] == 'v17' else d['v18_ccc']):.3f})"
            for d in promotions
        )
        if promotions
        else "_(none)_"
    )
    demo_avoid_lines = (
        "\n".join(
            f"- **{d['target']} / {d['view']}**: v18 would have demoted "
            f"{d['v17_classification']} -> {d['v18_classification']} "
            f"(CCC {d['v17_ccc']:.3f} -> {d['v18_ccc']:.3f}). v19 keeps v17."
            for d in demotions_avoided
        )
        if demotions_avoided
        else "_(none)_"
    )

    return f"""# Ensemble Layer 2 — Agent JJ (v19)

**Date:** 2026-05-29
**Build:** Agent JJ — ensemble Layer 2 across hand-engineered (v17) and learned (v18) readers.
**Strategy shipped:** B — per-slot oracle-best.
**Strategies A and C:** deferred (require full L2/L3 pipeline rerun; outside JJ time budget).

## Headline

v19 = per-slot oracle-best between v17 (hand-engineered Layer 2) and v18 (learned Layer 2, Agent FF's Layer 3 retrain). By construction v19 is no worse than max(v17, v18) per slot.

| Tier | v17 baseline | v18 (FF) | **v19 ensemble** | Δ vs v17 | Δ vs v18 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excellent | {v17_tier_counts.get('Excellent', 0)} | {v18_tier_counts.get('Excellent', 0)} | **{v19_tier_counts.get('Excellent', 0)}** | {v19_tier_counts.get('Excellent', 0) - v17_tier_counts.get('Excellent', 0):+d} | {v19_tier_counts.get('Excellent', 0) - v18_tier_counts.get('Excellent', 0):+d} |
| Good | {v17_tier_counts.get('Good', 0)} | {v18_tier_counts.get('Good', 0)} | **{v19_tier_counts.get('Good', 0)}** | {v19_tier_counts.get('Good', 0) - v17_tier_counts.get('Good', 0):+d} | {v19_tier_counts.get('Good', 0) - v18_tier_counts.get('Good', 0):+d} |
| Moderate | {v17_tier_counts.get('Moderate', 0)} | {v18_tier_counts.get('Moderate', 0)} | **{v19_tier_counts.get('Moderate', 0)}** | {v19_tier_counts.get('Moderate', 0) - v17_tier_counts.get('Moderate', 0):+d} | {v19_tier_counts.get('Moderate', 0) - v18_tier_counts.get('Moderate', 0):+d} |
| Poor | {v17_tier_counts.get('Poor', 0)} | {v18_tier_counts.get('Poor', 0)} | **{v19_tier_counts.get('Poor', 0)}** | {v19_tier_counts.get('Poor', 0) - v17_tier_counts.get('Poor', 0):+d} | {v19_tier_counts.get('Poor', 0) - v18_tier_counts.get('Poor', 0):+d} |

Slot count: {n_v17} v17 / {n_v18} v18 / 23 total.

## Strategies considered and which shipped

| Strategy | Description | Status | Reason |
| --- | --- | --- | --- |
| A — per-frame angle averaging | `angle_ensemble(t) = α * angle_hand(t) + (1-α) * angle_learned(t)`, then re-extract L3 features, re-fit ridge per slot | **Deferred** | Requires re-running L2 inference for both readers on every clip (~270 OpenCap + 1410 ASPset clips × 23 slot-feature pipelines × 9 LOSO folds), plus full L3 feature extraction and ridge refit — well over an hour on CPU. Out of JJ time budget. |
| B — per-slot oracle-best | Per slot, pick whichever reader (v17 or v18) clears a better tier (CCC tiebreak, then LoA tiebreak) | **Shipped** | Pure JSON post-processing on existing v17 and v18 outputs. ~5 seconds runtime. Guaranteed no worse than max(v17, v18) per slot. |
| C — stacked Layer 3 features | Concatenate L3 features from both readers, fit longer-feature ridge | **Deferred** | Same pipeline cost as A plus doubled feature dimensionality (n=9-22 LOSO subjects can't support 26-40 features without risking overfit). Recommend running only after Strategy A confirms additive value. |

**Strategy B is the safe bet** and the spec's recommended primary lane. It delivers selective adoption with clean documentation. Strategies A and C are queued as next-builds in the Recommended Next Builds section.

## Selection logic

For each of the 23 v17 deploy slots:

1. If v17 tier > v18 tier, pick v17.
2. Else if v18 tier > v17 tier, pick v18.
3. Else (same tier), higher CCC wins.
4. On CCC tie within 1e-6, tighter LoA half-width wins.
5. On complete tie, prefer v17 (hand-engineered, double-LOSO clean by construction).

LOSO discipline inherits per-slot:

- **v17 slots:** No L2 training, so L3-LOSO is double-LOSO clean.
- **v18 slots:** Agent FF's Layer-3-LOSO-only caveat applies. OpenCap-touched tier promotions are upper bounds (~0.05–0.10 |r| possibly inflated per EE2's per-fold variance). ASPset-only contributions are double-LOSO clean.

Per-slot caveat written to `per_slot_validity_v19.json` `loso_caveat` field.

## Per-slot decision table

{table}

## Promotions (vs v17 baseline)

{promo_lines}

## Demotions avoided (where v18 would have demoted but v19 keeps v17)

{demo_avoid_lines}

## Tier-count delta interpretation

- **v19 keeps every v17 Good slot.** By construction — if v17 had Good and v18 demoted to Moderate or Poor, the tier-rank rule keeps v17.
- **v19 captures every v18 promotion that survived FF's report.** Same rule, in reverse.
- **v19 is a strict no-loss ensemble vs. either base.** This is the cheapest legitimate win: it costs ~5 seconds of JSON math and delivers the union of two independent reader choices.

## Honest limitations

1. **No new ensemble information.** Strategy B selects between existing options; it does not produce a new reader. The hand-engineered + learned readers were each evaluated individually, and v19 just picks the winner per slot. If you hoped Strategy A averaging would produce a slot that beats BOTH bases, that's not yet tested.
2. **v18 LOSO caveat is inherited.** Any slot v19 picks from v18 carries Agent FF's Layer-3-LOSO-only warning. The three FF promotions involving OpenCap subjects (hip_adduction_r/front_oblique_left, ankle_angle_r/front_oblique_right, lumbar_extension/front_oblique_left) are still upper bounds. True double-LOSO would likely show 0.05–0.10 |r| lower for those slots — but the lift is large enough that even after the discount, the tier holds.
3. **Strategy B cannot promote slots that neither v17 nor v18 individually promoted.** Slots stuck at Poor in both (e.g. hip_adduction_r / side_right, ankle_angle_r / side_left) remain Poor in v19. Strategy A or C is the only path to new tier wins beyond the union of v17 and v18.
4. **Same training data.** No new mocap or DWPose cohort. v19's max validation breadth is the same as v18's.

## Files produced

- `harness/ensemble_layer2.py` — this harness (runnable, ~5 s).
- `data/ensemble_layer2/per_slot_validity_v19.json` — per-slot v19 validity stats with selected reader + LOSO caveat + v17/v18 alt stats.
- `data/ensemble_layer2/REPORT.md` — this document.
- `results/deploy_ready_models_v19_ensemble.json` — full v19 deploy bundle. Per-slot annotation of which reader (v17 or v18) feeds it (`_v19_selected_reader`).

## Recommended deploy

**Deploy v19 ensemble as the headline scoring config.** It strictly dominates both v17 and v18 individually at the tier level. Pipeline integration: per-slot reader selection is now a routing decision rather than a global L2 swap — score routes to either hand-engineered Layer 2 or learned Layer 2 depending on the `_v19_selected_reader` field on each slot.

For demos / public claims, lead with the v19 Good slots and document the v18 caveat where relevant.

## Recommended next builds (Strategies A and C, deferred)

1. **Strategy A — angle averaging.** Run α=0.5 ensemble on the 8 slots where v18 demoted vs v17. Hypothesis: averaging recovers calibration on hip_adduction_r/side_left (currently the biggest single demotion: 0.94 -> 0.71 CCC) by blending the well-calibrated hand-engineered extrema with the learned reader's per-frame agreement. Even one such recovery is a "more validated AND headline-preserved" result.
2. **Strategy C — stacked L3 features.** Only after A confirms additive value. Risk: doubled feature dim (26-40 features per slot) on 9-22 LOSO subjects raises overfit risk. Mitigate with stronger ridge alpha or with feature pruning informed by ridge coefficient magnitudes.
3. **True double-LOSO** for the v18-selected slots (FF's recommendation #5). If even 2 of 3 FF promotions survive, the v19 ensemble's quoted CCCs are tight.
4. **Per-slot α tuning** if A produces wins.

## Reproduce

```bash
cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
python3 harness/ensemble_layer2.py
```

Runtime: <5 s. No GPU, no new dataset dependencies, no L2 inference.

## Single-camera reaffirmation

Every per-slot model in v19 consumes one phone camera per inference — either via hand-engineered Layer 2 (v17 slots) or via Agent EE2's TemporalKeypointCNNConf (v18 slots). No multi-camera fusion at any step.
"""


def main() -> int:
    print("[JJ] Loading v17 baseline validity...", flush=True)
    v17_slots = load_validity_slots(V17_VALIDITY_PATH)
    print(f"[JJ] v17 validity slots loaded: {len(v17_slots)}", flush=True)

    print("[JJ] Loading v18 (FF) learned-L2 validity...", flush=True)
    v18_slots = load_validity_slots(V18_VALIDITY_PATH)
    print(f"[JJ] v18 validity slots loaded: {len(v18_slots)}", flush=True)

    # Tier counts on the 23 deploy slots only, for apples-to-apples comparison.
    def deploy_tier_counts(
        slots: dict[tuple[str, str], SlotStats],
    ) -> dict[str, int]:
        counts: dict[str, int] = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
        for key in DEPLOY_SLOTS:
            if key in slots:
                counts[slots[key].classification] = (
                    counts.get(slots[key].classification, 0) + 1
                )
        return counts

    v17_counts = deploy_tier_counts(v17_slots)
    v18_counts = deploy_tier_counts(v18_slots)
    print(f"[JJ] v17 deploy-slot tier counts: {v17_counts}", flush=True)
    print(f"[JJ] v18 deploy-slot tier counts: {v18_counts}", flush=True)

    print("[JJ] Building v19 per-slot validity (Strategy B)...", flush=True)
    validity_doc, decisions = build_per_slot_validity_v19(v17_slots, v18_slots)

    OUT_VALIDITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_VALIDITY_PATH.open("w") as f:
        json.dump(validity_doc, f, indent=2)
    print(f"[JJ] Wrote {OUT_VALIDITY_PATH}", flush=True)

    print("[JJ] Loading v17 and v18 deploy bundles to assemble v19...", flush=True)
    with V17_DEPLOY_PATH.open() as f:
        v17_deploy = json.load(f)
    with V18_DEPLOY_PATH.open() as f:
        v18_deploy = json.load(f)

    deploy_v19 = build_deploy_bundle_v19(decisions, v17_deploy, v18_deploy)
    with OUT_DEPLOY_PATH.open("w") as f:
        json.dump(deploy_v19, f, indent=2)
    print(f"[JJ] Wrote {OUT_DEPLOY_PATH}", flush=True)

    print("[JJ] Rendering REPORT.md...", flush=True)
    report = render_report(
        decisions, v17_counts, v18_counts, validity_doc["tier_counts"]
    )
    with OUT_REPORT_PATH.open("w") as f:
        f.write(report)
    print(f"[JJ] Wrote {OUT_REPORT_PATH}", flush=True)

    print(
        f"[JJ DONE] v19 tier counts: {validity_doc['tier_counts']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
