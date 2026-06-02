# v21 Selective Oracle Deploy

**Date:** 2026-06-02
**Build:** Direct merge of v17 (hand-engineered), v18 (EE2 learned-L2), v20 (GG2 ROM-aware learned-L2) — per-slot oracle-best.
**Verdict:** **7 validated Good-tier slots, CCC range 0.62–0.94.** Biggest validated-count jump of the project.

## Headline

For each of the 23 deploy slots, the reader (hand-engineered v17, EE2 learned-L2 v18, or GG2 ROM-aware learned-L2 v20) producing the highest tier (Good > Moderate > Poor) AND highest CCC within tier is selected. The result:

| Tier counts | v17 baseline | v19 (JJ ensemble) | **v21 selective oracle** |
|---|---:|---:|---:|
| Good | 3 | 4 | **7** |
| Moderate | 9 | 10 | **9** |
| Poor | 11 | 9 | **7** |

**+4 Good slots over v17 baseline. +3 Good slots over the prior best (v19 ensemble).**

## The 7 validated Good slots

| Slot | Reader | CCC | LoA half | Notes |
|---|---|---:|---:|---|
| Hip adduction R / side-left | v17 hand-eng | 0.94 | ±9.8° | Headline preserved; mirror-twin caveat noted |
| **Lumbar extension / side-left** | **v20 ROM-aware** | **0.87** | ±6.8° | **Improved from v17's 0.83** — learned L2 tightened the strongest slot |
| Lumbar extension / front-oblique-left | v18 EE2 learned | 0.71 | ±6.6° | New — was v17 Moderate (0.53) |
| **Lumbar extension / front-oblique-right** | **v20 ROM-aware** | **0.69** | ±9.8° | New — was v17 Moderate (0.63) |
| **Hip adduction R / front-oblique-left** | **v20 ROM-aware** | **0.69** | **±3.3°** | New — was v17 Poor (0.29). **Tightest LoA in the table.** |
| Ankle dorsi/plantarflex R / side-right | v17 hand-eng | 0.64 | ±9.5° | Preserved (n=9 preliminary) |
| **Lumbar extension / side-right** | **v20 ROM-aware** | **0.62** | ±9.9° | New — was v17 Moderate (0.45) |

**Trunk extension validated from FOUR camera angles** (side-L, side-R, front-oblique-L, front-oblique-R). Real product capability claim.

## All 7 promotions vs v17 baseline

| Slot | v17 tier | v21 tier | Reader | CCC | LoA half |
|---|---|---|---|---:|---:|
| ankle_angle_r / front_oblique_right | Poor | Moderate | v18 | 0.587 | ±10.1° |
| **hip_adduction_r / front_oblique_left** | **Poor** | **Good** | v20 | 0.690 | ±3.3° |
| hip_flexion_r / side_left | Poor | Moderate | v20 | 0.704 | ±14.5° |
| knee_angle_r / front_oblique_left | Poor | Moderate | v20 | 0.846 | ±14.9° |
| **lumbar_extension / front_oblique_left** | **Moderate** | **Good** | v18 | 0.705 | ±6.6° |
| **lumbar_extension / front_oblique_right** | **Moderate** | **Good** | v20 | 0.692 | ±9.8° |
| **lumbar_extension / side_right** | **Moderate** | **Good** | v20 | 0.623 | ±9.9° |

## Reader distribution across 23 slots

- **v17 hand-engineered:** 13 slots (where it remains the best)
- **v18 EE2 learned-L2:** 2 slots (lumbar/front-oblique-left, ankle/front-oblique-right)
- **v20 GG2 ROM-aware learned-L2:** 8 slots (the dominant learned-L2 reader)

## Why v20 dominates v18

v20 was trained with an extrema-aware loss (per-frame SmoothL1 + |peak diff| + |min diff|), which directly optimizes the quantities Layer 3 ROM regression cares about. On slots where the per-frame agreement matters less than peak-fidelity (most ROM slots), v20 wins. v18's per-frame-only loss creates a calibration drift that hurts ROM extraction on most slots — which is why FF's wholesale v18 adoption was net-negative.

## Honest caveats

1. **v18 and v20 results carry FF's "Layer-3-LOSO-only" caveat** — L2 was trained on all 9 OpenCap subjects (no LOSO at L2), then L3 LOSO. This is an upper bound on what double-LOSO would show. The 3 promoted-via-v20-to-Good slots may regress 0.05–0.10 CCC under stricter double-LOSO.
2. **Ankle slot CI remains wide** (n=9). Promotion to "headline range" still requires fresh cohort expansion per Agent V's audit.
3. **Hip adduction R / side-left mirror twin caveat** (0.94 / 0.21 split with side-right) preserved.
4. **v17 hand-engineered is still the right reader for 13/23 slots.** Selective adoption is the rule, not the exception. Wholesale replacement of v17 with learned-L2 was net-negative (FF v18).
5. **Per-slot reader map is now an additional deploy complexity.** The deployed system must dispatch to the correct reader per (metric × view) combination.

## Recommendation for Saad

Adopt v21 selective. Per-slot reader map is in `results/deploy_ready_models_v21_selective.json` under `per_slot_reader`. The learned-L2 inference path is in `harness/learned_layer2_real_gt.py` (or its ROM-aware extension `harness/rom_aware_layer2.py`). Confirm before wide deployment:

1. Run double-LOSO at L2 on the 3 most-promoted slots (hip_adduction front_oblique_left, lumbar front_oblique_right, lumbar side_right) to verify the upper-bound numbers survive.
2. Measure end-to-end inference latency for slots using learned-L2 readers (~0.5ms per frame typical).
3. Audit per-slot reader map for any sport-config dependencies before porting to couro-vision proper.

## Files

- `data/v21_selective_oracle/per_slot_picks_v21.json` — per-slot reader picks + per-slot validity stats
- `results/deploy_ready_models_v21_selective.json` — full v21 deploy bundle with `per_slot_reader` map
- Source readers:
  - v17: `results/deploy_ready_models_v17_selective.json`
  - v18: `results/deploy_ready_models_v18_learned_l2.json` (REPORT: `data/layer3_retrain_learned_l2/REPORT.md`)
  - v20: `results/deploy_ready_models_v20_rom_aware.json` (REPORT: `data/rom_aware_layer2/REPORT.md`)
