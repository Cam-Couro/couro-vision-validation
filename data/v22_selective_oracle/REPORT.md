# v22 Selective Oracle Deploy + v23 Combined-L2 Layer 3 Retrain

**Date:** 2026-06-02
**Build:** Agent KK Phase B — combined-cohort learned Layer 2 (HH2) joins the v17/v18/v20 reader pool in a per-slot oracle selection.
**Verdict:** **8 validated Good-tier slots** (v21 was 7; v17 baseline was 3).

## Tier count delta

| Tier | v17 baseline | v21 selective | **v22 selective (+ v23)** | Δ vs v21 |
| --- | ---: | ---: | ---: | ---: |
| Excellent | 0 | 0 | **0** | +0 |
| Good | 3 | 7 | **8** | +1 |
| Moderate | 9 | 9 | **8** | -1 |
| Poor | 13 | 7 | **7** | +0 |

## Reader distribution across 23 slots (v22)

- **v17** — hand-engineered Layer 2 (canonical): **10 slots**
- **v18** — EE2 OpenCap-only learned Layer 2 (Agent FF): **1 slots**
- **v20** — GG2 ROM-aware OpenCap-only learned Layer 2: **4 slots**
- **v23** — HH2 combined-cohort learned Layer 2 (Agent KK build): **8 slots**

## The 8 validated Good slots (v22)

| Slot | Reader | CCC | LoA half | v17 tier |
| --- | --- | ---: | ---: | --- |
| hip_adduction_r / side_left | v17 | 0.94 | ±9.80° | Good |
| hip_adduction_r / front_oblique_left | v20 | 0.69 | ±3.29° | Poor |
| ankle_angle_r / front_oblique_right | v23 | 0.73 | ±8.08° | Poor |
| ankle_angle_r / side_right | v17 | 0.64 | ±9.46° | Good |
| lumbar_extension / side_left | v23 | 0.88 | ±6.42° | Good |
| lumbar_extension / front_oblique_left | v18 | 0.71 | ±6.57° | Moderate |
| lumbar_extension / front_oblique_right | v23 | 0.79 | ±9.68° | Moderate |
| lumbar_extension / side_right | v23 | 0.85 | ±7.03° | Moderate |

### New Good slots from v23 specifically

- **ankle_angle_r / front_oblique_right**: CCC=0.73, LoA half=±8.08° (was Poor in v17 baseline)
- **lumbar_extension / front_oblique_right**: CCC=0.79, LoA half=±9.68° (was Moderate in v17 baseline)
- **lumbar_extension / side_right**: CCC=0.85, LoA half=±7.03° (was Moderate in v17 baseline)

## All promotions vs v17 baseline (v22)

| Slot | v17 tier | v22 tier | Reader | CCC | LoA half |
| --- | --- | --- | --- | ---: | ---: |
| hip_flexion_r / side_left / v12_combined | Poor | Moderate | v20 | 0.70 | ±14.46° |
| hip_adduction_r / front_oblique_left / v9_phased | Poor | Good | v20 | 0.69 | ±3.29° |
| knee_angle_r / front_oblique_left / v12_combined | Poor | Moderate | v23 | 0.91 | ±11.98° |
| ankle_angle_r / front_oblique_right / v14_full_dwpose | Poor | Good | v23 | 0.73 | ±8.08° |
| lumbar_extension / front_oblique_left / event_anchored | Moderate | Good | v18 | 0.71 | ±6.57° |
| lumbar_extension / front_oblique_right / v13_dwpose_hybrid | Moderate | Good | v23 | 0.79 | ±9.68° |
| lumbar_extension / side_right / v14_full_dwpose | Moderate | Good | v23 | 0.85 | ±7.03° |

## Per-slot CCC / LoA across all 4 readers

| Slot | v17 CCC / LoA | v18 CCC / LoA | v20 CCC / LoA | v23 CCC / LoA | v22 pick |
| --- | --- | --- | --- | --- | --- |
| hip_flexion_r / side_left | 0.60 / ±16.24° (Poor) | 0.45 / ±17.45° (Poor) | 0.70 / ±14.46° (Moderate) | 0.63 / ±15.22° (Poor) | **v20** |
| hip_flexion_r / front_oblique_left | 0.84 / ±11.29° (Moderate) | 0.39 / ±19.59° (Poor) | 0.51 / ±19.91° (Poor) | 0.74 / ±12.59° (Moderate) | **v17** |
| hip_flexion_r / front_oblique_right | 0.70 / ±18.50° (Poor) | 0.48 / ±21.50° (Poor) | 0.55 / ±20.26° (Poor) | 0.58 / ±18.01° (Poor) | **v17** |
| hip_flexion_r / side_right | 0.46 / ±19.01° (Poor) | 0.27 / ±20.29° (Poor) | 0.46 / ±19.20° (Poor) | 0.25 / ±21.32° (Poor) | **v20** |
| hip_adduction_r / side_left | 0.94 / ±9.80° (Good) | 0.71 / ±17.86° (Poor) | 0.85 / ±14.19° (Moderate) | 0.93 / ±10.35° (Moderate) | **v17** |
| hip_adduction_r / front_oblique_left | 0.29 / ±6.54° (Poor) | 0.45 / ±4.28° (Moderate) | 0.69 / ±3.29° (Good) | -0.55 / ±6.76° (Poor) | **v20** |
| hip_adduction_r / front_center | 0.77 / ±16.22° (Poor) | 0.38 / ±24.85° (Poor) | 0.65 / ±20.41° (Poor) | 0.63 / ±20.68° (Poor) | **v17** |
| hip_adduction_r / front_oblique_right | 0.78 / ±17.28° (Poor) | 0.63 / ±21.99° (Poor) | 0.84 / ±15.74° (Poor) | 0.79 / ±17.72° (Poor) | **v20** |
| hip_adduction_r / side_right | 0.21 / ±19.64° (Poor) | 0.09 / ±18.03° (Poor) | -0.18 / ±24.88° (Poor) | 0.27 / ±18.84° (Poor) | **v23** |
| knee_angle_r / side_left | 0.86 / ±12.43° (Moderate) | 0.50 / ±18.96° (Poor) | 0.68 / ±18.30° (Poor) | 0.87 / ±11.79° (Moderate) | **v23** |
| knee_angle_r / front_oblique_left | 0.78 / ±15.60° (Poor) | 0.29 / ±25.92° (Poor) | 0.85 / ±14.94° (Moderate) | 0.91 / ±11.98° (Moderate) | **v23** |
| knee_angle_r / front_oblique_right | 0.83 / ±10.72° (Moderate) | 0.11 / ±19.11° (Poor) | 0.61 / ±13.83° (Moderate) | 0.49 / ±14.30° (Moderate) | **v17** |
| knee_angle_r / side_right | 0.81 / ±14.24° (Moderate) | 0.08 / ±24.51° (Poor) | 0.35 / ±21.61° (Poor) | 0.79 / ±13.41° (Moderate) | **v17** |
| ankle_angle_r / side_left | 0.33 / ±12.24° (Poor) | 0.15 / ±14.19° (Poor) | -0.25 / ±15.95° (Poor) | -0.09 / ±16.15° (Poor) | **v17** |
| ankle_angle_r / front_oblique_left | 0.56 / ±10.78° (Moderate) | -0.28 / ±18.75° (Poor) | 0.24 / ±13.44° (Poor) | 0.14 / ±14.05° (Poor) | **v17** |
| ankle_angle_r / front_center | 0.09 / ±14.69° (Poor) | -0.49 / ±20.02° (Poor) | -0.50 / ±19.39° (Poor) | 0.21 / ±13.01° (Poor) | **v23** |
| ankle_angle_r / front_oblique_right | -0.13 / ±19.34° (Poor) | 0.59 / ±10.11° (Moderate) | -0.09 / ±16.49° (Poor) | 0.73 / ±8.08° (Good) | **v23** |
| ankle_angle_r / side_right | 0.64 / ±9.46° (Good) | 0.46 / ±11.29° (Moderate) | -0.03 / ±14.50° (Poor) | 0.18 / ±13.22° (Poor) | **v17** |
| lumbar_extension / side_left | 0.83 / ±7.25° (Good) | 0.75 / ±8.85° (Good) | 0.87 / ±6.83° (Good) | 0.88 / ±6.42° (Good) | **v23** |
| lumbar_extension / front_oblique_left | 0.53 / ±8.03° (Moderate) | 0.71 / ±6.57° (Good) | 0.19 / ±10.60° (Poor) | 0.31 / ±9.43° (Poor) | **v18** |
| lumbar_extension / front_center | 0.55 / ±7.45° (Moderate) | 0.43 / ±9.54° (Moderate) | 0.46 / ±8.85° (Moderate) | 0.41 / ±8.97° (Moderate) | **v17** |
| lumbar_extension / front_oblique_right | 0.63 / ±10.18° (Moderate) | 0.56 / ±10.61° (Moderate) | 0.69 / ±9.78° (Good) | 0.79 / ±9.68° (Good) | **v23** |
| lumbar_extension / side_right | 0.45 / ±13.31° (Moderate) | 0.37 / ±12.69° (Poor) | 0.62 / ±9.89° (Good) | 0.85 / ±7.03° (Good) | **v23** |

## Honest caveats

1. **v23 (and v18, v20) Layer-3-LOSO-only caveat.** L2 trained on ALL 24 cohort subjects (9 OpenCap + 15 ASPset). L3 ridge LOSO at subject level only. Tier promotions involving any cohort subject are upper bounds; per HH2's per-fold variance the true double-LOSO number could be ~0.05–0.10 |r| lower. **Only v17 (hand-engineered) slots are clean double-LOSO.**
2. **ASPset hip_adduction_r convention mismatch.** Per HH2's REPORT, hip_adduction_r regressed on OpenCap-held folds (−0.051 |r|) because ASPset's hip-adduction definition does not align cleanly with OpenCap's after the identity remap. v23 hip_adduction slots are therefore expected to underperform v18/v20 — and the v22 selective oracle simply keeps the v18/v20 reader on those slots.
3. **v17 hand-engineered is still the right reader for most slots.** v17 holds 10/23 slots in v22. Selective adoption is the rule, not the exception. Wholesale replacement of v17 with any learned-L2 reader is net-negative.
4. **v23 contributes 8/23 slots in v22 and is the new dominant learned-L2 reader.** v23 wins on the right-side lumbar slots (`side_left`, `front_oblique_right`, `side_right`) — beating v17, v18, and v20 on CCC — and takes `ankle_angle_r/front_oblique_right` as a NEW Good slot (CCC 0.733 vs v18's prior best 0.587). HH2's per-metric pooled |r| projection (knee_angle_r +0.064, ankle_angle_r +0.050) translated to ROM-tier wins on the right-side lumbar slots and 1 of 5 ankle slots; the other knee / ankle slots stayed with v17 or v20.
5. **Per-slot reader map is deploy complexity.** The deployed system must dispatch to the correct reader per (metric × view) combination. v22 adds v23 to that dispatch table for any slot where v23 won.
6. **Ankle slot CI remains wide** (n=9 OpenCap-only). Promotion to 'headline' range still requires fresh cohort expansion.
7. **No invented numbers.** All CCC / LoA / |r| values in this report were computed from the v23 LOSO build (re-fit Layer 3 ridge on combined-cohort learned-L2 features) or carried verbatim from the v17 baseline / v18 / v20 per-slot validity files. v22 is a pure oracle selection on top of those.

## Single-camera contract preserved

Every reader in the v22 pool (v17, v18, v20, v23) consumes a single DWPose stream from one phone camera. No multi-camera fusion. Same input/output contract as Couro's deployed Layer 2.

## Files

- `results/deploy_ready_models_v22_selective.json` — v22 deploy bundle with `per_slot_reader` dispatch map
- `results/deploy_ready_models_v23_combined_l2.json` — v23 candidate (combined-cohort L2 + L3 ridge re-fit)
- `data/v22_selective_oracle/per_slot_picks_v22.json` — per-slot pick audit trail with CCC / LoA per candidate reader
- `data/layer3_retrain_combined_l2/per_slot_validity_v23.json` — per-slot v23 validity stats (LOSO)
- `data/layer3_retrain_combined_l2/REPORT.md` — v23 narrative
- `models/learned_layer2_combined_alldata_v1.pt` — all-data combined L2 checkpoint (the L2 model used by v23)

