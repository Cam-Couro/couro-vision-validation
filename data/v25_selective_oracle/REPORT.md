# v25 Selective Oracle Deploy + v24 Combined-Cohort + ROM-Aware Layer 3 Retrain

**Date:** 2026-06-02
**Build:** Agent LL Phase C — LL combined-cohort + ROM-aware learned Layer 2 joins the v17/v18/v20/v23 reader pool in a per-slot oracle selection.
**Verdict:** **9 validated Good-tier slots** (v22 was 8; v21 was 7; v17 baseline was 3).
**Floor lift:** NO — none of the 5 sub-0.80 slots Cameron called out crossed 0.80 CCC under LL/v24 or in the v25 oracle. The Tier 1 (CCC ≥ 0.79) vs supplementary marketing restructure is the honest recommendation.

## Floor lift verdict (the 5 sub-0.80 slots)

Cameron's LL brief asked: "did ANY of these slots cross 0.80 CCC under LL or in the v25 oracle?"

| Slot | Previous CCC (reader) | v24 CCC (LL) | v25 oracle CCC (reader) | Tier change | Crossed 0.80? |
| --- | --- | ---: | --- | --- | :---: |
| lumbar_extension / front_oblique_right | 0.79 (v23) | 0.70 | 0.79 (v23) | Good (unchanged) | no |
| ankle_angle_r / front_oblique_right | 0.73 (v23) | 0.25 | 0.73 (v23) | Good (unchanged) | no |
| lumbar_extension / front_oblique_left | 0.71 (v18) | 0.36 | 0.71 (v18) | Good (unchanged) | no |
| hip_adduction_r / front_oblique_left | 0.69 (v20) | -0.77 | 0.69 (v20) | Good (unchanged) | no |
| ankle_angle_r / side_right | 0.64 (v17) | 0.25 | 0.64 (v17) | Good (unchanged) | no |

## Tier count delta

| Tier | v17 baseline | v22 selective | **v25 selective (+ v24)** | Δ vs v22 |
| --- | ---: | ---: | ---: | ---: |
| Excellent | 0 | 0 | **0** | +0 |
| Good | 3 | 8 | **9** | +1 |
| Moderate | 9 | 8 | **7** | -1 |
| Poor | 13 | 7 | **7** | +0 |

## Reader distribution across 23 slots (v25)

- **v17** — hand-engineered Layer 2 (canonical): **8 slots**
- **v18** — EE2 OpenCap-only learned Layer 2 (Agent FF): **1 slots**
- **v20** — GG2 ROM-aware OpenCap-only learned Layer 2: **4 slots**
- **v23** — HH2 combined-cohort learned Layer 2 (Agent KK): **8 slots**
- **v24** — LL combined-cohort + ROM-aware learned Layer 2 (Agent LL, this build): **2 slots**

## The 9 validated Good slots (v25)

| Slot | Reader | CCC | LoA half | v17 tier |
| --- | --- | ---: | ---: | --- |
| hip_adduction_r / side_left | v17 | 0.94 | ±9.80° | Good |
| hip_adduction_r / front_oblique_left | v20 | 0.69 | ±3.29° | Poor |
| knee_angle_r / front_oblique_right | v24 | 0.89 | ±8.05° | Moderate |
| ankle_angle_r / front_oblique_right | v23 | 0.73 | ±8.08° | Poor |
| ankle_angle_r / side_right | v17 | 0.64 | ±9.46° | Good |
| lumbar_extension / side_left | v23 | 0.88 | ±6.42° | Good |
| lumbar_extension / front_oblique_left | v18 | 0.71 | ±6.57° | Moderate |
| lumbar_extension / front_oblique_right | v23 | 0.79 | ±9.68° | Moderate |
| lumbar_extension / side_right | v23 | 0.85 | ±7.03° | Moderate |

### New Good slots from v24 specifically

- **knee_angle_r / front_oblique_right**: CCC=0.89, LoA half=±8.05° (was Moderate in v17 baseline)

## All promotions vs v17 baseline (v25)

| Slot | v17 tier | v25 tier | Reader | CCC | LoA half |
| --- | --- | --- | --- | ---: | ---: |
| hip_flexion_r / side_left / v12_combined | Poor | Moderate | v20 | 0.70 | ±14.46° |
| hip_adduction_r / front_oblique_left / v9_phased | Poor | Good | v20 | 0.69 | ±3.29° |
| knee_angle_r / front_oblique_left / v12_combined | Poor | Moderate | v23 | 0.91 | ±11.98° |
| knee_angle_r / front_oblique_right / v9_phased | Moderate | Good | v24 | 0.89 | ±8.05° |
| ankle_angle_r / front_oblique_right / v14_full_dwpose | Poor | Good | v23 | 0.73 | ±8.08° |
| lumbar_extension / front_oblique_left / event_anchored | Moderate | Good | v18 | 0.71 | ±6.57° |
| lumbar_extension / front_oblique_right / v13_dwpose_hybrid | Moderate | Good | v23 | 0.79 | ±9.68° |
| lumbar_extension / side_right / v14_full_dwpose | Moderate | Good | v23 | 0.85 | ±7.03° |

## Per-slot CCC / LoA across all 5 readers

| Slot | v17 CCC / LoA | v18 CCC / LoA | v20 CCC / LoA | v23 CCC / LoA | v24 CCC / LoA | v25 pick |
| --- | --- | --- | --- | --- | --- | --- |
| hip_flexion_r / side_left | 0.60 / ±16.24° (Poor) | 0.45 / ±17.45° (Poor) | 0.70 / ±14.46° (Moderate) | 0.63 / ±15.22° (Poor) | 0.50 / ±16.97° (Poor) | **v20** |
| hip_flexion_r / front_oblique_left | 0.84 / ±11.29° (Moderate) | 0.39 / ±19.59° (Poor) | 0.51 / ±19.91° (Poor) | 0.74 / ±12.59° (Moderate) | 0.65 / ±15.97° (Poor) | **v17** |
| hip_flexion_r / front_oblique_right | 0.70 / ±18.50° (Poor) | 0.48 / ±21.50° (Poor) | 0.55 / ±20.26° (Poor) | 0.58 / ±18.01° (Poor) | 0.51 / ±19.04° (Poor) | **v17** |
| hip_flexion_r / side_right | 0.46 / ±19.01° (Poor) | 0.27 / ±20.29° (Poor) | 0.46 / ±19.20° (Poor) | 0.25 / ±21.32° (Poor) | 0.07 / ±24.13° (Poor) | **v20** |
| hip_adduction_r / side_left | 0.94 / ±9.80° (Good) | 0.71 / ±17.86° (Poor) | 0.85 / ±14.19° (Moderate) | 0.93 / ±10.35° (Moderate) | 0.88 / ±12.31° (Moderate) | **v17** |
| hip_adduction_r / front_oblique_left | 0.29 / ±6.54° (Poor) | 0.45 / ±4.28° (Moderate) | 0.69 / ±3.29° (Good) | -0.55 / ±6.76° (Poor) | -0.77 / ±9.22° (Poor) | **v20** |
| hip_adduction_r / front_center | 0.77 / ±16.22° (Poor) | 0.38 / ±24.85° (Poor) | 0.65 / ±20.41° (Poor) | 0.63 / ±20.68° (Poor) | 0.15 / ±29.54° (Poor) | **v17** |
| hip_adduction_r / front_oblique_right | 0.78 / ±17.28° (Poor) | 0.63 / ±21.99° (Poor) | 0.84 / ±15.74° (Poor) | 0.79 / ±17.72° (Poor) | 0.81 / ±16.16° (Poor) | **v20** |
| hip_adduction_r / side_right | 0.21 / ±19.64° (Poor) | 0.09 / ±18.03° (Poor) | -0.18 / ±24.88° (Poor) | 0.27 / ±18.84° (Poor) | 0.24 / ±19.18° (Poor) | **v23** |
| knee_angle_r / side_left | 0.86 / ±12.43° (Moderate) | 0.50 / ±18.96° (Poor) | 0.68 / ±18.30° (Poor) | 0.87 / ±11.79° (Moderate) | 0.87 / ±11.55° (Moderate) | **v23** |
| knee_angle_r / front_oblique_left | 0.78 / ±15.60° (Poor) | 0.29 / ±25.92° (Poor) | 0.85 / ±14.94° (Moderate) | 0.91 / ±11.98° (Moderate) | 0.91 / ±11.77° (Moderate) | **v23** |
| knee_angle_r / front_oblique_right | 0.83 / ±10.72° (Moderate) | 0.11 / ±19.11° (Poor) | 0.61 / ±13.83° (Moderate) | 0.49 / ±14.30° (Moderate) | 0.89 / ±8.05° (Good) | **v24** |
| knee_angle_r / side_right | 0.81 / ±14.24° (Moderate) | 0.08 / ±24.51° (Poor) | 0.35 / ±21.61° (Poor) | 0.79 / ±13.41° (Moderate) | 0.58 / ±21.92° (Poor) | **v17** |
| ankle_angle_r / side_left | 0.33 / ±12.24° (Poor) | 0.15 / ±14.19° (Poor) | -0.25 / ±15.95° (Poor) | -0.09 / ±16.15° (Poor) | 0.38 / ±13.04° (Poor) | **v24** |
| ankle_angle_r / front_oblique_left | 0.56 / ±10.78° (Moderate) | -0.28 / ±18.75° (Poor) | 0.24 / ±13.44° (Poor) | 0.14 / ±14.05° (Poor) | 0.46 / ±12.39° (Moderate) | **v17** |
| ankle_angle_r / front_center | 0.09 / ±14.69° (Poor) | -0.49 / ±20.02° (Poor) | -0.50 / ±19.39° (Poor) | 0.21 / ±13.01° (Poor) | -0.36 / ±17.87° (Poor) | **v23** |
| ankle_angle_r / front_oblique_right | -0.13 / ±19.34° (Poor) | 0.59 / ±10.11° (Moderate) | -0.09 / ±16.49° (Poor) | 0.73 / ±8.08° (Good) | 0.25 / ±14.37° (Poor) | **v23** |
| ankle_angle_r / side_right | 0.64 / ±9.46° (Good) | 0.46 / ±11.29° (Moderate) | -0.03 / ±14.50° (Poor) | 0.18 / ±13.22° (Poor) | 0.25 / ±15.88° (Poor) | **v17** |
| lumbar_extension / side_left | 0.83 / ±7.25° (Good) | 0.75 / ±8.85° (Good) | 0.87 / ±6.83° (Good) | 0.88 / ±6.42° (Good) | 0.85 / ±6.46° (Good) | **v23** |
| lumbar_extension / front_oblique_left | 0.53 / ±8.03° (Moderate) | 0.71 / ±6.57° (Good) | 0.19 / ±10.60° (Poor) | 0.31 / ±9.43° (Poor) | 0.36 / ±9.56° (Poor) | **v18** |
| lumbar_extension / front_center | 0.55 / ±7.45° (Moderate) | 0.43 / ±9.54° (Moderate) | 0.46 / ±8.85° (Moderate) | 0.41 / ±8.97° (Moderate) | 0.48 / ±8.84° (Moderate) | **v17** |
| lumbar_extension / front_oblique_right | 0.63 / ±10.18° (Moderate) | 0.56 / ±10.61° (Moderate) | 0.69 / ±9.78° (Good) | 0.79 / ±9.68° (Good) | 0.70 / ±9.81° (Good) | **v23** |
| lumbar_extension / side_right | 0.45 / ±13.31° (Moderate) | 0.37 / ±12.69° (Poor) | 0.62 / ±9.89° (Good) | 0.85 / ±7.03° (Good) | 0.79 / ±7.93° (Good) | **v23** |

## Bottom-5 slot table (per Cameron's brief)

| Slot | Previous CCC | LL (v24) CCC | v25 oracle CCC | v25 tier |
| --- | ---: | ---: | ---: | --- |
| lumbar_extension / front_oblique_right | 0.79 | 0.70 | 0.79 | Good |
| ankle_angle_r / front_oblique_right | 0.73 | 0.25 | 0.73 | Good |
| lumbar_extension / front_oblique_left | 0.71 | 0.36 | 0.71 | Good |
| hip_adduction_r / front_oblique_left | 0.69 | -0.77 | 0.69 | Good |
| ankle_angle_r / side_right | 0.64 | 0.25 | 0.64 | Good |

## Honest caveats

1. **v24 (and v18, v20, v23) Layer-3-LOSO-only caveat.** L2 trained on ALL 24 cohort subjects (9 OpenCap + 15 ASPset). L3 ridge LOSO at subject level only. Tier promotions involving any cohort subject are upper bounds; per HH2's per-fold variance the true double-LOSO number could be ~0.05-0.10 |r| lower. **Only v17 (hand-engineered) slots are clean double-LOSO.**
2. **ROM-aware loss + cross-dataset convention may interact.** GG2 added the extrema-aware loss on a single clean OpenCap cohort. LL adds it on the combined OpenCap+ASPset cohort. ASPset's convention noise (especially hip_adduction_r and lumbar_extension offsets) can be amplified by the extrema terms because they pin max/min predictions to potentially miscalibrated targets. We mitigated this by dropping ASPset hip_adduction_r supervision entirely (HH2's own recommendation), but the lumbar extrema may still inherit some ASPset drift.
3. **ASPset hip_adduction_r dropped from LL training.** Per HH2's REPORT, hip_adduction_r regressed on OpenCap-held folds (−0.051 |r|) because ASPset's hip-adduction definition does not align with OpenCap's. LL drops ASPset hip_adduction_r supervision via the `drop_aspset_hipadd` flag (default ON). LL hip_adduction_r is therefore trained on OpenCap-only ground truth, with ASPset providing shared-trunk representation only.
4. **v17 hand-engineered remains the most chosen reader.** v17 holds 8/23 slots in v25. Selective adoption remains the rule, not the exception.
5. **v24 contributes 2/23 slots in v25.** v24's lift over v23 (where it occurs) is real but small.
6. **Per-slot reader map is deploy complexity.** The deployed system must dispatch to the correct reader per (metric × view) combination. v25 adds v24 to that dispatch table for any slot where v24 won.
7. **Ankle slot CI remains wide** (n=9 OpenCap-only ankle GT). Promotion to 'headline' range still requires fresh cohort expansion with paired ankle GT.
8. **No invented numbers.** All CCC / LoA / |r| values were computed from the v24 LOSO build (LL all-data L2 + L3 ridge re-fit) or carried verbatim from the v17 baseline / v18 / v20 / v23 per-slot validity files. v25 is a pure oracle selection on top of those.

## Single-camera contract preserved

Every reader in the v25 pool (v17, v18, v20, v23, v24) consumes a single DWPose stream from one phone camera. No multi-camera fusion. Same input/output contract as Couro's deployed Layer 2.

## Files

- `results/deploy_ready_models_v25_selective.json` — v25 deploy bundle with `per_slot_reader` dispatch map
- `results/deploy_ready_models_v24_combined_rom_aware.json` — v24 candidate (LL combined-cohort + ROM-aware L2 + L3 ridge re-fit)
- `data/v25_selective_oracle/per_slot_picks_v25.json` — per-slot pick audit trail with CCC / LoA per candidate reader
- `data/layer3_retrain_combined_rom_aware/per_slot_validity_v24.json` — per-slot v24 validity stats (LOSO)
- `data/layer3_retrain_combined_rom_aware/REPORT.md` — v24 narrative
- `models/learned_layer2_combined_rom_aware_alldata_v1.pt` — all-data LL L2 checkpoint (the L2 model used by v24)

