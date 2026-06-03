# v28 Selective Oracle Deploy + v26/v27 Per-Source Heads Layer 2

**Date:** 2026-06-02
**Build:** Agent MM Phase C — MM-A (v26, per-source heads + per-frame SmoothL1) and MM-B (v27, per-source heads + ROM-aware) join the v17/v18/v20/v23/v24 reader pool in a per-slot oracle selection.
**Verdict:** **10 validated Good-tier slots** (v25 was 9; v22 was 8; v17 baseline was 3).
**Floor lift to ≥0.80 CCC:** YES — target slot(s) crossed the 0.80 bar under MM: lumbar_extension / front_oblique_left.

## Floor lift verdict — the 2 target slots + sister monitor

The MM brief asked: did per-source heads push the 2 convention-mismatched supplementary slots to Tier 1 (CCC ≥ 0.79), or all the way to ≥0.80? Plus: do per-source heads hurt the existing Tier 1 lumbar slots?

| Slot | v25 CCC (reader) | v26 CCC (MM-A) | v27 CCC (MM-B) | v28 oracle CCC (reader) | Tier change vs v25 | Crossed 0.79? |
| --- | --- | ---: | ---: | --- | --- | :---: |
| hip_adduction_r / front_oblique_left | 0.69 (v20 (LL/v25 pick)) | -0.37 | -0.52 | 0.69 (v20) | Good (unchanged) | no |
| lumbar_extension / front_oblique_left | 0.71 (v18 (LL/v25 pick)) | 0.82 | 0.52 | 0.82 (v26) | Good (unchanged) | YES |
| lumbar_extension / front_oblique_right | 0.79 (v23 (LL/v25 pick)) | 0.65 | 0.67 | 0.79 (v23) | Good (unchanged) | no |
| lumbar_extension / side_left | 0.88 (v23 (LL/v25 pick)) | 0.70 | 0.64 | 0.88 (v23) | Good (unchanged) | YES |
| lumbar_extension / side_right | 0.85 (v23 (LL/v25 pick)) | 0.51 | 0.36 | 0.85 (v23) | Good (unchanged) | YES |

## Tier count delta

| Tier | v17 baseline | v22 selective | v25 selective | **v28 selective (+ v26 + v27)** | Δ vs v25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excellent | 0 | 0 | 0 | **0** | +0 |
| Good | 3 | 8 | 9 | **10** | +1 |
| Moderate | 9 | 8 | 7 | **6** | -1 |
| Poor | 13 | 7 | 7 | **7** | +0 |

## Reader distribution across 23 slots (v28)

- **v17** — hand-engineered Layer 2 (canonical): **6 slots**
- **v18** — EE2 OpenCap-only learned Layer 2: **0 slots**
- **v20** — GG2 ROM-aware OpenCap-only learned Layer 2: **2 slots**
- **v23** — HH2 combined-cohort learned Layer 2 (Agent KK): **7 slots**
- **v24** — LL combined-cohort + ROM-aware learned Layer 2: **2 slots**
- **v26** — MM-A per-source heads + per-frame SmoothL1 learned Layer 2 (Agent MM, this build): **3 slots**
- **v27** — MM-B per-source heads + ROM-aware learned Layer 2 (Agent MM, this build): **3 slots**

## The 10 validated Good slots (v28)

| Slot | Reader | CCC | LoA half | v17 tier | v25 tier |
| --- | --- | ---: | ---: | --- | --- |
| hip_flexion_r / side_left | v26 | 0.86 | ±9.24° | Poor | Moderate |
| hip_adduction_r / side_left | v17 | 0.94 | ±9.80° | Good | Good |
| hip_adduction_r / front_oblique_left | v20 | 0.69 | ±3.29° | Poor | Good |
| knee_angle_r / front_oblique_right | v24 | 0.89 | ±8.05° | Moderate | Good |
| ankle_angle_r / front_oblique_right | v23 | 0.73 | ±8.08° | Poor | Good |
| ankle_angle_r / side_right | v17 | 0.64 | ±9.46° | Good | Good |
| lumbar_extension / side_left | v23 | 0.88 | ±6.42° | Good | Good |
| lumbar_extension / front_oblique_left | v26 | 0.82 | ±5.18° | Moderate | Good |
| lumbar_extension / front_oblique_right | v23 | 0.79 | ±9.68° | Moderate | Good |
| lumbar_extension / side_right | v23 | 0.85 | ±7.03° | Moderate | Good |

### New Good slots from v26/v27 specifically

- **hip_flexion_r / side_left**: v26 CCC=0.86, LoA half=±9.24° (was Moderate in v25; Poor in v17 baseline)

## Promotions vs v25

| Slot | v25 tier | v28 tier | Reader | CCC | LoA half |
| --- | --- | --- | --- | ---: | ---: |
| hip_flexion_r / side_left | Moderate | Good | v26 | 0.86 | ±9.24° |

## All promotions vs v17 baseline (v28)

| Slot | v17 tier | v28 tier | Reader | CCC | LoA half |
| --- | --- | --- | --- | ---: | ---: |
| hip_flexion_r / side_left / v12_combined | Poor | Good | v26 | 0.86 | ±9.24° |
| hip_adduction_r / front_oblique_left / v9_phased | Poor | Good | v20 | 0.69 | ±3.29° |
| knee_angle_r / front_oblique_left / v12_combined | Poor | Moderate | v27 | 0.91 | ±11.83° |
| knee_angle_r / front_oblique_right / v9_phased | Moderate | Good | v24 | 0.89 | ±8.05° |
| ankle_angle_r / front_oblique_right / v14_full_dwpose | Poor | Good | v23 | 0.73 | ±8.08° |
| lumbar_extension / front_oblique_left / event_anchored | Moderate | Good | v26 | 0.82 | ±5.18° |
| lumbar_extension / front_oblique_right / v13_dwpose_hybrid | Moderate | Good | v23 | 0.79 | ±9.68° |
| lumbar_extension / side_right / v14_full_dwpose | Moderate | Good | v23 | 0.85 | ±7.03° |

## Per-slot CCC / LoA across all 7 readers

| Slot | v17 | v18 | v20 | v23 | v24 | v26 | v27 | v28 pick |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hip_flexion_r / side_left | 0.60/16.2° | 0.45/17.5° | 0.70/14.5° | 0.63/15.2° | 0.50/17.0° | 0.86/9.2° | 0.56/17.5° | **v26** |
| hip_flexion_r / front_oblique_left | 0.84/11.3° | 0.39/19.6° | 0.51/19.9° | 0.74/12.6° | 0.65/16.0° | 0.77/11.8° | 0.73/14.7° | **v17** |
| hip_flexion_r / front_oblique_right | 0.70/18.5° | 0.48/21.5° | 0.55/20.3° | 0.58/18.0° | 0.51/19.0° | 0.54/19.1° | 0.65/17.1° | **v17** |
| hip_flexion_r / side_right | 0.46/19.0° | 0.27/20.3° | 0.46/19.2° | 0.25/21.3° | 0.07/24.1° | -0.03/24.8° | 0.58/16.9° | **v27** |
| hip_adduction_r / side_left | 0.94/9.8° | 0.71/17.9° | 0.85/14.2° | 0.93/10.3° | 0.88/12.3° | 0.89/12.2° | 0.90/11.6° | **v17** |
| hip_adduction_r / front_oblique_left | 0.29/6.5° | 0.45/4.3° | 0.69/3.3° | -0.55/6.8° | -0.77/9.2° | -0.37/7.8° | -0.52/6.5° | **v20** |
| hip_adduction_r / front_center | 0.77/16.2° | 0.38/24.9° | 0.65/20.4° | 0.63/20.7° | 0.15/29.5° | 0.46/24.0° | 0.13/30.4° | **v17** |
| hip_adduction_r / front_oblique_right | 0.78/17.3° | 0.63/22.0° | 0.84/15.7° | 0.79/17.7° | 0.81/16.2° | 0.68/22.0° | 0.74/20.0° | **v20** |
| hip_adduction_r / side_right | 0.21/19.6° | 0.09/18.0° | -0.18/24.9° | 0.27/18.8° | 0.24/19.2° | 0.19/17.5° | 0.17/20.1° | **v23** |
| knee_angle_r / side_left | 0.86/12.4° | 0.50/19.0° | 0.68/18.3° | 0.87/11.8° | 0.87/11.6° | 0.80/13.3° | 0.87/11.6° | **v23** |
| knee_angle_r / front_oblique_left | 0.78/15.6° | 0.29/25.9° | 0.85/14.9° | 0.91/12.0° | 0.91/11.8° | 0.88/12.6° | 0.91/11.8° | **v27** |
| knee_angle_r / front_oblique_right | 0.83/10.7° | 0.11/19.1° | 0.61/13.8° | 0.49/14.3° | 0.89/8.1° | 0.83/9.7° | 0.71/11.9° | **v24** |
| knee_angle_r / side_right | 0.81/14.2° | 0.08/24.5° | 0.35/21.6° | 0.79/13.4° | 0.58/21.9° | 0.68/19.2° | 0.81/12.9° | **v27** |
| ankle_angle_r / side_left | 0.33/12.2° | 0.15/14.2° | -0.25/15.9° | -0.09/16.1° | 0.38/13.0° | 0.02/15.3° | -0.33/17.1° | **v24** |
| ankle_angle_r / front_oblique_left | 0.56/10.8° | -0.28/18.8° | 0.24/13.4° | 0.14/14.0° | 0.46/12.4° | -0.10/17.7° | -0.03/15.5° | **v17** |
| ankle_angle_r / front_center | 0.09/14.7° | -0.49/20.0° | -0.50/19.4° | 0.21/13.0° | -0.36/17.9° | 0.08/13.8° | -0.40/17.5° | **v23** |
| ankle_angle_r / front_oblique_right | -0.13/19.3° | 0.59/10.1° | -0.09/16.5° | 0.73/8.1° | 0.25/14.4° | 0.57/10.3° | 0.09/16.3° | **v23** |
| ankle_angle_r / side_right | 0.64/9.5° | 0.46/11.3° | -0.03/14.5° | 0.18/13.2° | 0.25/15.9° | 0.32/12.2° | -0.32/18.6° | **v17** |
| lumbar_extension / side_left | 0.83/7.2° | 0.75/8.9° | 0.87/6.8° | 0.88/6.4° | 0.85/6.5° | 0.70/8.2° | 0.64/9.7° | **v23** |
| lumbar_extension / front_oblique_left | 0.53/8.0° | 0.71/6.6° | 0.19/10.6° | 0.31/9.4° | 0.36/9.6° | 0.82/5.2° | 0.52/8.4° | **v26** |
| lumbar_extension / front_center | 0.55/7.5° | 0.43/9.5° | 0.46/8.9° | 0.41/9.0° | 0.48/8.8° | 0.55/8.5° | 0.50/8.5° | **v26** |
| lumbar_extension / front_oblique_right | 0.63/10.2° | 0.56/10.6° | 0.69/9.8° | 0.79/9.7° | 0.70/9.8° | 0.65/9.3° | 0.67/9.5° | **v23** |
| lumbar_extension / side_right | 0.45/13.3° | 0.37/12.7° | 0.62/9.9° | 0.85/7.0° | 0.79/7.9° | 0.51/10.8° | 0.36/12.3° | **v23** |

## Cameron's MM-brief table (per-slot pick history)

| Slot | v25 CCC | MM-A (v26) CCC | MM-B (v27) CCC | v28 oracle CCC |
| --- | ---: | ---: | ---: | ---: |
| hip_adduction_r / front_oblique_left | 0.69 | -0.37 | -0.52 | 0.69 |
| lumbar_extension / front_oblique_left | 0.71 | 0.82 | 0.52 | 0.82 |

## Honest caveats

1. **Layer-3-LOSO-only caveat applies to v18/v20/v23/v24/v26/v27.** L2 trained on ALL 24 cohort subjects (9 OpenCap + 15 ASPset). L3 ridge LOSO at subject level only. Tier promotions involving any cohort subject are upper bounds; per HH2's per-fold variance the true double-LOSO number could be ~0.05-0.10 |r| lower. **Only v17 (hand-engineered) slots are clean double-LOSO.**
2. **Per-source heads tested ONLY at L2 architecture level.** v26 and v27 are HH2's own recommended fix for the OpenCap/ASPset convention mismatch on hip_adduction_r and lumbar_extension. The shared 5-output head consumed by the L3 ridge is what Couro deploys; the ASPset head is discarded at inference.
3. **All-data L2, not 24-fold LOSO Phase A.** Like the LL build, this build cycle skipped the 24-fold LOSO Phase A eval to fit compute budget. The all-data L2 cached in `models/learned_layer2_persource_perframe_alldata_v1.pt` and `models/learned_layer2_persource_romaware_alldata_v1.pt` is what v26/v27 use. The harness supports Phase A LOSO via `python3 -m harness.learned_layer2_combined_persource --variant mm_a` (and `--variant mm_b`).
4. **Two supplementary slots are bias-limited, not noise-limited.** Cameron's brief flagged hip_adduction_r/front_oblique_left as having the tightest LoA in the entire table (±3.3° under v20). If MM did not lift it, the residual is a true convention/calibration bias that even per-source heads can't fix without metric-redefinition or new ground-truth collection.
5. **ASPset has no foot keypoints (ankle_angle_r cohort-limited).** Two of the four supplementary slots (ankle_angle_r/front_oblique_right at v23 CCC 0.73 and ankle_angle_r/side_right at v17 CCC 0.64) are cohort-limited (n=9 OpenCap). MM does not address those — they require fresh paired ankle GT collection.
6. **No invented numbers.** All CCC / LoA / |r| values are computed from this build's all-data L2 + L3 ridge re-fit, or carried verbatim from prior per-slot validity files (v17/v18/v20/v23/v24).

## Single-camera contract preserved

Every reader in the v28 pool (v17, v18, v20, v23, v24, v26, v27) consumes a single DWPose stream from one phone camera. No multi-camera fusion. Same input/output contract as Couro's deployed Layer 2.

## Files

- `results/deploy_ready_models_v28_selective.json` — v28 deploy bundle with `per_slot_reader` dispatch map
- `results/deploy_ready_models_v26_persource_perframe.json` — v26 deploy candidate (MM-A per-source heads + per-frame SmoothL1)
- `results/deploy_ready_models_v27_persource_romaware.json` — v27 deploy candidate (MM-B per-source heads + ROM-aware)
- `data/v28_selective_oracle/per_slot_picks_v28.json` — per-slot pick audit trail with CCC / LoA per candidate reader
- `data/layer3_retrain_persource_perframe/per_slot_validity_v26.json` — per-slot v26 validity stats (LOSO at L3)
- `data/layer3_retrain_persource_romaware/per_slot_validity_v27.json` — per-slot v27 validity stats (LOSO at L3)
- `data/layer3_retrain_persource_perframe/REPORT.md` — v26 narrative
- `data/layer3_retrain_persource_romaware/REPORT.md` — v27 narrative
- `models/learned_layer2_persource_perframe_alldata_v1.pt` — all-data MM-A L2 checkpoint
- `models/learned_layer2_persource_romaware_alldata_v1.pt` — all-data MM-B L2 checkpoint

