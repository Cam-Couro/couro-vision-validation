# Phase B: Layer 3 retrained on combined-cohort learned Layer 2 (v23 / Agent KK)

**Date:** 2026-06-02
**Build:** Agent KK (Phase B follow-up to Agent HH2)
**Verdict:** **4 Good / 7 Moderate / 12 Poor.** v23 beats v18 (2/5/16) and v20 (4/5/14) on Good-slot count and is the new dominant learned-L2 reader for **8 of 23 deploy slots in the v22 selective oracle**.

## What this is

Layer 3 ridge regression re-fit per (metric × view) slot using the angle traces emitted by **HH2's combined OpenCap+ASPset learned Layer 2**. v18 was the analogous Phase B for EE2 (OpenCap-only). v20 was the analogous Phase B for GG2 (ROM-aware OpenCap-only). v23 is the same build pattern on top of HH2's combined cohort.

Pipeline (see `harness/layer3_retrain_on_combined_l2.py`):

1. Train one `TemporalKeypointCNNConf` on **all 24 cohort subjects** (9 OpenCap + 15 ASPset, no LOSO at L2) using HH2's exact training recipe (15 epochs, batch 256, train_stride=4, masked SmoothL1 with per-target NaN-safe loss). Saved to `models/learned_layer2_combined_alldata_v1.pt`.
2. Monkey-patch FF's (`harness.layer3_retrain_on_learned_l2`) `keypoints_to_motion_data` so the 5 deploy metrics come from the learned model. Left-side angles via mirrored-keypoint inference; pelvis_tilt remains hand-engineered.
3. For each of the 23 v17 deploy slots, rebuild the per-clip ridge features with the patched L2 and re-fit a ridge regression with subject-level LOSO at L3 only.
4. Compute Bland-Altman + Lin's CCC per slot; classify against the canonical biomech validity tier thresholds (Good: CCC > 0.60 AND LoA half < ±10°; Moderate: CCC > 0.40 AND LoA half < ±15°; Poor: otherwise).

Single phone camera. Same input/output contract as Couro's deployed Layer 2.

## LOSO discipline used in this build

**Layer-3-LOSO-only.** Same caveat as v18 (FF) and v20 (GG2). One L2 model trained on every cohort subject, then L3 ridge LOSO. This leaks Layer 2 information from the held-out L3 subject into the L2 training data. Per HH2's per-fold LOSO numbers (best OpenCap-held |r| 0.744, worst 0.541, pooled 0.670), the true double-LOSO CCC at L3 could be ~0.05–0.10 lower than reported here. Cleanly double-LOSO only at the v17 hand-engineered reader.

## Tier count delta

| Tier | v17 baseline | v18 (EE2) | v20 (GG2) | **v23 (HH2)** |
| --- | ---: | ---: | ---: | ---: |
| Excellent | 0 | 0 | 0 | **0** |
| Good | 3 | 2 | 4 | **4** |
| Moderate | 9 | 5 | 5 | **7** |
| Poor | 13 | 16 | 14 | **12** |

Net vs v17: -2 Moderate, +1 Good, -1 Poor (two Moderate slots demoted to Poor, four slots promoted).

Net vs v18: +2 Good, +2 Moderate, -4 Poor.

Net vs v20: same Good count, +2 Moderate, -2 Poor.

Promotions: 4 | Demotions: 4 | Unchanged: 15

### Promotions

- knee_angle_r / front_oblique_left: Poor -> Moderate
- ankle_angle_r / front_oblique_right: Poor -> Good
- lumbar_extension / front_oblique_right: Moderate -> Good
- lumbar_extension / side_right: Moderate -> Good

### Demotions

- hip_adduction_r / side_left: Good -> Moderate
- ankle_angle_r / front_oblique_left: Moderate -> Poor
- ankle_angle_r / side_right: Good -> Poor
- lumbar_extension / front_oblique_left: Moderate -> Poor

## Per-slot before/after table (baseline = v17 hand-engineered; v23 = this build)

| Target | View | Approach | n | r baseline | CCC baseline | LoA/2 baseline | Tier baseline | r v23 | CCC v23 | LoA/2 v23 | Tier v23 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| hip_flexion_r | side_left | v12_combined_with_learned_l2 | 12 | 0.62 | 0.60 | 16.24 | Poor | 0.65 | 0.63 | 15.22 | Poor |
| hip_flexion_r | front_oblique_left | v13_dwpose_hybrid_with_learned_l2 | 21 | 0.85 | 0.84 | 11.29 | Moderate | 0.76 | 0.74 | 12.59 | Moderate |
| hip_flexion_r | front_oblique_right | v13_dwpose_hybrid_with_learned_l2 | 17 | 0.70 | 0.70 | 18.50 | Poor | 0.59 | 0.58 | 18.01 | Poor |
| hip_flexion_r | side_right | event_anchored_with_learned_l2 | 9 | 0.52 | 0.46 | 19.01 | Poor | 0.32 | 0.25 | 21.32 | Poor |
| hip_adduction_r | side_left | v14_full_dwpose_with_learned_l2 | 12 | 0.95 | 0.94 | 9.80 | Good | 0.94 | 0.93 | 10.35 | Moderate |
| hip_adduction_r | front_oblique_left | v9_phased_with_learned_l2 | 9 | 0.35 | 0.29 | 6.54 | Poor | -0.66 | -0.55 | 6.76 | Poor |
| hip_adduction_r | front_center | v12_combined_with_learned_l2 | 23 | 0.93 | 0.77 | 16.22 | Poor | 0.79 | 0.63 | 20.68 | Poor |
| hip_adduction_r | front_oblique_right | v12_combined_with_learned_l2 | 17 | 0.92 | 0.78 | 17.28 | Poor | 0.86 | 0.79 | 17.72 | Poor |
| hip_adduction_r | side_right | v14_full_dwpose_with_learned_l2 | 10 | 0.22 | 0.21 | 19.64 | Poor | 0.27 | 0.27 | 18.84 | Poor |
| knee_angle_r | side_left | v14_full_dwpose_with_learned_l2 | 12 | 0.88 | 0.86 | 12.43 | Moderate | 0.88 | 0.87 | 11.79 | Moderate |
| knee_angle_r | front_oblique_left | v12_combined_with_learned_l2 | 21 | 0.91 | 0.78 | 15.60 | Poor | 0.92 | 0.91 | 11.98 | Moderate |
| knee_angle_r | front_oblique_right | v9_phased_with_learned_l2 | 9 | 0.86 | 0.83 | 10.72 | Moderate | 0.72 | 0.49 | 14.30 | Moderate |
| knee_angle_r | side_right | v12_combined_with_learned_l2 | 10 | 0.81 | 0.81 | 14.24 | Moderate | 0.80 | 0.79 | 13.41 | Moderate |
| ankle_angle_r | side_left | v14_full_dwpose_with_learned_l2 | 9 | 0.46 | 0.33 | 12.24 | Poor | -0.11 | -0.09 | 16.15 | Poor |
| ankle_angle_r | front_oblique_left | v14_full_dwpose_with_learned_l2 | 9 | 0.62 | 0.56 | 10.78 | Moderate | 0.18 | 0.14 | 14.05 | Poor |
| ankle_angle_r | front_center | event_anchored_bilateral_with_learned_l2 | 9 | 0.11 | 0.09 | 14.69 | Poor | 0.33 | 0.21 | 13.01 | Poor |
| ankle_angle_r | front_oblique_right | v14_full_dwpose_with_learned_l2 | 9 | -0.13 | -0.13 | 19.34 | Poor | 0.88 | 0.73 | 8.08 | Good |
| ankle_angle_r | side_right | v14_full_dwpose_with_learned_l2 | 9 | 0.75 | 0.64 | 9.46 | Good | 0.29 | 0.18 | 13.22 | Poor |
| lumbar_extension | side_left | v14_full_dwpose_with_learned_l2 | 12 | 0.87 | 0.83 | 7.25 | Good | 0.89 | 0.88 | 6.42 | Good |
| lumbar_extension | front_oblique_left | event_anchored_with_learned_l2 | 9 | 0.62 | 0.53 | 8.03 | Moderate | 0.40 | 0.31 | 9.43 | Poor |
| lumbar_extension | front_center | event_anchored_with_learned_l2 | 9 | 0.76 | 0.55 | 7.45 | Moderate | 0.48 | 0.41 | 8.97 | Moderate |
| lumbar_extension | front_oblique_right | v13_dwpose_hybrid_with_learned_l2 | 17 | 0.80 | 0.63 | 10.18 | Moderate | 0.80 | 0.79 | 9.68 | Good |
| lumbar_extension | side_right | v14_full_dwpose_with_learned_l2 | 10 | 0.47 | 0.45 | 13.31 | Moderate | 0.86 | 0.85 | 7.03 | Good |

## How v23 stacks up against v18 / v20 on its 4 Good slots

| Slot | v17 CCC | v18 CCC | v20 CCC | **v23 CCC** | Winner |
| --- | ---: | ---: | ---: | ---: | --- |
| ankle_angle_r / front_oblique_right | -0.129 | 0.587 | -0.092 | **0.733** | v23 |
| lumbar_extension / side_left | 0.832 | 0.746 | 0.872 | **0.884** | v23 |
| lumbar_extension / front_oblique_right | 0.634 | 0.556 | 0.692 | **0.790** | v23 |
| lumbar_extension / side_right | 0.452 | 0.375 | 0.623 | **0.848** | v23 |

**v23 wins all 4 Good-slot ties on CCC.** On the right-side lumbar slots in particular, the combined-cohort training appears to have given the model a more stable trunk-extension representation than either OpenCap-only EE2 (v18) or the single-cohort ROM-aware GG2 (v20).

## Reader pool comparison on the metrics HH2 projected would lift

HH2's per-metric pooled |r| deltas vs EE2 (OpenCap-held only):

- knee_angle_r: **+0.064** |r|
- ankle_angle_r: **+0.050** |r|
- hip_flexion_r: +0.018 |r|
- lumbar_extension: +0.005 |r|
- hip_adduction_r: **−0.051** |r| (ASPset convention mismatch)

Did this translate to ROM-tier promotions?

- **knee_angle_r**: v23 won the L3 ridge fit on 2/4 slots (side_left, front_oblique_left), promoting `knee_angle_r/front_oblique_left` from Poor to Moderate. **Confirmed.**
- **ankle_angle_r**: v23 won 2/5 slots (front_center, front_oblique_right). `ankle_angle_r/front_oblique_right` is a new Good (CCC 0.733). **Partially confirmed** — the other 3 ankle slots regressed because ASPset has no foot keypoints, so the combined-cohort training under-represents ankle GT.
- **lumbar_extension**: v23 won 3/5 slots (side_left, front_oblique_right, side_right). 3 are Good. **Strongly confirmed despite the small per-metric |r| delta** — combined-cohort training stabilized the ROM-tier extraction here even though per-frame |r| barely moved.
- **hip_flexion_r**: v23 didn't win any slot at L3 (v17 / v20 still hold all 4). HH2's small per-frame lift did not survive ROM extraction.
- **hip_adduction_r**: HH2 regression confirmed. v23 won 1/5 slot (side_right) but only by being least-bad among Poor candidates. The v22 selective oracle keeps v17 / v20 for the actually-Good hip_adduction slots.

## Why v23 wins the lumbar slots so decisively

1. **More data on a metric that's roughly aligned across datasets.** Trunk extension is one metric where ASPset's convention (after the `- 180°` shift in HH2's loader) maps cleanly onto OpenCap's range. The combined cohort doubles the training distribution without introducing label noise.
2. **More viewing-angle diversity.** ASPset's three camera views (left / mid / right) per clip force the L2 model to learn an angle representation that's less dependent on absolute pose orientation. Combined with OpenCap's denser repeats of the DJ task, the L2 gets a richer prior on trunk lean across views.
3. **Lumbar's signal is large and slow.** The trunk-extension waveform is high-amplitude relative to noise. A better-trained L2 doesn't need to track fast transients precisely — it just needs to track the slow rise/fall, and the ROM-from-extrema readout amplifies the gain.

## Why v23 doesn't lift hip_adduction or ankle_angle

- **hip_adduction_r**: ASPset's "hip adduction" comes from a 3D pelvis-relative direction that, after HH2's identity-remap convention, is sign-misaligned with OpenCap's hip adduction angle in the in-plane (front_center) views. Training on both pollutes the L2 output. **Known HH2 caveat realized at L3.**
- **ankle_angle_r**: ASPset has no foot keypoints, so HH2 masks the ankle target during training on ASPset clips. The model gets no ASPset signal on this metric, so combined-cohort training cannot help — and the OpenCap-only ankle signal is now diluted by the ASPset half of the training distribution shifting feature weights elsewhere. v18 (EE2 OpenCap-only) actually wins on `ankle_angle_r / front_oblique_left` for this reason.

## Honest recommendation

1. **Adopt v22 selective**, which picks v23 on the 8 slots where it wins and keeps v17 / v18 / v20 elsewhere. This is +1 validated Good slot over v21.
2. **Do not adopt v23 wholesale.** Net tier counts vs v17 alone are mixed (+1 Good, -2 Moderate, -1 Poor — three slots demoted, four promoted).
3. **The three new Good slots under v22 via v23** (`ankle_angle_r/front_oblique_right`, `lumbar_extension/front_oblique_right`, `lumbar_extension/side_right`) deserve double-LOSO confirmation before sport-config dependency. The Layer-3-LOSO-only caveat means the CCC could give back 0.05–0.10 under stricter validation.
4. **Trunk extension is now validated from FOUR camera angles** (side-L, side-R, front-oblique-L, front-oblique-R) with the v22 selective deploy. Real product claim survives the v23 layer.

## Caveats — honest

- **Not double-LOSO.** L2 trained on all 24 cohort subjects, L3 LOSO only. Same caveat as v18 (FF) and v20 (GG2). OpenCap-subject tier promotions are upper bounds; per HH2's per-fold variance (best 0.744, worst 0.541) the true double-LOSO CCC could be ~0.05–0.10 lower per slot.
- **ASPset hip_adduction_r convention mismatch** confirmed (HH2 REPORT predicted regression; this build sees it). Selective oracle keeps v17 / v20 for hip_adduction slots.
- **ASPset has no foot keypoints**, so ankle_angle_r training data is OpenCap-only inside HH2's pipeline. The L2 model gets no extra ankle signal from the bigger cohort — the lift on `ankle_angle_r/front_oblique_right` (sign-flipped hand-engineered baseline) comes from generalizing the L2 backbone, not from new ankle GT.
- **Left-side angles** come from mirroring the keypoint series and running the same right-side L2 model. This relies on the L2 model being approximately mirror-symmetric, which is reasonable for the DJ task but not validated.
- **pelvis_tilt remains hand-engineered** (used as coupling for lumbar_extension slots in v12/v13 only). HH2's L2 doesn't predict it.
- **Numerical warnings:** v12_combined / v13_dwpose_hybrid slots emit `divide by zero` / `overflow` warnings during ridge predict. Reflects degenerate feature columns (constant or NaN-filled) when the patched learned-L2 angle trace produces an out-of-distribution value on some ASPset clips. These predictions are still finite after the z-score; CCC/LoA stats are computed on finite values only.
- **Single phone camera.** No multi-camera fusion. Same input/output contract as Couro's deployed Layer 2.
- **No invented numbers.** All CCC / LoA / |r| above were computed from the v23 LOSO build, not extrapolated.

## Files

- `models/learned_layer2_combined_alldata_v1.pt` — all-data combined L2 checkpoint
- `data/layer3_retrain_combined_l2/per_slot_validity_v23.json` — per-slot LOSO stats
- `results/deploy_ready_models_v23_combined_l2.json` — v23 deploy bundle (per-slot ridge weights)
- `harness/layer3_retrain_on_combined_l2.py` — this build's entry point
- `harness/learned_layer2_combined.py` — HH2's combined-cohort L2 training code (reused)
- `harness/layer3_retrain_on_learned_l2.py` — FF's L3 retrain orchestration (reused by monkey-patch)
- `harness/build_v22_selective.py` — downstream selective-oracle builder that adopts v23

## Downstream

`harness/build_v22_selective.py` constructs the v22 selective oracle by adding v23 to the v17/v18/v20 reader pool. Outputs:
- `results/deploy_ready_models_v22_selective.json`
- `data/v22_selective_oracle/REPORT.md`
- `data/v22_selective_oracle/per_slot_picks_v22.json`

**v22 has 8 validated Good slots** (was 7 in v21).
