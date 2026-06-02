# Phase B: Layer 3 retrained on LL combined-cohort + ROM-aware Layer 2 (v24 / Agent LL)

**Date:** 2026-06-02
**Build:** Agent LL (Phase B follow-up to HH2 + GG2)
**Verdict:** **4 Good / 5 Moderate / 14 Poor.** v24 = HH2's combined cohort + GG2's ROM-aware loss.

## What this is

Layer 3 ridge regression re-fit per (metric × view) slot using the angle traces emitted by **LL's combined-cohort + ROM-aware learned Layer 2**, which is the hybrid of:

- **HH2 (v23)**: doubled cohort (9 OpenCap + 15 ASPset = 24 subjects, ~1670 clips), masked SmoothL1 per-frame loss.
- **GG2 (v20)**: extrema-aware extra loss term, lam=1.0, computed per-(clip, metric) on the differentiable max/min of the predicted trajectory.

Per HH2's own recommendation, ASPset hip_adduction_r supervision is dropped from training (convention mismatch with OpenSim's lumped-rotation definition). All other metrics use the full combined cohort. Loss = `SmoothL1(per_frame, masked) + lam * |peak_pred - peak_gt| + lam * |min_pred - min_gt|`. Extrema respect the same NaN mask as the per-frame term.

Pipeline (see `harness/layer3_retrain_on_combined_rom_aware_l2.py`):

1. Train one `TemporalKeypointCNNConf` on **all 24 cohort subjects** (no LOSO at L2) with the combined + ROM-aware loss. Saved to `models/learned_layer2_combined_rom_aware_alldata_v1.pt`.
2. Monkey-patch FF's `keypoints_to_motion_data` so the 5 deploy metrics come from the learned model. Left-side angles via mirrored-keypoint inference; pelvis_tilt remains hand-engineered.
3. For each of the 23 v17 deploy slots, rebuild the per-clip ridge features with the patched L2 and re-fit a ridge regression with subject-level LOSO at L3 only.
4. Compute Bland-Altman + Lin's CCC per slot; classify against the canonical biomech validity tier thresholds (Good: CCC > 0.60 AND LoA half < ±10°; Moderate: CCC > 0.40 AND LoA half < ±15°; Poor: otherwise).

Single phone camera. Same input/output contract as Couro's deployed Layer 2.

## LOSO discipline used in this build

**Layer-3-LOSO-only.** Same caveat as v18 (FF), v20 (GG2), and v23 (KK). One L2 model trained on every cohort subject, then L3 ridge LOSO. This leaks Layer 2 information from the held-out L3 subject into the L2 training data. Per HH2's per-fold LOSO numbers (best OpenCap-held |r| 0.744, worst 0.541, pooled 0.670), the true double-LOSO CCC at L3 could be ~0.05-0.10 lower than reported here. Cleanly double-LOSO only at the v17 hand-engineered reader.

## Tier count delta

| Tier | v17 baseline | v18 (EE2) | v20 (GG2) | v23 (HH2) | **v24 (LL)** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excellent | 0 | 0 | 0 | 0 | **0** |
| Good | 3 | 2 | 4 | 4 | **4** |
| Moderate | 9 | 5 | 5 | 7 | **5** |
| Poor | 13 | 16 | 14 | 12 | **14** |

Promotions vs v17: 4 | Demotions vs v17: 5 | Unchanged vs v17: 14

### Promotions vs v17

- knee_angle_r / front_oblique_left: Poor -> Moderate
- knee_angle_r / front_oblique_right: Moderate -> Good
- lumbar_extension / front_oblique_right: Moderate -> Good
- lumbar_extension / side_right: Moderate -> Good

### Demotions vs v17

- hip_flexion_r / front_oblique_left: Moderate -> Poor
- hip_adduction_r / side_left: Good -> Moderate
- knee_angle_r / side_right: Moderate -> Poor
- ankle_angle_r / side_right: Good -> Poor
- lumbar_extension / front_oblique_left: Moderate -> Poor

## Per-slot before/after table (baseline = v17 hand-engineered; v24 = this build)

| Target | View | n | r baseline | CCC baseline | LoA/2 baseline | Tier baseline | r v24 | CCC v24 | LoA/2 v24 | Tier v24 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| hip_flexion_r | side_left | 12 | 0.62 | 0.60 | 16.24 | Poor | 0.53 | 0.50 | 16.97 | Poor |
| hip_flexion_r | front_oblique_left | 21 | 0.85 | 0.84 | 11.29 | Moderate | 0.67 | 0.65 | 15.97 | Poor |
| hip_flexion_r | front_oblique_right | 17 | 0.70 | 0.70 | 18.50 | Poor | 0.53 | 0.51 | 19.04 | Poor |
| hip_flexion_r | side_right | 9 | 0.52 | 0.46 | 19.01 | Poor | 0.08 | 0.07 | 24.13 | Poor |
| hip_adduction_r | side_left | 12 | 0.95 | 0.94 | 9.80 | Good | 0.93 | 0.88 | 12.31 | Moderate |
| hip_adduction_r | front_oblique_left | 9 | 0.35 | 0.29 | 6.54 | Poor | -0.78 | -0.77 | 9.22 | Poor |
| hip_adduction_r | front_center | 23 | 0.93 | 0.77 | 16.22 | Poor | 0.35 | 0.15 | 29.54 | Poor |
| hip_adduction_r | front_oblique_right | 17 | 0.92 | 0.78 | 17.28 | Poor | 0.92 | 0.81 | 16.16 | Poor |
| hip_adduction_r | side_right | 10 | 0.22 | 0.21 | 19.64 | Poor | 0.25 | 0.24 | 19.18 | Poor |
| knee_angle_r | side_left | 12 | 0.88 | 0.86 | 12.43 | Moderate | 0.88 | 0.87 | 11.55 | Moderate |
| knee_angle_r | front_oblique_left | 21 | 0.91 | 0.78 | 15.60 | Poor | 0.92 | 0.91 | 11.77 | Moderate |
| knee_angle_r | front_oblique_right | 9 | 0.86 | 0.83 | 10.72 | Moderate | 0.91 | 0.89 | 8.05 | Good |
| knee_angle_r | side_right | 10 | 0.81 | 0.81 | 14.24 | Moderate | 0.58 | 0.58 | 21.92 | Poor |
| ankle_angle_r | side_left | 9 | 0.46 | 0.33 | 12.24 | Poor | 0.41 | 0.38 | 13.04 | Poor |
| ankle_angle_r | front_oblique_left | 9 | 0.62 | 0.56 | 10.78 | Moderate | 0.49 | 0.46 | 12.39 | Moderate |
| ankle_angle_r | front_center | 9 | 0.11 | 0.09 | 14.69 | Poor | -0.46 | -0.36 | 17.87 | Poor |
| ankle_angle_r | front_oblique_right | 9 | -0.13 | -0.13 | 19.34 | Poor | 0.27 | 0.25 | 14.37 | Poor |
| ankle_angle_r | side_right | 9 | 0.75 | 0.64 | 9.46 | Good | 0.25 | 0.25 | 15.88 | Poor |
| lumbar_extension | side_left | 12 | 0.87 | 0.83 | 7.25 | Good | 0.91 | 0.85 | 6.46 | Good |
| lumbar_extension | front_oblique_left | 9 | 0.62 | 0.53 | 8.03 | Moderate | 0.40 | 0.36 | 9.56 | Poor |
| lumbar_extension | front_center | 9 | 0.76 | 0.55 | 7.45 | Moderate | 0.52 | 0.48 | 8.84 | Moderate |
| lumbar_extension | front_oblique_right | 17 | 0.80 | 0.63 | 10.18 | Moderate | 0.79 | 0.70 | 9.81 | Good |
| lumbar_extension | side_right | 10 | 0.47 | 0.45 | 13.31 | Moderate | 0.82 | 0.79 | 7.93 | Good |

## v18 vs v20 vs v23 vs v24 head-to-head

| Slot | v17 CCC | v18 CCC | v20 CCC | v23 CCC | **v24 CCC** | Best of (v18/v20/v23/v24) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hip_flexion_r / side_left | 0.60 | 0.45 | 0.70 | 0.63 | **0.50** | v20 |
| hip_flexion_r / front_oblique_left | 0.84 | 0.39 | 0.51 | 0.74 | **0.65** | v23 |
| hip_flexion_r / front_oblique_right | 0.70 | 0.48 | 0.55 | 0.58 | **0.51** | v23 |
| hip_flexion_r / side_right | 0.46 | 0.27 | 0.46 | 0.25 | **0.07** | v20 |
| hip_adduction_r / side_left | 0.94 | 0.71 | 0.85 | 0.93 | **0.88** | v23 |
| hip_adduction_r / front_oblique_left | 0.29 | 0.45 | 0.69 | -0.55 | **-0.77** | v20 |
| hip_adduction_r / front_center | 0.77 | 0.38 | 0.65 | 0.63 | **0.15** | v20 |
| hip_adduction_r / front_oblique_right | 0.78 | 0.63 | 0.84 | 0.79 | **0.81** | v20 |
| hip_adduction_r / side_right | 0.21 | 0.09 | -0.18 | 0.27 | **0.24** | v23 |
| knee_angle_r / side_left | 0.86 | 0.50 | 0.68 | 0.87 | **0.87** | v23 |
| knee_angle_r / front_oblique_left | 0.78 | 0.29 | 0.85 | 0.91 | **0.91** | v23 |
| knee_angle_r / front_oblique_right | 0.83 | 0.11 | 0.61 | 0.49 | **0.89** | v24 |
| knee_angle_r / side_right | 0.81 | 0.08 | 0.35 | 0.79 | **0.58** | v23 |
| ankle_angle_r / side_left | 0.33 | 0.15 | -0.25 | -0.09 | **0.38** | v24 |
| ankle_angle_r / front_oblique_left | 0.56 | -0.28 | 0.24 | 0.14 | **0.46** | v24 |
| ankle_angle_r / front_center | 0.09 | -0.49 | -0.50 | 0.21 | **-0.36** | v23 |
| ankle_angle_r / front_oblique_right | -0.13 | 0.59 | -0.09 | 0.73 | **0.25** | v23 |
| ankle_angle_r / side_right | 0.64 | 0.46 | -0.03 | 0.18 | **0.25** | v18 |
| lumbar_extension / side_left | 0.83 | 0.75 | 0.87 | 0.88 | **0.85** | v23 |
| lumbar_extension / front_oblique_left | 0.53 | 0.71 | 0.19 | 0.31 | **0.36** | v18 |
| lumbar_extension / front_center | 0.55 | 0.43 | 0.46 | 0.41 | **0.48** | v24 |
| lumbar_extension / front_oblique_right | 0.63 | 0.56 | 0.69 | 0.79 | **0.70** | v23 |
| lumbar_extension / side_right | 0.45 | 0.37 | 0.62 | 0.85 | **0.79** | v23 |

## Where v24 beat v23 (LL beat HH2)

| Slot | v23 CCC | v24 CCC | Δ |
| --- | ---: | ---: | ---: |
| ankle_angle_r / side_left | -0.09 | 0.38 | 0.46 |
| knee_angle_r / front_oblique_right | 0.49 | 0.89 | 0.40 |
| ankle_angle_r / front_oblique_left | 0.14 | 0.46 | 0.33 |
| ankle_angle_r / side_right | 0.18 | 0.25 | 0.07 |
| lumbar_extension / front_center | 0.41 | 0.48 | 0.07 |
| lumbar_extension / front_oblique_left | 0.31 | 0.36 | 0.04 |
| hip_adduction_r / front_oblique_right | 0.79 | 0.81 | 0.02 |

## Where v23 beat v24 (HH2 beat LL)

| Slot | v23 CCC | v24 CCC | Δ |
| --- | ---: | ---: | ---: |
| ankle_angle_r / front_center | 0.21 | -0.36 | -0.57 |
| ankle_angle_r / front_oblique_right | 0.73 | 0.25 | -0.48 |
| hip_adduction_r / front_center | 0.63 | 0.15 | -0.48 |
| hip_adduction_r / front_oblique_left | -0.55 | -0.77 | -0.22 |
| knee_angle_r / side_right | 0.79 | 0.58 | -0.21 |
| hip_flexion_r / side_right | 0.25 | 0.07 | -0.18 |
| hip_flexion_r / side_left | 0.63 | 0.50 | -0.13 |
| hip_flexion_r / front_oblique_left | 0.74 | 0.65 | -0.09 |
| lumbar_extension / front_oblique_right | 0.79 | 0.70 | -0.09 |
| hip_flexion_r / front_oblique_right | 0.58 | 0.51 | -0.07 |
| lumbar_extension / side_right | 0.85 | 0.79 | -0.06 |
| hip_adduction_r / side_left | 0.93 | 0.88 | -0.05 |
| lumbar_extension / side_left | 0.88 | 0.85 | -0.03 |
| hip_adduction_r / side_right | 0.27 | 0.24 | -0.02 |
| knee_angle_r / side_left | 0.87 | 0.87 | -0.01 |

## Honest caveats

1. **Layer-3-LOSO-only caveat.** See discipline section. L2 trained on ALL 24 cohort subjects; L3 ridge LOSO at subject level only. Not double-LOSO. Cohort-subject tier promotions are upper bounds.
2. **ROM-aware loss + cross-dataset interaction.** GG2 demonstrated ROM-aware loss on a clean single OpenCap cohort. LL adds it on the combined OpenCap+ASPset cohort. Where ASPset's convention noise exists (lumbar offset, hip_adduction definition), the extrema terms can pin predictions to noisy targets. We mitigated this by dropping ASPset hip_adduction_r supervision; lumbar still inherits some risk.
3. **ASPset hip_adduction_r dropped from LL training** (HH2's recommendation). LL's hip_adduction_r is therefore trained on OpenCap-only ground truth, with ASPset providing shared-trunk representation only.
4. **Ankle GT remains OpenCap-only** (ASPset has no foot KPs). The learned L2 was trained on OpenCap ankle GT; ASPset clips have ankle masked from loss but contribute pose representation.
5. **Per-source target heads not implemented.** A natural next experiment is to add per-source output heads for hip_adduction_r and lumbar_extension to absorb convention mismatch without discarding ASPset data.
6. **No invented numbers.** All CCC / LoA / |r| values were computed from the v24 LOSO build.

## Files

- `harness/learned_layer2_combined_rom_aware.py` — LL Layer 2 trainer (combined cohort + ROM-aware loss)
- `harness/layer3_retrain_on_combined_rom_aware_l2.py` — LL Layer 3 retrain wrapper
- `models/learned_layer2_combined_rom_aware_alldata_v1.pt` — all-data LL L2 checkpoint
- `data/layer3_retrain_combined_rom_aware/per_slot_validity_v24.json` — per-slot v24 validity stats
- `results/deploy_ready_models_v24_combined_rom_aware.json` — v24 deploy candidate (full v17 base + v24 learned-L2 ridge re-fits)

