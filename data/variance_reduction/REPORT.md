# Variance Reduction Pass for Moderate-Tier Layer 3 Slots

Three Moderate-tier slots whose CCC already exceeds 0.83 but whose Bland-Altman LoA half-width sits over the Good-tier gate of +/- 10 degrees. We test three levers to tighten LoA. Tier-promotion claims require LOSO-CV CCC > 0.60 AND LoA half-width < 10 degrees on the full subject cohort.

## Targets

| Slot | n subjects | CCC | LoA half-width (deg) | Gap to Good (deg) |
| --- | ---: | ---: | ---: | ---: |
| knee_angle_r / front_oblique_right / v9_phased | 8 | 0.83 | 10.72 | 0.72 |
| hip_flexion_r / front_oblique_left / v13_dwpose_hybrid | 21 | 0.84 | 11.29 | 1.29 |
| knee_angle_r / side_left / v14_full_dwpose | 12 | 0.86 | 12.43 | 2.43 |

## Per-slot results

### knee_angle_r / front_oblique_right / v9_phased

Baseline (this run, LOSO per-subject): n=8, CCC=0.83, LoA half-width=10.72 deg, MAE=4.53 deg, bias=0.15 deg.

**Lever 1: outlier subject leave-one-out (diagnostic, NOT validation)**

- Highest-leverage subject: `subject11` (abs error in LOSO = 7.22 deg)
- After excluding that subject: CCC=0.87, LoA half-width=9.87 deg (crosses Good gate)
- Note: dropping subjects post hoc is a leverage diagnostic, not a validated tier change. Promotion claims below come only from Levers 2 and 3 on the full subject cohort.

| Dropped subject | n remain | CCC | LoA half (deg) | abs err on drop (deg) |
| --- | ---: | ---: | ---: | ---: |
| subject11 | 7 | 0.87 | 9.87 | 7.22 |
| subject7 | 7 | 0.86 | 9.92 | 6.83 |
| subject5 | 7 | 0.75 | 10.26 | 6.41 |
| subject3 | 7 | 0.84 | 10.30 | 6.03 |
| subject8 | 7 | 0.85 | 10.77 | 5.11 |
| subject9 | 7 | 0.79 | 11.47 | 1.70 |
| subject2 | 7 | 0.80 | 11.49 | 1.52 |
| subject10 | 7 | 0.82 | 11.50 | 1.44 |

**Lever 2: confidence-weighted phased-angle features**

- LOSO CCC = 0.80 (baseline 0.83), LoA half-width = 11.26 deg (baseline 10.72 deg).
- delta LoA = 0.54 deg, delta CCC = -0.03.
- Tier verdict (full cohort): Moderate (no promotion)

**Lever 3: temporal smoothing sweep (Savgol window 0, 5, 9 frames)**

| window | n subj | CCC | LoA half (deg) | MAE (deg) | tier |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 8 | 0.83 | 10.72 | 4.53 | Moderate |
| 5 | 8 | 0.82 | 11.10 | 4.69 | Moderate |
| 9 | 8 | 0.81 | 11.40 | 4.76 | Moderate |

- Best window = 0: CCC=0.83, LoA half-width=10.72 deg (delta LoA = 0.00 deg, delta CCC = 0.00).
- Tier verdict (full cohort): Moderate (no promotion)

### hip_flexion_r / front_oblique_left / v13_dwpose_hybrid

Baseline (this run, LOSO per-subject): n=21, CCC=0.84, LoA half-width=11.29 deg, MAE=4.36 deg, bias=0.18 deg.

**Lever 1: outlier subject leave-one-out (diagnostic, NOT validation)**

- Highest-leverage subject: `opencap_subject11` (abs error in LOSO = 14.27 deg)
- After excluding that subject: CCC=0.88, LoA half-width=9.59 deg (crosses Good gate)
- Note: dropping subjects post hoc is a leverage diagnostic, not a validated tier change. Promotion claims below come only from Levers 2 and 3 on the full subject cohort.

| Dropped subject | n remain | CCC | LoA half (deg) | abs err on drop (deg) |
| --- | ---: | ---: | ---: | ---: |
| opencap_subject11 | 20 | 0.88 | 9.59 | 14.27 |
| aspset_eb61 | 20 | 0.87 | 10.26 | 11.47 |
| opencap_subject2 | 20 | 0.83 | 10.51 | 10.40 |
| opencap_subject5 | 20 | 0.85 | 11.24 | 5.91 |
| aspset_b3c1 | 20 | 0.84 | 11.26 | 6.07 |
| aspset_5ff4 | 20 | 0.85 | 11.30 | 5.70 |
| opencap_subject9 | 20 | 0.84 | 11.34 | 5.25 |
| opencap_subject8 | 20 | 0.85 | 11.44 | 4.07 |
| opencap_subject3 | 20 | 0.84 | 11.47 | 3.34 |
| aspset_14ce | 20 | 0.85 | 11.48 | 3.57 |
| aspset_4d9e | 20 | 0.84 | 11.49 | 2.94 |
| opencap_subject4 | 20 | 0.84 | 11.50 | 2.85 |
| aspset_b8e1 | 20 | 0.84 | 11.52 | 2.82 |
| opencap_subject10 | 20 | 0.82 | 11.52 | 2.46 |
| aspset_bae6 | 20 | 0.84 | 11.54 | 2.46 |
| aspset_11ac | 20 | 0.83 | 11.55 | 1.77 |
| aspset_c9f8 | 20 | 0.83 | 11.55 | 1.64 |
| opencap_subject7 | 20 | 0.83 | 11.56 | 1.66 |
| aspset_7b5d | 20 | 0.84 | 11.57 | 1.44 |
| aspset_fb7c | 20 | 0.83 | 11.57 | 1.03 |
| aspset_d26c | 20 | 0.84 | 11.58 | 0.37 |

**Lever 2: confidence-weighted phased-angle features**

- LOSO CCC = 0.82 (baseline 0.84), LoA half-width = 11.57 deg (baseline 11.29 deg).
- delta LoA = 0.28 deg, delta CCC = -0.02.
- Tier verdict (full cohort): Moderate (no promotion)

**Lever 3: temporal smoothing sweep (Savgol window 0, 5, 9 frames)**

| window | n subj | CCC | LoA half (deg) | MAE (deg) | tier |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 21 | 0.84 | 11.29 | 4.36 | Moderate |
| 5 | 21 | 0.84 | 11.49 | 4.45 | Moderate |
| 9 | 21 | 0.85 | 11.36 | 4.38 | Moderate |

- Best window = 0: CCC=0.84, LoA half-width=11.29 deg (delta LoA = 0.00 deg, delta CCC = 0.00).
- Tier verdict (full cohort): Moderate (no promotion)

### knee_angle_r / side_left / v14_full_dwpose

Baseline (this run, LOSO per-subject): n=12, CCC=0.86, LoA half-width=12.43 deg, MAE=4.99 deg, bias=-0.48 deg.

**Lever 1: outlier subject leave-one-out (diagnostic, NOT validation)**

- Highest-leverage subject: `opencap_subject5` (abs error in LOSO = 12.75 deg)
- After excluding that subject: CCC=0.91, LoA half-width=9.84 deg (crosses Good gate)
- Note: dropping subjects post hoc is a leverage diagnostic, not a validated tier change. Promotion claims below come only from Levers 2 and 3 on the full subject cohort.

| Dropped subject | n remain | CCC | LoA half (deg) | abs err on drop (deg) |
| --- | ---: | ---: | ---: | ---: |
| opencap_subject5 | 11 | 0.91 | 9.84 | 12.75 |
| opencap_subject11 | 11 | 0.89 | 11.52 | 8.96 |
| opencap_subject2 | 11 | 0.88 | 11.83 | 8.95 |
| opencap_subject7 | 11 | 0.87 | 12.48 | 6.34 |
| opencap_subject4 | 11 | 0.87 | 12.64 | 5.41 |
| aspset_7b5d | 11 | 0.83 | 12.68 | 4.20 |
| opencap_subject8 | 11 | 0.87 | 12.88 | 3.65 |
| opencap_subject9 | 11 | 0.84 | 12.89 | 3.50 |
| opencap_subject10 | 11 | 0.85 | 13.00 | 1.15 |
| aspset_4d9e | 11 | 0.84 | 13.00 | 2.09 |
| opencap_subject3 | 11 | 0.86 | 13.02 | 1.67 |
| aspset_4448 | 11 | 0.84 | 13.03 | 1.17 |

**Lever 2: confidence-weighted phased-angle features**

- LOSO CCC = 0.87 (baseline 0.86), LoA half-width = 11.74 deg (baseline 12.43 deg).
- delta LoA = -0.69 deg, delta CCC = 0.00.
- Tier verdict (full cohort): Moderate (no promotion)

**Lever 3: temporal smoothing sweep (Savgol window 0, 5, 9 frames)**

| window | n subj | CCC | LoA half (deg) | MAE (deg) | tier |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 12 | 0.86 | 12.43 | 4.99 | Moderate |
| 5 | 12 | 0.87 | 11.87 | 4.94 | Moderate |
| 9 | 12 | 0.89 | 11.20 | 4.67 | Moderate |

- Best window = 9: CCC=0.89, LoA half-width=11.20 deg (delta LoA = -1.24 deg, delta CCC = 0.03).
- Tier verdict (full cohort): Moderate (no promotion)

## Summary

| Slot | Baseline tier | Lever 2 tier | Lever 3 best tier | Best lever | Best LoA half (deg) |
| --- | --- | --- | --- | --- | ---: |
| knee_angle_r / front_oblique_right | Moderate | Moderate | Moderate | lever3_smoothing(w=0) | 10.72 |
| hip_flexion_r / front_oblique_left | Moderate | Moderate | Moderate | lever3_smoothing(w=0) | 11.29 |
| knee_angle_r / side_left | Moderate | Moderate | Moderate | lever3_smoothing(w=9) | 11.20 |

## Honest verdict

- **knee_angle_r / front_oblique_right**: no tier promotion on full cohort. Lever 1 diagnostic: excluding subject `subject11` (abs LOSO error 7.22 deg) would cross the gate, but this is post-hoc and does NOT warrant a Good-tier claim. Subject warrants a clip-level audit to determine if there's a systematic cause.
- **hip_flexion_r / front_oblique_left**: no tier promotion on full cohort. Lever 1 diagnostic: excluding subject `opencap_subject11` (abs LOSO error 14.27 deg) would cross the gate, but this is post-hoc and does NOT warrant a Good-tier claim. Subject warrants a clip-level audit to determine if there's a systematic cause.
- **knee_angle_r / side_left**: no tier promotion on full cohort. Lever 1 diagnostic: excluding subject `opencap_subject5` (abs LOSO error 12.75 deg) would cross the gate, but this is post-hoc and does NOT warrant a Good-tier claim. Subject warrants a clip-level audit to determine if there's a systematic cause.

No slot crossed the Good-tier gate on the full subject cohort. The variance is real (not driven by a single outlier on at least some slots) and is not eliminated by confidence weighting or smoothing parameter tuning. The outlier diagnostics (Lever 1) surface candidates for further clip-level investigation but do not constitute validation of any slot at the Good tier.

## Constraints respected

- LOSO discipline preserved on every tier-change claim.
- Single phone camera only.
- No modification of v15/v16/v17 or biomech_validity_stats source files.
- Outlier exclusion treated as a diagnostic, not a validation method.

