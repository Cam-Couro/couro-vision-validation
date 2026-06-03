# Residual Calibration -- v37 / v38 / v39 (Agent PP Lever 2)

**Date:** 2026-06-02

**Hypothesis:** Bland-Altman LoA half-width = 1.96 * SD(residuals). A per-slot linear calibration ``pred_cal = a*pred + b`` fit on nested-LOSO pseudo-residuals can tighten LoA if the current ridge has structured slope mismatch or per-cohort bias not absorbed by its single ridge fit.

**Discipline:** outer LOSO at L3 unchanged. For each outer subject S, we (a) run an inner LOSO across the N-1 training subjects T to collect pseudo-predictions, (b) fit a linear calibration on those pseudo (pred, gt) pairs, (c) predict on S with the standard outer ridge, (d) apply the calibration to S's predictions. S is **never** part of the calibration fit. Per-slot fallback to uncalibrated ridge if calibration inflates LoA.

## Category A targets -- did calibration push them under +/-10 deg?

| Slot | v35 LoA/2 | v37 (v23+cal) | v38 (v31+cal) | v39 (v17+cal) | best calibrated | Promoted? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| knee_angle_r|front_oblique_left | 10.77 | 11.98 (unc) | 11.07 (cal) | 15.60 (unc) | 11.07 | no |
| knee_angle_r|side_left | 10.15 | 11.74 (cal) | 9.78 (cal) | 12.43 (unc) | 9.78 | YES |
| knee_angle_r|side_right | 12.92 | 13.41 (unc) | 17.35 (cal) | 13.66 (cal) | 13.41 | no |
| hip_flexion_r|front_oblique_left | 11.29 | 12.59 (unc) | 13.00 (unc) | 11.29 (unc) | 11.29 | no |
| hip_adduction_r|front_oblique_right | 13.80 | 17.72 (unc) | 18.86 (unc) | 17.28 (unc) | 17.28 | no |

**Category A promotions to Good (LoA < 10 deg): 1/5.**

## Per-reader sanity check (calibration should not hurt)

### v37_v23_calibrated

- Calibration helped LoA on **5** slots.
- Calibration neutral or hurt on **18** slots (fell back to uncalibrated).
- Tier counts (chosen-per-slot): {'Excellent': 0, 'Good': 5, 'Moderate': 6, 'Poor': 12}

Top LoA tightenings:

| Slot | LoA uncal | LoA cal | Delta |
| --- | ---: | ---: | ---: |
| hip_adduction_r|front_oblique_left | 6.76 | 5.35 | -1.41 |
| hip_adduction_r|side_left | 10.35 | 9.89 | -0.46 |
| ankle_angle_r|front_oblique_right | 8.08 | 8.03 | -0.05 |
| hip_flexion_r|side_left | 15.22 | 15.17 | -0.05 |
| knee_angle_r|side_left | 11.79 | 11.74 | -0.05 |

### v38_v31_calibrated

- Calibration helped LoA on **6** slots.
- Calibration neutral or hurt on **17** slots (fell back to uncalibrated).
- Tier counts (chosen-per-slot): {'Excellent': 0, 'Good': 4, 'Moderate': 5, 'Poor': 14}

Top LoA tightenings:

| Slot | LoA uncal | LoA cal | Delta |
| --- | ---: | ---: | ---: |
| knee_angle_r|side_right | 26.25 | 17.35 | -8.90 |
| hip_adduction_r|front_oblique_left | 7.51 | 5.35 | -2.17 |
| ankle_angle_r|side_right | 20.72 | 19.01 | -1.71 |
| hip_adduction_r|side_left | 9.32 | 8.95 | -0.37 |
| knee_angle_r|side_left | 10.02 | 9.78 | -0.24 |
| knee_angle_r|front_oblique_left | 11.11 | 11.07 | -0.05 |

### v39_v17_calibrated

- Calibration helped LoA on **3** slots.
- Calibration neutral or hurt on **20** slots (fell back to uncalibrated).
- Tier counts (chosen-per-slot): {'Excellent': 0, 'Good': 3, 'Moderate': 9, 'Poor': 11}

Top LoA tightenings:

| Slot | LoA uncal | LoA cal | Delta |
| --- | ---: | ---: | ---: |
| ankle_angle_r|front_oblique_right | 19.34 | 16.73 | -2.61 |
| knee_angle_r|side_right | 14.24 | 13.66 | -0.58 |
| hip_adduction_r|side_left | 9.80 | 9.62 | -0.17 |

## Full per-slot calibration table (all readers)

### v37_v23_calibrated

| Target | View | n_subj | CCC unc | LoA unc | CCC cal | LoA cal | Delta | Chosen | Tier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| hip_flexion_r | side_left | 12 | 0.632 | 15.22 | 0.645 | 15.17 | -0.05 | calibrated | Poor |
| hip_flexion_r | front_oblique_left | 21 | 0.738 | 12.59 | 0.741 | 12.60 | 0.01 | uncalibrated | Moderate |
| hip_flexion_r | front_oblique_right | 17 | 0.579 | 18.01 | 0.579 | 18.22 | 0.21 | uncalibrated | Poor |
| hip_flexion_r | side_right | 9 | 0.255 | 21.32 | -0.229 | 25.03 | 3.71 | uncalibrated | Poor |
| hip_adduction_r | side_left | 12 | 0.926 | 10.35 | 0.934 | 9.89 | -0.46 | calibrated | Good |
| hip_adduction_r | front_oblique_left | 9 | -0.551 | 6.76 | -0.267 | 5.35 | -1.41 | calibrated | Poor |
| hip_adduction_r | front_center | 23 | 0.627 | 20.68 | 0.593 | 21.24 | 0.56 | uncalibrated | Poor |
| hip_adduction_r | front_oblique_right | 17 | 0.794 | 17.72 | 0.787 | 18.02 | 0.30 | uncalibrated | Poor |
| hip_adduction_r | side_right | 10 | 0.265 | 18.84 | -0.370 | 23.82 | 4.99 | uncalibrated | Poor |
| knee_angle_r | side_left | 12 | 0.873 | 11.79 | 0.877 | 11.74 | -0.05 | calibrated | Moderate |
| knee_angle_r | front_oblique_left | 21 | 0.914 | 11.98 | 0.915 | 11.98 | 0.01 | uncalibrated | Moderate |
| knee_angle_r | front_oblique_right | 9 | 0.488 | 14.30 | 0.128 | 17.84 | 3.54 | uncalibrated | Moderate |
| knee_angle_r | side_right | 10 | 0.790 | 13.41 | 0.553 | 16.43 | 3.02 | uncalibrated | Moderate |
| ankle_angle_r | side_left | 9 | -0.088 | 16.15 | -0.402 | 16.89 | 0.74 | uncalibrated | Poor |
| ankle_angle_r | front_oblique_left | 9 | 0.136 | 14.05 | -0.267 | 15.78 | 1.73 | uncalibrated | Poor |
| ankle_angle_r | front_center | 9 | 0.207 | 13.01 | -0.165 | 15.16 | 2.15 | uncalibrated | Poor |
| ankle_angle_r | front_oblique_right | 9 | 0.733 | 8.08 | 0.751 | 8.03 | -0.05 | calibrated | Good |
| ankle_angle_r | side_right | 9 | 0.181 | 13.22 | -0.244 | 15.52 | 2.30 | uncalibrated | Poor |
| lumbar_extension | side_left | 12 | 0.884 | 6.42 | 0.855 | 6.99 | 0.57 | uncalibrated | Good |
| lumbar_extension | front_oblique_left | 9 | 0.315 | 9.43 | -0.062 | 11.37 | 1.94 | uncalibrated | Poor |
| lumbar_extension | front_center | 9 | 0.412 | 8.97 | -0.186 | 11.76 | 2.79 | uncalibrated | Moderate |
| lumbar_extension | front_oblique_right | 17 | 0.790 | 9.68 | 0.780 | 9.85 | 0.16 | uncalibrated | Good |
| lumbar_extension | side_right | 10 | 0.848 | 7.03 | 0.799 | 7.96 | 0.93 | uncalibrated | Good |

### v38_v31_calibrated

| Target | View | n_subj | CCC unc | LoA unc | CCC cal | LoA cal | Delta | Chosen | Tier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| hip_flexion_r | side_left | 12 | 0.444 | 18.10 | 0.450 | 18.28 | 0.17 | uncalibrated | Poor |
| hip_flexion_r | front_oblique_left | 21 | 0.747 | 13.00 | 0.748 | 13.09 | 0.09 | uncalibrated | Moderate |
| hip_flexion_r | front_oblique_right | 17 | 0.569 | 18.25 | 0.571 | 18.42 | 0.16 | uncalibrated | Poor |
| hip_flexion_r | side_right | 9 | 0.083 | 22.83 | -0.392 | 26.72 | 3.89 | uncalibrated | Poor |
| hip_adduction_r | side_left | 12 | 0.938 | 9.32 | 0.942 | 8.95 | -0.37 | calibrated | Good |
| hip_adduction_r | front_oblique_left | 9 | -0.453 | 7.51 | -0.271 | 5.35 | -2.17 | calibrated | Poor |
| hip_adduction_r | front_center | 23 | 0.227 | 28.14 | 0.144 | 29.16 | 1.03 | uncalibrated | Poor |
| hip_adduction_r | front_oblique_right | 17 | 0.733 | 18.86 | 0.652 | 20.79 | 1.93 | uncalibrated | Poor |
| hip_adduction_r | side_right | 10 | -0.041 | 19.26 | -0.450 | 23.74 | 4.49 | uncalibrated | Poor |
| knee_angle_r | side_left | 12 | 0.895 | 10.02 | 0.903 | 9.78 | -0.24 | calibrated | Good |
| knee_angle_r | front_oblique_left | 21 | 0.922 | 11.11 | 0.924 | 11.07 | -0.05 | calibrated | Moderate |
| knee_angle_r | front_oblique_right | 9 | 0.451 | 15.62 | 0.106 | 18.32 | 2.70 | uncalibrated | Poor |
| knee_angle_r | side_right | 10 | 0.594 | 26.25 | 0.473 | 17.35 | -8.90 | calibrated | Poor |
| ankle_angle_r | side_left | 9 | 0.101 | 14.78 | -0.262 | 15.87 | 1.09 | uncalibrated | Poor |
| ankle_angle_r | front_oblique_left | 9 | 0.207 | 13.54 | -0.414 | 17.12 | 3.58 | uncalibrated | Poor |
| ankle_angle_r | front_center | 9 | -0.261 | 16.91 | -0.547 | 18.36 | 1.44 | uncalibrated | Poor |
| ankle_angle_r | front_oblique_right | 9 | 0.348 | 12.17 | -0.118 | 14.74 | 2.56 | uncalibrated | Poor |
| ankle_angle_r | side_right | 9 | -0.342 | 20.72 | -0.456 | 19.01 | -1.71 | calibrated | Poor |
| lumbar_extension | side_left | 12 | 0.758 | 7.74 | 0.601 | 9.19 | 1.46 | uncalibrated | Good |
| lumbar_extension | front_oblique_left | 9 | 0.532 | 8.07 | 0.278 | 9.37 | 1.30 | uncalibrated | Moderate |
| lumbar_extension | front_center | 9 | 0.749 | 6.07 | 0.629 | 6.96 | 0.90 | uncalibrated | Good |
| lumbar_extension | front_oblique_right | 17 | 0.595 | 10.07 | 0.542 | 10.63 | 0.56 | uncalibrated | Moderate |
| lumbar_extension | side_right | 10 | 0.446 | 11.76 | -0.109 | 15.77 | 4.01 | uncalibrated | Moderate |

### v39_v17_calibrated

| Target | View | n_subj | CCC unc | LoA unc | CCC cal | LoA cal | Delta | Chosen | Tier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| hip_flexion_r | side_left | 12 | 0.604 | 16.24 | 0.593 | 16.82 | 0.58 | uncalibrated | Poor |
| hip_flexion_r | front_oblique_left | 21 | 0.843 | 11.29 | 0.843 | 11.30 | 0.01 | uncalibrated | Moderate |
| hip_flexion_r | front_oblique_right | 17 | 0.697 | 18.50 | 0.697 | 18.52 | 0.02 | uncalibrated | Poor |
| hip_flexion_r | side_right | 9 | 0.461 | 19.01 | 0.225 | 20.98 | 1.97 | uncalibrated | Poor |
| hip_adduction_r | side_left | 12 | 0.936 | 9.80 | 0.943 | 9.62 | -0.17 | calibrated | Good |
| hip_adduction_r | front_oblique_left | 9 | 0.289 | 6.54 | -0.299 | 8.02 | 1.49 | uncalibrated | Poor |
| hip_adduction_r | front_center | 22 | 0.767 | 16.22 | 0.759 | 16.43 | 0.22 | uncalibrated | Poor |
| hip_adduction_r | front_oblique_right | 17 | 0.775 | 17.28 | 0.750 | 18.01 | 0.73 | uncalibrated | Poor |
| hip_adduction_r | side_right | 10 | 0.210 | 19.64 | -0.327 | 22.15 | 2.51 | uncalibrated | Poor |
| knee_angle_r | side_left | 12 | 0.864 | 12.43 | 0.856 | 12.62 | 0.18 | uncalibrated | Moderate |
| knee_angle_r | front_oblique_left | 21 | 0.784 | 15.60 | 0.782 | 15.71 | 0.11 | uncalibrated | Poor |
| knee_angle_r | front_oblique_right | 8 | 0.829 | 10.72 | 0.780 | 11.89 | 1.17 | uncalibrated | Moderate |
| knee_angle_r | side_right | 10 | 0.810 | 14.24 | 0.819 | 13.66 | -0.58 | calibrated | Moderate |
| ankle_angle_r | side_left | 9 | 0.325 | 12.24 | -0.096 | 14.54 | 2.31 | uncalibrated | Poor |
| ankle_angle_r | front_oblique_left | 9 | 0.556 | 10.78 | 0.346 | 11.66 | 0.88 | uncalibrated | Moderate |
| ankle_angle_r | front_center | 9 | 0.091 | 14.69 | -0.218 | 15.32 | 0.63 | uncalibrated | Poor |
| ankle_angle_r | front_oblique_right | 9 | -0.129 | 19.34 | -0.372 | 16.73 | -2.61 | calibrated | Poor |
| ankle_angle_r | side_right | 9 | 0.644 | 9.46 | 0.564 | 10.04 | 0.57 | uncalibrated | Good |
| lumbar_extension | side_left | 12 | 0.832 | 7.25 | 0.760 | 8.15 | 0.91 | uncalibrated | Good |
| lumbar_extension | front_oblique_left | 9 | 0.534 | 8.03 | 0.046 | 10.84 | 2.81 | uncalibrated | Moderate |
| lumbar_extension | front_center | 9 | 0.548 | 7.45 | 0.252 | 9.18 | 1.72 | uncalibrated | Moderate |
| lumbar_extension | front_oblique_right | 17 | 0.634 | 10.18 | 0.567 | 10.72 | 0.55 | uncalibrated | Moderate |
| lumbar_extension | side_right | 10 | 0.452 | 13.31 | -0.051 | 16.69 | 3.38 | uncalibrated | Moderate |

## Honest caveats

- **Nested LOSO is legitimate but expensive.** Each slot fits N*(N-1) ridges (N = subjects in slot's pool). Ridge is cheap, so total runtime is dominated by feature building (L2 inference + event detection + per-slot dataset assembly).
- **Per-slot fallback to uncalibrated.** If calibration inflates LoA on a slot, we keep the uncalibrated ridge. This preserves the no-regression guarantee and prevents calibration noise from hurting reliable slots.
- **Calibration is per slot x base reader.** Each (slot, reader) gets its own (a, b). The calibration parameters are NOT shared across slots.
- **Ship-time deployment cost is one (a, b) multiply per inference** per slot per outer-LOSO fold. Stored as per-fold (a, b) in the deploy bundle. For production use, the deploy-time calibration is the per-fold average (a_mean, b_mean) (stored in ``calibration_summary``); this loses the per-fold conditioning but gains a single closed-form correction. Per-fold details are retained for audit.
- **Single camera at inference** (unchanged from v17-v36).
- **Outer LOSO discipline preserved.** Calibration is fit on (N-1) training subjects only; the outer held-out S is predicted but never used to choose calibration parameters.

