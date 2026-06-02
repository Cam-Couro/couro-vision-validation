# Couro v14 Deploy: Biomech-Standard Validity Statistics

Bland-Altman 95% Limits of Agreement (LoA), Lin's CCC, and mean bias for every (metric x camera view) slot in the v14 deploy bundle. These are the device-validation stats Oura/Polar/force-plate papers report and that diligence reviewers expect alongside Pearson r.

## Methodology

Per slot we reconstruct the leave-one-subject-out (LOSO) cross-validation that produced the deploy ridge regression (using the same dataset builder as the original trainer — v12/v13/v14/v9_phased/event_anchored/event_anchored_bilateral). Each held-out subject's trials yield paired (predicted_ROM, observed_ROM) values.

**Two aggregation levels are reported per slot:**

- **Per-subject** (the headline, matches Atkinson & Nevill 1998 device-validation convention): trials of each subject are averaged into one (pred, obs) point before Bland-Altman; n = unique subjects.
- **Per-trial** (matches `deploy_ready_models.json` LOSO summary): every trial is a separate point; n = total LOSO test trials. Wider LoA because within-subject trial variability adds in.

Subject-level LoA is the appropriate accuracy claim for an athlete-level metric (one ROM number per session). Trial-level LoA is the lower bound for per-rep accuracy.

Statistics computed per slot:

- **Pearson r** — same as existing doc; measures linear association only.
- **CCC** (Lin 1989) = `2·cov(p, o) / (var(p) + var(o) + (mean(p) − mean(o))²)`. Penalises systematic bias and scale mismatch — what Pearson r misses.
- **Mean bias** = mean(predicted − observed). Sign tells direction of systematic offset.
- **SD(differences)** = SD(predicted − observed). The random-error component.
- **95% LoA** = mean_bias ± 1.96 × SD(differences). 95% of paired differences fall in this range.
- **MAE** = mean(|predicted − observed|).
- **RMSE** = sqrt(mean((predicted − observed)²)).
- **p-value (bias ≠ 0)** — one-sample t-test on the differences.

**Classification (biomech device-validation conventions):**

| Tier | CCC | LoA half-width |
| --- | --- | --- |
| Excellent | > 0.75 | < ±5° |
| Good | 0.60–0.75 | ±5–10° |
| Moderate | 0.40–0.60 | ±10–15° |
| Poor | ≤ 0.40 | > ±15° |

## Headline slots (recommended for validation doc)

Per-subject Bland-Altman + Lin's CCC. n = unique LOSO-held-out subjects.

| Slot | n | r | CCC | bias (°) | 95% LoA (°) | MAE (°) | RMSE (°) | Tier |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| hip_flexion_r / front_oblique_left | 21 | 0.85 | 0.84 | 0.18 | [-11.1, 11.5] | 4.36 | 5.62 | Moderate |
| hip_adduction_r / side_left | 12 | 0.95 | 0.94 | 1.31 | [-8.5, 11.1] | 4.44 | 4.96 | Good |
| knee_angle_r / side_left | 12 | 0.88 | 0.86 | -0.48 | [-12.9, 12.0] | 4.99 | 6.09 | Moderate |
| ankle_angle_r / side_right | 9 | 0.75 | 0.64 | 0.33 | [-9.1, 9.8] | 3.89 | 4.56 | Good |
| lumbar_extension / side_left | 12 | 0.87 | 0.83 | -1.38 | [-8.6, 5.9] | 3.08 | 3.80 | Good |
| hip_flexion_r / front_center | 22 | -0.21 | -0.15 | -9.74 | [-42.6, 23.1] | 12.69 | 19.06 | Poor |
| hip_flexion_r / front_oblique_right | 17 | 0.70 | 0.70 | -0.21 | [-18.7, 18.3] | 7.01 | 9.16 | Poor |

### Same headline slots, per-trial (matches existing deploy doc)

| Slot | n trials | r | CCC | bias (°) | 95% LoA (°) | MAE (°) | RMSE (°) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| hip_flexion_r / front_oblique_left | 432 | 0.85 | 0.84 | 0.03 | [-32.8, 32.9] | 12.78 | 16.73 |
| hip_adduction_r / side_left | 105 | 0.81 | 0.78 | -0.27 | [-25.3, 24.7] | 9.69 | 12.70 |
| knee_angle_r / side_left | 139 | 0.81 | 0.80 | -0.11 | [-24.7, 24.4] | 9.45 | 12.48 |
| ankle_angle_r / side_right | 54 | 0.62 | 0.54 | 0.33 | [-11.4, 12.0] | 4.68 | 5.92 |
| lumbar_extension / side_left | 140 | 0.66 | 0.63 | -0.30 | [-16.2, 15.6] | 5.79 | 8.10 |
| hip_flexion_r / front_center | 615 | 0.78 | 0.75 | 0.03 | [-40.7, 40.7] | 16.25 | 20.75 |
| hip_flexion_r / front_oblique_right | 279 | 0.84 | 0.83 | 0.01 | [-30.9, 30.9] | 12.22 | 15.75 |

## All 25 deploy slots — per-subject Bland-Altman summary

Per-subject aggregation: trials of each subject averaged into one paired point before computing Bland-Altman + CCC.

| Target | View | Approach | n subj | mean obs | mean pred | bias (°) | SD diff (°) | lower LoA (°) | upper LoA (°) | r | CCC | MAE (°) | RMSE (°) | Tier |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ankle_angle_r | side_left | v14_full_dwpose | 9 | 72.06 | 73.03 | 0.97 | 6.24 | -11.27 | 13.20 | 0.46 | 0.33 | 4.67 | 5.96 | Poor |
| ankle_angle_r | front_oblique_left | v14_full_dwpose | 9 | 72.06 | 72.08 | 0.02 | 5.50 | -10.76 | 10.81 | 0.62 | 0.56 | 4.38 | 5.19 | Moderate |
| ankle_angle_r | front_center | event_anchored_bilateral | 9 | 72.06 | 72.70 | 0.64 | 7.49 | -14.05 | 15.32 | 0.11 | 0.09 | 6.42 | 7.09 | Poor |
| ankle_angle_r | front_oblique_right | v14_full_dwpose | 9 | 72.06 | 72.13 | 0.07 | 9.87 | -19.27 | 19.41 | -0.13 | -0.13 | 8.87 | 9.30 | Poor |
| ankle_angle_r | side_right | v14_full_dwpose | 9 | 72.06 | 72.39 | 0.33 | 4.83 | -9.13 | 9.80 | 0.75 | 0.64 | 3.89 | 4.56 | Good |
| hip_adduction_r | side_left | v14_full_dwpose | 12 | 19.80 | 21.11 | 1.31 | 5.00 | -8.49 | 11.10 | 0.95 | 0.94 | 4.44 | 4.96 | Good |
| hip_adduction_r | front_oblique_left | v9_phased | 9 | 11.79 | 11.29 | -0.50 | 3.33 | -7.04 | 6.03 | 0.35 | 0.29 | 2.80 | 3.18 | Poor |
| hip_adduction_r | front_center | v12_combined | 22 | 30.79 | 34.39 | 3.60 | 8.27 | -12.62 | 19.82 | 0.93 | 0.77 | 7.21 | 8.85 | Poor |
| hip_adduction_r | front_oblique_right | v12_combined | 17 | 26.52 | 30.49 | 3.98 | 8.82 | -13.31 | 21.26 | 0.92 | 0.78 | 7.76 | 9.43 | Poor |
| hip_adduction_r | side_right | v14_full_dwpose | 10 | 13.84 | 14.27 | 0.43 | 10.02 | -19.21 | 20.08 | 0.22 | 0.21 | 7.76 | 9.52 | Poor |
| hip_flexion_r | side_left | v12_combined | 12 | 75.93 | 77.66 | 1.73 | 8.28 | -14.51 | 17.97 | 0.62 | 0.60 | 6.28 | 8.12 | Poor |
| hip_flexion_r | front_oblique_left | v13_dwpose_hybrid | 21 | 70.38 | 70.56 | 0.18 | 5.76 | -11.11 | 11.47 | 0.85 | 0.84 | 4.36 | 5.62 | Moderate |
| hip_flexion_r | front_center | v12_combined | 22 | 69.82 | 60.08 | -9.74 | 16.77 | -42.61 | 23.13 | -0.21 | -0.15 | 12.69 | 19.06 | Poor |
| hip_flexion_r | front_oblique_right | v13_dwpose_hybrid | 17 | 70.90 | 70.69 | -0.21 | 9.44 | -18.71 | 18.29 | 0.70 | 0.70 | 7.01 | 9.16 | Poor |
| hip_flexion_r | side_right | event_anchored | 9 | 77.11 | 78.25 | 1.14 | 9.70 | -17.86 | 20.15 | 0.52 | 0.46 | 8.25 | 9.21 | Poor |
| knee_angle_r | side_left | v14_full_dwpose | 12 | 95.99 | 95.51 | -0.48 | 6.34 | -12.91 | 11.96 | 0.88 | 0.86 | 4.99 | 6.09 | Moderate |
| knee_angle_r | front_oblique_left | v12_combined | 21 | 86.38 | 83.18 | -3.20 | 7.96 | -18.81 | 12.40 | 0.91 | 0.78 | 6.95 | 8.40 | Poor |
| knee_angle_r | front_center | v12_combined | 22 | 84.48 | 75.64 | -8.84 | 14.75 | -37.74 | 20.06 | 0.22 | 0.12 | 11.40 | 16.90 | Poor |
| knee_angle_r | front_oblique_right | v9_phased | 8 | 100.82 | 100.97 | 0.15 | 5.47 | -10.57 | 10.87 | 0.86 | 0.83 | 4.53 | 5.12 | Moderate |
| knee_angle_r | side_right | v12_combined | 10 | 99.30 | 99.48 | 0.18 | 7.27 | -14.06 | 14.42 | 0.81 | 0.81 | 5.50 | 6.89 | Moderate |
| lumbar_extension | side_left | v14_full_dwpose | 12 | 33.03 | 31.65 | -1.38 | 3.70 | -8.63 | 5.87 | 0.87 | 0.83 | 3.08 | 3.80 | Good |
| lumbar_extension | front_oblique_left | event_anchored | 9 | 36.26 | 35.96 | -0.29 | 4.10 | -8.32 | 7.73 | 0.62 | 0.53 | 2.88 | 3.87 | Moderate |
| lumbar_extension | front_center | event_anchored | 9 | 36.25 | 36.23 | -0.02 | 3.80 | -7.47 | 7.44 | 0.76 | 0.55 | 2.74 | 3.58 | Moderate |
| lumbar_extension | front_oblique_right | v13_dwpose_hybrid | 17 | 29.73 | 27.18 | -2.55 | 5.19 | -12.72 | 7.63 | 0.80 | 0.63 | 4.14 | 5.64 | Moderate |
| lumbar_extension | side_right | v14_full_dwpose | 10 | 34.66 | 33.08 | -1.58 | 6.79 | -14.88 | 11.73 | 0.47 | 0.45 | 5.20 | 6.63 | Moderate |

## All 25 deploy slots — per-trial summary (matches deploy_ready_models.json)

Every LOSO test trial is one point. Wider LoA than per-subject because within-subject variance is included.

| Target | View | n trials | bias (°) | SD diff (°) | lower LoA (°) | upper LoA (°) | r | CCC | MAE (°) | RMSE (°) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ankle_angle_r | side_left | 54 | 0.97 | 7.13 | -13.00 | 14.93 | 0.41 | 0.36 | 5.88 | 7.13 |
| ankle_angle_r | front_oblique_left | 54 | 0.02 | 6.76 | -13.22 | 13.26 | 0.50 | 0.47 | 5.42 | 6.69 |
| ankle_angle_r | front_center | 46 | 0.10 | 8.03 | -15.64 | 15.83 | 0.15 | 0.14 | 6.46 | 7.94 |
| ankle_angle_r | front_oblique_right | 54 | 0.07 | 10.07 | -19.67 | 19.82 | 0.02 | 0.02 | 8.87 | 9.98 |
| ankle_angle_r | side_right | 54 | 0.33 | 5.97 | -11.37 | 12.03 | 0.62 | 0.54 | 4.68 | 5.92 |
| hip_adduction_r | side_left | 105 | -0.27 | 12.76 | -25.28 | 24.73 | 0.81 | 0.78 | 9.69 | 12.70 |
| hip_adduction_r | front_oblique_left | 48 | -0.12 | 4.18 | -8.31 | 8.07 | 0.11 | 0.10 | 3.15 | 4.14 |
| hip_adduction_r | front_center | 445 | 0.03 | 15.89 | -31.12 | 31.19 | 0.63 | 0.58 | 11.85 | 15.88 |
| hip_adduction_r | front_oblique_right | 212 | 0.11 | 18.15 | -35.46 | 35.69 | 0.58 | 0.52 | 14.19 | 18.11 |
| hip_adduction_r | side_right | 77 | -5.02 | 17.83 | -39.96 | 29.93 | 0.15 | 0.09 | 11.89 | 18.41 |
| hip_flexion_r | side_left | 142 | 0.10 | 16.02 | -31.29 | 31.49 | 0.82 | 0.80 | 12.29 | 15.96 |
| hip_flexion_r | front_oblique_left | 432 | 0.03 | 16.75 | -32.79 | 32.85 | 0.85 | 0.84 | 12.78 | 16.73 |
| hip_flexion_r | front_center | 615 | 0.03 | 20.76 | -40.67 | 40.72 | 0.78 | 0.75 | 16.25 | 20.75 |
| hip_flexion_r | front_oblique_right | 279 | 0.01 | 15.78 | -30.92 | 30.94 | 0.84 | 0.83 | 12.22 | 15.75 |
| hip_flexion_r | side_right | 53 | 0.91 | 10.90 | -20.45 | 22.27 | 0.50 | 0.46 | 9.42 | 10.83 |
| knee_angle_r | side_left | 139 | -0.11 | 12.53 | -24.67 | 24.44 | 0.81 | 0.80 | 9.45 | 12.48 |
| knee_angle_r | front_oblique_left | 440 | -0.13 | 15.36 | -30.23 | 29.97 | 0.78 | 0.75 | 12.31 | 15.34 |
| knee_angle_r | front_center | 615 | 0.18 | 17.08 | -33.30 | 33.67 | 0.70 | 0.67 | 13.00 | 17.07 |
| knee_angle_r | front_oblique_right | 40 | 0.54 | 7.11 | -13.38 | 14.47 | 0.80 | 0.76 | 5.48 | 7.04 |
| knee_angle_r | side_right | 83 | 0.87 | 12.33 | -23.30 | 25.03 | 0.81 | 0.78 | 8.81 | 12.28 |
| lumbar_extension | side_left | 140 | -0.30 | 8.13 | -16.23 | 15.63 | 0.66 | 0.63 | 5.79 | 8.10 |
| lumbar_extension | front_oblique_left | 54 | -0.29 | 4.74 | -9.59 | 9.00 | 0.54 | 0.48 | 3.75 | 4.71 |
| lumbar_extension | front_center | 53 | -0.07 | 4.83 | -9.52 | 9.39 | 0.53 | 0.48 | 3.85 | 4.78 |
| lumbar_extension | front_oblique_right | 287 | 0.01 | 9.10 | -17.83 | 17.84 | 0.58 | 0.53 | 6.53 | 9.08 |
| lumbar_extension | side_right | 84 | 0.96 | 8.94 | -16.56 | 18.47 | 0.50 | 0.46 | 7.24 | 8.93 |

## Where CCC changes the story (per-subject)

Pearson r ignores systematic bias. CCC penalises it. Slots with large (r − CCC) gaps are correlated-but-biased — the model is tracking the shape of the ROM across subjects but offsetting it systematically.

| Slot | r | CCC | r − CCC | bias (°) | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| lumbar_extension / front_center | 0.76 | 0.55 | 0.21 | -0.02 | Scale mismatch (slope ≠ 1) |
| lumbar_extension / front_oblique_right | 0.80 | 0.63 | 0.16 | -2.55 | Scale mismatch (slope ≠ 1) |
| hip_adduction_r / front_center | 0.93 | 0.77 | 0.16 | 3.60 | Scale mismatch (slope ≠ 1) |
| hip_adduction_r / front_oblique_right | 0.92 | 0.78 | 0.14 | 3.98 | Scale mismatch (slope ≠ 1) |
| ankle_angle_r / side_left | 0.46 | 0.33 | 0.14 | 0.97 | Scale mismatch (slope ≠ 1) |
| knee_angle_r / front_oblique_left | 0.91 | 0.78 | 0.13 | -3.20 | Scale mismatch (slope ≠ 1) |
| knee_angle_r / front_center | 0.22 | 0.12 | 0.10 | -8.84 | Large systematic bias — calibration needed |
| ankle_angle_r / side_right | 0.75 | 0.64 | 0.10 | 0.33 | Scale mismatch (slope ≠ 1) |

## Where the per-subject view changes the story

Per-trial Pearson r mixes between-subject and within-subject variance. Per-subject Pearson r asks: does this model differentiate athletes? Slots where per-trial r is good but per-subject r collapses are the model tracking exercise shape (e.g. DJ trial-to-trial variation) but failing to rank athletes — a common failure mode for ridge regression on small subject-pool datasets.

| Slot | per-trial r | per-subject r | Δ (trial − subj) | per-subject CCC | bias (°) | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hip_flexion_r / front_center | 0.78 | -0.21 | 0.99 | -0.15 | -9.74 | Model ranks athletes wrong — between-subject signal is absent |
| knee_angle_r / front_center | 0.70 | 0.22 | 0.48 | 0.12 | -8.84 | Tracks reps but not athletes — limited discriminative validity |
| hip_flexion_r / side_left | 0.82 | 0.62 | 0.20 | 0.60 | 1.73 | Some between-subject signal, mostly within-subject |
| ankle_angle_r / front_oblique_right | 0.02 | -0.13 | 0.15 | -0.13 | 0.07 | Model ranks athletes wrong — between-subject signal is absent |

## Slots warranting the 'validated' label

By biomech device-validation conventions (CCC > 0.60 AND 95% LoA half-width < ±10°), the following slots are validated:

- **hip_adduction_r / side_left** (n=12, r=0.95, CCC=0.94, 95% LoA = [-8.5°, 11.1°], tier=Good)
- **ankle_angle_r / side_right** (n=9, r=0.75, CCC=0.64, 95% LoA = [-9.1°, 9.8°], tier=Good)
- **lumbar_extension / side_left** (n=12, r=0.87, CCC=0.83, 95% LoA = [-8.6°, 5.9°], tier=Good)

Moderate tier (CCC 0.40–0.60, LoA ±10–15°) — usable with caveats:

- hip_flexion_r / front_oblique_left (n=21, r=0.85, CCC=0.84, 95% LoA = [-11.1°, 11.5°])
- knee_angle_r / side_left (n=12, r=0.88, CCC=0.86, 95% LoA = [-12.9°, 12.0°])
- knee_angle_r / front_oblique_right (n=8, r=0.86, CCC=0.83, 95% LoA = [-10.6°, 10.9°])
- knee_angle_r / side_right (n=10, r=0.81, CCC=0.81, 95% LoA = [-14.1°, 14.4°])
- ankle_angle_r / front_oblique_left (n=9, r=0.62, CCC=0.56, 95% LoA = [-10.8°, 10.8°])
- lumbar_extension / front_oblique_left (n=9, r=0.62, CCC=0.53, 95% LoA = [-8.3°, 7.7°])
- lumbar_extension / front_center (n=9, r=0.76, CCC=0.55, 95% LoA = [-7.5°, 7.4°])
- lumbar_extension / front_oblique_right (n=17, r=0.80, CCC=0.63, 95% LoA = [-12.7°, 7.6°])
- lumbar_extension / side_right (n=10, r=0.47, CCC=0.45, 95% LoA = [-14.9°, 11.7°])

## MPI-INF-3DHP rear-view cohort (cam7, n=8 subjects)

Per-frame joint-angle validation against MPI-INF-3DHP mocap on the rear oblique camera (cam7, azimuth −62°). Methodology differs from the deploy ROM slots above: this is a per-frame waveform-shape correlation, not a per-trial ROM regression. Reported as per-subject Pearson r summarised across the 8 subjects.

### Per-subject Pearson r (waveform shape agreement, |r|)

| Metric | n subj | n frames | mean abs(r) ± SD | range |
| --- | ---: | ---: | --- | --- |
| hip_flexion_r | 8 | 28718 | 0.87 ± 0.04 | [0.81, 0.91] |
| hip_flexion_l | 8 | 30611 | 0.86 ± 0.07 | [0.75, 0.92] |
| knee_angle_r | 8 | 27921 | 0.79 ± 0.12 | [0.51, 0.89] |
| knee_angle_l | 8 | 30605 | 0.75 ± 0.11 | [0.59, 0.88] |
| ankle_angle_r | 8 | 26953 | 0.23 ± 0.13 | [0.05, 0.44] |
| ankle_angle_l | 8 | 30125 | 0.24 ± 0.08 | [0.12, 0.36] |
| hip_adduction_r | 8 | 29095 | 0.41 ± 0.12 | [0.26, 0.59] |
| hip_adduction_l | 8 | 31238 | 0.39 ± 0.12 | [0.22, 0.54] |
| lumbar_extension | 8 | 31084 | 0.83 ± 0.07 | [0.71, 0.91] |

Notes: |r| is reported because some metrics flip sign with coordinate convention (an absolute correlation of 0.9 means the waveform is tracked, just inverted). The MPI rear validation is a waveform-shape claim — it does not warrant absolute-angle Bland-Altman because the predicted angle streams are not calibrated to MPI's mocap zero pose. A Bland-Altman accuracy claim on MPI rear would require a per-subject calibration step that the current pipeline does not perform.

## Notes

- All stats computed on LOSO-CV held-out subjects, n = number of unique subjects in that slot.
- For ankle_angle_r slots, training is OpenCap-only (ASPset has no foot markers).
- For slots where the LOSO rerun was skipped (e.g. cost-prohibitive combined datasets), CCC and MAE are reported as `—` and Pearson r / LoA come from the deploy_ready_models.json summary stats.
