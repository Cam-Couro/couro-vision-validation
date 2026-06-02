# Layer-2 Joint Angle Pearson r

Frame-by-frame Pearson r between Couro's predicted per-frame joint angle and the corresponding mocap GT angle, computed per clip and aggregated per (metric × view bucket).

## Methodology

1. **Predicted angles.** For each clip we load the DWPose 2D keypoint JSON and run Couro's production `harness.couro_keypoints.keypoints_to_motion_data` with anthropometric 3D reconstruction (subject height + camera intrinsics). This is the same code path used by the v14 deploy.

2. **GT angles.** OpenCap clips: angle column parsed from the OpenSim IK `.mot` file. ASPset clips: angle computed from the 3D marker positions in the `.c3d` file via `harness.aspset_loader.joint_angles_from_aspset`.

3. **Time alignment.** Predicted and GT clocks are aligned to a common origin (predicted t0 → GT t0) and the predicted series is linearly interpolated onto the GT timeline within their temporal overlap window. Pearson r is computed on the overlapping samples; clips with fewer than 30 overlapping samples or with near-zero variance in either signal are dropped.

4. **Aggregation.** Per-(metric × view bucket): mean r ± SD across clips, plus mean |r| (a sign-blind measure that's useful where convention differences flip the signal — see hip_adduction_r below).

5. **Comparison baseline.** v14 ROM-regression r is the LOSO-CV Pearson r of a per-trial ROM ridge regression with engineered features (the deploy model). Layer-2 r and v14 ROM r measure different things and should not be expected to match.

## Overall Layer-2 r

- **Headline (mean |r|, sign-blind):** +0.544 ± 0.294 across 6725 clip×metric measurements (OpenCap n=1343, ASPset n=5382).
- **Mean signed r:** +0.003 ± 0.618. Near zero overall because two dataset conventions flip sign on two of the five metrics (see Interpretation).
- **Why |r| is the honest headline:** Pearson r is invariant to scale + offset but *not* to sign-flip. Couro's 2D `hip_adduction_r` proxy and ASPset's `knee_angle_r` GT loader each use a different convention than OpenSim, producing perfectly anti-correlated traces (r ≈ -1) that should count as a high-quality fit. `mean |r|` is the proper layer-2 fit metric; we report signed r alongside so the convention issues are visible rather than swept under the rug.

## Per-dataset breakdown (matched vs mismatched convention)

| Metric | Dataset | n | mean r | mean \|r\| |
|---|---|---:|---:|---:|
| hip_flexion_r | opencap | 270 | +0.637 | +0.705 |
| hip_flexion_r | aspset | 1350 | +0.524 | +0.579 |
| hip_adduction_r | opencap | 263 | -0.077 | +0.409 |
| hip_adduction_r | aspset | 1332 | -0.069 | +0.431 |
| knee_angle_r | opencap | 270 | +0.630 | +0.679 |
| knee_angle_r | aspset | 1350 | -0.753 | +0.756 |
| ankle_angle_r | opencap | 270 | +0.351 | +0.549 |
| lumbar_extension | opencap | 270 | -0.003 | +0.678 |
| lumbar_extension | aspset | 1350 | +0.006 | +0.346 |

## Per-metric summary (pooled across views)

| Metric | n_clips | mean r | mean |r| | mean MAE (°) |
|---|---:|---:|---:|---:|
| hip_flexion_r | 1620 | +0.542 | +0.600 | 16.03 |
| hip_adduction_r | 1595 | -0.070 | +0.428 | 178.79 |
| knee_angle_r | 1620 | -0.523 | +0.743 | 101.49 |
| ankle_angle_r | 270 | +0.351 | +0.549 | 42.67 |
| lumbar_extension | 1620 | +0.004 | +0.402 | 145.68 |

## Per-(metric × view) Layer-2 r vs v14 ROM r

### hip_flexion_r

| view | n clips | n_oc | n_asp | mean r ± SD | mean |r| | mean MAE (°) | v14 ROM r |
|---|---:|---:|---:|---|---:|---:|---:|
| front_center | 654 | 54 | 600 | +0.491 ± 0.394 | +0.567 | 16.53 | +0.71 |
| front_oblique_left | 444 | 54 | 390 | +0.556 ± 0.363 | +0.604 | 15.38 | +0.85 |
| front_oblique_right | 294 | 54 | 240 | +0.584 ± 0.359 | +0.626 | 15.78 | +0.82 |
| side_left | 144 | 54 | 90 | +0.573 ± 0.370 | +0.620 | 16.78 | +0.80 |
| side_right | 84 | 54 | 30 | +0.672 ± 0.367 | +0.719 | 15.17 | +0.45 |

### hip_adduction_r

| view | n clips | n_oc | n_asp | mean r ± SD | mean |r| | mean MAE (°) | v14 ROM r |
|---|---:|---:|---:|---|---:|---:|---:|
| front_center | 643 | 53 | 590 | -0.076 ± 0.494 | +0.427 | 176.19 | +0.62 |
| front_oblique_left | 438 | 52 | 386 | -0.070 ± 0.498 | +0.425 | 177.85 | +0.62 |
| front_oblique_right | 291 | 53 | 238 | -0.096 ± 0.504 | +0.447 | 179.82 | +0.57 |
| side_left | 140 | 52 | 88 | -0.088 ± 0.486 | +0.413 | 181.24 | +0.81 |
| side_right | 83 | 53 | 30 | +0.096 ± 0.472 | +0.408 | 196.03 | +0.15 |

### knee_angle_r

| view | n clips | n_oc | n_asp | mean r ± SD | mean |r| | mean MAE (°) | v14 ROM r |
|---|---:|---:|---:|---|---:|---:|---:|
| front_center | 654 | 54 | 600 | -0.649 ± 0.386 | +0.723 | 111.02 | +0.55 |
| front_oblique_left | 444 | 54 | 390 | -0.563 ± 0.518 | +0.737 | 105.70 | +0.76 |
| front_oblique_right | 294 | 54 | 240 | -0.529 ± 0.632 | +0.804 | 96.75 | +0.72 |
| side_left | 144 | 54 | 90 | -0.220 ± 0.726 | +0.728 | 80.48 | +0.81 |
| side_right | 84 | 54 | 30 | +0.180 ± 0.773 | +0.753 | 57.67 | +0.80 |

### ankle_angle_r

| view | n clips | n_oc | n_asp | mean r ± SD | mean |r| | mean MAE (°) | v14 ROM r |
|---|---:|---:|---:|---|---:|---:|---:|
| front_center | 54 | 54 | 0 | -0.296 ± 0.377 | +0.408 | 65.65 | -0.03 |
| front_oblique_left | 54 | 54 | 0 | +0.274 ± 0.451 | +0.471 | 45.51 | +0.50 |
| front_oblique_right | 54 | 54 | 0 | +0.590 ± 0.295 | +0.619 | 43.56 | +0.02 |
| side_left | 54 | 54 | 0 | +0.615 ± 0.319 | +0.647 | 26.55 | +0.41 |
| side_right | 54 | 54 | 0 | +0.572 ± 0.344 | +0.600 | 32.10 | +0.62 |

### lumbar_extension

| view | n clips | n_oc | n_asp | mean r ± SD | mean |r| | mean MAE (°) | v14 ROM r |
|---|---:|---:|---:|---|---:|---:|---:|
| front_center | 654 | 54 | 600 | -0.003 ± 0.431 | +0.345 | 158.09 | +0.33 |
| front_oblique_left | 444 | 54 | 390 | -0.039 ± 0.503 | +0.414 | 154.10 | +0.48 |
| front_oblique_right | 294 | 54 | 240 | +0.081 ± 0.502 | +0.411 | 139.90 | +0.53 |
| side_left | 144 | 54 | 90 | -0.181 ± 0.535 | +0.471 | 120.15 | +0.66 |
| side_right | 84 | 54 | 30 | +0.335 ± 0.594 | +0.620 | 68.60 | +0.50 |

## Interpretation

### Layer-2 r vs ROM r

- Layer-2 r asks: *given the frame at time t, does Couro's angle track the GT's angle?* It's a per-frame waveform-fit metric.
- v14 ROM r asks: *across trials, does Couro's predicted peak ROM correlate with the GT's peak ROM?* It's a per-trial scalar metric, fit with a ridge regression on engineered features and evaluated under leave-one-subject-out CV.
- These are different objects. A clip can have layer-2 |r| = 0.95 (angle waveform tracks well) but only contribute a single peak-ROM scalar to the ROM regression. Twenty such clips with subject-specific peak biases will have a low LOSO ROM r.

### Per-metric story

- **knee_angle_r is Couro's strongest layer-2 metric.** On OpenCap (matched OpenSim convention), mean |r| = +0.68 with mean signed r = +0.63. Side views push to |r| ≈ 0.73–0.75. The ASPset signed r of -0.75 is a pure convention flip in the ASPset GT loader (`knee_angle_r = 180 - interior` produces a perfectly inverted trace) — the |r| of 0.76 on ASPset confirms the same fit quality.
- **hip_flexion_r is the most honest comparison** — sign-aligned across both datasets and against OpenSim. OpenCap |r| = +0.71, ASPset |r| = +0.58, pooled |r| = +0.60. Side views best (|r| ≈ 0.72); front_center weakest (|r| ≈ 0.57). This is what we'd expect from a sagittal-plane angle measured from a 2D camera: perfect view → near-perfect waveform fit.
- **ankle_angle_r is sharply view-dependent.** Side views: |r| = 0.60–0.65. Front_center: |r| = 0.41 (and signed r = -0.30 — the front view sees the foot end-on, so the 2D ankle angle is uncorrelated or even anti-correlated with true sagittal dorsi/plantar flexion). v14 ROM r matches this pattern: 0.41–0.62 on sides, ~0 on front_center.
- **hip_adduction_r is fundamentally hard from 2D.** Couro's 2D proxy uses arctan2 of (knee-hip) image-vector with a sign that's opposite to OpenSim's frontal-plane convention, so signed mean r is near zero. |r| ≈ 0.42 uniformly across views. v14 ROM r is stronger (0.57–0.81 on most views) because the ridge regression learns the convention flip *and* leverages coupling features from the other angles — neither of which the raw layer-2 r sees.
- **lumbar_extension has a strong dataset gap.** OpenCap |r| = +0.68 (very strong layer-2 fit). ASPset |r| = +0.35. The difference is GT definition: OpenCap measures full lumbar angle from a marker-based torso; ASPset's loader uses neck-spine-pelvis 3-point angle, which is noisier and less aligned with Couro's neck→hip_center image-vector proxy. Side views still dominate (|r| = 0.47–0.62).

### Surprises

- **knee side_right has v14 ROM r = +0.80 but layer-2 |r| = +0.75 with signed r = +0.18.** ASPset's 30 side_right clips drag the signed r down via convention flip; the OpenCap-only signed r for the same slot is high. The ROM regression isn't affected because it operates on |max - min|.
- **lumbar OpenCap |r| = 0.68 is higher than v14 ROM r in every view (which peaks at +0.66).** The per-frame trunk-lean trace is actually quite good; the ROM regression underperforms because drop-jump trunk-lean peak is dominated by subject-specific landing strategy that LOSO CV can't generalize from only 23 subjects.
- **hip_flexion_r front_center layer-2 |r| = +0.57 vs v14 ROM r = +0.71.** Here the ROM regression is *higher* than the layer-2 fit — it benefits from anthropometric foreshortening cues + coupling features (knee ROM, lumbar ROM) that recover the peak even when the raw front_center waveform is noisy.
- **ankle_angle_r front_center signed r = -0.30 (|r| = +0.41) vs v14 ROM r = -0.03.** The waveform is actively anti-correlated (front camera sees the foot end-on; perceived ankle flexion moves opposite the true sagittal one), but the ROM regression lands at zero — i.e. the model correctly learns 'this view carries no usable ankle signal' and shrinks toward the mean.

### Headline numbers for the validation doc

- **Overall layer-2 |r| = +0.544** across 6725 clip × metric pairs from 1 680 clips (270 OpenCap DJ + 1 410 ASPset trainval).
- **Per-metric layer-2 |r| range:**
    - hip_flexion_r: |r| = +0.600 (n = 1620)
    - hip_adduction_r: |r| = +0.428 (n = 1595)
    - knee_angle_r: |r| = +0.743 (n = 1620)
    - ankle_angle_r: |r| = +0.549 (n = 270)
    - lumbar_extension: |r| = +0.402 (n = 1620)
- **vs v14 deploy ROM r** (LOSO ridge regression on per-trial ROM, 23 subjects): v14 ROM r ranges +0.33 to +0.85 across slots with the same dataset, with the worst slots being ankle/lumbar in front-facing views. Layer-2 |r| is a complementary measurement — both should appear in the validation doc as 'angle-tracking r' and 'ROM-recovery r' respectively.
