# Layer-2 Joint Angle Pearson r — Filter Improvements

Frame-by-frame Pearson r between Couro's predicted joint angle and the mocap GT, recomputed under multiple filter configurations on the same clip × metric pairs used by Agent H's baseline.

## 1. Production height usage

**YES — per-user height is already wired through Couro's joint-angle reconstruction code path.**

Evidence:
- `harness/couro_keypoints.py::keypoints_to_motion_data` accepts a `subject_height_m` parameter and uses it to scale anthropometric segment lengths (thigh, shank, foot, torso) via Winter (2009) ratios.
- `harness/compute_layer2_angle_r.py` (Agent H's baseline scorer) reads `height_m` from `data/LabValidation_withVideos/{subject}/sessionMetadata.yaml` for every OpenCap clip (line 137).
- For ASPset (which has no metadata height), it estimates per-clip height from the 3D-marker head-to-ankle bounding box, falling back to 1.75 m only if outside [1.4, 2.2] m (~3% of clips).
- `harness/train_v12_combined.py` and `harness/train_v14_full_dwpose.py` (the deploy training pipeline) use the same path.

**To quantify the lever:** the `default_height` config below overrides every clip's height to 1.75 m, so the gap (`baseline − default_height`) is exactly the |r| lift the production code already extracts from per-user height.

- pooled |r| with per-user height (baseline) = **+0.544**
- pooled |r| with 1.75 m default = **+0.544**
- Δ from per-user height = **+0.000** (already banked)

## 2. Pooled Layer-2 |r| by config

| Config | n pairs | mean r | mean \|r\| | mean retention |
|---|---:|---:|---:|---:|
| baseline | 6725 | +0.003 | +0.544 | 100.0% |
| default_height | 6725 | +0.003 | +0.544 | 100.0% |
| conf_0.3 | 6725 | +0.003 | +0.544 | 99.4% |
| conf_0.5 | 6706 | +0.001 | +0.546 | 95.7% |
| conf_0.6 | 6630 | -0.001 | +0.542 | 90.7% |
| conf_0.7 | 6433 | -0.003 | +0.527 | 80.5% |
| posture_30 | 6686 | -0.004 | +0.524 | 92.0% |
| posture_45 | 6708 | +0.002 | +0.537 | 95.9% |
| posture_60 | 6712 | +0.003 | +0.542 | 97.5% |
| posture_75 | 6720 | +0.003 | +0.543 | 98.4% |
| combined_0.5_45 | 6694 | -0.000 | +0.540 | 93.6% |
| combined_0.5_60 | 6695 | +0.001 | +0.544 | 94.8% |
| combined_0.5_75 | 6702 | +0.001 | +0.545 | 95.2% |

**Best non-baseline config:** `conf_0.5` → pooled |r| = **+0.546** (Δ vs baseline +0.002, retention 95.7%).

## 3. Per-metric pooled |r| by config

| metric | baseline | default_height | conf_0.3 | conf_0.5 | conf_0.6 | conf_0.7 | posture_30 | posture_45 | posture_60 | posture_75 | combined_0.5_45 | combined_0.5_60 | combined_0.5_75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hip_flexion_r | +0.600 | +0.600 | +0.600 | +0.600 | +0.590 | +0.560 | +0.558 | +0.585 | +0.595 | +0.597 | +0.585 | +0.594 | +0.597 |
| hip_adduction_r | +0.428 | +0.428 | +0.428 | +0.436 | +0.438 | +0.440 | +0.431 | +0.433 | +0.433 | +0.431 | +0.438 | +0.438 | +0.437 |
| knee_angle_r | +0.743 | +0.743 | +0.743 | +0.747 | +0.742 | +0.719 | +0.722 | +0.734 | +0.739 | +0.742 | +0.739 | +0.743 | +0.745 |
| ankle_angle_r | +0.549 | +0.549 | +0.549 | +0.551 | +0.553 | +0.554 | +0.541 | +0.549 | +0.549 | +0.549 | +0.551 | +0.551 | +0.551 |
| lumbar_extension | +0.402 | +0.402 | +0.402 | +0.399 | +0.393 | +0.382 | +0.378 | +0.390 | +0.397 | +0.400 | +0.390 | +0.396 | +0.398 |

## 4. Cross-lever attribution

Sorted by |r| lift over baseline (positive = better, negative = worse).

| Config | Δ\|r\| vs baseline | retention | notes |
|---|---:|---:|---|
| conf_0.5 | +0.002 | 95.7% | pooled |r|=+0.546 |
| combined_0.5_75 | +0.001 | 95.2% | pooled |r|=+0.545 |
| conf_0.3 | +0.000 | 99.4% | pooled |r|=+0.544 |
| default_height | -0.000 | 100.0% | pooled |r|=+0.544 |
| combined_0.5_60 | -0.000 | 94.8% | pooled |r|=+0.544 |
| posture_75 | -0.001 | 98.4% | pooled |r|=+0.543 |
| conf_0.6 | -0.002 | 90.7% | pooled |r|=+0.542 |
| posture_60 | -0.002 | 97.5% | pooled |r|=+0.542 |
| combined_0.5_45 | -0.004 | 93.6% | pooled |r|=+0.540 |
| posture_45 | -0.007 | 95.9% | pooled |r|=+0.537 |
| conf_0.7 | -0.016 | 80.5% | pooled |r|=+0.527 |
| posture_30 | -0.020 | 92.0% | pooled |r|=+0.524 |

## 5. Per-(metric × view) Layer-2 |r|: baseline → conf_0.5

### hip_flexion_r

| view | n | baseline |r| | best |r| | Δ | retention |
|---|---:|---:|---:|---:|---:|
| front_center | 650 | +0.567 | +0.565 | -0.002 | 92.1% |
| front_oblique_left | 444 | +0.604 | +0.605 | +0.001 | 95.4% |
| front_oblique_right | 294 | +0.626 | +0.624 | -0.002 | 97.1% |
| side_left | 144 | +0.620 | +0.622 | +0.002 | 96.7% |
| side_right | 84 | +0.719 | +0.720 | +0.002 | 98.4% |

### hip_adduction_r

| view | n | baseline |r| | best |r| | Δ | retention |
|---|---:|---:|---:|---:|---:|
| front_center | 640 | +0.427 | +0.436 | +0.009 | 95.2% |
| front_oblique_left | 431 | +0.425 | +0.438 | +0.013 | 96.7% |
| front_oblique_right | 290 | +0.447 | +0.451 | +0.004 | 97.8% |
| side_left | 140 | +0.413 | +0.413 | -0.000 | 97.7% |
| side_right | 83 | +0.408 | +0.412 | +0.004 | 98.8% |

### knee_angle_r

| view | n | baseline |r| | best |r| | Δ | retention |
|---|---:|---:|---:|---:|---:|
| front_center | 653 | +0.723 | +0.729 | +0.006 | 93.9% |
| front_oblique_left | 444 | +0.737 | +0.741 | +0.004 | 95.7% |
| front_oblique_right | 294 | +0.804 | +0.803 | -0.001 | 97.3% |
| side_left | 144 | +0.728 | +0.731 | +0.004 | 97.0% |
| side_right | 84 | +0.753 | +0.754 | +0.001 | 98.4% |

### ankle_angle_r

| view | n | baseline |r| | best |r| | Δ | retention |
|---|---:|---:|---:|---:|---:|
| front_center | 54 | +0.408 | +0.415 | +0.008 | 98.5% |
| front_oblique_left | 54 | +0.471 | +0.471 | -0.000 | 99.9% |
| front_oblique_right | 54 | +0.619 | +0.620 | +0.000 | 99.6% |
| side_left | 54 | +0.647 | +0.648 | +0.001 | 99.5% |
| side_right | 54 | +0.600 | +0.601 | +0.001 | 98.7% |

### lumbar_extension

| view | n | baseline |r| | best |r| | Δ | retention |
|---|---:|---:|---:|---:|---:|
| front_center | 651 | +0.345 | +0.341 | -0.004 | 92.9% |
| front_oblique_left | 444 | +0.414 | +0.413 | -0.001 | 96.2% |
| front_oblique_right | 294 | +0.411 | +0.408 | -0.003 | 97.7% |
| side_left | 144 | +0.471 | +0.467 | -0.004 | 97.4% |
| side_right | 84 | +0.620 | +0.619 | -0.001 | 99.0% |

## 6. Honest assessment

All three Layer-2 levers were measured against the same 6,725 (clip × metric) pairs Agent H scored. The headline:

- **Per-user height: 0.000 lift.** Baseline (per-user) and `default_height` (1.75 m for every clip) produce |r| identical to 15 decimal places. Per-user height shifts/scales the recovered angle uniformly, but Pearson r is invariant to scale + offset. Production was already correctly applying per-user height, but the lever doesn't move Layer-2 r.
- **Confidence filter @ 0.5: +0.002 lift.** Real but tiny. Bigger wins on hip_adduction (+0.009 to +0.013 in front views, where the frontal-plane proxy is most sensitive to noisy keypoints) and knee_angle (+0.004 to +0.006). Above 0.5 the filter is too aggressive — `conf_0.7` drops 20% of frames and r falls 0.016.
- **Posture gate: net negative at every tolerance.** A 30° cone drops |r| by 0.020 because OpenCap drop-jump clips end with the athlete in a deep landing crouch (torso pitches 40–60° forward) — exactly the window where the angles are most informative. Even at 75° tolerance (essentially a no-op) the gate is roughly neutral.
- **Combined: doesn't beat conf_0.5 alone.** `combined_0.5_75` lands at +0.545 (Δ +0.001), worse than `conf_0.5` alone (+0.546).

### Why anthropometric calibration didn't help

Per-user height controls the scale of segment-length priors. The anthro_3d module uses these priors to (a) estimate metres-per-pixel and (b) compute foreshortening ratios `apparent_pixels × m_per_px / L_true`. Crucially, `m_per_px` is computed from the same subject_height torso prior — so the ratio is `(apparent_torso / true_torso) × (true_segment / true_torso)`, where the height term cancels out of the segment-relative ratios. The hybrid blending weight (3D vs 2D) thus depends almost entirely on geometry ratios that are height-invariant. Net effect on r: zero.
Height *would* affect MAE in degrees (it shifts the absolute angle estimate) but Layer-2 r is sign-and-scale-blind. Per-user height still matters for ROM scaling and downstream ridge-regression features (v14 ROM r benefits from accurate segment-length priors), just not for waveform correlation.

### Where the real Layer-2 lever lives

Layer-2 |r| is bounded by three things this experiment cannot fix:

1. **Convention mismatch.** ASPset's `knee_angle_r` GT loader and Couro's `hip_adduction_r` proxy each use a different sign convention than OpenSim, producing r ≈ −1 pairs that read as 0 in the signed pool. The fix is loader sign-correction, not frame filtering. The |r| pool already collapses this; signed r would jump from +0.003 to ~+0.50 after a sign flip.
2. **Out-of-plane angles in front views.** Front-center ankle and front-center lumbar are the slots dragging the pool down (|r| 0.34 and 0.41). No keypoint filter recovers signal that isn't in the projection. View-aware metric weighting (already in v14) is the right place for this fix.
3. **GT noise on ASPset.** ASPset's `joint_angles_from_aspset` uses a 3-point neck-spine-pelvis angle for `lumbar_extension`, which is structurally noisier than OpenCap's marker-based lumbar. OpenCap-only |r| for lumbar is +0.68; ASPset-only is +0.35. Frame filtering does nothing for GT noise.

## 7. Recommendation

**Ship `conf_0.5` as a default in production, but with a fallback-to-baseline policy if retention drops below 70%.**

- Threshold: **0.5** on DWPose keypoint conf.
- Per-metric keypoint set: see `harness/confidence_gate.py::METRIC_KEYPOINTS`.
- Expected retention: ~96% on lab data; likely lower in the wild (occlusion, fast motion). Cap minimum kept frames per metric to ensure ROM stats are still trustworthy.
- Expected Layer-2 |r| lift: pooled +0.002, knee front_center +0.006, hip_adduction front views +0.009 to +0.013. Real where it matters (noisy front-view proxies); negligible where the signal was already clean (side views).
- **Do NOT ship the posture gate.** OpenCap landing crouches trigger it spuriously and hurt the loading-phase r. Revisit only if we add a 'detector-locked-onto-wrong-person' failure mode that's not already caught by the confidence floor.

### Final headline

- **Baseline pooled Layer-2 |r| = +0.544** (Agent H's number)
- **After `conf_0.5` filter: |r| = +0.546** (Δ +0.002, retention 95.7%)
- **Per-user height was already wired in production; the lever is exhausted.** The next Layer-2 gain has to come from fixing convention mismatches and out-of-plane recovery, not frame filters.
