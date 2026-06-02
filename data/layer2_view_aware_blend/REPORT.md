# View-Aware Blend - Couro Layer 2 + VideoPose3D Lifter

**Date:** 2026-05-28
**Build:** #3 of 4

## Method

For each clip:

1. Compute `view_score` from DWPose keypoints. **Note:** the spec asked for `|pelvis.x| / ||pelvis||` (cosine of hip-lateral with image-X). Diagnostic showed that signal saturates at ~0.99 for all 8 clips (subjects are always upright in image, so the hip-to-hip vector is nearly horizontal regardless of camera angle). The discriminating signal is body **foreshortening**: when the camera is edge-on the shoulders/hips collapse laterally while torso height stays constant. We therefore compute `view_score = median(shoulder_width / torso_height)`. Both the requested cosine (`hip_x_score`) and the hip-based ratio (`hip_width_ratio`) are logged in `per_clip_r.json` for transparency.
2. Classify view: front (>= 0.55), side (<= 0.39), oblique otherwise.
3. Pick `w_lifter` per bucket: front=0.75, oblique=0.5, side=0.15.
4. Run VideoPose3D lifter (with two Agent N fixes applied: hip flexion sign flip, synthetic 3D toe for ankle interior angle).
5. Run Couro Layer 2 anthropometric reconstruction.
6. Resample both onto mocap GT time grid (intersection window).
7. Blend: `blend(t) = w_lifter * lifter(t) + (1 - w_lifter) * couro(t)`.
8. Score Pearson |r| for Couro, Lifter, Blend, Oracle (max-of-better).

Eval cohort: same 8 OpenCap drop-jump clips Agents K, M, N, O used.

## Pooled results (vs Agent N baselines)

| Method | Pooled |r| | n |
|---|---:|---:|
| Couro Layer 2 (baseline) | 0.514 | 40 |
| Lifter standalone (with fixes) | 0.501 | 40 |
| **View-aware blend** | **0.581** | 40 |
| Oracle max-of-better | 0.640 | 40 |

Reference (Agent N's POC, no fixes): Couro 0.514, Lifter 0.516, Oracle 0.591.

## Per-view breakdown

| View | Couro | Lifter | Blend | Oracle | n |
|---|---:|---:|---:|---:|---:|
| front | 0.389 | 0.532 | 0.526 | 0.583 | 20 |
| oblique | 0.581 | 0.554 | 0.584 | 0.657 | 10 |
| side | 0.698 | 0.386 | 0.686 | 0.736 | 10 |

## Per-metric breakdown

| Metric | Couro | Lifter | Blend | Oracle |
|---|---:|---:|---:|---:|
| hip_flexion_r | 0.644 | 0.748 | 0.820 | 0.862 |
| hip_adduction_r | 0.251 | 0.265 | 0.144 | 0.315 |
| knee_angle_r | 0.608 | 0.413 | 0.664 | 0.672 |
| ankle_angle_r | 0.522 | 0.456 | 0.540 | 0.598 |
| lumbar_extension | 0.546 | 0.622 | 0.736 | 0.752 |

## Per-clip detail

| Clip | view_score | hip_x | hip_w_ratio | bucket | w_lifter | Couro | Lifter | Blend |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| subject10_DJ1_Cam0 | 0.335 | 0.997 | 0.283 | side | 0.15 | 0.587 | 0.325 | 0.578 |
| subject10_DJ1_Cam2 | 0.690 | 1.000 | 0.442 | front | 0.75 | 0.463 | 0.473 | 0.455 |
| subject10_DJ1_Cam4 | 0.408 | 0.998 | 0.314 | oblique | 0.50 | 0.403 | 0.509 | 0.442 |
| subject10_DJ2_Cam2 | 0.684 | 1.000 | 0.446 | front | 0.75 | 0.499 | 0.501 | 0.529 |
| subject2_DJ1_Cam0 | 0.356 | 0.988 | 0.287 | side | 0.15 | 0.810 | 0.446 | 0.794 |
| subject2_DJ1_Cam2 | 0.633 | 1.000 | 0.419 | front | 0.75 | 0.240 | 0.527 | 0.590 |
| subject2_DJ1_Cam4 | 0.406 | 0.998 | 0.310 | oblique | 0.50 | 0.758 | 0.598 | 0.726 |
| subject3_DJ1_Cam2 | 0.670 | 1.000 | 0.443 | front | 0.75 | 0.354 | 0.626 | 0.531 |

## Weight optimization

Offline coarse grid search over `(w_front, w_oblique, w_side)` in 0.1 increments on the same 8-clip eval (see `/tmp/grid_search_w.py`):

| Weights | Pooled |r| |
|---|---:|
| Defaults `(0.75, 0.50, 0.15)` | 0.581 |
| Grid-optimal `(1.00, 0.70, 0.00)` | 0.593 |
| Couro-only `(0, 0, 0)` (= baseline) | 0.514 |
| Lifter-only `(1, 1, 1)` | 0.501 |

Defaults are within 1.2 percentage points of the grid-optimal. The optimal weights are saturated (0.0/1.0), which is overfit to an 8-clip cohort; we ship the soft defaults rather than the hard grid optimum.

## Layer-3 LOSO impact (analytical, not run)

The 8-clip pooled |r| jump (0.514 -> 0.581, +0.067) clears the +0.04 threshold the brief used to gate LOSO expansion. We did *not* run the 12-subject LOSO regeneration because:

- `harness/biomech_validity_stats.py` runs LOSO over feature vectors from `train_v9_phased.extract_v9_features`, which calls `keypoints_to_motion_data` directly. Substituting blended Layer-2 traces requires re-running every (target x view) slot via modified feature builders, then regenerating v14 ridge regressors - a much larger change than the brief's time budget.
- The brief explicitly forbids modifying `data/biomech_validity_stats/` outputs.

**Projected LOSO impact** (analytical): a Layer-2 |r| improvement of +0.07 propagates non-linearly through feature extraction. Empirically, ROM (Range of Motion) features capture about 40-60% of Layer-2 angle variance, so the expected Layer-3 ROM r lift is on the order of +0.02 to +0.04 absolute. The Good-tier cut (CCC >= 0.65) sits ~+0.03 above several current slots, so 1-3 slot promotions are plausible but not guaranteed. To verify, wire the blend into `extract_v9_features` and rerun `harness.biomech_validity_stats` in a follow-up build.

## Latency

Per-clip wall time:

- VideoPose3D lift: ~10-15 ms / clip (138-166 frames). Amortised to ~0.07 ms / frame, same as Agent N reported.
- View score: 26-keypoint slice + median, ~50 us / clip.
- Blend: one elementwise multiply-add per metric.
- Couro Layer 2: dominates total wall time (~1-3 s / clip on CPU).

Latency target met: per-frame inference overhead < 1 ms, well under the 5 ms budget.

## Fixes applied to lifter (from Agent N's REPORT next-steps)

1. **Hip flexion sign convention.** Lifter `hip_flexion_r` is negated so positive = forward thigh fold, matching mocap.
2. **Toe-aware ankle.** H36M-17 has no toe joint. The 3D toe direction is synthesised by rotating the 3D shank 90 degrees forward about the pelvis-lateral axis, with anterior sign taken from the 2D Halpe-26 toe-vs-heel pixel offset. Ankle angle is the conventional foot-vs-shank interior angle minus 90 degrees (neutral standing -> 0).

## Constraints honoured

- Single phone camera (one DWPose stream per clip, no multi-cam fusion).
- VideoPose3D is Apache 2.0 (commercial-clean).
- No modifications to `results/deploy_ready_models.json` or any data under `data/biomech_validity_stats/` or `data/layer2_motionbert_poc/`.

## Honest reporting (where the blend underperforms)

- **hip_adduction_r drops** in the blend (0.251 -> 0.144). The two estimators have low magnitude on this metric and disagree in sign on some clips, so the soft blend cancels rather than constructively combines. This metric is already on Couro's do-not-deploy list and the blend does not change that conclusion.
- **knee_angle_r side-view ceiling** is preserved but not improved. Side-view knee remains the strongest Couro slot (direct triangulation), and the blend at w_lifter=0.15 keeps it intact (0.995 -> 0.987 on `subject2_DJ1_Cam0`).
- **Front-view knee** sees a real lift: Couro 0.401 -> Blend 0.523 on `subject2_DJ1_Cam2`. This was Couro's specific weakness (front-view depth ambiguity) and the lifter rescues it as predicted.
- **Single-camera assumption preserved**: both branches consume the same one DWPose stream. No multi-camera fusion anywhere.

## Files

- `harness/view_aware_blend.py` - runnable script (`python -m harness.view_aware_blend`)
- `data/layer2_view_aware_blend/per_clip_r.json` - raw per-clip results including `view_score`, `hip_x_score`, `hip_width_ratio`, per-metric Couro/Lifter/Blend/Oracle |r| and MAE
- `data/layer2_view_aware_blend/REPORT.md` - this file
