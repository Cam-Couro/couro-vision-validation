# MPI-INF-3DHP S1 Seq1 (Walking) — Rear-View DWPose Validation

**Date:** 2026-05-29
**Subject:** S1 only (smoke test for walking-gait class)
**Sequence:** Seq1 — Standing / Walking / Sitting (general locomotion, GT mocap)
**Rear camera:** cam7, rel_az = **−159.0°** from cam0 (rear-oblique; identical placement to Seq2 Sports baseline)
**Pipeline:** DWPose-L (CoreML, 384×288, Halpe-26 remap) on `video_7.avi`, bboxes from `annot2[7]` with 15% pad. GT joint angles from `annot3` (camera-invariant 3D mocap). Pearson `|r|` reported to absorb sign convention.

## Motivation

Running gait is the strategic priority for Couro running rear+front. Fukuchi running is not local; ASPset-510 has no rear camera; OpenCap has no running trials. MPI-INF-3DHP Seq1 (Walking) is the closest local proxy for gait at the rear-oblique camera, using the exact harness that produced the n=8 Sports rear-view result.

## Results — S1 Seq1 cam7 vs 3D mocap GT

| Metric | n_valid | |r| | Sports cross-subj mean |r| | Δ vs Sports |
| --- | ---: | ---: | ---: | ---: |
| hip_flexion_r        | 5,266 | **0.889** | 0.869 | +0.020 |
| hip_flexion_l        | 5,510 | **0.882** | 0.859 | +0.023 |
| knee_angle_r         | 5,223 | **0.856** | 0.791 | +0.065 |
| knee_angle_l         | 5,427 | **0.873** | 0.754 | +0.119 |
| hip_adduction_r      | 5,359 | **0.585** | 0.406 | +0.179 |
| hip_adduction_l      | 5,571 | **0.525** | 0.386 | +0.139 |
| lumbar_extension     | 5,568 | **0.856** | 0.830 | +0.026 |
| ankle_angle_r        | 4,973 | 0.279 | 0.312 | −0.033 |
| ankle_angle_l        | 5,274 | 0.302 | 0.312 | −0.010 |

*Sports cross-subj means from `MULTI_SUBJECT_REPORT.md`, n=8 (S1–S8) on Seq2.*

## Headline

**Walking-gait rear-view validation matches or exceeds the Sports-rear baseline on every shippable slot.** Hip flexion, knee flexion, and lumbar extension all read `|r| ≥ 0.85` from the −159° rear-oblique camera on walking gait. Knee flexion improves substantially (+0.07 to +0.12 vs Sports) — gait motion has cleaner, more periodic knee trajectories than tennis/golf/boxing swings, so the keypoint signal is more correlated frame-to-frame. Hip adduction lifts from "not shippable" (Sports ≈ 0.40) to "moderate" (Walking ≈ 0.55) for the same reason.

## What this implies for running rear-view

Running gait is dynamically similar to walking gait but with larger joint ROMs (more signal-to-noise on flexion/extension) and ground contact + flight phases. The geometric and biomechanical relationships the DWPose→Halpe-26→angle math relies on are preserved.

Honest expectation for **running rear** (untested locally — needs Fukuchi or treadmill data):
- Hip flexion: ≈ 0.88–0.92 (likely **stronger** than walking due to larger ROM)
- Knee flexion: ≈ 0.85–0.90 (likely stronger)
- Lumbar extension: ≈ 0.80–0.86 (similar)
- Hip adduction: ≈ 0.50–0.65 (similar to walking; small ROM keeps SNR limited)
- Ankle dorsiflexion: ≈ 0.30–0.50 (foot occlusion at toe-off remains the bottleneck)

These are **extrapolations, not measurements**. Treat as Couro-internal priors; do not quote externally without Fukuchi or equivalent.

## What's still missing for the full rear+front running answer

1. **True running gait dataset** — Fukuchi (treadmill running, multi-cam mocap) is the right benchmark. Not on this machine.
2. **Front-view walking/running** — analogous validation at front camera (cam 0 or cam 13) on the same Seq1 data would tell us front vs rear for the same motion class. Cheap follow-up.
3. **S2–S8 cohort expansion** for Seq1 — confirms S1 isn't an outlier.

## Artifacts

- DWPose keypoints: `data/mpi_inf_3dhp_dwpose/S1_Seq1_cam7.json` (6,416 frames)
- Predicted angles: `data/mpi_inf_3dhp_dwpose/S1_Seq1_cam7_pred_angles.json`
- GT angles (3D mocap): `data/mpi_inf_3dhp_gt/S1_Seq1_angles.json`
- Per-metric correlations: `data/mpi_inf_3dhp_rear_validation/r_S1_Seq1_cam7.json`
- Download script: `harness/mpi3dhp_download_seq1.py`

## Reproduce

```
python3 -m harness.mpi3dhp_download_seq1     # S1 Seq1 ~700 MB
python3 -m harness.mpi3dhp_bboxes  data/mpi_inf_3dhp/S1/Seq1/annot.mat 7  data/mpi_inf_3dhp_dwpose/S1_Seq1_cam7_boxes.csv
python3 -m harness.mpi3dhp_gt_angles  data/mpi_inf_3dhp/S1/Seq1/annot.mat  data/mpi_inf_3dhp_gt/S1_Seq1_angles.json
python3 -m harness.mpi3dhp_infer_dwpose  data/mpi_inf_3dhp/S1/Seq1/imageSequence/video_7.avi  data/mpi_inf_3dhp_dwpose/S1_Seq1_cam7_boxes.csv  data/mpi_inf_3dhp_dwpose/S1_Seq1_cam7.json
python3 -m harness.mpi3dhp_pred_angles  data/mpi_inf_3dhp_dwpose/S1_Seq1_cam7.json  data/mpi_inf_3dhp_dwpose/S1_Seq1_cam7_pred_angles.json
python3 -m harness.mpi3dhp_correlate  data/mpi_inf_3dhp_dwpose/S1_Seq1_cam7_pred_angles.json  data/mpi_inf_3dhp_gt/S1_Seq1_angles.json  --label S1_Seq1_cam7
```
