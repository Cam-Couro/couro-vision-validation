# MPI-INF-3DHP Seq1 (Walking) — Cross-Subject Rear-View + Front-View Validation

**Subjects:** S1–S8 (n=8 rear) + S1 (front, single-subject)
**Sequence:** Seq1 — Standing / Walking / Sitting (general locomotion, mocap GT)
**Rear camera:** cam7, rel_az = **−159.0°** from cam0 (rear-oblique; identical placement to existing Seq2 Sports baseline)
**Front camera:** cam0, rel_az = **0.0°** (subject-facing)
**Pipeline:** DWPose-L (CoreML, 384×288, Halpe-26 remap) on cam7/cam0 video, bboxes from `annot2[cam]` with 15% pad. GT angles from `annot3` 3D mocap.

## TL;DR — three measured headlines

1. **Walking-rear matches Sports-rear on every shippable slot.** Hip flexion (0.87), knee flexion (0.81–0.86), lumbar extension (0.88) all hold at production-ready accuracy across n=8. Cross-subject SD is small (0.03–0.05) — the floor is the metric, not the subject.

2. **Lumbar extension lifts +0.05 on walking vs sports**, and **knee flexion L lifts +0.10**. Gait's periodic, lower-velocity motion gives cleaner keypoint signal than tennis/golf/boxing swings.

3. **Walking ankle dorsi is *worse* than sports** (-0.10 to -0.14). Walking has small ankle ROM at toe-off compared to dynamic sports motions — signal-to-noise drops below the keypoint noise floor. Ankle from rear is not shippable on either activity class.

## Headline table — cross-subject n=8 walking-rear

| Metric | Walk mean \|r\| | SD | min | max | Sports baseline \|r\| | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **hip_flexion_r** | **0.871** | 0.032 | 0.826 | 0.940 | 0.869 | +0.002 |
| **hip_flexion_l** | **0.856** | 0.053 | 0.759 | 0.949 | 0.859 | −0.003 |
| **knee_angle_r** | **0.807** | 0.054 | 0.680 | 0.856 | 0.791 | +0.016 |
| **knee_angle_l** | **0.857** | 0.044 | 0.777 | 0.926 | 0.754 | **+0.103** |
| **lumbar_extension** | **0.875** | 0.032 | 0.809 | 0.927 | 0.830 | **+0.045** |
| hip_adduction_r | 0.397 | 0.131 | 0.118 | 0.585 | 0.406 | −0.009 |
| hip_adduction_l | 0.359 | 0.099 | 0.159 | 0.525 | 0.386 | −0.027 |
| ankle_angle_r | 0.210 | 0.088 | 0.090 | 0.333 | 0.312 | **−0.102** |
| ankle_angle_l | 0.169 | 0.060 | 0.108 | 0.302 | 0.312 | **−0.143** |

## Per-subject rear-cam |r| (matches the Sports n=8 reporting format)

| Subj | hip_flex_r | hip_flex_l | knee_r | knee_l | ank_r | ank_l | hipadd_r | hipadd_l | lumbar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 0.889 | 0.882 | 0.856 | 0.873 | 0.279 | 0.302 | 0.585 | 0.525 | 0.856 |
| S2 | 0.887 | 0.889 | 0.844 | 0.892 | 0.210 | 0.118 | 0.428 | 0.314 | 0.875 |
| S3 | 0.862 | 0.759 | 0.805 | 0.870 | 0.097 | 0.155 | 0.460 | 0.386 | 0.878 |
| S4 | 0.826 | 0.826 | 0.680 | 0.777 | 0.333 | 0.112 | 0.341 | 0.350 | 0.885 |
| S5 | 0.858 | 0.814 | 0.839 | 0.818 | 0.312 | 0.201 | 0.389 | 0.405 | 0.809 |
| S6 | 0.858 | 0.852 | 0.802 | 0.830 | 0.218 | 0.188 | 0.343 | 0.310 | 0.865 |
| S7 | 0.851 | 0.877 | 0.783 | 0.926 | 0.139 | 0.108 | 0.118 | 0.159 | 0.903 |
| S8 | 0.940 | 0.949 | 0.847 | 0.873 | 0.090 | 0.169 | 0.513 | 0.419 | 0.927 |

S4 knee R (0.68) is the lone weak knee result — likely the same near-leg-behind-torso occlusion that flagged S4/S8 in the Sports n=8 report. Worth a 30-frame eyeball before shipping rear-cam knee deploys for any production athlete.

## S1 front-view (cam0) vs rear-view (cam7) on identical walking motion

| Metric | Front \|r\| | Rear \|r\| | Better view |
| --- | ---: | ---: | --- |
| hip_flexion_r | 0.902 | 0.889 | front (≈tie) |
| hip_flexion_l | 0.887 | 0.882 | tie |
| knee_angle_r | 0.872 | 0.856 | tie |
| knee_angle_l | 0.824 | 0.873 | **rear** |
| **lumbar_extension** | 0.764 | **0.856** | **rear (+0.09)** |
| hip_adduction_r | 0.483 | 0.585 | **rear** |
| hip_adduction_l | 0.558 | 0.525 | front |
| ankle_angle_r | 0.344 | 0.279 | front |
| ankle_angle_l | 0.266 | 0.302 | tie |

**Front view is NOT strictly better than rear.** Front wins hip flexion + ankle R (marginal). Rear wins lumbar (+0.09), knee L (+0.05), hip add R (+0.10). For the "single-camera anywhere" product story, rear is a legitimate first-class angle, not a fallback. Lumbar extension specifically is **substantially better** from rear — the front-on view sees trunk lean as small in-plane rotation, while the rear-oblique view sees it as large up-down motion of the head/shoulders relative to pelvis.

## What this means for running specifically

Walking is the closest local proxy for running gait. Running has:
- Larger joint ROMs (better SNR on flexion/extension → likely better |r| on hip and knee)
- Ground contact + flight phases (briefly clearer foot/ankle visibility → ankle may improve marginally)
- Higher angular velocities (more motion blur risk, but Couro pipeline is keypoint-based, not edge-based → probably similar)
- Forward translation (subject moves across frame → bbox tracking matters, but already handled by the annot2 GT or production tracker)

Honest prior (extrapolation, not measurement) for **running rear**:
- Hip flexion: ≈ 0.88–0.93 (walk 0.87 → likely up with larger ROM)
- Knee flexion: ≈ 0.85–0.92 (walk 0.81–0.86 → likely up)
- Lumbar extension: ≈ 0.80–0.88 (walk 0.88 → similar or slightly down with more dynamic trunk)
- Hip adduction: ≈ 0.40–0.55 (walk 0.36–0.40 → similar; small ROM stays the bottleneck)
- Ankle dorsi: ≈ 0.30–0.50 (walk 0.17–0.21 → larger ROM in running likely helps but stays weak)

Treat these as Couro-internal priors. Do not quote externally until a real running-rear measurement lands.

## Closing the loop to "real running rear video"

Recapping the dead end and the real paths:
- **Fukuchi 2017 (running)** — mocap C3D + force + EMG only, **no video files**. Same for **Fukuchi 2018 (walking)**.
- **ASPset-510** — outdoor sport video with mocap, but only left/mid/right cameras, **no rear**.
- **OpenCap LabValidation** — multi-cam mocap, but trials are DJ / squat / STS / walking — **no running trials**.

Real paths forward:
1. **Self-collect** treadmill running with phone cameras at rear/front and existing trusted-side-cam as soft GT. Fastest unblock for a real running-rear |r|.
2. **AMASS synthetic** — render BMLrub locomotion subjects (running gait included) at front/rear virtual cameras via the SMPL-Body pipeline. Blocked on Phase 1 of [PIPELINE_PLAN.md](../../synthetic_amass/PIPELINE_PLAN.md) (~1–2 days). Once unblocked, gives unlimited camera angle variations.
3. **Fukuchi-via-SMPL-fit** — fit SMPL body to Fukuchi mocap markers, render synthetic video at rear/front, run DWPose. Real running biomechanics, synthetic visuals. Same blocker as #2 plus marker-fit step.

## Artifacts (all today, 2026-05-29 → 2026-05-30)

- Per-subject DWPose keypoints: `data/mpi_inf_3dhp_dwpose/S{N}_Seq1_cam7.json` (rear, n=8)
- S1 front-cam keypoints: `data/mpi_inf_3dhp_dwpose/S1_Seq1_cam0.json`
- Per-subject GT angles: `data/mpi_inf_3dhp_gt/S{N}_Seq1_angles.json`
- Per-subject correlations: `data/mpi_inf_3dhp_rear_validation/r_S{N}_Seq1_cam7.json`
- S1 front-cam correlation: `data/mpi_inf_3dhp_rear_validation/r_S1_Seq1_cam0.json`
- Single-subject deep-dive: `data/mpi_inf_3dhp_rear_validation/WALKING_S1_REPORT.md`
- Download scripts: `harness/mpi3dhp_download_seq1.py`, `harness/mpi3dhp_download_multi_seq1.py`
- Per-subject pipeline runner: `harness/run_seq1_rear_pipeline.sh`

## Reproduce

```
# Single subject smoke test
python3 -m harness.mpi3dhp_download_seq1

# Cross-subject pull (parallel, 3 workers)
python3 -m harness.mpi3dhp_download_multi_seq1 --subjects 2 3 4 5 6 7 8 --workers 3

# Per-subject pipeline (bbox + GT + DWPose + pred + correlate)
for S in S1 S2 S3 S4 S5 S6 S7 S8; do
  bash harness/run_seq1_rear_pipeline.sh "$S"
done
```
