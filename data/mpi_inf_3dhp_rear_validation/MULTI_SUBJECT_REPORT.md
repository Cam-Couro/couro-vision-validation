# MPI-INF-3DHP Multi-Subject Rear-View DWPose Validation

Subjects included (8 of 8): S2, S3, S4, S5, S6, S7, S8, S1. Sequence: Seq2 (Sports — boxing, tennis, golf, soccer). Rear camera per subject auto-selected from camera.calibration (cam closest to -159 deg rel-az from cam 0).

Pipeline: DWPose-L (CoreML, 384x288 input, Halpe-26 remap) with bounding boxes derived from MPI-INF-3DHP 2D GT joints (15% pad). GT joint angles computed from 3D mocap (`annot3`, camera-invariant). Predicted 2D angles from DWPose keypoints in rear-camera image space via Couro's production `keypoints_to_motion_data` math. Pearson r reported as `|r|` to absorb sign-convention mismatch between 2D interior angles and 3D mocap angles.

## Rear-camera selection (per subject)

| Subject | cam_id | rel_az from cam0 | height (mm) | radius (mm) |
| --- | --- | --- | --- | --- |
| S2 | 7 | -159.0° | 1165.8 | 3013.7 |
| S3 | 7 | -159.0° | 1165.8 | 3013.7 |
| S4 | 7 | -159.0° | 1165.8 | 3013.7 |
| S5 | 7 | -159.0° | 1165.8 | 3013.7 |
| S6 | 7 | -159.0° | 1165.8 | 3013.7 |
| S7 | 7 | -159.0° | 1165.8 | 3013.7 |
| S8 | 7 | -159.0° | 1165.8 | 3013.7 |
| S1 | 7 | -159.0° | 1166 | 3014 |

## Per-subject Pearson |r| (rear camera vs 3D mocap)

| Metric | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hip flexion R | 0.897 | 0.806 | 0.808 | 0.913 | 0.898 | 0.915 | 0.842 | 0.870 |
| Hip flexion L | 0.900 | 0.767 | 0.750 | 0.924 | 0.896 | 0.842 | 0.876 | 0.914 |
| Knee flexion R | 0.895 | 0.792 | 0.509 | 0.849 | 0.858 | 0.812 | 0.802 | 0.814 |
| Knee flexion L | 0.879 | 0.712 | 0.590 | 0.828 | 0.838 | 0.787 | 0.587 | 0.809 |
| Ankle dorsiflex R | 0.408 | 0.274 | 0.170 | 0.436 | 0.499 | 0.292 | 0.111 | 0.305 |
| Ankle dorsiflex L | 0.449 | 0.173 | 0.222 | 0.347 | 0.322 | 0.298 | 0.297 | 0.388 |
| Hip adduction R | 0.472 | 0.534 | 0.587 | 0.280 | 0.446 | 0.260 | 0.390 | 0.281 |
| Hip adduction L | 0.489 | 0.397 | 0.539 | 0.216 | 0.475 | 0.390 | 0.345 | 0.240 |
| Lumbar extension | 0.897 | 0.780 | 0.862 | 0.906 | 0.837 | 0.847 | 0.707 | 0.805 |

## Cross-subject summary

| Metric | n subj | mean |r| | SD | min | max |
| --- | --- | --- | --- | --- | --- |
| Hip flexion R | 8 | 0.869 | 0.045 | 0.806 | 0.915 |
| Hip flexion L | 8 | 0.859 | 0.067 | 0.750 | 0.924 |
| Knee flexion R | 8 | 0.791 | 0.119 | 0.509 | 0.895 |
| Knee flexion L | 8 | 0.754 | 0.112 | 0.587 | 0.879 |
| Ankle dorsiflex R | 8 | 0.312 | 0.132 | 0.111 | 0.499 |
| Ankle dorsiflex L | 8 | 0.312 | 0.088 | 0.173 | 0.449 |
| Hip adduction R | 8 | 0.406 | 0.124 | 0.260 | 0.587 |
| Hip adduction L | 8 | 0.386 | 0.116 | 0.216 | 0.539 |
| Lumbar extension | 8 | 0.830 | 0.065 | 0.707 | 0.906 |

## Assessment

**Verdict (n=8): rear-view (cam 7, -159°) is shippable for hip flexion and lumbar extension, conditionally shippable for knee flexion, and still unreliable for ankle dorsiflexion and hip adduction — consistent with the S1 baseline. Cross-subject SD is small for the strong metrics (0.05–0.07) and large for the weak ones (0.09–0.13), so the floor is the metric, not the camera.**

- **Hip flexion R**: cross-subject mean |r| = 0.869 ± 0.045 (range 0.806–0.915, n=8).
- **Knee flexion R**: cross-subject mean |r| = 0.791 ± 0.119 (range 0.509–0.895, n=8).
- **Lumbar extension**: cross-subject mean |r| = 0.830 ± 0.065 (range 0.707–0.906, n=8).
- **Ankle dorsiflex R**: cross-subject mean |r| = 0.312 ± 0.132 (range 0.111–0.499, n=8).

### Strongest / weakest subjects (by hip flexion R)

- Strongest: S7 (|r|=0.915)
- Weakest:   S3 (|r|=0.806)

### Outliers worth investigating

- **S4**: knee R=0.509, knee L=0.590
- **S8**: knee L=0.587

Knee regressions below 0.65 in rear view likely reflect occlusion of the near-side leg by the torso (rear-oblique view puts one leg behind the other across part of the gait/swing cycle), or foreshortening of the contralateral shank when the subject is angled obliquely. Worth pulling 30-frame clips to eyeball pose stability before shipping rear-cam knee deploys.

### S1 (original baseline) vs cross-subject

- S1 hip flexion R |r|=0.870 vs cross-subject mean 0.869 (delta +0.001). S1 was a representative-not-exceptional baseline; expanding to n=8 did NOT inflate the apparent quality of rear-view DWPose.

### Camera consistency check

Rear-camera rel-az across subjects: mean -159.0°, SD 0.0°, range [-159.0°, -159.0°].
Studio rig is essentially identical across subjects (rear-cam angle SD < 1°), so cross-subject variation in r reflects subject biomechanics + body shape, not camera differences.

## Artifacts

- Per-subject DWPose keypoints: `data/mpi_inf_3dhp_dwpose/S{N}_Seq2_cam{X}.json`
- Per-subject GT angles: `data/mpi_inf_3dhp_gt/S{N}_Seq2_angles.json`
- Per-subject correlations: `data/mpi_inf_3dhp_rear_validation/r_S{N}_cam{X}.json`
- Aggregated summary: `data/mpi_inf_3dhp_rear_validation/multi_subject_r_summary.json`
- Per-subject camera layout: `data/mpi_inf_3dhp_rear_validation/camera_layout_S{N}.json`
