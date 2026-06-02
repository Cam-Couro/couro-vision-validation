# Rear-View Validation Path — Commercial-Clean

## What this pipeline unlocks

Couro's commercial deployment includes a rear-view camera position for softball pitching (camera behind the catcher, pitcher facing away) and other sports where the rear is the natural viewpoint. Until now, the only real-data rear-view validation set we had was the **MPI-INF-3DHP rear cohort** — which is academic-only license and cannot ship in production marketing, pitch decks, or diligence materials without violating the MPI license.

The synthetic Layer 2 pipeline in `harness/synthetic_layer2_production.py` renders pose-to-2D projections through arbitrary virtual cameras, with perfect ground-truth joint angles for every frame. This means:

**We can generate a rear-view validation set that is 100% commercial-clean.** Same `SMPL_NEUTRAL.pkl` (CC-BY 4.0) pose distribution, but the virtual camera placed behind the subject. Ground truth comes from `smpl_joints_to_metrics(joints)` on the 3D joint positions — analytical truth, not a separate mocap measurement.

## How to build the rear-view dataset

1. Modify `_sample_camera()` in `synthetic_layer2_production.py` to restrict yaw to a rear arc:
   - **Loose:** yaw ∈ [135°, 225°] (back-half hemisphere)
   - **Strict:** yaw ∈ [165°, 195°] (true rear view)
2. Sample bursts as needed (5K bursts ≈ 45K frames runs in ~50s on CPU)
3. Save the `(clean_keypoints, GT_angles, camera_params)` tuples for reuse in publication/marketing-grade rear-view validation tables

For softball pitching specifically, swap `_sample_pose()` for an overhand-throw-biased pose distribution (shoulder external rotation ROM, trunk rotation, stride length). The pipeline machinery (FK, projection, normalize, angle-extraction, evaluation) is unchanged.

## Why this matters for Couro pitches

- **ESV / AUSL / Cal Berkeley:** rear-view validation numbers can be produced without an academic-license blocker on the data they're computed from.
- **SJ Sharks / NHL pilots:** on-ice biomech often has the camera angled behind the play; the same rear-view framework applies.
- **Any future client where camera placement is unconstrained:** gets validated rear-view metrics without a separate real-data collection effort up front.
- **Replaces the MPI-INF-3DHP rear cohort in the v2 validation doc footnote** with a citable, commercial-clean alternative.

## Limitations — honest caveats

- Synthetic rear-view ground truth uses the same SMPL kinematic model as the training data. A rear-view evaluation against synthetic truth primarily measures the model's geometric correctness, not its robustness to DWPose's real rear-view error distribution.
- For external/publication validation we still want at least one small real rear-view test set per sport. The synthetic path **eliminates the license blocker** but does not eliminate the need for a real 50–100 clip reference set per sport — that's now a smaller, separable data-collection problem.
- The pitching-specific rear-view configs need calibration against the rear-view targets in the `project_rear_view_calibration` memory note. 3D lab targets do not transfer to 2D rear-view; calibration is per-camera-angle, not per-sport-config alone.

## TL;DR

The synthetic Layer 2 pipeline IS the commercial-clean rear-view validation path. One config flag change to `_sample_camera()` and we're producing infinite labeled rear-view data for any pose distribution we can biased-sample. This is the v2 doc's footnoted "commercial-clean rear-view validation pending via synthetic AMASS pipeline" claim, now made concrete and runnable.
