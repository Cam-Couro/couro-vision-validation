# AMASS Synthetic Data Pipeline — Plan

> Goal: render Halpe-26 keypoints + 3D joint-angle ground truth from AMASS
> motion at arbitrary camera angles (especially rear views) to provide
> training data for Couro's weaker CV biomech slots.

## TL;DR

- Proof-of-concept renderer is built and working end-to-end (synthetic
  skeleton -> Halpe-26 mapping -> rear-view camera projection -> GT angle
  extraction). See `proof_frame.json` and `proof_frame_viz.png`.
- The only swap-in needed for "real" AMASS data is replacing the synthetic
  skeleton generator with `smplx.create('smplh', ...).forward(...)`. The
  Halpe-26 mapping, camera projection, and angle extraction modules already
  consume the smplx 144-joint output layout natively.
- Blocker: SMPL+H model file and AMASS subjects both require academic
  registration at https://mano.is.tue.mpg.de/ and https://amass.is.tue.mpg.de/.
  Commercial use requires a separate license from MPG (ps-licensing@tue.mpg.de).

## Architecture

```
                    PROOF-OF-CONCEPT (built, runs today)
                    ===================================
SyntheticPose       generate_synthetic_joints()        (144, 3) array
(input angles) ---> procedural T-pose with hip/   ---> in smplx layout
                    knee/ankle/lumbar transforms

                              FULL PIPELINE (planned, swap-in)
                              ================================
AMASS .npz          smplx.create('smplh', ...)         (144, 3) array
(pose, betas,  ---> body_model(global_orient,    --->  in smplx layout
 trans, dmpls)      body_pose, betas, transl).
                    joints.detach().numpy()
                                |
                                v
                  -------------------------------
                  | Halpe-26 mapping             |
                  | - 25 direct indices          |
                  | - 1 synthesized (head_top)   |
                  | (halpe26_mapping.py)         |
                  -------------------------------
                                |
                                v
                  -------------------------------
                  | GT joint angles              |
                  | (joint_angles.py)            |
                  | hip flex, knee flex,         |
                  | ankle dorsiflex, hip adduct, |
                  | lumbar extension             |
                  -------------------------------
                                |
                                v
                  -------------------------------
                  | Virtual camera + projection  |
                  | (camera.py)                  |
                  | (azimuth, elevation, dist)   |
                  | -> Halpe-26 2D pixels        |
                  -------------------------------
                                |
                                v
                       proof_frame.json  +
                       proof_frame_viz.png
```

## Halpe-26 -> SMPL+H joint mapping

Of the 26 Halpe-26 keypoints, 25 are directly available in the smplx
extended joint output (the 144-joint array returned when no `joint_mapper`
is passed to `body_model.forward`). 1 requires synthesis:

| Halpe-26      | Source in smplx output    | Notes |
| ------------- | ------------------------- | ----- |
| nose          | idx 55 (vertex 332 on smplh mesh) | direct |
| L/R eye       | idx 57 / 56               | direct |
| L/R ear       | idx 59 / 58               | direct |
| L/R shoulder  | idx 16 / 17               | direct |
| L/R elbow     | idx 18 / 19               | direct |
| L/R wrist     | idx 20 / 21               | direct |
| L/R hip       | idx 1 / 2                 | direct |
| L/R knee      | idx 4 / 5                 | direct |
| L/R ankle     | idx 7 / 8                 | direct |
| neck          | idx 12                    | direct |
| hip_center    | idx 0 (pelvis)            | direct |
| L/R big toe   | idx 60 / 63               | direct |
| L/R small toe | idx 61 / 64               | direct |
| L/R heel      | idx 62 / 65               | direct |
| head_top      | --- synthesized ---       | head joint (15) + 0.13m along +Y skull axis (rotate by head orientation when implementing for real); could alternatively pick a specific vertex of the scalp mesh |

See `halpe26_mapping.py` for the dict.

## Implementation plan — proof to production

### Phase 0 — already done (in this session)
- [x] Install `smplx`, `numpy`, `trimesh`, `pyrender`, `matplotlib`
- [x] Build `halpe26_mapping.py` (Halpe-26 -> smplx index table)
- [x] Build `camera.py` (CameraIntrinsics, CameraPose, project_points)
- [x] Build `joint_angles.py` (5 angles from 144-joint array)
- [x] Build `synthetic_skeleton.py` (procedural placeholder for SMPL+H)
- [x] Build `render_proof_frame.py` (end-to-end driver)
- [x] Render proof_frame.json + viz at azimuth=180/elev=15/dist=4m

### Phase 1 — unblock SMPL+H (Cameron decision, ~1-2 days elapsed)
- [ ] Register at https://mano.is.tue.mpg.de/ (for SMPL+H model file)
- [ ] Download SMPL+H neutral model file (`SMPLH_neutral.npz`) and place
      at `synthetic_amass/models/smplh/SMPLH_neutral.npz`
- [ ] **Licensing decision**: registration is free for academic, but Couro
      is commercial. Options:
      (a) Use SMPL-Body (CC-BY 4.0, free for commercial) — simpler joint
          model (no hands), 24 joints instead of 52, but still produces all
          25 needed Halpe-26 indices for body+face+toes. RECOMMENDED for
          fastest path to production.
      (b) Email ps-licensing@tue.mpg.de for SMPL+H commercial license.
      (c) Use BEDLAM/AGORA pre-rendered images instead of rolling our own
          rendering — but loses arbitrary camera control.

### Phase 2 — single-pose real render (~0.5 day after model arrives)
- [ ] Replace `synthetic_skeleton.generate_synthetic_joints()` with smplx
      forward pass. Module becomes ~15 lines:
      ```python
      def generate_smplh_joints(pose_params, betas, transl):
          body = smplx.create(MODEL_PATH, model_type='smplh',
                              gender='neutral', use_pca=False)
          out = body(body_pose=pose_params, betas=betas, transl=transl)
          return out.joints[0].detach().numpy()  # (144, 3)
      ```
- [ ] Re-run `render_proof_frame.py` and verify GT angles match input
      pose params within rounding error.
- [ ] Spot-check viz against a known-good AMASS frame visualization.

### Phase 3 — AMASS ingestion (~1-2 days)
- [ ] Register at https://amass.is.tue.mpg.de/ (academic) or license
      commercially. Recommended free-academic subsets to start:
      - **BMLrub** (Biomotion Lab Rub) — free for research, 522 subjects
        of locomotion + general motion. Strongest free coverage.
      - **CMU** — academic via AMASS, with original CMU Mocap free
        commercially under their own terms (but AMASS's *fitted* version
        is still MPG-licensed).
      - **HumanEva**, **MPI_HDM05** — small, good for smoke tests.
- [ ] Build `amass_loader.py` to read AMASS .npz files:
      keys are `poses` (T x 156), `betas` (16,), `trans` (T x 3),
      `dmpls` (T x 8), `mocap_framerate` (scalar), `gender`.
- [ ] Subsample frames at the chosen output framerate (e.g., 30 fps).
- [ ] Per frame: feed pose/betas/trans to smplx, get joints, run through
      existing mapping + angle + projection pipeline.

### Phase 4 — multi-view batch renderer (~1-2 days)
- [ ] Build `multiview_batch.py`:
      - Takes a list of AMASS subjects + a list of camera configs.
      - Iterates frames, projects to each camera config.
      - Output schema: one JSON per (subject, camera, frame) with the
        same shape as `proof_frame.json`, plus subject_id, frame_idx,
        framerate, original AMASS source.
- [ ] Standard camera configs to render at first (5 views per subject):
      - Front (az=0, el=10)
      - Rear (az=180, el=15)
      - Sagittal L (az=90, el=10)
      - Sagittal R (az=270, el=10)
      - 3/4 angle (az=135, el=20)
- [ ] Add small jitter to camera params (+/- 5 deg azimuth, +/- 0.3m
      distance, +/- 50px principal point) to mimic real-world placement
      variability.

### Phase 5 — training-set generation + integration (~2-3 days)
- [ ] Choose target slot(s): pick from the 24/25 slots; start with the
      weakest data-scarce ones (slots with r=0.41-0.55).
- [ ] Render 1000 motion clips x 5 camera angles = 5000 trials. At
      ~50 frames/clip avg, that's ~250k frames. Output filesize estimate:
      ~150 bytes/keypoint * 26 kpts * 250k frames = ~1 GB JSON. Switch to
      parquet/numpy for production.
- [ ] Build `dataset_export.py` to convert the per-frame JSONs into the
      same training-tensor format the existing ridge regression
      validation harness expects.
- [ ] Re-run ridge regression on the augmented training set vs original
      baseline. Measure r-lift on the targeted slots.
- [ ] Ablation: train with each view added separately to quantify the
      marginal value of rear views.

### Phase 6 — realism upgrades (optional, ~2-4 days each)
- [ ] **Noise modeling**: add 2D Gaussian noise to projected keypoints
      calibrated to match DWPose's empirical error distribution at varying
      distances and resolutions. Currently confidence=1.0 across the board.
- [ ] **Occlusion**: zero-out confidences for self-occluded keypoints
      using a fast ray-cast through the SMPL+H mesh (trimesh ray test).
      Critical for rear view where face landmarks are partially occluded.
- [ ] **Realistic background rendering**: pyrender + textured SMPL+H
      mesh + background image. Required only if we want the *image*
      rendered (not just keypoints) for retraining DWPose end-to-end.
      For pure ridge-regression retraining, keypoints suffice.
- [ ] **Shape variation**: sample betas from a learned distribution to
      vary body proportions across the dataset.

## Realistic timeline

| Phase                              | Days (elapsed) |
| ---------------------------------- | -------------- |
| 0 — proof-of-concept (done)        | 0.5            |
| 1 — registration + model download  | 1-2            |
| 2 — single-pose real render        | 0.5            |
| 3 — AMASS ingestion                | 1-2            |
| 4 — multi-view batch renderer      | 1-2            |
| 5 — full training set + retrain    | 2-3            |
| **Subtotal to first r-lift number**| **6-9 days**   |
| 6 — realism upgrades (per upgrade) | 2-4            |

These are *engineering days* assuming one focused developer. Multi-week
calendar time if interleaved with other work.

## Blockers / open questions

1. **Commercial licensing.** SMPL/SMPL+H and AMASS are academic-only by
   default. SMPL-Body (CC-BY 4.0) is the cleanest commercial-OK path and
   covers all Halpe-26 needs. Decision needed: pursue SMPL-Body (easiest)
   or email MPG for commercial license (more capability, includes hands).
2. **Ground-truth angle definition.** The proof computes angles from 3D
   joint positions; we could alternately read them from SMPL+H pose
   params directly (axis-angle). Position-based matches how Couro extracts
   angles from CV reconstruction, so it's the right call for training
   consistency. Worth documenting as a deliberate choice.
3. **Rear-view depth ambiguity for some slots.** Rear views inherently
   can't recover sagittal-plane angles well (e.g., knee flexion). The
   r-lift hypothesis is mainly for frontal-plane slots (hip adduction,
   trunk lateral lean, foot strike width). Worth measuring per-slot
   r-lift, not just aggregate.
4. **No domain randomization yet.** The proof uses fixed camera and
   anthropometric defaults. Production needs camera jitter and SMPL+H
   shape sampling. Phases 4 and 6 cover this but cost time.
5. **smplx model file size.** ~250MB; don't commit to git. Use a download
   script with checksum verification.

## Files in this directory

| File                     | Purpose                                          |
| ------------------------ | ------------------------------------------------ |
| `halpe26_mapping.py`     | Halpe-26 -> smplx joint index table              |
| `camera.py`              | CameraIntrinsics, CameraPose, projection         |
| `joint_angles.py`        | 5 GT joint angles from 144-joint array           |
| `synthetic_skeleton.py`  | Procedural placeholder for SMPL+H (Phase 0 only) |
| `render_proof_frame.py`  | End-to-end driver, produces proof_frame.json     |
| `proof_frame.json`       | Proof artifact: Halpe-26 2D + GT angles          |
| `proof_frame_viz.png`    | Proof artifact: rendered skeleton at az=180      |
| `PIPELINE_PLAN.md`       | This document                                    |

## Honest assessment

**Worth pursuing** — but only if the licensing path is acceptable.
Engineering risk is low (the pipeline is straightforward once the model
file is in hand; the proof demonstrates all the hard parts work).

The decision hinges on:
- **Commercial license**: SMPL-Body (CC-BY 4.0) is the path of least
  resistance and almost certainly sufficient for Couro's needs. SMPL+H
  via MPG commercial license would cost time and money to negotiate.
- **Expected r-lift**: yesterday's roadmap estimated +0.05 to +0.15. The
  weakest slots (r=0.41) are mostly rear-view/frontal-plane metrics
  precisely the slots synthetic rear-view data should help most.
- **Cheaper alternative**: BEDLAM ships pre-rendered images with SMPL-X
  GT, but the camera angles are fixed by the BEDLAM dataset. If we want
  *arbitrary* camera control (which is the whole point), we need our
  own renderer.

Recommendation: pursue Phase 1 with SMPL-Body (CC-BY 4.0) immediately;
plan a 1-week sprint through Phases 1-5 to get the first measured r-lift
number before committing to Phase 6 polish.
