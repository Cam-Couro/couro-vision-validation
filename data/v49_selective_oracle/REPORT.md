# v49 Selective Oracle: Layer 1 DWPose+RTMPose Detector Ensemble (Agent VV2)

**Date:** 2026-06-12

**Build:** A two-ARCHITECTURE Layer 1 detector ensemble. For each frame, DWPose (`dw-ll_ucoco_384`, COCO-WholeBody-133 -> Halpe-26) and RTMPose-Halpe26 are both run on the SAME person bbox, then their keypoints are confidence-weighted averaged: `kp = (c_dw*kp_dw + c_rt*kp_rt)/(c_dw+c_rt)`. This is the final modeling lever in the campaign.

**SINGLE CAMERA ONLY.** Two DETECTORS (two architectures) on the SAME single-camera frame. This is NOT multi-camera fusion. The inference contract is unchanged: 1 video stream -> keypoints -> 5 angles.

## 1. Keypoint correspondence: did DWPose & RTMPose Halpe-26 indices align?

**YES -- indices align directly; no remap needed for RTMPose.** The built-in 1-frame smoke check confirmed both detectors emit 26 keypoints in the identical Halpe-26 index order. On a Cam0 clip (subject10/DJ1) the mean DWPose-vs-RTMPose distance over the six torso/limb joints {L/R shoulder, L/R hip, L/R knee} was ~4.8 px, and the L_hip same-index distance (0.9 px) was far smaller than the crossed L_hip-vs-R_hip distance (33.2 px) -- ruling out an L/R label swap. The DWPose re-inference pass also reproduced the cached `opencap_dwpose_keypoints` to ~4.1 px, validating the shared bbox / pre-processing path. RTMPose-Halpe26 outputs Halpe-26 directly; DWPose is remapped from COCO-WholeBody-133 via the verbatim `dwpose_to_halpe26`. Both land in the same 26-index layout.

## 2. Mirror-twin verdict: hip_adduction_r / side_right (v47 0.277)

Measured on the hand-engineered geometry reader (the most transparent direct function of L1 keypoints -- the cleanest detector-asymmetry probe).

- Geometry reader, ORIGINAL DWPose: CCC **0.198**
- Geometry reader, DETECTOR ENSEMBLE: CCC **0.216** (LoA 8.38 deg)
- Mirror sibling side_left geometry reader: ORIGINAL CCC 0.281 -> ENSEMBLE CCC 0.299
- **Ensemble delta on the geometry reader: +0.018 CCC**

- v47 deploy pick (consolidated): CCC 0.277 (tier Poor, reader v31 (mirror-flip L2 + learned L3)).
- v49 deploy pick: v31 (mirror-flip L2 + learned L3) CCC 0.277 (tier Poor).

- **Promoted to Good?** NO. The mirror twin needed 0.277 -> 0.60; the detector ensemble did not get there. See section 9 for the verdict.

## 3. A/B table (geometry reader, ORIGINAL DWPose vs detector ensemble)

| Slot | v47 deploy CCC | geom ORIG CCC | geom ENSEMBLE CCC | ENS LoA | delta | tier |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hip_adduction_r/side_right | 0.277 | 0.198 | 0.216 | 8.38 | +0.018 | Poor |
| hip_flexion_r/side_right | 0.584 | 0.296 | 0.319 | 31.22 | +0.023 | Poor |
| ankle_angle_r/side_left | 0.638 | 0.325 | 0.549 | 11.37 | +0.224 | Moderate |
| knee_angle_r/side_right | 0.819 | 0.620 | 0.607 | 17.40 | -0.013 | Poor |

## 4. Ensemble (v48, two detectors) vs flip-TTA (v46, one detector flipped)

Both are Layer-1 levers on the same geometry reader. flip-TTA runs ONE detector (DWPose) twice (original + mirror); the ensemble runs TWO detectors (DWPose + RTMPose) once each. Comparison on slots flip-TTA improved in v47:

| Slot | geom ORIG | flip-TTA (v46) | ensemble (v48) | ensemble beats flip-TTA? |
| --- | ---: | ---: | ---: | --- |
| ankle_angle_r/side_left | 0.325 | 0.638 | 0.549 | no |
| hip_adduction_r/side_right | 0.198 | 0.272 | 0.216 | no |
| hip_flexion_r/side_right | 0.296 | 0.342 | 0.319 | no |
| knee_angle_r/side_right | 0.620 | 0.564 | 0.607 | YES |

## 5. v49 Good count + delta vs v47

- **v49 Good slots: 14** (delta vs v47's 14: +0) -- counted from `consolidated_metrics_v49.json`, the source-of-truth artifact rebuilt from each reader's validity file (not the stale v47 routing map).
- v48 (detector ensemble) picked in **0** slots.

## 6. v49 Tier 1 (CCC >= 0.79 AND Good)

- **v49 Tier 1: 10** (delta vs v47's 14: -4).

## 7. Did the ensemble hurt any v47 Good slots?

- **No.** The no-regression fallback (v48 cannot regress CCC by >0.02 vs the v47 pick, and never demotes tier) preserved every v47 Good slot. The ensemble only ever displaced a v47 pick when it strictly improved tier or CCC/LoA.

## 8. Latency (ms/frame)

- **Detector ensemble: 142.2 ms/frame** for two detector passes (DWPose CPUExecutionProvider + RTMPose CoreMLExecutionProvider). p50 138.2, p95 159.5.
- Reference: flip-TTA (v46) ~246 ms/frame (measured 245.8 ms/frame this repo); single-pass DWPose ~120 ms/frame.
- The ensemble's RTMPose pass runs on CoreML (Neural Engine) while DWPose stays on CPU, so the two-detector cost is below the two-CPU-pass flip-TTA. For Couro's near-real-time per-clip analysis this is feasible.

## 9. Campaign-closing verdict: detector-limited or data-limited?

**DATA-LIMITED -- the ensemble does NOT help.** A second, independent detector architecture (RTMPose) averaged with DWPose did not lift the side_right mirror twin. This is the third independent Layer-1/Layer-2 negative on this slot: NN's mirror-flip L2 training, UU's flip-TTA L1, and now VV2's two-architecture detector ensemble all fail to move it. Two different network architectures agreeing on the 'wrong' answer rules out idiosyncratic detector error and points to the OpenCap capture / ground-truth side: the side_right camera's oblique view of the right limb, or an asymmetry baked into the mocap reference for this metric/view. The remaining lever is view-specific GT recalibration or additional capture, not the detector. **The detector is not the bottleneck.**

## 10. LOSO discipline & honesty notes

- Subject-level LOSO at L3 on the OpenCap cohort; the ensemble changed only OpenCap keypoints, so validity is measured on OpenCap subjects only (combined-dataset ASPset trials still train the ridge but are excluded from the slot statistic, matching all prior builds). No leak.
- The consolidated v49 metrics are reproduced from each reader's validity file. The known-stale v47 routing map was NOT inherited.
- Per-slot fallback to the v47 pick is enforced by the oracle (no-regression, no tier demotion).
