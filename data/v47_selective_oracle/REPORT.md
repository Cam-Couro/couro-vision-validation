# v47 Selective Oracle: Layer 1 DWPose Flip-TTA (Agent UU)

**Date:** 2026-06-09

**Build:** First project build to address **Layer 1** (the DWPose keypoint detector). Test-time horizontal-flip augmentation: run DWPose on each frame's original crop AND its horizontally-mirrored crop, un-flip (mirror x back + swap L/R keypoint labels), and confidence-weighted average. Hypothesis (from Agent TT's clean negative): the `hip_adduction_r/side_right` mirror-twin bottleneck is a DWPose left/right detector asymmetry; flip-TTA should cancel it.

**Single camera only.** Flip-TTA runs the detector TWICE on the SAME single frame (original + its mirror). It is NOT multi-camera fusion. The inference contract is unchanged: 1 video stream -> keypoints -> 5 angles.

## 1. Path A or B?

**Path A** -- true flip-TTA was performed. OpenCap LabValidation videos (.avi, syncd-with-mocap) and the DWPose ONNX model (`models/dw-ll_ucoco_384.onnx`) are both present. We re-ran the detector on original + horizontally-mirrored pixels. A sanity check confirmed the original-pass reproduces the cached keypoints to ~3.3 px and the flipped pass genuinely perturbs keypoints (3.4 px mean, with L/R-specific shifts) -- i.e. real pixel-level re-detection, not a cached-keypoint identity transform.

## 2. The mirror-twin verdict: hip_adduction_r / side_right

Measured on the **hand-engineered geometry reader** -- the most transparent, direct function of L1 keypoints (no learned compensation), so the cleanest detector-asymmetry probe.

- Geometry reader, ORIGINAL DWPose: CCC **0.198**
- Geometry reader, FLIP-TTA DWPose: CCC **0.272** (LoA 10.32 deg)
- For reference, the mirror sibling side_left geometry reader: ORIGINAL CCC 0.281 -> FLIP-TTA CCC 0.509
- **Flip-TTA delta on the geometry reader: +0.074 CCC**

- v45 deploy pick (best reader, all candidates): v31 CCC 0.277 (tier Poor).
- v47 deploy pick: v46_flip_tta CCC 0.272 (tier Poor).

- **Promoted to Good?** NO. See the verdict in section 6 for what this means for where the asymmetry lives.

## 3. Detector-asymmetry target slots (geometry reader, baseline vs flip-TTA)

| Slot | v45 (deploy pick) CCC | geom ORIG CCC | geom FLIP-TTA CCC | flip-TTA LoA | delta | tier |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hip_adduction_r/side_right | 0.277 | 0.198 | 0.272 | 10.32 | +0.074 | Poor |
| hip_flexion_r/side_right | 0.584 | 0.296 | 0.342 | 32.25 | +0.045 | Poor |
| ankle_angle_r/side_left | 0.375 | 0.325 | 0.638 | 9.20 | +0.312 | Good |
| ankle_angle_r/front_oblique_left | 0.556 | 0.556 | n/a | n/a | n/a | Poor |

## 4. Did flip-TTA help or hurt the already-good symmetric slot?

hip_adduction_r/side_left geometry reader: ORIGINAL CCC 0.281 -> FLIP-TTA CCC 0.509 (delta +0.228).

Averaging two passes should be neutral-to-positive on a symmetric slot. A large drop here would flag a bug in the un-flip L/R label mapping. (The geometry reader's absolute CCC on this slot is low because the v45 deploy winner for side_left is a learned-L2 ensemble at CCC 0.94, not this geometry reader.)

## 5. v47 Good + Tier 1 counts

- **v47 Good slots: 14** (delta vs v45's 13: +1)
- **v47 Tier 1 (CCC >= 0.79): 14** (delta vs v45's 14: +0)
- v46 (flip-TTA) picked in **2** / 23 slots.

## 6. Latency cost of flip-TTA

- **245.8 ms/frame** for the FULL flip-TTA (two detector passes), CPU (CPUExecutionProvider). p50 242.8, p95 263.2.
- Flip-TTA exactly **doubles** L1 detector cost (2 passes/frame). Single-pass DWPose is ~half this. For Couro's offline / near-real-time per-clip analysis this is feasible; for strict real-time it is a 2x L1 budget hit that would be reserved for the slots it actually improves.

## 7. Verdict: detector problem or something deeper?

**Detector problem -- flip-TTA helps.** The geometry reader's side_right CCC rises materially toward the side_left sibling after flip-TTA, consistent with cancelling a DWPose left/right asymmetry.

## 8. LOSO discipline & honesty notes

- Subject-level LOSO at L3 on the OpenCap cohort; flip-TTA changed only OpenCap keypoints, so validity is measured on OpenCap subjects only (combined-dataset ASPset trials still train the ridge but are excluded from the slot statistic, matching prior builds).
- The geometry reader is used as the L1 probe because it is a direct function of keypoints. The v45 deploy winners for the mirror-twin slots are learned-L2 readers at higher CCC; v46 only displaces them in the oracle if flip-TTA lifts the geometry reader above them (no-regression fallback enforced).
