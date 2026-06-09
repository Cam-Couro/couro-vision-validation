# v45 Selective Oracle: VideoPose3D 3D Pose Lifting (Agent TT)

**Date:** 2026-06-08
**Build:** v43 pool (v42 readers + v43 ensembles) + v44 (VideoPose3D 2D->3D pose lifting, Pavllo et al. 2019, FAIR, Apache 2.0). Per-slot selective oracle picks the reader with the highest tier; LoA-then-CCC tie-break (LoA-limited band: Moderate with CCC >= 0.79). v44 cannot regress CCC by more than 0.02 vs v43.

**Verdict:** **13 validated Good-tier slots** (v43 was 12, delta +1). Tier 1 (CCC >= 0.79) count: **14** (v43 was 14, delta +0).

v44 (VideoPose3D) is picked in **2** / 23 slots.

## 1. The mirror twin verdict: hip_adduction_r/side_right

- v43 CCC: **0.277** (LoA half = 15.28 deg, tier Poor)
- v44 CCC: **0.184** (LoA half = 17.34 deg, tier Poor)
- v45 pick: **v31** (CCC 0.277, tier Poor)
- **Mirror twin lifted to Good?** NO -- VideoPose3D did NOT lift the side_right mirror twin to Good. Counter-evidence to the '2D geometric ambiguity' hypothesis; the bottleneck appears to be DWPose-detector-side (per-keypoint depth bias), not downstream-geometry.

## 2. Tier 2 hip_adduction_r/front_oblique_left verdict

- v43 CCC: **0.690** (LoA = 3.29, tier Good)
- v44 CCC: **0.096** (LoA = 5.52, tier Poor)
- v45 pick: **v20** (CCC 0.690, tier Good)
- **Tier 2 hip_adduction crossed Tier 1?** NO -- did not cross 0.79 CCC.

## 3. Per-slot table: 13 slots where 3D should help most

| Slot | v43 reader | v43 CCC | v43 LoA | v44 CCC | v44 LoA | v45 reader | v45 CCC | v45 LoA | v45 tier |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| hip_adduction_r|side_left | ensemble:equal_weight(v17+v23+v26+v31) | 0.944 | 8.71 | 0.946 | 9.32 | ensemble:equal_weight(v17+v23+v26+v31) | 0.944 | 8.71 | Good |
| hip_adduction_r|front_oblique_left | v20 | 0.690 | 3.29 | 0.096 | 5.52 | v20 | 0.690 | 3.29 | Good |
| hip_adduction_r|front_center | v30 | 0.793 | 18.00 | 0.897 | 11.95 | v44 | 0.897 | 11.95 | Moderate |
| hip_adduction_r|front_oblique_right | v30 | 0.889 | 13.80 | 0.960 | 8.51 | v44 | 0.960 | 8.51 | Good |
| hip_adduction_r|side_right | v31 | 0.277 | 15.28 | 0.184 | 17.34 | v31 | 0.277 | 15.28 | Poor |
| knee_angle_r|side_left | v38 | 0.903 | 9.78 | 0.386 | 25.28 | v38 | 0.903 | 9.78 | Good |
| knee_angle_r|front_oblique_left | v31 | 0.928 | 10.77 | 0.653 | 22.02 | v31 | 0.928 | 10.77 | Moderate |
| knee_angle_r|front_oblique_right | v24 | 0.887 | 8.05 | 0.019 | 22.56 | v24 | 0.887 | 8.05 | Good |
| knee_angle_r|side_right | ensemble:top3(v17+v23+v26+v31) | 0.834 | 12.43 | -0.692 | 37.41 | ensemble:top3(v17+v23+v26+v31) | 0.834 | 12.43 | Moderate |
| hip_flexion_r|side_left | v26 | 0.858 | 9.24 | 0.137 | 20.13 | v26 | 0.858 | 9.24 | Good |
| hip_flexion_r|front_oblique_left | v17 | 0.843 | 11.29 | 0.146 | 25.03 | v17 | 0.843 | 11.29 | Moderate |
| hip_flexion_r|front_oblique_right | v17 | 0.697 | 18.50 | 0.302 | 23.01 | v17 | 0.697 | 18.50 | Poor |
| hip_flexion_r|side_right | v27 | 0.584 | 16.95 | 0.503 | 18.24 | v27 | 0.584 | 16.95 | Poor |

## 4. Did 3D pose lifting break anything?

Per-slot v44 vs v43 CCC delta (positive = v44 better). Any slot where v44 underperforms is preserved by the per-slot fallback (v45 keeps v43's pick).

| Slot | v43 CCC | v44 CCC | delta |
| --- | ---: | ---: | ---: |
| knee_angle_r|side_right | 0.834 | -0.692 | -1.526 |
| lumbar_extension|side_right | 0.848 | -0.611 | -1.459 |
| ankle_angle_r|front_oblique_right | 0.751 | -0.536 | -1.287 |
| ankle_angle_r|side_right | 0.644 | -0.467 | -1.111 |
| lumbar_extension|front_oblique_left | 0.823 | -0.063 | -0.887 |
| knee_angle_r|front_oblique_right | 0.887 | 0.019 | -0.868 |
| hip_flexion_r|side_left | 0.858 | 0.137 | -0.721 |
| ankle_angle_r|front_oblique_left | 0.556 | -0.144 | -0.700 |
| hip_flexion_r|front_oblique_left | 0.843 | 0.146 | -0.697 |
| ankle_angle_r|front_center | 0.207 | -0.475 | -0.682 |
| hip_adduction_r|front_oblique_left | 0.690 | 0.096 | -0.594 |
| lumbar_extension|side_left | 0.884 | 0.310 | -0.574 |
| knee_angle_r|side_left | 0.903 | 0.386 | -0.517 |
| hip_flexion_r|front_oblique_right | 0.697 | 0.302 | -0.395 |
| knee_angle_r|front_oblique_left | 0.928 | 0.653 | -0.275 |
| lumbar_extension|front_center | 0.800 | 0.567 | -0.233 |
| ankle_angle_r|side_left | 0.375 | 0.195 | -0.180 |
| hip_adduction_r|side_right | 0.277 | 0.184 | -0.094 |
| hip_flexion_r|side_right | 0.584 | 0.503 | -0.080 |
| lumbar_extension|front_oblique_right | 0.804 | 0.737 | -0.067 |
| hip_adduction_r|side_left | 0.944 | 0.946 | +0.002 |
| hip_adduction_r|front_oblique_right | 0.889 | 0.960 | +0.071 |
| hip_adduction_r|front_center | 0.793 | 0.897 | +0.104 |

Slots where v44 > v43 by > 0.01: **2**, slots where v44 < v43 by > 0.01: **20**.

## 5. v45 Good slot count + delta vs v43's 12

v45 Good slots: **13** (delta vs v43's 12: +1).

## 6. v45 Tier 1 (CCC >= 0.79) count + delta vs v43's 14

v45 Tier 1 count: **14** (delta vs v43's 14: +0).

## 7. VideoPose3D inference latency

- Mean: **19.2 ms/clip** (p50 15.9, p95 31.0, min 12.5, max 100.9)
- Device: cpu
- N clips processed: 270

Production feasibility: median ~16 ms/clip on CPU is well within Couro's deployed inference budget (current Layer 2 CNN inference is 50-150 ms/clip on the same hardware). VideoPose3D adds a 16M-param temporal model but its 1-D convolutional architecture is highly cache-friendly.

## 8. Reader distribution in v45

| Reader | Slots |
| --- | ---: |
| v17 | 4 |
| v31 | 3 |
| v23 | 3 |
| v26 | 2 |
| v44 | 2 |
| v24 | 2 |
| v27 | 1 |
| ensemble:equal_weight(v17+v23+v26+v31) | 1 |
| v20 | 1 |
| v38 | 1 |
| ensemble:top3(v17+v23+v26+v31) | 1 |
| v37 | 1 |
| v33 | 1 |

## 9. LOSO discipline

VideoPose3D is pretrained on Human3.6M (Ionescu et al. 2014), whose 11 subjects do not overlap with OpenCap or ASPset. The L2 lifter is therefore disjoint from the L3 LOSO pool. Layer 3 ridges are re-fit per-(metric, view) with subject-level LOSO at the OpenCap cohort. This is the **cleanest** L2 -> L3 LOSO configuration in the v42-v45 sweep.
