# Lever (c) — 3D positions as Layer-3 ridge features

**Date:** 2026-05-31
**Verdict:** **Honest fail.** No slot promoted to Good. Joins calibration fix, slope correction, variance reduction, and view-aware blend as the 5th unsuccessful Layer-3 improvement attempt.

## What was tested

The blend → Layer-3 integration report (v16) diagnosed why view-aware blending didn't promote slots:

> *"Layer-3 ridges distill the trace into scalar features (ROM, at_contact, phased samples). When the lifter wins it smooths the trace shape; when it loses it adds high-frequency noise."*

That report listed three unexplored levers. Lever (c) — "feed VideoPose3D's 3D positions directly as ridge features rather than collapsing through Layer-2 angles" — was the most structural.

We added 6 per-clip 3D summary features to the v9_phased baseline:
- pelvis–knee Z-distance max (forward stride depth)
- knee–knee X-distance range (frontal knee separation)
- hip–hip 3D width median (subject-size normalizer)
- right ankle Y range (foot lift)
- pelvis–shoulder distance range (trunk lean energy)
- right hip Z SD (sagittal hip motion energy)

LOSO ridge with alpha=10 (same regularization as v17). OpenCap-only for apples-to-apples comparison.

## Results — 5 Mid-Poor slots (CCC ≥ 0.60 in v17)

| Slot | Baseline CCC | Baseline LoA± | Lever (c) CCC | Lever (c) LoA± | Δ CCC | Δ LoA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hip_adduction_r / front_center | −0.31 | 13.5° | −0.34 | 14.5° | −0.03 | +0.97° |
| hip_adduction_r / front_oblique_right | +0.11 | 9.1° | +0.10 | 9.6° | −0.01 | +0.51° |
| knee_angle_r / front_oblique_left | +0.27 | 21.0° | −0.03 | 26.5° | **−0.30** | **+5.55°** |
| hip_flexion_r / front_oblique_right | +0.15 | 33.0° | +0.20 | 32.4° | +0.05 | −0.63° |
| **hip_flexion_r / side_left** | +0.42 | 22.6° | **+0.53** | **19.8°** | **+0.11** | **−2.84°** |

None promoted to Good (CCC > 0.60 AND LoA half < 10° required). One slot (hip_flexion_r/side_left) improved meaningfully (+0.11 CCC, −2.84° LoA) but remained Poor.

## The real finding — v17 numbers are mostly ASPset-driven

The OpenCap-only baselines in the table above look much weaker than the published v17 numbers for the same slots:

| Slot | Published v17 CCC (n=445) | OpenCap-only CCC (n≈32–54) |
| --- | ---: | ---: |
| hip_adduction_r / front_center | **0.77** | **−0.31** |
| hip_adduction_r / front_oblique_right | 0.78 | +0.11 |
| knee_angle_r / front_oblique_left | 0.78 | +0.27 |
| hip_flexion_r / front_oblique_right | 0.70 | +0.15 |
| hip_flexion_r / side_left | 0.60 | +0.42 |

ASPset contributes 413 of 445 training rows for the v12_combined approach (92%). The OpenCap rows alone carry near-zero or negative signal on hip_adduction_r/front_center and weak signal on knee_angle_r/front_oblique_left. The v17 CCC numbers are mostly carried by ASPset.

**Implications worth surfacing:**

1. **The hip-adduction-from-front-center signal isn't really in OpenCap drop-jump data.** Whether this is because front-center is geometrically wrong for hip adduction, or because OpenCap subjects didn't produce informative variation, the data says it's not there. ASPset's larger-N + outdoor athletic motion is filling the gap.
2. **Apples-to-apples Lever (c) testing requires ASPset 3D features.** ASPset has C3D markers but no synchronized video, so the 3D feature path would need to project ASPset's 3D markers through the appropriate virtual camera. That's a known synthetic-projection technique (Phase 1 of the AMASS pipeline plan covers similar territory).
3. **The +0.11 CCC lift on hip_flexion_r/side_left is the only signal-positive result.** Side-left is a side-view slot where the lifter typically loses, so this is a counterintuitive lift. Worth a closer look in a follow-up, possibly with subject11 outlier exclusion (the variance_reduction_pass report flagged the same subject driving leverage on knee_angle_r/front_oblique_right).

## Status of the 5-attempt Layer-3 promotion campaign

| # | Build | Lever | Slots promoted | Net Good |
| --- | --- | --- | --- | --- |
| 1 | Build #5 (Agent Q) | Calibration fix | 0 | 0 |
| 2 | Build #8 (Agent Z) | Slope correction | 0 | 0 |
| 3 | Variance reduction pass | Lever 1/2/3 sweep | 0 | 0 |
| 4 | v16 blend → Layer-3 | View-aware angle blend | +1 (selectively adopted as v17) | +1 |
| 5 | This: Lever (c) 3D features | 3D positions as features | 0 | 0 |

Net Good slot count after 5 attempts: **4** (the 3 v15 originals + lumbar/front_oblique_right from build #4).

## Honest next moves

The pattern across 5 attempts is consistent: **Layer-3 ridges resist Layer-2 improvements.** Scalar feature collapse + small n + LOSO regularization is a robust ceiling.

The realistic paths forward, ranked by expected payoff:

1. **Fresh data — same slot, larger cohort.** The Poor slots with high CCC (0.7–0.8) have CCC ranking right; the LoA gate is failing on small n with high subject variance. More subjects with mocap-grade GT would directly tighten LoA. Ankle cohort expansion (Agent V's audit) is the existing example.
2. **Per-athlete calibration as product feature.** Removes the +3 to +4° biases on the front-view hip-adduction slots without needing a model fix. Multi-week product UX work, not research.
3. **Lever (a) — per-clip selective blending.** Untried, lower expected payoff than Lever (c) but cheaper to test.
4. **Accept "monitoring-grade" vs "ranking-grade".** Some of these slots are honestly per-athlete consistent across sessions but cross-athlete noisy. That's a useful product framing.

## Artifacts

- Script: `harness/lever_c_3d_ridge_features.py`
- Per-slot results: `data/lever_c_3d_ridge/per_slot_results.json`
- This report.

## Reproduce

```
cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
python3 -m harness.lever_c_3d_ridge_features
# Runtime: ~2 min for 5 slots on CPU
```
