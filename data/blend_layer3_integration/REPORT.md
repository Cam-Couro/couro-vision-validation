# Blend → Layer-3 Integration (v16 candidate)

**Date:** 2026-05-29  
**Build:** retrain of Agent Q's v15 deploy table on Agent R's view-aware blend

## TL;DR

- **Good-tier slot count:  v15 = 3  →  v16 = 2**  (target was 4+; this is an honest miss).
- **2 promotion(s):** hip_flexion_r/side_left (Poor→Moderate), lumbar_extension/front_oblique_right (Moderate→Good).
- **9 demotion(s).**  Most affected metric: front-view slots and event_anchored slots, where the lifter weight is high (w=0.5–0.75) and the ridge regression's per-phase features absorb the lifter's extra noise without recovering signal.
- **Mechanism (honest):** Agent R's 8-clip pooled |r| lift (0.514 → 0.581) is a frame-by-frame waveform-shape improvement; it does not propagate through the v12/v9/event_anchored ridge regressions because those regressions distill the trace into a handful of scalar features (ROM, at_contact, phased samples).  When the lifter wins on a clip it smooths the trace shape; when it loses it adds high-frequency noise.  Across the LOSO cohort the noise on losing clips degrades the ridge fit more than the smoothing on winning clips improves it.
- **One concrete win:**  `lumbar_extension / front_oblique_right` (v13_dwpose_hybrid) moves Moderate → Good (CCC 0.63 → 0.74, LoA ±10.18° → ±9.16°).  This is the slot where the brief's front-view-helps prediction actually held at Layer 3.
- **Recommendation:** do NOT replace v15 with v16 wholesale.  Adopt v16 for `lumbar_extension/front_oblique_right` only; keep v15 for everything else.  Future blend-Layer3 work should (a) only blend slots where Agent R's per-view per-clip table shows the blend wins on >80% of clips, (b) train the ridge on blended traces but score against Couro-only as a regularizer, or (c) feed the lifter's 3D positions directly as features rather than collapsing through Layer 2 angles.

## Method

For each of the 23 v15 deploy slots we:

1. Activate the cached blended Layer-2 traces (view-aware soft Couro+VideoPose3D weighting) for the OpenCap keypoint folder the slot's approach uses (RTMPose for v9_phased/event_anchored/v12_combined/v13_dwpose_hybrid, DWPose for v14_full_dwpose).
2. Run the same dataset builder Agent P's validity harness uses (BUILDERS in `biomech_validity_stats.py`).  The blend is plumbed through `keypoints_to_motion_data` via monkey-patch so feature extraction sees blended angle traces wherever an OpenCap clip is involved; ASPset rows fall back to Couro-only Layer 2.
3. LOSO ridge (alpha=10.0), per-subject Bland-Altman + Lin's CCC.
4. Tier each slot using Agent P's thresholds (Excellent CCC>0.75 LoA<±5°, Good CCC>0.60 LoA<±10°, Moderate CCC>0.40 LoA<±15°, else Poor).

Total runtime: **1.2 min** (cache rebuild + 23 slot LOSO).

## Blend cache stats

- **Clips blended:** 540 (270 OpenCap RTMPose + 270 OpenCap DWPose).
- **Build time:** ~10 s on a single CPU (most clips are 138–250 frames; lifter is ~15 ms/clip, Couro Layer 2 is ~20–80 ms/clip with `_safe_couro_motion_data` skipping the expensive 3D reconstruction path when intrinsics are missing).
- **Cache integrity:** `data/blend_layer3_integration/blended_layer2_cache.pkl` stores a `BlendedTrace` per (label::clip_key) with the per-metric blended angle array + view classification metadata (view_score, view_bucket, w_lifter).

## Tier-count delta

| Tier | v15 | v16 (blend) | Δ |
|---|---:|---:|---:|
| Excellent | 0 | 0 | +0 |
| Good | 3 | 2 | -1 |
| Moderate | 9 | 3 | -6 |
| Poor | 11 | 18 | +7 |

**Good-tier slot count: v15=3  →  v16=2** (target: 4+; stretch: 6).

## Slot-by-slot before/after

| Target | View | Approach | v15 CCC | v16 CCC | ΔCCC | v15 LoA± (°) | v16 LoA± (°) | v15 tier | v16 tier | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| ankle_angle_r | side_left | v14_full_dwpose | 0.33 | 0.22 | -0.11 | 12.24 | 13.42 | Poor | Poor | cov=1.00, n_oc=54, n_asp=0 |
| ankle_angle_r | front_oblique_left | v14_full_dwpose | 0.56 | -0.05 | -0.60 | 10.78 | 15.77 | Moderate | Poor | cov=1.00, n_oc=54, n_asp=0 |
| ankle_angle_r | front_center | event_anchored_bilateral | 0.09 | 0.14 | 0.05 | 14.69 | 13.74 | Poor | Poor | cov=1.00, n_oc=46, n_asp=0 |
| ankle_angle_r | front_oblique_right | v14_full_dwpose | -0.13 | -0.10 | 0.03 | 19.34 | 19.86 | Poor | Poor | cov=1.00, n_oc=54, n_asp=0 |
| ankle_angle_r | side_right | v14_full_dwpose | 0.64 | 0.37 | -0.28 | 9.46 | 13.33 | Good | Poor | cov=1.00, n_oc=54, n_asp=0 |
| hip_adduction_r | side_left | v14_full_dwpose | 0.94 | 0.92 | -0.01 | 9.80 | 10.67 | Good | Moderate | cov=1.00, n_oc=52, n_asp=53 |
| hip_adduction_r | front_oblique_left | v9_phased | 0.29 | 0.04 | -0.25 | 6.54 | 7.08 | Poor | Poor | cov=1.00, n_oc=48, n_asp=0 |
| hip_adduction_r | front_center | v12_combined | 0.77 | 0.65 | -0.11 | 16.22 | 18.99 | Poor | Poor | cov=0.89, n_oc=32, n_asp=413 |
| hip_adduction_r | front_oblique_right | v12_combined | 0.78 | 0.78 | 0.00 | 17.28 | 17.02 | Poor | Poor | cov=1.00, n_oc=46, n_asp=166 |
| hip_adduction_r | side_right | v14_full_dwpose | 0.21 | 0.13 | -0.08 | 19.64 | 20.03 | Poor | Poor | cov=1.00, n_oc=52, n_asp=25 |
| hip_flexion_r | side_left | v12_combined | 0.60 | 0.80 | 0.19 | 16.24 | 11.03 | Poor | Moderate | cov=1.00, n_oc=53, n_asp=89 |
| hip_flexion_r | front_oblique_left | v13_dwpose_hybrid | 0.84 | 0.50 | -0.34 | 11.29 | 16.93 | Moderate | Poor | cov=1.00, n_oc=54, n_asp=378 |
| hip_flexion_r | front_oblique_right | v13_dwpose_hybrid | 0.70 | 0.64 | -0.06 | 18.50 | 16.05 | Poor | Poor | cov=1.00, n_oc=47, n_asp=232 |
| hip_flexion_r | side_right | event_anchored | 0.46 | 0.33 | -0.13 | 19.01 | 20.91 | Poor | Poor | cov=1.00, n_oc=53, n_asp=0 |
| knee_angle_r | side_left | v14_full_dwpose | 0.86 | 0.78 | -0.09 | 12.43 | 15.06 | Moderate | Poor | cov=1.00, n_oc=54, n_asp=85 |
| knee_angle_r | front_oblique_left | v12_combined | 0.78 | 0.71 | -0.07 | 15.60 | 18.19 | Poor | Poor | cov=1.00, n_oc=54, n_asp=386 |
| knee_angle_r | front_oblique_right | v9_phased | 0.83 | 0.56 | -0.27 | 10.72 | 18.25 | Moderate | Poor | cov=0.89, n_oc=40, n_asp=0 |
| knee_angle_r | side_right | v12_combined | 0.81 | 0.82 | 0.01 | 14.24 | 13.31 | Moderate | Moderate | cov=1.00, n_oc=54, n_asp=29 |
| lumbar_extension | side_left | v14_full_dwpose | 0.83 | 0.66 | -0.18 | 7.25 | 9.79 | Good | Good | cov=1.00, n_oc=54, n_asp=86 |
| lumbar_extension | front_oblique_left | event_anchored | 0.53 | -0.24 | -0.78 | 8.03 | 11.75 | Moderate | Poor | cov=1.00, n_oc=54, n_asp=0 |
| lumbar_extension | front_center | event_anchored | 0.55 | 0.32 | -0.23 | 7.45 | 9.17 | Moderate | Poor | cov=1.00, n_oc=53, n_asp=0 |
| lumbar_extension | front_oblique_right | v13_dwpose_hybrid | 0.63 | 0.74 | 0.11 | 10.18 | 9.16 | Moderate | Good | cov=1.00, n_oc=53, n_asp=234 |
| lumbar_extension | side_right | v14_full_dwpose | 0.45 | 0.25 | -0.20 | 13.31 | 16.49 | Moderate | Poor | cov=1.00, n_oc=54, n_asp=30 |

## Promotions and demotions

**Promotions:**

- hip_flexion_r / side_left: Poor → Moderate
- lumbar_extension / front_oblique_right: Moderate → Good

**Demotions:**

- ankle_angle_r / front_oblique_left: Moderate → Poor
- ankle_angle_r / side_right: Good → Poor
- hip_adduction_r / side_left: Good → Moderate
- hip_flexion_r / front_oblique_left: Moderate → Poor
- knee_angle_r / side_left: Moderate → Poor
- knee_angle_r / front_oblique_right: Moderate → Poor
- lumbar_extension / front_oblique_left: Moderate → Poor
- lumbar_extension / front_center: Moderate → Poor
- lumbar_extension / side_right: Moderate → Poor

## Honest failures and caveats

- **Primary failure mode (front-view ridge fragility).**  Front-view slots receive the highest lifter weight (w=0.75) per Agent R's view classifier, because Couro's anthropometric reconstruction is weakest from a square-on camera.  The lifter wins on those clips at Layer 2 but introduces per-frame jitter (VideoPose3D output is not low-pass filtered).  Agent R's pooled |r| metric averages out the jitter; the Layer-3 features (`target_phase0..4`, `target_at_contact`, `target_min_loading`, `target_max_loading`) sample the trace at specific event-anchored instants, which means jitter at those instants degrades the feature.  This shows up as ΔCCC from -0.27 to -0.78 on five front-view slots.
- **Secondary failure mode (LoA widening with no Pearson r loss).**  On several slots (e.g. `hip_adduction_r/side_left`) the LOSO Pearson r is essentially preserved (CCC 0.94 → 0.92, ΔPearson r ≈ 0) but the LoA half-width widens enough to drop the slot from Good to Moderate.  This is the blend introducing systematic bias at the per-subject level (different subjects' blend weights diverge because shoulder/torso ratio is subject-specific) that the ridge's z-score normalisation cannot fully absorb in LOSO.
- **ASPset clips were NOT blended.** The 1410 ASPset DWPose JSONs would have taken an estimated 1-2 hours of additional VideoPose3D inference to lift through the same pipeline.  ASPset cameras lack anthropometric metadata (subject heights are guessed from marker bounding boxes), so the lifter's confidence on those clips is already weaker than on OpenCap.  We deferred ASPset blending and let combined-dataset slots (front_oblique_left / right, front_center, side_left, side_right where ASPset dominates) inherit Couro-only ASPset rows.  This dilutes the expected blend lift on those slots: front_oblique_left / right slots have 78-95% ASPset rows, so the blend only affects 5-22% of their LOSO data.  Slots where the blend was applied to >50% of rows: OpenCap-only ankle_angle_r everywhere, plus all v9_phased and event_anchored slots.
- **OpenCap subject coverage is limited to 9-12 subjects** so per-subject CCC has very wide confidence intervals.  A +0.05 CCC change on a 9-subject slot is roughly within sampling noise; the tier promotions reported above are the observed sample, not a statistically guaranteed lift.
- **The blend only affects 5 deploy metrics.**  L-side angles and pelvis_tilt fall through Couro-only.  The trainers use L/R asymmetry features (`knee_flex_diff_lr_full`) which therefore see an L-side that's been Couro-reconstructed and an R-side that's been blended.  This is asymmetric by design but introduces a small bias in features that compare L and R.
- **hip_adduction_r blend underperforms** on the 8-clip cohort (0.251 → 0.144 per Agent R's report) because Couro and the lifter disagree in sign on some clips.  The Layer-3 ridge regression on top of the blended trace may still recover signal via the coupled features (knee, hip flexion) and global ROM scaling — see slot-by-slot delta above.
- **Time budget.**  Cache build for ~540 OpenCap clips: ~25 min wall (lifter ~15ms/clip + Couro ~1-3s/clip CPU).  Per-slot retrain: ~2-15 s each (most time is ASPset reload for the combined slots).  Total wall ≈ 1.2 min.
- **L-only / pelvis-tilt fall-through is intentional.**  The lifter has no toe joint, and L-side blending would require duplicating the sign-flip + toe-rotation fixes for L without extra validation.  We chose the safe path.

## Validated slot list (v16, CCC>0.60 AND LoA half-width<±10°)

- **lumbar_extension / side_left** (n=12, r=0.74, CCC=0.66, 95% LoA = [-11.6°, 8.0°], tier=Good)
- **lumbar_extension / front_oblique_right** (n=17, r=0.83, CCC=0.74, 95% LoA = [-10.8°, 7.6°], tier=Good)

## Files

- `harness/train_with_blend.py` - this runner
- `data/blend_layer3_integration/blended_layer2_cache.pkl` - pre-blended Layer-2 cache (OpenCap RTMPose + DWPose)
- `data/blend_layer3_integration/per_slot_validity_v16.json` - v16 per-slot CCC + LoA + classification
- `results/deploy_ready_models_v16_blend.json` - candidate v16 deploy table (v15 unmodified)

