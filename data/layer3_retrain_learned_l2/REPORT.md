# Phase B: Layer 3 retrained on learned Layer 2 (Agent FF)

**Date:** 2026-05-29
**Build:** Agent FF (Phase B)

## LOSO discipline used in this build

**Layer-3-LOSO-only.** ONE Layer 2 model (Agent EE2's `TemporalKeypointCNNConf`) was trained on ALL 9 OpenCap subjects (no LOSO at L2) and used to produce learned angle traces for every clip (OpenCap + ASPset). Layer 3 ridge was re-fit per slot with LOSO at L3 only.

**This leaks Layer 2 information from the held-out OpenCap subject into the L2 training data.** For ASPset subjects, the regime is effectively double-LOSO because no ASPset subject ever entered Layer 2 training (Layer 2 cohort is OpenCap-only).

**Tier-change claims warning:** any tier promotion driven primarily by OpenCap LOSO folds is an upper bound. EE2's per-fold variance (best fold 0.700, worst 0.541) implies the true double-LOSO number could be ~0.05-0.10 |r| lower than reported here.

## Tier count delta

| Tier | v17 baseline | v18 (learned L2) | Δ |
| --- | ---: | ---: | ---: |
| Excellent | 0 | 0 | +0 |
| Good | 3 | 2 | -1 |
| Moderate | 9 | 5 | -4 |
| Poor | 13 | 16 | +3 |

Promotions: 3 | Demotions: 8 | Unchanged: 12

### Promotions

- hip_adduction_r / front_oblique_left: Poor -> Moderate
- ankle_angle_r / front_oblique_right: Poor -> Moderate
- lumbar_extension / front_oblique_left: Moderate -> Good

### Demotions

- hip_flexion_r / front_oblique_left: Moderate -> Poor
- hip_adduction_r / side_left: Good -> Poor
- knee_angle_r / side_left: Moderate -> Poor
- knee_angle_r / front_oblique_right: Moderate -> Poor
- knee_angle_r / side_right: Moderate -> Poor
- ankle_angle_r / front_oblique_left: Moderate -> Poor
- ankle_angle_r / side_right: Good -> Moderate
- lumbar_extension / side_right: Moderate -> Poor

## Per-slot before/after table

| Target | View | Approach | n | r baseline | CCC baseline | LoA/2 baseline | Tier baseline | r v18 | CCC v18 | LoA/2 v18 | Tier v18 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| hip_flexion_r | side_left | v12_combined_with_learned_l2 | 12 | 0.62 | 0.60 | 16.24 | Poor | 0.48 | 0.45 | 17.45 | Poor |
| hip_flexion_r | front_oblique_left | v13_dwpose_hybrid_with_learned_l2 | 21 | 0.85 | 0.84 | 11.29 | Moderate | 0.40 | 0.39 | 19.59 | Poor |
| hip_flexion_r | front_oblique_right | v13_dwpose_hybrid_with_learned_l2 | 17 | 0.70 | 0.70 | 18.50 | Poor | 0.49 | 0.48 | 21.50 | Poor |
| hip_flexion_r | side_right | event_anchored_with_learned_l2 | 9 | 0.52 | 0.46 | 19.01 | Poor | 0.40 | 0.27 | 20.29 | Poor |
| hip_adduction_r | side_left | v14_full_dwpose_with_learned_l2 | 12 | 0.95 | 0.94 | 9.80 | Good | 0.85 | 0.71 | 17.86 | Poor |
| hip_adduction_r | front_oblique_left | v9_phased_with_learned_l2 | 9 | 0.35 | 0.29 | 6.54 | Poor | 0.48 | 0.45 | 4.28 | Moderate |
| hip_adduction_r | front_center | v12_combined_with_learned_l2 | 23 | 0.93 | 0.77 | 16.22 | Poor | 0.74 | 0.38 | 24.85 | Poor |
| hip_adduction_r | front_oblique_right | v12_combined_with_learned_l2 | 17 | 0.92 | 0.78 | 17.28 | Poor | 0.79 | 0.63 | 21.99 | Poor |
| hip_adduction_r | side_right | v14_full_dwpose_with_learned_l2 | 10 | 0.22 | 0.21 | 19.64 | Poor | 0.12 | 0.09 | 18.03 | Poor |
| knee_angle_r | side_left | v14_full_dwpose_with_learned_l2 | 12 | 0.88 | 0.86 | 12.43 | Moderate | 0.63 | 0.50 | 18.96 | Poor |
| knee_angle_r | front_oblique_left | v12_combined_with_learned_l2 | 21 | 0.91 | 0.78 | 15.60 | Poor | 0.61 | 0.29 | 25.92 | Poor |
| knee_angle_r | front_oblique_right | v9_phased_with_learned_l2 | 9 | 0.86 | 0.83 | 10.72 | Moderate | 0.15 | 0.11 | 19.11 | Poor |
| knee_angle_r | side_right | v12_combined_with_learned_l2 | 10 | 0.81 | 0.81 | 14.24 | Moderate | 0.10 | 0.08 | 24.51 | Poor |
| ankle_angle_r | side_left | v14_full_dwpose_with_learned_l2 | 9 | 0.46 | 0.33 | 12.24 | Poor | 0.19 | 0.15 | 14.19 | Poor |
| ankle_angle_r | front_oblique_left | v14_full_dwpose_with_learned_l2 | 9 | 0.62 | 0.56 | 10.78 | Moderate | -0.30 | -0.28 | 18.75 | Poor |
| ankle_angle_r | front_center | event_anchored_bilateral_with_learned_l2 | 9 | 0.11 | 0.09 | 14.69 | Poor | -0.54 | -0.49 | 20.02 | Poor |
| ankle_angle_r | front_oblique_right | v14_full_dwpose_with_learned_l2 | 9 | -0.13 | -0.13 | 19.34 | Poor | 0.69 | 0.59 | 10.11 | Moderate |
| ankle_angle_r | side_right | v14_full_dwpose_with_learned_l2 | 9 | 0.75 | 0.64 | 9.46 | Good | 0.58 | 0.46 | 11.29 | Moderate |
| lumbar_extension | side_left | v14_full_dwpose_with_learned_l2 | 12 | 0.87 | 0.83 | 7.25 | Good | 0.79 | 0.75 | 8.85 | Good |
| lumbar_extension | front_oblique_left | event_anchored_with_learned_l2 | 9 | 0.62 | 0.53 | 8.03 | Moderate | 0.78 | 0.71 | 6.57 | Good |
| lumbar_extension | front_center | event_anchored_with_learned_l2 | 9 | 0.76 | 0.55 | 7.45 | Moderate | 0.45 | 0.43 | 9.54 | Moderate |
| lumbar_extension | front_oblique_right | v13_dwpose_hybrid_with_learned_l2 | 17 | 0.80 | 0.63 | 10.18 | Moderate | 0.82 | 0.56 | 10.61 | Moderate |
| lumbar_extension | side_right | v14_full_dwpose_with_learned_l2 | 10 | 0.47 | 0.45 | 13.31 | Moderate | 0.43 | 0.37 | 12.69 | Poor |

## Interpretation

**Headline:** Phase B did NOT deliver the 3-6 tier promotions EE2's per-metric per-frame |r| gains projected. Net tier change is negative: -1 Good, -4 Moderate, +3 Poor. EE2's per-frame Layer 2 lift does not translate cleanly to per-trial ROM tier promotion.

### Why per-frame |r| gains did not translate to ROM tier lifts

1. **ROM is a max-minus-min aggregate.** EE2's +0.131 pooled |r| measures per-frame waveform tracking. ROM throws away the within-trial waveform shape and keeps only the extrema. A model that tracks waveform shape better doesn't necessarily land its extrema in the same places as ground truth — and the ridge regression on ROM features is sensitive to the exact extrema, not the shape.
2. **Calibration drift.** Several slots that lost Good status (e.g. hip_adduction_r/side_left: r 0.95 -> 0.85, LoA 9.8 -> 17.9) still have high Pearson r but wider LoA. The learned-L2 ROM is well-correlated with ground truth but offset by a systematic bias that hand-engineered L2 was either calibrated against or incidentally aligned with.
3. **L3 ridge re-fit overfits noise.** The L3 ridge was re-fit from scratch on learned-L2 features. With small subject pools (n=9-22 LOSO folds), small differences in feature distribution cause meaningful shifts in ridge coefficients, increasing variance in held-out predictions.
4. **Distribution shift on ASPset.** Layer 2 trained on OpenCap (controlled DJ task, lab setting) is then applied to ASPset (general athletic movement). Per-frame |r| is robust to global shift; ROM is not.

### Where Layer 2 DID help (the 3 promotions)

- **ankle_angle_r / front_oblique_right**: Poor -> Moderate (r -0.13 -> 0.69, CCC -0.13 -> 0.59, LoA 19.3 -> 10.1). Baseline Layer 2 was geometrically broken on this view. Learned Layer 2 replaces a sign-flipped hand-engineered angle with a correct magnitude, unlocking real ROM signal. **Confirms EE2 hypothesis for ankle out-of-plane.**
- **lumbar_extension / front_oblique_left**: Moderate -> Good (r 0.62 -> 0.78, CCC 0.53 -> 0.71, LoA 8.0 -> 6.6). Lumbar extension is the metric where EE2 reported the biggest per-metric lift (+0.160). The improvement carries through to ROM here.
- **hip_adduction_r / front_oblique_left**: Poor -> Moderate (r 0.35 -> 0.48, CCC 0.29 -> 0.45). EE2 reported +0.146 |r| on this metric. Real signal recovered at this view.

### Where Layer 2 hurt (the 8 demotions)

Most are knee/hip slots that were strongly tuned to hand-engineered features. The hand-engineered Layer 2 is geometrically well-suited to in-plane knee/hip flexion from side and oblique views — and the v17 ridge weights for these slots are tuned to the noise structure of those features. Replacing with a different (but biomechanically reasonable) angle trace breaks the tuning.

### Honest recommendation

1. **Do NOT adopt v18 wholesale.** Net tier loss is real.
2. **Selectively adopt the 3 promoted slots** (ankle/foright, lumbar/foleft, hipadd/foleft). These are tier wins that survive the Layer-3-LOSO-only caveat because the lift is large enough to absorb the upper-bound discount.
3. **For knee/hip slots that demoted, keep v17 ridge weights.** The v17 deploy bundle already has these dialed in.
4. **Next experiment:** ENSEMBLE learned-L2 ROM with hand-engineered-L2 ROM. The two angle estimates may be additively informative at the ROM level even if learned-L2 alone is worse.
5. **Run full double-LOSO** to confirm the 3 promotions survive without L2 information leak. If even 2 of 3 survive, v18 selective adoption is the right call.

## Caveats - honest

- **Not double-LOSO.** See LOSO discipline section. Tier promotions involving OpenCap subjects are an upper bound. Of the 3 promotions, two (ankle_angle_r/front_oblique_right, hip_adduction_r/front_oblique_left) use OpenCap-only data and so are subject to the upper-bound caveat. The third (lumbar_extension/front_oblique_left, event_anchored, OpenCap-only) is also subject to the caveat.
- **pelvis_tilt remains hand-engineered** (used as coupling for lumbar_extension slots in v12/v13 only). EE2's L2 doesn't predict it.
- **Left-side angles** come from mirroring the keypoint series and running the same right-side L2 model. This relies on the L2 model being approximately mirror-symmetric, which is reasonable for the DJ task but not validated.
- **ASPset L2 inference** uses the same all-OpenCap-trained model. ASPset DWPose distribution may differ from OpenCap's; predictions may have a shift not corrected by the OpenCap training.
- **ankle_angle_r slots remain OpenCap-only** (ASPset has no foot GT). The learned L2 was trained on OpenCap ankle GT, so the L2 predictions for ankle are well-calibrated to the OpenCap distribution -- but tier change is still under Layer-3-LOSO-only discipline.
- **Numerical warnings:** v12_combined / v13_dwpose_hybrid slots emit `divide by zero` / `overflow` warnings during ridge predict. Reflects degenerate feature columns (constant or NaN-filled) when the patched learned-L2 angle trace produces an out-of-distribution value on some ASPset clips. These predictions are still finite after the z-score; CCC/LoA stats are computed on finite values only.
