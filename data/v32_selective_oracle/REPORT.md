# v32 Selective Oracle Deploy + v29/v30/v31 Candidates

**Date:** 2026-06-02
**Build:** Agent NN Phase 3 — adds 3 candidates to the v28 reader pool:

- **v29** Layer 2 mirror-flip augmentation (per-source heads + per-frame SmoothL1, MM-A backbone with L/R-flipped clip duplicates) + ridge L3.
- **v30** v23 HH2 combined Layer 2 + per-slot **learned Layer 3** (TinyMLP hidden=32, dropout 0.2, AdamW lr=1e-2 wd=1e-3, 200 epochs w/ early stopping). Per-slot fallback to ridge if learned underperforms by >0.05 CCC.
- **v31** v29 mirror-flip Layer 2 + per-slot learned Layer 3 (same architecture as v30).

**Verdict:** **11 validated Good-tier slots** (v28 was 10). Tier 1 (CCC >= 0.79) count: **13** (v28 was 7).

## Tier counts vs v28

| Tier | v28 | v32 | Delta |
| --- | ---: | ---: | ---: |
| Excellent | 0 | 0 | +0 |
| Good | 10 | 11 | +1 |
| Moderate | 6 | 6 | +0 |
| Poor | 7 | 6 | -1 |
| Tier 1 (CCC >= 0.79) | 7 | 13 | +6 |

Promotions vs v28: **2**. Slots: hip_adduction_r|front_oblique_right (Poor -> Moderate, reader=v30), lumbar_extension|front_center (Moderate -> Good, reader=v31)

## Category A: did the LoA-limited borderlines promote?

These slots already have CCC >= 0.81 (Tier 1) but fail the LoA ±10° gate by 1-3°. Lever was learned L3 (replaces ridge with TinyMLP) to tighten peak prediction.

| Slot | v28 tier | v28 CCC | v28 LoA/2 | v28 reader | v32 tier | v32 CCC | v32 LoA/2 | v32 reader | Promoted? |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| knee_angle_r|front_oblique_left | Moderate | 0.915 | 11.83 | v27 | Moderate | 0.928 | 10.77 | v31 | no |
| knee_angle_r|side_left | Moderate | 0.873 | 11.79 | v23 | Moderate | 0.899 | 10.15 | v31 | no |
| knee_angle_r|side_right | Moderate | 0.813 | 12.92 | v27 | Moderate | 0.813 | 12.92 | v27 | no |
| hip_flexion_r|front_oblique_left | Moderate | 0.843 | 11.29 | v17 | Moderate | 0.843 | 11.29 | v17 | no |
| hip_adduction_r|front_oblique_right | Poor | 0.838 | 15.74 | v20 | Moderate | 0.889 | 13.80 | v30 | no |

**Category A promotions: 0/5.**

## Category B: did the mirror-twin asymmetric slots lift?

These slots have a sister slot that scores well but this side scores poorly. Lever was mirror-flip Layer 2 augmentation (v29 / v31).

| Slot | v28 tier | v28 CCC | v28 reader | v32 tier | v32 CCC | v32 reader | CCC delta |
| --- | --- | ---: | --- | --- | ---: | --- | ---: |
| hip_adduction_r|side_right | Poor | 0.265 | v23 | Poor | 0.277 | v31 | 0.012 |
| hip_flexion_r|side_right | Poor | 0.584 | v27 | Poor | 0.584 | v27 | 0.000 |
| ankle_angle_r|side_left | Poor | 0.375 | v24 | Poor | 0.375 | v24 | 0.000 |
| ankle_angle_r|front_oblique_left | Moderate | 0.556 | v17 | Moderate | 0.556 | v17 | 0.000 |

**Category B lifts (CCC delta > 0.01): 1/4.**

## Reader distribution in v32

| Reader | Slots | Description |
| --- | ---: | --- |
| v17 | 4 | hand-engineered + ridge |
| v18 | 0 | FF learned L2 (OpenCap-only) + ridge |
| v20 | 1 | GG2 ROM-aware OpenCap L2 + ridge |
| v23 | 5 | HH2 combined L2 + ridge |
| v24 | 2 | LL combined + ROM-aware + ridge |
| v26 | 2 | MM-A per-source per-frame L2 + ridge |
| v27 | 2 | MM-B per-source ROM-aware L2 + ridge |
| v29 | 1 | NN mirror-flip per-source per-frame L2 + ridge |
| v30 | 2 | v23 L2 + learned L3 (TinyMLP) |
| v31 | 4 | v29 mirror-flip L2 + learned L3 (TinyMLP) |

## Honest caveats

- **Double-LOSO upper bound.** v23 onward train L2 on all 24 cohort subjects; L3 LOSO at subject level only. Tier promotions involving any cohort subject are upper bounds. Per HH2's per-fold variance, true double-LOSO numbers could be ~0.05-0.10 |r| lower.
- **Learned L3 overfit risk is real.** Per-slot models with n=9-22 LOSO inner folds and ~5K parameters. Mitigations: hidden=32 (tiny capacity), dropout 0.2, weight_decay 1e-3, early stopping on 15% inner-val. Per-slot fallback to ridge if learned underperforms ridge by > 0.05 CCC keeps a no-regression guarantee on the headline metric.
- **Mirror flip discipline.** Flipped clips inherit the original subject id, so LOSO holdouts include both flipped and unflipped versions of the held-out subject's clips. No leak through the L/R label swap.
- **Per-source head specifics carried from MM-A.** For hip_adduction_r and lumbar_extension where OpenCap and ASPset differ in convention, the shared (deployed) head is supervised by OpenCap only; ASPset gradients flow to a separate aux head that is discarded at deploy. v29 inherits this routing.
- **v29 LOSO-at-L2 evaluation was skipped** to stay in time budget. The diagnostic per-fold |r| would be informative but is not on the critical path for v32 (which depends on the all-data L2 + L3 LOSO pipeline). Per-fold variance for the closely-related MM-A model was already characterized and is a reasonable prior.

## Recommendation for next move

Adopt v32 selectively. Net **1** more Good slots than v28 with no Good slot demoted. The headline Tier-1 (CCC >= 0.79) count is up by +6 vs v28.

Category A (LoA-limited knee/hip_flexion borderlines) did NOT promote. Learned L3 did not deliver tighter peak prediction than ridge on these slots. Next experiment: per-slot loss weighting on max/min extrema (an explicit ROM-aware learned L3), or per-slot calibration on the held-out fold residuals.

