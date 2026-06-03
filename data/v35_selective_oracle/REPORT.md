# v35 Selective Oracle Deploy + v33/v34 Extrema-Aware Candidates

**Date:** 2026-06-02
**Build:** Agent OO -- adds 2 candidates to the v32 reader pool:

- **v33** v23 HH2 combined L2 + per-slot **extrema-aware** learned L3 (TinyMLP, two heads pred_max/pred_min, loss = SmoothL1(ROM) + 0.5*SmoothL1(max) + 0.5*SmoothL1(min)). Hidden=32, dropout 0.2, AdamW lr=1e-2 wd=1e-3, 200 epochs w/ early stopping on 15% inner-val. Per-slot fallback to ridge if extrema-aware CCC underperforms by > 0.05.
- **v34** v29 mirror-flip L2 + extrema-aware learned L3 (same architecture as v33).

**Verdict:** **11 validated Good-tier slots** (v32 was 11). Tier 1 (CCC >= 0.79) count: **14** (v32 was 13).

## Tier counts vs v32

| Tier | v32 | v35 | Delta |
| --- | ---: | ---: | ---: |
| Excellent | 0 | 0 | +0 |
| Good | 11 | 11 | +0 |
| Moderate | 6 | 6 | +0 |
| Poor | 6 | 6 | +0 |
| Tier 1 (CCC >= 0.79) | 13 | 14 | +1 |

Promotions vs v32: **0**. 
Demotions vs v32: **0**. 

## Category A: did the extrema-aware L3 break the LoA wall?

These slots have strong CCC (0.81-0.93) in v32 but miss the LoA +/-10 deg gate by 1-3 deg. Agent OO's lever was a per-slot TinyMLP with two heads (pred_max, pred_min) directly supervised against per-clip ground-truth extrema, hypothesising that max/min supervision tightens LoA where ROM-only supervision cannot.

| Slot | v32 tier | v32 CCC | v32 LoA/2 | v32 reader | v33 CCC | v33 LoA/2 | v34 CCC | v34 LoA/2 | v35 reader | v35 tier | Promoted? |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| knee_angle_r|front_oblique_left | Moderate | 0.928 | 10.77 | v31 | 0.894 | 13.11 | 0.902 | 12.44 | v31 | Moderate | no |
| knee_angle_r|side_left | Moderate | 0.899 | 10.15 | v31 | 0.873 | 11.79 | 0.895 | 10.02 | v31 | Moderate | no |
| knee_angle_r|side_right | Moderate | 0.813 | 12.92 | v27 | 0.790 | 13.41 | 0.594 | 26.25 | v27 | Moderate | no |
| hip_flexion_r|front_oblique_left | Moderate | 0.843 | 11.29 | v17 | 0.737 | 12.77 | 0.711 | 13.89 | v17 | Moderate | no |
| hip_adduction_r|front_oblique_right | Moderate | 0.889 | 13.80 | v30 | 0.864 | 15.57 | 0.840 | 16.53 | v30 | Moderate | no |

**Category A promotions to Good: 0/5.**

## Extrema-prediction diagnostics on Category A slots

Per-slot LOSO mean absolute error on max and min predictions (from v33 and v34 heads). This tells us whether the new heads are actually learning extrema or just being clamped to ROM-implied averages.

| Slot | v33 MAE_max | v33 MAE_min | v33 n | v34 MAE_max | v34 MAE_min | v34 n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| knee_angle_r|front_oblique_left | 9.21 | 12.99 | 443 | 8.62 | 12.21 | 443 |
| knee_angle_r|side_left | 9.10 | 9.45 | 144 | 12.57 | 13.72 | 144 |
| knee_angle_r|side_right | 37.82 | 40.05 | 84 | 38.05 | 39.72 | 84 |
| hip_flexion_r|front_oblique_left | 7.72 | 4.85 | 444 | 7.21 | 4.93 | 444 |
| hip_adduction_r|front_oblique_right | 6.19 | 10.32 | 220 | 6.37 | 11.23 | 220 |

## Reader distribution in v35

| Reader | Slots | Description |
| --- | ---: | --- |
| v17 | 4 | hand-engineered + ridge |
| v18 | 0 | FF learned L2 (OpenCap-only) + ridge |
| v20 | 1 | GG2 ROM-aware OpenCap L2 + ridge |
| v23 | 4 | HH2 combined L2 + ridge |
| v24 | 2 | LL combined + ROM-aware + ridge |
| v26 | 2 | MM-A per-source per-frame L2 + ridge |
| v27 | 2 | MM-B per-source ROM-aware L2 + ridge |
| v29 | 1 | NN mirror-flip per-source per-frame L2 + ridge |
| v30 | 2 | v23 L2 + learned L3 (TinyMLP, ROM-only) |
| v31 | 4 | v29 mirror-flip L2 + learned L3 (TinyMLP, ROM-only) |
| v33 | 1 | v23 L2 + extrema-aware learned L3 (max/min heads) |
| v34 | 0 | v29 mirror-flip L2 + extrema-aware learned L3 |

## Honest caveats

- **Double-LOSO upper bound** (unchanged from v32). v23/v29 L2 trained on all 24 cohort subjects; L3 LOSO at subject level only. Per-fold L2 variance from HH2 suggests true double-LOSO numbers could be ~0.05-0.10 |r| lower.
- **Extrema-aware L3 overfit risk is real.** Per-slot models with n=9-22 LOSO inner folds and ~5K params, now with two output heads instead of one. Mitigations carried from NN: hidden=32 (tiny capacity), dropout 0.2, weight_decay 1e-3, early stopping on 15% inner-val. Per-slot fallback to ridge if extrema-aware CCC underperforms ridge by > 0.05 keeps a no-regression guarantee.
- **Extrema GT was newly computed.** For OpenCap, max/min of the target IK angle column from the .mot file (same source as the existing gt_rom_col which returns max - min). For ASPset, max/min of joint_angles_from_aspset(clip)[target_gt] (same source as the v12_combined inline GT). Per-row alignment was sanity-checked: |gt_max - gt_min - y| < 1e-3 on all surviving rows; mismatching rows are filtered.
- **No multi-camera fusion.** Single DWPose stream at inference (Couro's core single-camera differentiator).
- **No new sport thresholds.** Extrema-aware L3 only affects how ROM is predicted; downstream sport-specific risk multipliers are unchanged.

## Recommendation for next move

Hold at v32. Extrema-aware L3 matched v32 Good count (11) but did not net-gain. The per-slot fallback to ridge prevented regressions on slots where the two-head MLP overfit, but extrema supervision did not break the LoA wall on Category A targets.

**Category A verdict: extrema-aware L3 did NOT crack the LoA wall.** Promoting these slots from Moderate to Good will require a different lever -- candidates include (a) per-slot residual calibration on held-out folds, (b) richer per-clip feature vectors that capture peak timing more directly, or (c) revisiting Layer 2 with an extrema-aware loss (GG2 style) targeted at the affected joints.

