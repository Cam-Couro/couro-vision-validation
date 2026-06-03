# v42 Selective Oracle -- v20 Calibration Extension (Agent QQ)

**Date:** 2026-06-03
**Build:** Agent QQ -- v40 reader pool (15 readers) + v41 (v20 + nested-LOSO calibration). Selection rule preserved from v40: LoA-then-CCC tie-break within the Moderate tier when all top-tier candidates have CCC >= 0.79.

**Verdict:** **12 validated Good-tier slots** (v40 was 12). Tier 1 (CCC >= 0.79) count: **14** (v40 was 14).

## 1. Did `hip_adduction_r / front_oblique_left` promote to Tier 1 (CCC >= 0.79)?

This slot has the tightest LoA in the validation table (+/-3.3 deg) but a modest CCC of 0.69 -- the textbook bias-dominated pattern that residual calibration is designed to fix.

Per-slot table (raw calibrated stats, before the no-regression fallback is applied):

| Reader on this slot | CCC | LoA half |
| --- | ---: | ---: |
| v20 uncalibrated (v40 pick) | 0.690 | 3.29 |
| v41 v20 calibrated (raw) | 0.213 | 4.30 |

After per-slot fallback rule (require LoA tighten AND CCC not regress by more than 0.05), the v41 reader for this slot chose **uncalibrated** and publishes CCC=0.690, LoA/2=3.29 (identical to v20 uncalibrated).

**v42 oracle pick for hip_adduction_r|front_oblique_left:** reader=**v20**, CCC=0.690, LoA/2=3.29, tier=Good.

**Verdict: NO.** Slot did NOT promote to Tier 1 (CCC >= 0.79). Residual calibration is not the right lever for this v20-based slot.

Bias hypothesis **REFUTED**: raw calibration dropped CCC from 0.690 to 0.213. The inner-LOSO (a, b) fit varied wildly across the 9 outer folds (a_mean=0.808 +/- 0.250, b_mean=2.245 +/- 2.632) -- the pseudo-residual fit is overfit to inner-fold noise given the n=9 OpenCap subject pool. Per-slot fallback to uncalibrated v20 holds, so v42 ships the same v20 stat as v40.

## 2. Tier counts vs v40

| Tier | v40 | v42 | Delta |
| --- | ---: | ---: | ---: |
| Excellent | 0 | 0 | +0 |
| Good | 12 | 12 | +0 |
| Moderate | 5 | 5 | +0 |
| Poor | 6 | 6 | +0 |
| Tier 1 (CCC >= 0.79) | 14 | 14 | +0 |

Promotions vs v40: **0**.
Demotions vs v40: **0**.

## 3. v41 picks in v42 oracle

**No slots picked v41 at the v42 oracle.** Every v41 candidate was either dominated by an uncalibrated reader or fell back to uncalibrated v20 internally (per-slot fallback rule inside v41), and v20 was already on the menu via the original v20 entry.

## 4. Sanity check: did calibrating v20 break v40 Good slots?

No demotions vs v40. Adding v41 to the pool is non-destructive: per-slot fallback inside v41 means calibrated stats only enter the pool when they tighten LoA without regressing CCC by more than 0.05.

## 5. Category A: did adding v41 crack any new LoA walls?

| Slot | v40 reader | v40 LoA/2 | v41 CCC | v41 LoA/2 | v41 tier | v42 reader | v42 LoA/2 | v42 CCC | v42 tier |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| knee_angle_r|front_oblique_left | v31 | 10.77 | 0.846 | 14.94 | Moderate | v31 | 10.77 | 0.928 | Moderate |
| knee_angle_r|side_left | v38 | 9.78 | 0.683 | 18.30 | Poor | v38 | 9.78 | 0.903 | Good |
| knee_angle_r|side_right | v27 | 12.92 | 0.354 | 21.61 | Poor | v27 | 12.92 | 0.813 | Moderate |
| hip_flexion_r|front_oblique_left | v17 | 11.29 | 0.506 | 19.72 | Poor | v17 | 11.29 | 0.843 | Moderate |
| hip_adduction_r|front_oblique_right | v30 | 13.80 | 0.838 | 15.74 | Poor | v30 | 13.80 | 0.889 | Moderate |

## 6. Reader shifts vs v40

**No reader shifts vs v40.** Adding v41 to the pool changed no oracle picks.

## 7. Reader distribution in v42

| Reader | Slots | Description |
| --- | ---: | --- |
| v17 | 4 | hand-engineered + ridge |
| v18 | 0 | FF learned L2 (OpenCap-only) + ridge |
| v20 | 1 | GG2 ROM-aware OpenCap L2 + ridge |
| v23 | 3 | HH2 combined L2 + ridge |
| v24 | 2 | LL combined ROM-aware + ridge |
| v26 | 2 | MM-A per-source per-frame L2 + ridge |
| v27 | 2 | MM-B per-source ROM-aware L2 + ridge |
| v29 | 0 | NN mirror-flip per-source per-frame L2 + ridge |
| v30 | 2 | v23 L2 + learned L3 (TinyMLP, ROM-only) |
| v31 | 3 | v29 mirror-flip L2 + learned L3 (TinyMLP, ROM-only) |
| v33 | 1 | v23 L2 + extrema-aware learned L3 (max/min heads) |
| v34 | 0 | v29 mirror-flip L2 + extrema-aware learned L3 |
| v37 | 1 | v23 L2 + ridge L3 + nested-LOSO calibration |
| v38 | 1 | v29 L2 + ridge L3 + nested-LOSO calibration |
| v39 | 1 | v17 hand-engineered L2 + ridge L3 + nested-LOSO calibration |
| v41 | 0 | v20 GG2 ROM-aware L2 + ridge L3 + nested-LOSO calibration |

## 8. Honest caveats

- **Single camera only.** Unchanged from v17-v40.
- **Per-slot fallback to uncalibrated v20.** If calibration fails to tighten LoA OR regresses CCC by more than 0.05, the v41 slot stat is the uncalibrated v20 stat. This preserves the no-regression guarantee.
- **L2 LOSO discipline.** GG2 L2 trained on all 9 OpenCap subjects; OpenCap-subject tier promotions remain upper bounds (true double-LOSO would likely show ~0.05-0.10 |r| lower).
- **L3 LOSO + nested LOSO discipline preserved.** Outer LOSO at L3 unchanged. Calibration is fit on pseudo-residuals from (N-1) training subjects only; outer held-out subject never used for calibration fitting.
- **Small n at L2.** For v20-based slots with n=9 OpenCap subjects, the inner LOSO calibration fit uses 7-8 training subjects. With so few subjects, the linear (a, b) fit is high-variance; this is the expected failure mode and the per-slot fallback handles it.

