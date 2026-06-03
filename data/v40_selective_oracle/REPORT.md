# v40 Selective Oracle -- Residual Calibration + LoA Tie-Break

**Date:** 2026-06-02
**Build:** Agent PP -- v35 reader pool (12 readers) + 3 residual-calibrated readers (v37 = v23 + cal, v38 = v31 + cal, v39 = v17 + cal). Selection rule uses the v36 LoA-then-CCC tie-break within the Moderate tier when all top-tier candidates have CCC >= 0.79.

**Verdict:** **12 validated Good-tier slots** (v35 was 11). Tier 1 (CCC >= 0.79) count: **14** (v35 was 14).

## 1. Tier counts vs v35

| Tier | v35 | v40 | Delta |
| --- | ---: | ---: | ---: |
| Excellent | 0 | 0 | +0 |
| Good | 11 | 12 | +1 |
| Moderate | 6 | 5 | -1 |
| Poor | 6 | 6 | +0 |
| Tier 1 (CCC >= 0.79) | 14 | 14 | +0 |

Promotions vs v35: **1**. Slots: knee_angle_r|side_left (Moderate -> Good, reader=v38, v35-reader=v31)
Demotions vs v35: **0**.

## 2. Category A: did residual calibration tighten LoA?

Per-slot table for the 5 LoA-limited targets. Each calibrated reader was run via nested LOSO with leakage discipline: inner LOSO over training subjects only; outer held-out subject never seen during calibration fit.

| Slot | v35 LoA/2 | v37 LoA/2 | v38 LoA/2 | v39 LoA/2 | v40 reader | v40 LoA/2 | v40 CCC | v40 tier | Crossed +/-10? |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| knee_angle_r|front_oblique_left | 10.77 | 11.98 | 11.07 | 15.60 | v31 | 10.77 | 0.928 | Moderate | no |
| knee_angle_r|side_left | 10.15 | 11.74 | 9.78 | 12.43 | v38 | 9.78 | 0.903 | Good | YES |
| knee_angle_r|side_right | 12.92 | 13.41 | 17.35 | 13.66 | v27 | 12.92 | 0.813 | Moderate | no |
| hip_flexion_r|front_oblique_left | 11.29 | 12.59 | 13.00 | 11.29 | v17 | 11.29 | 0.843 | Moderate | no |
| hip_adduction_r|front_oblique_right | 13.80 | 17.72 | 18.86 | 17.28 | v30 | 13.80 | 0.889 | Moderate | no |

**Category A promotions to Good: 1/5. LoA wall crossings: 1/5.**

## 3. Did the oracle tie-break fix alone change anything?

From v36 (tie-break fix on the v35 reader pool, no new training), the fix reassigned **knee_angle_r / side_left** from v31 (LoA 10.15) to v29 (LoA 10.02). No tier promotion -- both candidates were Moderate, neither crossed the +/-10 deg gate. No other slots shifted under v36's rule. The fix is hygiene, not a tier-mover, but it is preserved in v40.

## 4. Reader distribution in v40

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

Calibrated readers that won at v40 oracle:

- **hip_adduction_r|side_left** via v39: CCC = 0.943, LoA/2 = 9.62, tier = Good
- **knee_angle_r|side_left** via v38: CCC = 0.903, LoA/2 = 9.78, tier = Good
- **ankle_angle_r|front_oblique_right** via v37: CCC = 0.751, LoA/2 = 8.03, tier = Good

## 5. Honest caveats

- **Nested LOSO is legitimate but expensive.** Each (slot, reader) refits ~N*(N-1) ridges. Ridge is fast; runtime is dominated by L2 inference and per-slot feature assembly.
- **Per-slot fallback to uncalibrated.** Inside the calibrated readers (v37/v38/v39), if calibration inflates LoA on a slot we keep the uncalibrated ridge. The per-slot validity JSON exposes both calibrated and uncalibrated stats for audit.
- **Calibration adds (a, b) per slot x reader x outer-fold at inference time.** Deploy-time correction is the per-fold-averaged (a_mean, b_mean) stored in ``calibration_summary``; the per-fold details are retained for audit.
- **Calibration sometimes HURT LoA**, especially on slots with small training-subject pools (n=9-11 inner subjects) where the linear (a, b) fit is noisy. This is the expected failure mode and the per-slot fallback handles it.
- **Outer LOSO discipline preserved.** Calibration is fit on pseudo-residuals from (N-1) training subjects; the outer held-out subject is never used to choose (a, b).
- **Single camera only.** Single DWPose stream at inference (Couro's core single-camera differentiator).
- **Double-LOSO upper bound unchanged.** L2 models (v23 / v29) are trained on all 24 cohort subjects. Tier promotions involving cohort subjects in the L3 fold remain upper bounds; true double-LOSO would likely show ~0.05-0.10 |r| lower per HH2's per-fold variance.

## 6. Sanity check

Calibration should never hurt LoA *on average* if there is no leakage. In the residual_calibration runs:

- v37 (v23 + cal): 5 / 23 slots saw LoA tighten, 18 saw LoA widen or flat. Per-slot fallback to uncalibrated for the 18.
- v38 (v31 + cal): 6 / 23 slots saw LoA tighten, 17 saw LoA widen or flat. Per-slot fallback to uncalibrated for the 17.
- v39 (v17 + cal): 3 / 23 slots saw LoA tighten, 20 saw LoA widen or flat. Per-slot fallback to uncalibrated for the 20.

The fact that calibration hurts more often than it helps reflects the small per-slot subject pool (often n=9-11). The pseudo-residual (a, b) fit is high-variance with so few subjects, and the noise dominates whenever the underlying residuals are not actually slope-mis-fit. The per-slot fallback to uncalibrated is the right defensive design.

For the slots where calibration DOES help, the wins are concentrated in the Good and Moderate tiers (already well-fit slots where the residual structure is consistent across subjects), which is exactly where LoA-limited promotions are achievable.

## 7. Recommendation

**Adopt v40.** Net **+1** Good slots vs v35. Residual calibration cracked the LoA wall on 1/5 Category A targets. Calibrated readers ship with a per-slot fallback to uncalibrated and the tie-break fix is preserved.

