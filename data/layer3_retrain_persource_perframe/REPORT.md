# Phase B: Layer 3 retrained on MM-A (per-source heads + per-frame SmoothL1) learned Layer 2 (Agent MM → v26)

**Date:** 2026-06-02
**Build:** Agent MM Phase B — v26 candidate (sister of v27)

## LOSO discipline

**Layer-3-LOSO-only.** ONE L2 model (`TemporalKeypointCNNConfPerSource`) was trained on ALL 24 cohort subjects (9 OpenCap + 15 ASPset, no LOSO at L2) and used to produce learned angle traces for every clip. Layer 3 ridge was re-fit per slot with LOSO at L3 only. Same caveat as v18/v20/v23/v24: tier promotions involving any cohort subject are upper bounds; the true double-LOSO number could be ~0.05-0.10 |r| lower per HH2's per-fold variance.

## Per-source heads recap

L2 architecture has two output heads:

  * **Shared head (5 outputs)** — hip_flexion_r, hip_adduction_r, knee_angle_r, ankle_angle_r, lumbar_extension. **This is the deployed head** — produces OpenCap-convention output.
  * **ASPset head (2 outputs)** — hip_adduction_r_aspset, lumbar_extension_aspset. **Discarded at inference.** Absorbs ASPset's convention-divergent target during training so it doesn't pollute the shared head's gradient on those two metrics.

L3 ridge consumes the shared-head 5-output trace, exactly like v18/v23/v24. No deploy-side complexity change vs prior learned-L2 readers.

## Tier counts (this reader alone)

| Tier | v26 count |
| --- | ---: |
| Excellent | 0 |
| Good | 5 |
| Moderate | 7 |
| Poor | 11 |

(Tier counts evaluated reader-in-isolation. The v28 oracle selects the best reader per slot — see `data/v28_selective_oracle/REPORT.md`.)

## Target slot answers

| Slot | v25 CCC (reader) | v26 CCC | v26 LoA half | Tier | Crossed 0.79? |
| --- | --- | ---: | ---: | --- | :---: |
| hip_adduction_r / front_oblique_left | 0.69 (v20 (LL/v25)) | -0.374 | ±7.76° | Poor | no |
| lumbar_extension / front_oblique_left | 0.71 (v18 (LL/v25)) | 0.823 | ±5.18° | Good | YES |
| lumbar_extension / side_left | 0.88 (v23 (LL/v25)) | 0.695 | ±8.24° | Good | no |
| lumbar_extension / side_right | 0.85 (v23 (LL/v25)) | 0.506 | ±10.81° | Moderate | no |
| lumbar_extension / front_oblique_right | 0.79 (v23 (LL/v25)) | 0.653 | ±9.31° | Good | no |

## Promotions vs v17 baseline (v26 alone)

- hip_flexion_r / side_left: Poor -> Good
- knee_angle_r / front_oblique_left: Poor -> Moderate
- knee_angle_r / front_oblique_right: Moderate -> Good
- ankle_angle_r / front_oblique_right: Poor -> Moderate
- lumbar_extension / front_oblique_left: Moderate -> Good
- lumbar_extension / front_oblique_right: Moderate -> Good

## Demotions vs v17 baseline (v26 alone)

- hip_adduction_r / side_left: Good -> Moderate
- knee_angle_r / side_right: Moderate -> Poor
- ankle_angle_r / front_oblique_left: Moderate -> Poor
- ankle_angle_r / side_right: Good -> Poor

## Honest caveats

1. **Layer-3-LOSO-only.** L2 trained on all 24 cohort subjects. Tier promotions involving any cohort subject are upper bounds.
2. **Shared head sees OpenCap-only supervision for the 2 per-source metrics.** Gradient narrowing is intentional — the goal is a clean convention — but it narrows the training distribution for hip_adduction_r and lumbar_extension.
3. **Final verdict is in v28, not here.** This reader competes with v17/v18/v20/v23/v24 in the v28 oracle. A reader that regresses on slots already handled by v17/v23 still costs nothing at deploy because the oracle keeps the better reader.

## Files

- `results/deploy_ready_models_v26_persource_perframe.json` — v26 deploy candidate
- `data/layer3_retrain_persource_perframe/per_slot_validity_v26.json` — per-slot v26 validity stats
- `data/v28_selective_oracle/REPORT.md` — v28 narrative (the floor-lift verdict)

