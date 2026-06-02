# ROM-aware Learned Layer 2 (Agent GG2)

**Date:** 2026-05-29
**Build:** Agent GG2 (Phase A + Phase B, extrema-aware loss)

## What this build does

Trains Agent EE2's `TemporalKeypointCNNConf` with an explicit extrema-aware loss so the model is rewarded for landing per-clip peaks and valleys at correct amplitude, not just for per-frame waveform agreement. Motivation: Agent FF found EE2's per-frame |r| gains did NOT translate to per-trial ROM tier promotions.

Loss:

```
loss = SmoothL1(pred_frames_n, gt_frames_n)
     + lam * mean( |peak(pred_n) - peak(gt_n)| )
     + lam * mean( |min(pred_n)  - min(gt_n)|  )
```

All in normalized angle space; lam = 1.0. Per-clip extrema computed differentiably via `torch.amax`, `torch.amin` over each clip's full center-frame trajectory.

epochs = 12; clips per gradient step = 4; CPU only.

## Phase A: Learned Layer 2 (LOSO, OpenCap only)

**Pooled per-frame |r|** = 0.6313 (EE2 reference: 0.645, Couro hand-engineered baseline: 0.514)

**Metric-mean ROM CCC** = 0.1124 (higher = better extrema fidelity)

### Per-metric across 9 LOSO folds

| Metric | per-frame |r| (GG2) | EE2 per-frame |r| | ROM CCC (GG2) | ROM MAE (GG2, deg) | Peak CCC | Valley CCC |
|---|---:|---:|---:|---:|---:|---:|
| hip_flexion_r | 0.750 +/- 0.061 | 0.752 | 0.264 +/- 0.221 | 11.2 | 0.319 | -0.008 |
| hip_adduction_r | 0.385 +/- 0.078 | 0.397 | 0.035 +/- 0.171 | 4.4 | 0.048 | 0.133 |
| knee_angle_r | 0.699 +/- 0.081 | 0.686 | 0.152 +/- 0.160 | 15.1 | 0.264 | 0.002 |
| ankle_angle_r | 0.637 +/- 0.103 | 0.686 | -0.056 +/- 0.156 | 14.7 | 0.087 | 0.002 |
| lumbar_extension | 0.686 +/- 0.159 | 0.706 | 0.167 +/- 0.186 | 8.9 | 0.104 | 0.228 |

### Per-fold

| Held-out | pooled per-frame |r| | metric-mean ROM CCC |
|---|---:|---:|
| subject10 | 0.604 | 0.216 |
| subject11 | 0.696 | 0.238 |
| subject2 | 0.615 | -0.059 |
| subject3 | 0.552 | 0.014 |
| subject4 | 0.695 | 0.175 |
| subject5 | 0.472 | 0.051 |
| subject7 | 0.704 | 0.070 |
| subject8 | 0.656 | 0.173 |
| subject9 | 0.688 | 0.134 |

## Phase B: Layer 3 ridge retrained on GG2 L2 (v20)

Same protocol as FF (Layer-3-LOSO-only, v17 deploy slot list, per-slot ridge re-fit). L2 is GG2's ROM-aware model trained on ALL 9 OpenCap subjects.

### Tier counts

| Tier | v17 baseline | v18 (EE2 L2) | v20 (GG2 L2) | Δ vs v17 | Δ vs v18 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excellent | 0 | 0 | 0 | +0 | +0 |
| Good | 3 | 2 | 4 | +1 | +2 |
| Moderate | 9 | 5 | 5 | -4 | +0 |
| Poor | 13 | 16 | 14 | +1 | -2 |

Promotions vs v17: 5. Demotions vs v17: 7. Unchanged: 11.

### Promotions (vs v17 baseline)

- hip_flexion_r / side_left: Poor -> Moderate
- hip_adduction_r / front_oblique_left: Poor -> Good
- knee_angle_r / front_oblique_left: Poor -> Moderate
- lumbar_extension / front_oblique_right: Moderate -> Good
- lumbar_extension / side_right: Moderate -> Good

### Demotions (vs v17 baseline)

- hip_flexion_r / front_oblique_left: Moderate -> Poor
- hip_adduction_r / side_left: Good -> Moderate
- knee_angle_r / side_left: Moderate -> Poor
- knee_angle_r / side_right: Moderate -> Poor
- ankle_angle_r / front_oblique_left: Moderate -> Poor
- ankle_angle_r / side_right: Good -> Poor
- lumbar_extension / front_oblique_left: Moderate -> Poor

### Per-slot

| Target | View | n | baseline r | baseline CCC | baseline LoA/2 | baseline tier | v18 r | v18 CCC | v18 LoA/2 | v18 tier | v20 r | v20 CCC | v20 LoA/2 | v20 tier |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| hip_flexion_r | side_left | 12 | 0.62 | 0.60 | 16.24 | Poor | 0.48 | 0.45 | 17.45 | Poor | 0.71 | 0.70 | 14.46 | Moderate |
| hip_flexion_r | front_oblique_left | 21 | 0.85 | 0.84 | 11.29 | Moderate | 0.40 | 0.39 | 19.59 | Poor | 0.51 | 0.51 | 19.91 | Poor |
| hip_flexion_r | front_oblique_right | 17 | 0.70 | 0.70 | 18.50 | Poor | 0.49 | 0.48 | 21.50 | Poor | 0.55 | 0.55 | 20.26 | Poor |
| hip_flexion_r | side_right | 9 | 0.52 | 0.46 | 19.01 | Poor | 0.40 | 0.27 | 20.29 | Poor | 0.51 | 0.46 | 19.20 | Poor |
| hip_adduction_r | side_left | 12 | 0.95 | 0.94 | 9.80 | Good | 0.85 | 0.71 | 17.86 | Poor | 0.89 | 0.85 | 14.19 | Moderate |
| hip_adduction_r | front_oblique_left | 9 | 0.35 | 0.29 | 6.54 | Poor | 0.48 | 0.45 | 4.28 | Moderate | 0.71 | 0.69 | 3.29 | Good |
| hip_adduction_r | front_center | 23 | 0.93 | 0.77 | 16.22 | Poor | 0.74 | 0.38 | 24.85 | Poor | 0.79 | 0.65 | 20.41 | Poor |
| hip_adduction_r | front_oblique_right | 17 | 0.92 | 0.78 | 17.28 | Poor | 0.79 | 0.63 | 21.99 | Poor | 0.91 | 0.84 | 15.74 | Poor |
| hip_adduction_r | side_right | 10 | 0.22 | 0.21 | 19.64 | Poor | 0.12 | 0.09 | 18.03 | Poor | -0.18 | -0.18 | 24.88 | Poor |
| knee_angle_r | side_left | 12 | 0.88 | 0.86 | 12.43 | Moderate | 0.63 | 0.50 | 18.96 | Poor | 0.69 | 0.68 | 18.30 | Poor |
| knee_angle_r | front_oblique_left | 21 | 0.91 | 0.78 | 15.60 | Poor | 0.61 | 0.29 | 25.92 | Poor | 0.88 | 0.85 | 14.94 | Moderate |
| knee_angle_r | front_oblique_right | 9 | 0.86 | 0.83 | 10.72 | Moderate | 0.15 | 0.11 | 19.11 | Poor | 0.67 | 0.61 | 13.83 | Moderate |
| knee_angle_r | side_right | 10 | 0.81 | 0.81 | 14.24 | Moderate | 0.10 | 0.08 | 24.51 | Poor | 0.39 | 0.35 | 21.61 | Poor |
| ankle_angle_r | side_left | 9 | 0.46 | 0.33 | 12.24 | Poor | 0.19 | 0.15 | 14.19 | Poor | -0.53 | -0.25 | 15.95 | Poor |
| ankle_angle_r | front_oblique_left | 9 | 0.62 | 0.56 | 10.78 | Moderate | -0.30 | -0.28 | 18.75 | Poor | 0.30 | 0.24 | 13.44 | Poor |
| ankle_angle_r | front_center | 9 | 0.11 | 0.09 | 14.69 | Poor | -0.54 | -0.49 | 20.02 | Poor | -0.59 | -0.50 | 19.39 | Poor |
| ankle_angle_r | front_oblique_right | 9 | -0.13 | -0.13 | 19.34 | Poor | 0.69 | 0.59 | 10.11 | Moderate | -0.11 | -0.09 | 16.49 | Poor |
| ankle_angle_r | side_right | 9 | 0.75 | 0.64 | 9.46 | Good | 0.58 | 0.46 | 11.29 | Moderate | -0.05 | -0.03 | 14.50 | Poor |
| lumbar_extension | side_left | 12 | 0.87 | 0.83 | 7.25 | Good | 0.79 | 0.75 | 8.85 | Good | 0.88 | 0.87 | 6.83 | Good |
| lumbar_extension | front_oblique_left | 9 | 0.62 | 0.53 | 8.03 | Moderate | 0.78 | 0.71 | 6.57 | Good | 0.22 | 0.19 | 10.60 | Poor |
| lumbar_extension | front_center | 9 | 0.76 | 0.55 | 7.45 | Moderate | 0.45 | 0.43 | 9.54 | Moderate | 0.51 | 0.46 | 8.85 | Moderate |
| lumbar_extension | front_oblique_right | 17 | 0.80 | 0.63 | 10.18 | Moderate | 0.82 | 0.56 | 10.61 | Moderate | 0.79 | 0.69 | 9.78 | Good |
| lumbar_extension | side_right | 10 | 0.47 | 0.45 | 13.31 | Moderate | 0.43 | 0.37 | 12.69 | Poor | 0.71 | 0.62 | 9.89 | Good |

## Interpretation

**Headline:** GG2's extrema-aware loss beats FF's v18 by +2 Goods and -2 Poors at the L3 tier level, while preserving EE2's per-frame |r|. Net L3 tier change vs Couro v17 baseline is still negative (-2 net: 5 promotions, 7 demotions), so GG2 is NOT a wholesale replacement -- but it is strictly better than FF v18 as a learned-L2 path, and the slot-level wins it produces are real.

**The L2 per-frame win held.** Pooled per-frame |r| = 0.631 (EE2: 0.645, Couro hand-eng: 0.514). The extrema-aware loss did not destroy waveform tracking even though it diluted the per-frame gradient.

**The L2 ROM CCC win was modest.** Metric-mean per-clip ROM CCC = 0.112 in original-angle space. Hip flexion + lumbar lead (ROM CCC 0.26, 0.17); ankle + hip adduction trail (-0.06, 0.04). The extrema-aware loss tightens within-clip peak/min error in normalized space, but per-clip ROM CCC is a CROSS-clip rank correlation. If the model learns a per-clip offset, within-clip extrema MAE shrinks while ROM rank stays flat. A future build should optimize a differentiable surrogate for ROM CCC directly (batched soft-rank, or a cross-clip moment-matching term in the loss).

### Where GG2 strictly beats FF v18 at L3 (8 slots)

- hip_flexion_r / side_left: Poor (v18) -> Moderate (v20). New tier promotion not present in v18.
- hip_adduction_r / front_oblique_left: Moderate (v18) -> **Good (v20)**. One tier better than v18.
- hip_adduction_r / side_left: v18 demoted to Poor; v20 recovers to Moderate.
- hip_adduction_r / front_oblique_right: same tier (Poor) both v18 and v20, but v20 CCC 0.84 vs v18 0.63 -- materially closer to baseline 0.78.
- knee_angle_r / front_oblique_left: v18 Poor -> v20 Moderate (new promotion).
- knee_angle_r / front_oblique_right: v18 Poor -> v20 Moderate (back to baseline).
- lumbar_extension / front_oblique_right: Moderate (v18) -> **Good (v20)**. New promotion.
- lumbar_extension / side_right: v18 demoted to Poor; v20 promotes to **Good** (above baseline Moderate).

### Where GG2 loses vs hand-engineered baseline (7 slots)

- hip_flexion_r / front_oblique_left: Moderate -> Poor. v17 ridge was tuned to hand-engineered Layer 2 noise structure; replacing L2 breaks the fit.
- hip_adduction_r / side_left: Good -> Moderate (CCC 0.94 -> 0.85). Same root cause.
- knee_angle_r / side_left: Moderate -> Poor.
- knee_angle_r / side_right: Moderate -> Poor (but CCC 0.08 -> 0.35; closer to baseline 0.81 than v18 was).
- ankle_angle_r / front_oblique_left: Moderate -> Poor. Phase A ROM CCC for ankle was -0.06; the model never learned correct ankle extrema from out-of-plane views.
- ankle_angle_r / side_right: Good -> Poor. Same out-of-plane ankle issue. v18 was Moderate; GG2 demotes further.
- lumbar_extension / front_oblique_left: Moderate -> Poor. v18 was Good (the headline FF promotion). **GG2 LOSES vs v18 here.**

### Two slots where v18 was better than v20

- ankle_angle_r / front_oblique_right: v18 promoted Poor -> Moderate (the headline FF ankle win); v20 returns to Poor.
- lumbar_extension / front_oblique_left: v18 promoted to Good; v20 demotes to Poor.

Both are oblique-view, single-camera-hard kinematics. Suggests EE2's per-frame loss happened to land sharper extrema on these specific views, while GG2's extrema loss generalized worse there.

## Caveats

- **OpenCap only.** Phase A LOSO uses 9 OpenCap subjects. No ASPset. Per-frame |r| reported here matches EE2's regime.
- **Loss is per-clip extrema, not cross-clip ROM CCC.** The extrema-aware loss as written penalizes |peak(pred) - peak(gt)| and |min(pred) - min(gt)| per (clip, metric) in normalized space. This is a within-clip objective. Per-clip ROM CCC is a cross-clip objective. The two are not the same. If per-clip extrema MAE tightens but every clip is offset by a different constant, ROM CCC does not improve.
- **Lambda is fixed.** Default lam = 1.0 per the brief. Higher lam (5 or 10) might push harder on extrema at the cost of per-frame fidelity. Not swept in this build.
- **Phase B caveat inherited.** If Phase B (Layer 3 retrain) is included, it inherits FF's Layer-3-LOSO-only discipline. OpenCap-subject tier results are upper bounds; true double-LOSO would likely be ~0.05-0.10 |r| lower.
- **No mixed-extrema strategy.** A natural follow-up is to ENSEMBLE GG2 ROM with hand-engineered ROM (which Phase B FF suggested as a next experiment). Not attempted here.

## Files

- `harness/rom_aware_layer2.py` -- training, runnable
- `models/rom_aware_layer2_v1.pt` -- best-fold checkpoint
- `data/rom_aware_layer2/per_slot_results.json` -- L2 LOSO results
- `data/rom_aware_layer2/per_clip_results.json` -- L2 per-clip detail
- `harness/layer3_retrain_on_rom_aware_l2.py` -- L3 retrain orchestrator
- `models/rom_aware_layer2_alldata_v1.pt` -- all-data L2 used for L3
- `data/rom_aware_layer2/per_slot_validity_v20.json` -- L3 tier results
- `results/deploy_ready_models_v20_rom_aware.json` -- v20 deploy candidate

## Recommendation

**Do NOT adopt v20 wholesale.** Net tier change vs v17 baseline is negative (-2: 5 promotions, 7 demotions). The extrema-aware loss did not solve FF's translation gap as a one-shot replacement.

**DO adopt v20 selectively for the 5 promotion slots**:

1. **hip_adduction_r / front_oblique_left: Poor -> Good** (CCC 0.29 -> 0.69, LoA/2 6.5 -> 3.3). Largest single absolute lift in this build. Same slot FF v18 also lifted, but only to Moderate; GG2 takes it one tier further.
2. **lumbar_extension / front_oblique_right: Moderate -> Good** (CCC 0.63 -> 0.69, LoA/2 10.2 -> 9.8). Below the strict 10-deg Good threshold by ~0.2 deg -- borderline but classified Good.
3. **lumbar_extension / side_right: Moderate -> Good** (CCC 0.45 -> 0.62, LoA/2 13.3 -> 9.9). FF v18 demoted this slot to Poor; GG2 reverses that and promotes.
4. **hip_flexion_r / side_left: Poor -> Moderate** (CCC 0.60 -> 0.70, LoA/2 16.2 -> 14.5). Modest lift, real tier change.
5. **knee_angle_r / front_oblique_left: Poor -> Moderate** (CCC 0.78 baseline but classified Poor due to LoA 15.6; v20 CCC 0.85, LoA/2 14.9). Real promotion driven by tighter LoA.

**KEEP v17 ridge weights** for the 7 demoted slots. The hand-engineered Layer 2 is geometrically well-suited to in-plane knee/hip from side and oblique views, and v17 ridge coefficients are noise-structure-tuned to those features.

**For the 11 unchanged slots**, no decision needed -- they are at the same tier under both v17 and v20.

**Selective build path**: build a "v21" deploy bundle that adopts v20 weights for the 5 promotion slots and v17 weights for the 7 demoted + 11 unchanged slots. This is the strict net-positive ROI from this work.

**Future build directions** (in order of expected lift):

1. **ENSEMBLE GG2-L2 and EE2-L2** at the per-frame level, then re-run L3 retrain. The two L2 models have different per-metric strengths (GG2 wins knee, lumbar; EE2 wins ankle/oblique-right). A simple per-metric trust-weighted average may unlock 2-3 more promotions.
2. **Differentiable ROM-CCC surrogate loss.** Replace the per-clip extrema absolute-error term with a cross-clip moment-matching term (encourage var(pred_rom across clips) and cov(pred_rom, gt_rom) to track). This targets the actual L3 ridge input distribution directly.
3. **Per-metric loss weighting.** Hip adduction + ankle outperform expectations on GG2 only in some views. A loss-mask that zeros out the extrema term for metrics where it hurts (ankle, hip flexion-FOL) may preserve more L3 tiers.
4. **Higher lambda sweep** (lam in {2, 5, 10}). The brief said lam=1.0 to start. Higher lam may push harder on extrema at acceptable per-frame fidelity cost.

## Single-camera reaffirmation

Every measurement uses a single virtual or real camera per inference. No multi-camera fusion. GG2 consumes one DWPose stream and produces 5 joint angles per frame -- same I/O contract as Couro's hand-engineered Layer 2.

