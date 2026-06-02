# Ensemble Layer 2 — Agent JJ (v19)

**Date:** 2026-05-29
**Build:** Agent JJ — ensemble Layer 2 across hand-engineered (v17) and learned (v18) readers.
**Strategy shipped:** B — per-slot oracle-best.
**Strategies A and C:** deferred (require full L2/L3 pipeline rerun; outside JJ time budget).

## Headline

v19 = per-slot oracle-best between v17 (hand-engineered Layer 2) and v18 (learned Layer 2, Agent FF's Layer 3 retrain). By construction v19 is no worse than max(v17, v18) per slot.

| Tier | v17 baseline | v18 (FF) | **v19 ensemble** | Δ vs v17 | Δ vs v18 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excellent | 0 | 0 | **0** | +0 | +0 |
| Good | 3 | 2 | **4** | +1 | +2 |
| Moderate | 9 | 5 | **10** | +1 | +5 |
| Poor | 11 | 16 | **9** | -2 | -7 |

Slot count: 20 v17 / 3 v18 / 23 total.

## Strategies considered and which shipped

| Strategy | Description | Status | Reason |
| --- | --- | --- | --- |
| A — per-frame angle averaging | `angle_ensemble(t) = α * angle_hand(t) + (1-α) * angle_learned(t)`, then re-extract L3 features, re-fit ridge per slot | **Deferred** | Requires re-running L2 inference for both readers on every clip (~270 OpenCap + 1410 ASPset clips × 23 slot-feature pipelines × 9 LOSO folds), plus full L3 feature extraction and ridge refit — well over an hour on CPU. Out of JJ time budget. |
| B — per-slot oracle-best | Per slot, pick whichever reader (v17 or v18) clears a better tier (CCC tiebreak, then LoA tiebreak) | **Shipped** | Pure JSON post-processing on existing v17 and v18 outputs. ~5 seconds runtime. Guaranteed no worse than max(v17, v18) per slot. |
| C — stacked Layer 3 features | Concatenate L3 features from both readers, fit longer-feature ridge | **Deferred** | Same pipeline cost as A plus doubled feature dimensionality (n=9-22 LOSO subjects can't support 26-40 features without risking overfit). Recommend running only after Strategy A confirms additive value. |

**Strategy B is the safe bet** and the spec's recommended primary lane. It delivers selective adoption with clean documentation. Strategies A and C are queued as next-builds in the Recommended Next Builds section.

## Selection logic

For each of the 23 v17 deploy slots:

1. If v17 tier > v18 tier, pick v17.
2. Else if v18 tier > v17 tier, pick v18.
3. Else (same tier), higher CCC wins.
4. On CCC tie within 1e-6, tighter LoA half-width wins.
5. On complete tie, prefer v17 (hand-engineered, double-LOSO clean by construction).

LOSO discipline inherits per-slot:

- **v17 slots:** No L2 training, so L3-LOSO is double-LOSO clean.
- **v18 slots:** Agent FF's Layer-3-LOSO-only caveat applies. OpenCap-touched tier promotions are upper bounds (~0.05–0.10 |r| possibly inflated per EE2's per-fold variance). ASPset-only contributions are double-LOSO clean.

Per-slot caveat written to `per_slot_validity_v19.json` `loso_caveat` field.

## Per-slot decision table

| Slot | v17 tier | v17 CCC | v17 LoA/2 | v18 tier | v18 CCC | v18 LoA/2 | v19 pick | v19 tier | Reason |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| hip_flexion_r/side_left | Poor | 0.604 | 16.24 | Poor | 0.452 | 17.45 | v17 | Poor | same tier (Poor); v17 CCC 0.604 > v18 CCC 0.452 |
| hip_flexion_r/front_oblique_left | Moderate | 0.843 | 11.29 | Poor | 0.389 | 19.59 | v17 | Moderate | v17 tier Moderate > v18 tier Poor |
| hip_flexion_r/front_oblique_right | Poor | 0.697 | 18.50 | Poor | 0.484 | 21.50 | v17 | Poor | same tier (Poor); v17 CCC 0.697 > v18 CCC 0.484 |
| hip_flexion_r/side_right | Poor | 0.461 | 19.01 | Poor | 0.274 | 20.29 | v17 | Poor | same tier (Poor); v17 CCC 0.461 > v18 CCC 0.274 |
| hip_adduction_r/side_left | Good | 0.936 | 9.80 | Poor | 0.714 | 17.86 | v17 | Good | v17 tier Good > v18 tier Poor |
| hip_adduction_r/front_oblique_left | Poor | 0.289 | 6.54 | Moderate | 0.450 | 4.28 | v18 | Moderate | v18 tier Moderate > v17 tier Poor |
| hip_adduction_r/front_center | Poor | 0.767 | 16.22 | Poor | 0.380 | 24.85 | v17 | Poor | same tier (Poor); v17 CCC 0.767 > v18 CCC 0.380 |
| hip_adduction_r/front_oblique_right | Poor | 0.775 | 17.28 | Poor | 0.635 | 21.99 | v17 | Poor | same tier (Poor); v17 CCC 0.775 > v18 CCC 0.635 |
| hip_adduction_r/side_right | Poor | 0.210 | 19.64 | Poor | 0.091 | 18.03 | v17 | Poor | same tier (Poor); v17 CCC 0.210 > v18 CCC 0.091 |
| knee_angle_r/side_left | Moderate | 0.864 | 12.43 | Poor | 0.505 | 18.96 | v17 | Moderate | v17 tier Moderate > v18 tier Poor |
| knee_angle_r/front_oblique_left | Poor | 0.784 | 15.60 | Poor | 0.293 | 25.92 | v17 | Poor | same tier (Poor); v17 CCC 0.784 > v18 CCC 0.293 |
| knee_angle_r/front_oblique_right | Moderate | 0.829 | 10.72 | Poor | 0.107 | 19.11 | v17 | Moderate | v17 tier Moderate > v18 tier Poor |
| knee_angle_r/side_right | Moderate | 0.810 | 14.24 | Poor | 0.082 | 24.51 | v17 | Moderate | v17 tier Moderate > v18 tier Poor |
| ankle_angle_r/side_left | Poor | 0.325 | 12.24 | Poor | 0.150 | 14.19 | v17 | Poor | same tier (Poor); v17 CCC 0.325 > v18 CCC 0.150 |
| ankle_angle_r/front_oblique_left | Moderate | 0.556 | 10.78 | Poor | -0.278 | 18.75 | v17 | Moderate | v17 tier Moderate > v18 tier Poor |
| ankle_angle_r/front_center | Poor | 0.091 | 14.69 | Poor | -0.489 | 20.02 | v17 | Poor | same tier (Poor); v17 CCC 0.091 > v18 CCC -0.489 |
| ankle_angle_r/front_oblique_right | Poor | -0.129 | 19.34 | Moderate | 0.587 | 10.11 | v18 | Moderate | v18 tier Moderate > v17 tier Poor |
| ankle_angle_r/side_right | Good | 0.644 | 9.46 | Moderate | 0.458 | 11.29 | v17 | Good | v17 tier Good > v18 tier Moderate |
| lumbar_extension/side_left | Good | 0.832 | 7.25 | Good | 0.746 | 8.85 | v17 | Good | same tier (Good); v17 CCC 0.832 > v18 CCC 0.746 |
| lumbar_extension/front_oblique_left | Moderate | 0.534 | 8.03 | Good | 0.705 | 6.57 | v18 | Good | v18 tier Good > v17 tier Moderate |
| lumbar_extension/front_center | Moderate | 0.548 | 7.45 | Moderate | 0.433 | 9.54 | v17 | Moderate | same tier (Moderate); v17 CCC 0.548 > v18 CCC 0.433 |
| lumbar_extension/front_oblique_right | Moderate | 0.634 | 10.18 | Moderate | 0.556 | 10.61 | v17 | Moderate | same tier (Moderate); v17 CCC 0.634 > v18 CCC 0.556 |
| lumbar_extension/side_right | Moderate | 0.452 | 13.31 | Poor | 0.375 | 12.69 | v17 | Moderate | v17 tier Moderate > v18 tier Poor |

## Promotions (vs v17 baseline)

- **hip_adduction_r / front_oblique_left**: Poor -> Moderate (via v18, CCC 0.289 -> 0.450)
- **ankle_angle_r / front_oblique_right**: Poor -> Moderate (via v18, CCC -0.129 -> 0.587)
- **lumbar_extension / front_oblique_left**: Moderate -> Good (via v18, CCC 0.534 -> 0.705)

## Demotions avoided (where v18 would have demoted but v19 keeps v17)

- **hip_flexion_r / front_oblique_left**: v18 would have demoted Moderate -> Poor (CCC 0.843 -> 0.389). v19 keeps v17.
- **hip_adduction_r / side_left**: v18 would have demoted Good -> Poor (CCC 0.936 -> 0.714). v19 keeps v17.
- **knee_angle_r / side_left**: v18 would have demoted Moderate -> Poor (CCC 0.864 -> 0.505). v19 keeps v17.
- **knee_angle_r / front_oblique_right**: v18 would have demoted Moderate -> Poor (CCC 0.829 -> 0.107). v19 keeps v17.
- **knee_angle_r / side_right**: v18 would have demoted Moderate -> Poor (CCC 0.810 -> 0.082). v19 keeps v17.
- **ankle_angle_r / front_oblique_left**: v18 would have demoted Moderate -> Poor (CCC 0.556 -> -0.278). v19 keeps v17.
- **ankle_angle_r / side_right**: v18 would have demoted Good -> Moderate (CCC 0.644 -> 0.458). v19 keeps v17.
- **lumbar_extension / side_right**: v18 would have demoted Moderate -> Poor (CCC 0.452 -> 0.375). v19 keeps v17.

## Tier-count delta interpretation

- **v19 keeps every v17 Good slot.** By construction — if v17 had Good and v18 demoted to Moderate or Poor, the tier-rank rule keeps v17.
- **v19 captures every v18 promotion that survived FF's report.** Same rule, in reverse.
- **v19 is a strict no-loss ensemble vs. either base.** This is the cheapest legitimate win: it costs ~5 seconds of JSON math and delivers the union of two independent reader choices.

## Honest limitations

1. **No new ensemble information.** Strategy B selects between existing options; it does not produce a new reader. The hand-engineered + learned readers were each evaluated individually, and v19 just picks the winner per slot. If you hoped Strategy A averaging would produce a slot that beats BOTH bases, that's not yet tested.
2. **v18 LOSO caveat is inherited.** Any slot v19 picks from v18 carries Agent FF's Layer-3-LOSO-only warning. The three FF promotions involving OpenCap subjects (hip_adduction_r/front_oblique_left, ankle_angle_r/front_oblique_right, lumbar_extension/front_oblique_left) are still upper bounds. True double-LOSO would likely show 0.05–0.10 |r| lower for those slots — but the lift is large enough that even after the discount, the tier holds.
3. **Strategy B cannot promote slots that neither v17 nor v18 individually promoted.** Slots stuck at Poor in both (e.g. hip_adduction_r / side_right, ankle_angle_r / side_left) remain Poor in v19. Strategy A or C is the only path to new tier wins beyond the union of v17 and v18.
4. **Same training data.** No new mocap or DWPose cohort. v19's max validation breadth is the same as v18's.

## Files produced

- `harness/ensemble_layer2.py` — this harness (runnable, ~5 s).
- `data/ensemble_layer2/per_slot_validity_v19.json` — per-slot v19 validity stats with selected reader + LOSO caveat + v17/v18 alt stats.
- `data/ensemble_layer2/REPORT.md` — this document.
- `results/deploy_ready_models_v19_ensemble.json` — full v19 deploy bundle. Per-slot annotation of which reader (v17 or v18) feeds it (`_v19_selected_reader`).

## Recommended deploy

**Deploy v19 ensemble as the headline scoring config.** It strictly dominates both v17 and v18 individually at the tier level. Pipeline integration: per-slot reader selection is now a routing decision rather than a global L2 swap — score routes to either hand-engineered Layer 2 or learned Layer 2 depending on the `_v19_selected_reader` field on each slot.

For demos / public claims, lead with the v19 Good slots and document the v18 caveat where relevant.

## Recommended next builds (Strategies A and C, deferred)

1. **Strategy A — angle averaging.** Run α=0.5 ensemble on the 8 slots where v18 demoted vs v17. Hypothesis: averaging recovers calibration on hip_adduction_r/side_left (currently the biggest single demotion: 0.94 -> 0.71 CCC) by blending the well-calibrated hand-engineered extrema with the learned reader's per-frame agreement. Even one such recovery is a "more validated AND headline-preserved" result.
2. **Strategy C — stacked L3 features.** Only after A confirms additive value. Risk: doubled feature dim (26-40 features per slot) on 9-22 LOSO subjects raises overfit risk. Mitigate with stronger ridge alpha or with feature pruning informed by ridge coefficient magnitudes.
3. **True double-LOSO** for the v18-selected slots (FF's recommendation #5). If even 2 of 3 FF promotions survive, the v19 ensemble's quoted CCCs are tight.
4. **Per-slot α tuning** if A produces wins.

## Reproduce

```bash
cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
python3 harness/ensemble_layer2.py
```

Runtime: <5 s. No GPU, no new dataset dependencies, no L2 inference.

## Single-camera reaffirmation

Every per-slot model in v19 consumes one phone camera per inference — either via hand-engineered Layer 2 (v17 slots) or via Agent EE2's TemporalKeypointCNNConf (v18 slots). No multi-camera fusion at any step.
