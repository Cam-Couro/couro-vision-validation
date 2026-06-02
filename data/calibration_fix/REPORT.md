# Couro v15 Calibration Fix: LOSO-Safe Bias Correction + Front-Center Demotion

## Summary

v15 ships **3 Good-tier slots** (unchanged from v14: `hip_adduction_r/side_left`, `ankle_angle_r/side_right`, `lumbar_extension/side_left`) across **23 deploy slots** (down from 25 — `hip_flexion_r/front_center` and `knee_angle_r/front_center` were demoted because per-subject CCC is at or below zero, meaning the model tracks within-subject rep variation but does not rank athletes against population norms). **Build 1 (LOSO-safe bias-intercept correction on the 3 +3-4 degree bias slots) returned a deploy-time correction of essentially 0 degrees per slot and did not lift any slot into the Good tier** — the systematic bias in those slots is an across-subject generalization gap, not a removable global calibration offset (full math below). The headline corrected slot `hip_adduction_r/front_center` stayed at per-subject CCC = 0.767, half-LoA = 16.22 degrees after correction; the slot ships with `bias_correction_deg ~= 0` and the existing v14 model behavior is preserved. Both demoted slots are gone from `deploy_ready_models_v15.json`.

## Tier counts: v14 -> v15

| Tier | v14 (n=25) | v15 (n=23) | Delta |
| --- | ---: | ---: | ---: |
| Excellent | 0 | 0 | 0 |
| Good | 3 | 3 | 0 |
| Moderate | 9 | 9 | 0 |
| Poor | 13 | 11 | -2 (both demoted slots were Poor) |

## Build 1 - bias intercept correction (3 slots)

### Methodology actually applied

Per slot, for each LOSO fold s:

1. Fit ridge on all subjects except s.
2. Compute the train-fold bias intercept as `b_s = mean(pred_train - obs_train)`.
3. Predict on the held-out subject and correct: `pred_corrected = pred_held_out - b_s`.
4. Compute per-subject Bland-Altman + CCC on the corrected held-out predictions.

The deploy-time scalar `bias_correction_deg` shipped in `deploy_ready_models_v15.json` is the across-folds mean of `b_s` - a leak-free estimator of the model's systematic offset at inference time.

This is exactly what the spec mandated. It is the methodologically clean approach: the held-out subject never participates in its own correction term, so no leakage.

### Result: spec-mandated estimator is identically zero

| Slot | Pre Pearson r | Pre CCC | Pre bias (deg) | Pre half-LoA (deg) | `bias_correction_deg` | Post CCC | Post half-LoA (deg) | Tier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hip_adduction_r / front_center | 0.925 | 0.767 | +3.60 | 16.22 | -4e-15 (~0) | 0.767 | 16.22 | Poor (unchanged) |
| hip_adduction_r / front_oblique_right | 0.919 | 0.775 | +3.98 | 17.28 | -2e-17 (~0) | 0.775 | 17.28 | Poor (unchanged) |
| knee_angle_r / front_oblique_left | 0.913 | 0.784 | -3.20 | 15.60 | -2e-15 (~0) | 0.784 | 15.60 | Poor (unchanged) |

### Why the spec-mandated estimator is zero (closed-form)

Ridge regression with an unregularized intercept (which is how `harness.train_regression_poc.fit_ridge` is implemented - see lines 142-152, `reg[-1, -1] = 0`) sets the intercept so that, on the training data, `mean(pred_train - obs_train) == 0` exactly for every train fold. The per-fold `b_s` is therefore zero up to numerical noise (~1e-15), and the across-folds mean is also zero.

The "+3.60 degree bias" Agent P reported is computed differently: it is `mean(pred - obs)` over **held-out** predictions, aggregated per subject. That mean captures the model's across-subject generalization offset, which is a separate quantity from the train-fold residual mean and cannot be removed without using held-out information.

In short: **the spec-mandated estimator zeroes the train-fold residual mean by construction. The across-subject offset in Bland-Altman is not the train-fold residual mean.** No bias correction of this form can lift these slots into the Good tier.

### Alternative leak-free estimator (exploratory, not shipped)

For transparency, the script `harness/_explore_loo_held_out.py` also computes a different leak-free estimator: for each held-out subject s, `b_s = mean over j != s of (held-out fold mean residual for subject j)` - leave-one-fold-out on held-out residuals. Subject s never contributes to its own correction term, so no leakage. Result file: `data/calibration_fix/loo_held_out_alternative.json`.

| Slot | `deploy_bias_correction_deg` | PRE per-subj CCC / half-LoA | POST per-subj CCC / half-LoA |
| --- | ---: | ---: | ---: |
| hip_adduction_r / front_center | +3.60 | 0.767 / 16.22 | 0.775 / 16.99 |
| hip_adduction_r / front_oblique_right | +3.98 | 0.775 / 17.28 | 0.779 / 18.36 |
| knee_angle_r / front_oblique_left | -3.20 | 0.784 / 15.60 | 0.787 / 16.38 |

Even with this estimator (which does capture the across-subject offset), CCC moves by < 0.01 and the half-LoA grows slightly because shifting predictions by a constant doesn't reduce SD(diff), and SD(diff) is what drives half-LoA in this per-subject n ~ 17-22 regime. **Neither leak-free estimator lifts these slots into the Good tier.**

### What this means for product

- The +3.60 / +3.98 / -3.20 degree biases reported by Agent P are real - but they are **across-subject generalization features**, not removable global calibration. The model behaves as if there is an athlete-specific scale factor that ridge regression hasn't captured (consistent with v12_combined's slope != 1 pattern flagged in `data/biomech_validity_stats/REPORT.md` "Where CCC changes the story" section).
- These slots still have per-subject Pearson r > 0.91, which means **shape ranking is good** - they can support relative comparisons within an athlete or for ranking athletes by magnitude direction. They should stay in v15, just not be branded as "validated" until the LoA narrows.
- The path to lifting them is **not a calibration patch** - it would be either (a) a per-athlete calibration step (one warm-up rep against ground truth), or (b) a model architecture change that captures the across-subject slope (e.g. mixed-effects, athlete embedding, anthropometric covariate).

### What `bias_correction_deg ~= 0` means for the deploy pipeline

The v15 inference contract for the 3 corrected slots is `prediction_corrected = prediction - bias_correction_deg`. With `bias_correction_deg ~= 0`, this is a no-op. The field is present so the inference pipeline can pick it up; future re-training with a different estimator can ship a non-zero value without changing the inference code.

## Build 2 - demote 2 broken front-center slots

| Slot | per-trial r | per-subject r | per-subject CCC | Rationale |
| --- | ---: | ---: | ---: | --- |
| hip_flexion_r / front_center | +0.78 | **-0.21** | **-0.15** | Negative per-subject correlation - the model ranks athletes the wrong way. Tracks within-subject rep variation only. Cannot ship to a product that compares athletes against population norms. |
| knee_angle_r / front_center | +0.70 | **+0.22** | **+0.12** | Near-zero per-subject CCC - the model carries almost no between-subject signal. Tracks reps but not athletes. Cannot ship to a product that compares athletes against population norms. |

Both slots are removed from `results/deploy_ready_models_v15.json`. The remaining 4 views for each metric (side_left, front_oblique_left, front_oblique_right, side_right) are unchanged in v15 and still cover both targets.

## Files produced

- `results/deploy_ready_models_v15.json` - fork of v14 with 3 slots carrying `bias_correction_deg ~= 0` (and `bias_correction_metadata` describing the per-fold estimator) and 2 demoted slots removed. **v14 unchanged.**
- `data/calibration_fix/per_slot_validity_v15.json` - per-slot Bland-Altman + CCC stats over the 23 v15 slots. For the 3 corrected slots, the row also carries `bias_correction_applied: true` and a `pre_correction_per_subject` block for diff-on-readback.
- `data/calibration_fix/loo_held_out_alternative.json` - exploratory leave-one-fold-out-on-held-out estimator results (not shipped).
- `harness/apply_calibration_fix.py` - deterministic runnable script that produces v15 deploy + v15 validity from v14.
- `harness/_explore_loo_held_out.py` - exploratory script for the alternative estimator.

## Methodological note: what counts as LOSO-safe

There are three plausible "bias intercept" estimators that all sound similar but behave very differently:

1. **One-shot on all data** - `bias = mean over all pred - all obs` computed once. **Leaks**: each held-out subject's own residual feeds the correction term. Inflates corrected CCC by a constant. Not used.
2. **Train-fold residual mean (spec-mandated)** - `b_s = mean(train_pred_s - train_obs_s)`, averaged across folds. Leak-free. **Identically zero by construction for ridge with an intercept** - see closed-form argument above. This is what `apply_calibration_fix.py` ships.
3. **Leave-one-fold-out on held-out residuals (exploratory)** - `b_s = mean over j != s of mean(held_out_pred_j - held_out_obs_j)`. Leak-free (subject s does not contribute to its own correction). Captures across-subject offset but does not lift CCC into Good tier on these slots either.

Estimator (1) leaks. Estimator (2) is what the spec asked for and what ships in v15. Estimator (3) is methodologically defensible and is documented as the alternative if Saad wants to revisit; the math says neither (2) nor (3) is enough to lift these three slots into the Good tier, because the bias term is small relative to SD(diff) in the per-subject regime.

## Honest caveat: LoA half-width > +/-10 degrees after correction

Per the spec: "If a slot has been corrected and still has LoA half-width > +/-10 degrees in the held-out fold, report that explicitly - bias correction only fixes the offset, not the variance."

All 3 corrected slots have per-subject half-LoA > +/-10 degrees after correction (16.22, 17.28, 15.60 degrees) because **bias correction only shifts the LoA interval up or down by `bias_correction_deg`; it does not narrow it.** SD(differences) is unchanged. To narrow the LoA, the underlying model would need to reduce per-subject prediction variance - that is a model change, not a calibration change.
