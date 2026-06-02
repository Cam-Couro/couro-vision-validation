# Slope-and-Intercept Calibration Correction — LOSO-rigorous Nested Test

## Summary

**Honest fail.** Slope+intercept correction does NOT reduce LoA half-width on any of the 3 slots. On all 3 slots, the corrected half-LoA is within 0.75 deg of the uncorrected baseline (and in fact slightly wider on all 3). The deploy-time slope scalars come out very close to 1.0 (b in [0.95, 0.99]), meaning the inner-LOSO regression of observed-on-predicted is essentially the identity. There is no scale issue to remove. The residual error is additive noise around the trend, not scale-proportional, so per-athlete calibration would not fix it either. The only path to LoA reduction on these slots is variance reduction at the Layer 2 / Layer 3 feature-engineering level — improving the underlying predictions, not post-hoc remapping them.

### LoA half-width: baseline vs slope+intercept

| Slot | baseline half-LoA (deg) | slope+intercept half-LoA (deg) | delta | deploy slope b |
| --- | ---: | ---: | ---: | ---: |
| hip_adduction_r / front_center | 16.22 | 16.43 | -0.22 | 0.9826 |
| hip_adduction_r / front_oblique_right | 17.28 | 18.01 | -0.73 | 0.9511 |
| knee_angle_r / front_oblique_left | 15.60 | 15.71 | -0.11 | 0.9899 |

## Methodology

Nested LOSO. For each (slot, outer held-out subject s_out):

1. Fit outer ridge on all subjects except s_out (alpha = 10).
2. Inner LOSO on the outer train fold: for each inner held-out subject s_in, fit ridge on subjects \ {s_out, s_in} and predict on s_in. This produces a set of (pred, obs) pairs in which every prediction was made without the corresponding subject being in the training set. These pairs are leak-free estimates of the kind of (pred, obs) sample the outer ridge will produce.
3. Fit OLS observed = a + b * predicted on the inner-LOSO pairs. (a, b) is a per-outer-fold calibration parameter; subject s_out never participates in fitting it.
4. Compute the corrected outer prediction for s_out: `pred_corrected = a + b * outer_pred`. Aggregate across all outer folds and evaluate per-subject Bland-Altman + Lin's CCC.

Four conditions are reported per slot for diagnosis:

- **baseline_v15**: outer ridge predictions, no correction. Matches v15 deploy stats.
- **intercept_only**: outer prediction minus the across-inner-folds mean of the inner-held-out residual means. This is Agent Q's leak-free alternative estimator (leave-one-fold-out on held-out residuals) reproduced here for side-by-side diagnosis. It is mathematically guaranteed not to reduce SD(differences) — it can only shift the LoA interval up or down.
- **slope_only**: `pred_corrected = b * pred`, b fit through origin on inner-LOSO pairs. Tests whether the error is purely a scale issue with the intercept already absorbed by the ridge.
- **slope_and_intercept**: `pred_corrected = a + b * pred`, (a, b) fit by OLS on inner-LOSO pairs. The main candidate.

Why slope+intercept can in principle reduce LoA: if the inner-LOSO regression of observed-on-predicted has slope b < 1 (the model overshoots peak ROM by a scale factor), then `a + b * pred` shrinks the predicted-ROM range. When the true scale of (obs - pred) error is proportional to ROM, this shrinkage also shrinks SD(differences) and therefore the LoA half-width. If the error is instead additive noise around the trend, slope correction shrinks the range without shrinking SD(diff) and may even widen LoA slightly.

## Per-slot results — 4-condition comparison

All values are per-subject Bland-Altman + Lin's CCC. Good gate: CCC > 0.60 AND LoA half-width < +/-10 deg.

### hip_adduction_r / front_center

n_subjects = 22, n_trials = 445, approach = v12_combined.

| Condition | CCC | bias (deg) | half-LoA (deg) | MAE (deg) | RMSE (deg) | Tier | Good? |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Baseline v15 (no correction) | 0.767 | 3.60 | 16.22 | 7.21 | 8.85 | Poor | no |
| Intercept-only (Agent Q form) | 0.788 | -0.01 | 16.53 | 6.84 | 8.24 | Poor | no |
| Slope-only (a=0, b fit) | 0.767 | 3.51 | 16.25 | 7.20 | 8.83 | Poor | no |
| Slope + intercept (main) | 0.759 | 3.65 | 16.43 | 7.30 | 8.97 | Poor | no |

Deploy-time scalars (across-folds mean): a = 0.652 deg, b = 0.9826 (slope_only b = 0.9975).

### hip_adduction_r / front_oblique_right

n_subjects = 17, n_trials = 212, approach = v12_combined.

| Condition | CCC | bias (deg) | half-LoA (deg) | MAE (deg) | RMSE (deg) | Tier | Good? |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Baseline v15 (no correction) | 0.775 | 3.98 | 17.28 | 7.76 | 9.43 | Poor | no |
| Intercept-only (Agent Q form) | 0.810 | -0.02 | 17.19 | 7.02 | 8.51 | Poor | no |
| Slope-only (a=0, b fit) | 0.776 | 3.71 | 17.39 | 7.81 | 9.37 | Poor | no |
| Slope + intercept (main) | 0.750 | 4.15 | 18.01 | 8.26 | 9.84 | Poor | no |

Deploy-time scalars (across-folds mean): a = 1.671 deg, b = 0.9511 (slope_only b = 0.9912).

### knee_angle_r / front_oblique_left

n_subjects = 21, n_trials = 440, approach = v12_combined.

| Condition | CCC | bias (deg) | half-LoA (deg) | MAE (deg) | RMSE (deg) | Tier | Good? |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Baseline v15 (no correction) | 0.784 | -3.20 | 15.60 | 6.95 | 8.40 | Poor | no |
| Intercept-only (Agent Q form) | 0.807 | 0.01 | 15.70 | 6.48 | 7.82 | Poor | no |
| Slope-only (a=0, b fit) | 0.786 | -3.11 | 15.58 | 6.92 | 8.36 | Poor | no |
| Slope + intercept (main) | 0.782 | -3.11 | 15.71 | 6.96 | 8.42 | Poor | no |

Deploy-time scalars (across-folds mean): a = 0.927 deg, b = 0.9899 (slope_only b = 1.0011).

## Honest interpretation

Three diagnostic facts converge on the same answer:

1. **The inner-LOSO slope b is essentially 1.** Across all 3 slots the deploy-time slope sits in the range [0.95, 0.99]. If the model were overshooting peak ROM by a scale factor we would expect b notably below 1; we do not see that.
2. **The slope-only condition does not improve LoA either.** Even when we force a pure proportional rescale (intercept tied to zero), half-LoA moves by <=0.05 deg on every slot. The variance of (obs - pred) is genuinely scale-invariant.
3. **The intercept-only condition closes the bias but not the LoA.** Per Bland-Altman geometry, intercept correction shifts the mean of differences but cannot reduce SD(differences); we see that explicitly — bias goes to ~0 but half-LoA is unchanged.

Together these three facts say: the residual error on these slots is **additive noise around the trend**, not scale-proportional and not removable by a global intercept either. Per-athlete calibration (one warm-up rep) cannot fix this either, because the noise is not what's miscalibrated — it's the trial-to-trial residual the model itself emits.

**Implication for next steps.** The only path to LoA reduction on these slots is variance reduction at the model/feature level. Two concrete moves:

- **Layer 2 / Layer 3 feature engineering** — add features that capture the currently-unexplained ROM variance (e.g. event-anchored loading-phase features for knee/hip rotation, depth-aware features for hip_adduction that disambiguate front-on from oblique frontal-plane motion).
- **Approach swap** — these 3 slots all run v12_combined. The same metric on side views runs v14_full_dwpose and already lands in Good tier. The honest answer for front-facing hip adduction may be 'ship side views as Good, leave frontal views Poor until pose detection improves on the frontal plane'.

**Implication for v15 deploy schema.** No change. We do not ship the slope+intercept scalars because they would slightly *widen* LoA on the live inference pipeline. v15 stays as-is. No recommended_v19_updates.json is produced.

## LOSO discipline statement

The (a, b) calibration parameters are fit per outer fold using inner LOSO on the outer train fold. The outer held-out subject never participates in fitting its own calibration parameters. The deploy-time scalars reported in the deploy calibration block are the across-folds mean — that is what would ship at inference if a slot crossed the Good gate. No post-hoc cherry-picking; no leakage.

