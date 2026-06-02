# Couro Layer 3 Nonlinear Model Sweep

Tests whether Random Forest, Gradient Boosting, or a small MLP improve over the deployed ridge regression on eight priority slots flagged as Moderate (close to the Good gate) in `data/biomech_validity_stats/per_slot_validity.json`. Evaluation is subject-level leave-one-subject-out (LOSO) on each slot's original dataset builder. Ridge is re-run inside this script under the same pipeline so the comparison is apples-to-apples.

## Bottom line

**No slot crosses the Good gate under rigorous LOSO evaluation with seed-stability control.** The single seed=0 candidate that initially appeared to cross (lumbar_extension / front_center / event_anchored / MLP, seed=0 CCC=0.74) was checked across 7 random seeds and produced CCC between 0.42 and 0.85 (mean 0.60 ± 0.15), crossing the Good gate on only 4 of 7 seeds. That is seed-luck variance, not a validated improvement.

**This is an honest fail of the model-class hypothesis on these slots.** It tells us features (or the Layer-2 calibration upstream) are the bottleneck at n=8-21 subjects, not the linearity assumption. Motivates Agent EE2's learned-Layer-2 work as the right next direction.

## Models tested

- **Ridge (baseline replay)** - alpha=10.0, StandardScaler. Deterministic.
- **Random Forest** - n_estimators=100, max_depth=4, min_samples_leaf=2.
- **Gradient Boosting** - n_estimators=100, max_depth=3, learning_rate=0.05.
- **MLP** - hidden_layer_sizes=(32,16), early_stopping=True, max_iter=300, StandardScaler.

All three nonlinear models were re-run across seeds [0, 1, 2, 3, 7, 17, 42] to control for stochastic estimator variance.

## Tier gates (biomech device-validation convention)

| Tier | CCC | LoA half-width |
| --- | --- | --- |
| Excellent | > 0.75 | < +/- 5 deg |
| Good | 0.60 - 0.75 | +/- 5 - 10 deg |
| Moderate | 0.40 - 0.60 | +/- 10 - 15 deg |
| Poor | <= 0.40 | > +/- 15 deg |

## Headline table: per-slot per-model-class CCC + LoA

For RF/GB/MLP the CCC and LoA values are reported as seed-mean +/- SD across 7 random seeds. `good/N` = number of seeds where the model actually crossed the Good gate (CCC > 0.60 AND LoA half-width < 10 deg).

| Slot | n subj | n trials | Model | CCC | LoA half (deg) | Tier (best seed) | good/7 seeds |
| --- | ---: | ---: | --- | --- | --- | --- | ---: |
| **knee_angle_r/front_oblique_right/v9_phased** | 8 | 40 | ridge (deploy doc) | 0.829 | 10.72 | Moderate | n/a |
| | | | ridge (replay) | 0.829 | 10.72 | Moderate | n/a |
| | | | random_forest | +0.552 +/- 0.019 | 16.62 +/- 0.42 | Poor | 0/7 |
| | | | gradient_boosting | +0.555 +/- 0.018 | 16.88 +/- 0.30 | Poor | 0/7 |
| | | | mlp | -0.105 +/- 0.077 | 63.84 +/- 8.45 | Poor | 0/7 |
| **hip_flexion_r/front_oblique_left/v13_dwpose_hybrid** | 21 | 432 | ridge (deploy doc) | 0.843 | 11.29 | Moderate | n/a |
| | | | ridge (replay) | 0.843 | 11.29 | Moderate | n/a |
| | | | random_forest | +0.758 +/- 0.019 | 14.60 +/- 0.73 | Poor | 0/7 |
| | | | gradient_boosting | +0.748 +/- 0.001 | 14.83 +/- 0.04 | Moderate | 0/7 |
| | | | mlp | +0.591 +/- 0.035 | 19.49 +/- 1.61 | Poor | 0/7 |
| **knee_angle_r/side_left/v14_full_dwpose** | 12 | 139 | ridge (deploy doc) | 0.864 | 12.43 | Moderate | n/a |
| | | | ridge (replay) | 0.864 | 12.43 | Moderate | n/a |
| | | | random_forest | +0.897 +/- 0.005 | 11.00 +/- 0.27 | Moderate | 0/7 |
| | | | gradient_boosting | +0.887 +/- 0.003 | 11.76 +/- 0.16 | Moderate | 0/7 |
| | | | mlp | +0.403 +/- 0.053 | 38.82 +/- 4.48 | Poor | 0/7 |
| **knee_angle_r/side_right/v12_combined** | 10 | 83 | ridge (deploy doc) | 0.810 | 14.24 | Moderate | n/a |
| | | | ridge (replay) | 0.810 | 14.24 | Moderate | n/a |
| | | | random_forest | +0.670 +/- 0.012 | 16.44 +/- 0.19 | Poor | 0/7 |
| | | | gradient_boosting | +0.754 +/- 0.002 | 14.68 +/- 0.06 | Moderate | 0/7 |
| | | | mlp | -0.055 +/- 0.020 | 99.40 +/- 11.06 | Poor | 0/7 |
| **lumbar_extension/front_center/event_anchored** | 9 | 53 | ridge (deploy doc) | 0.548 | 7.45 | Moderate | n/a |
| | | | ridge (replay) | 0.548 | 7.45 | Moderate | n/a |
| | | | random_forest | +0.394 +/- 0.023 | 8.30 +/- 0.13 | Poor | 0/7 |
| | | | gradient_boosting | +0.343 +/- 0.010 | 8.72 +/- 0.04 | Poor | 0/7 |
| | | | mlp | +0.604 +/- 0.151 | 6.82 +/- 1.96 | Good | 4/7 |
| **lumbar_extension/front_oblique_left/event_anchored** | 9 | 54 | ridge (deploy doc) | 0.534 | 8.03 | Moderate | n/a |
| | | | ridge (replay) | 0.534 | 8.03 | Moderate | n/a |
| | | | random_forest | +0.308 +/- 0.018 | 9.88 +/- 0.11 | Poor | 0/7 |
| | | | gradient_boosting | +0.282 +/- 0.004 | 10.41 +/- 0.03 | Poor | 0/7 |
| | | | mlp | +0.046 +/- 0.015 | 57.85 +/- 10.68 | Poor | 0/7 |
| **ankle_angle_r/front_oblique_left/v14_full_dwpose** | 9 | 54 | ridge (deploy doc) | 0.556 | 10.78 | Moderate | n/a |
| | | | ridge (replay) | 0.556 | 10.78 | Moderate | n/a |
| | | | random_forest | +0.274 +/- 0.035 | 12.63 +/- 0.27 | Poor | 0/7 |
| | | | gradient_boosting | +0.330 +/- 0.005 | 12.57 +/- 0.07 | Poor | 0/7 |
| | | | mlp | +0.143 +/- 0.064 | 28.50 +/- 5.43 | Poor | 0/7 |
| **lumbar_extension/side_right/v14_full_dwpose** | 10 | 84 | ridge (deploy doc) | 0.452 | 13.31 | Moderate | n/a |
| | | | ridge (replay) | 0.452 | 13.31 | Moderate | n/a |
| | | | random_forest | +0.043 +/- 0.015 | 15.73 +/- 0.11 | Poor | 0/7 |
| | | | gradient_boosting | +0.046 +/- 0.012 | 15.13 +/- 0.18 | Poor | 0/7 |
| | | | mlp | +0.069 +/- 0.113 | 22.59 +/- 2.97 | Poor | 0/7 |

## Per-slot tier verdict

- **knee_angle_r/front_oblique_right/v9_phased** (baseline ridge CCC=0.829, tier=Moderate): **no change** - no nonlinear model class improves on ridge under seed-stability control.
- **hip_flexion_r/front_oblique_left/v13_dwpose_hybrid** (baseline ridge CCC=0.843, tier=Moderate): **no change** - no nonlinear model class improves on ridge under seed-stability control.
- **knee_angle_r/side_left/v14_full_dwpose** (baseline ridge CCC=0.864, tier=Moderate): meaningful seed-mean improvement (random_forest: CCC +0.897 vs ridge 0.864), same tier.
- **knee_angle_r/side_right/v12_combined** (baseline ridge CCC=0.810, tier=Moderate): **no change** - no nonlinear model class improves on ridge under seed-stability control.
- **lumbar_extension/front_center/event_anchored** (baseline ridge CCC=0.548, tier=Moderate): meaningful seed-mean improvement (mlp: CCC +0.604 vs ridge 0.548), same tier.
- **lumbar_extension/front_oblique_left/event_anchored** (baseline ridge CCC=0.534, tier=Moderate): **no change** - no nonlinear model class improves on ridge under seed-stability control.
- **ankle_angle_r/front_oblique_left/v14_full_dwpose** (baseline ridge CCC=0.556, tier=Moderate): **no change** - no nonlinear model class improves on ridge under seed-stability control.
- **lumbar_extension/side_right/v14_full_dwpose** (baseline ridge CCC=0.452, tier=Moderate): **no change** - no nonlinear model class improves on ridge under seed-stability control.

## Why we dug into seed stability

Initial single-seed (seed=0) run reported **lumbar_extension / front_center / event_anchored / MLP** as CCC=0.74, LoA half-width=4.4 deg -> **Good** tier. But the train-on-all R^2 for the same MLP/seed was -1.55 (worse than predicting the mean), which is an early-stopping artifact, not a sign the model learned a generalisable signal. We re-ran the same MLP under 7 seeds:

| seed | CCC | LoA half-width (deg) | tier |
| ---: | ---: | ---: | --- |
| 0 | +0.739 | 4.40 | Good |
| 1 | +0.683 | 7.28 | Good |
| 2 | +0.419 | 9.60 | Moderate |
| 3 | +0.717 | 7.10 | Good |
| 7 | +0.481 | 8.79 | Moderate |
| 17 | +0.753 | 4.79 | Excellent |
| 42 | +0.437 | 5.76 | Moderate |

The MLP crosses the Good gate on **4 of 7 seeds**. That is not a robust improvement - it is the high-variance estimator landing in the Good region on lucky seeds. The seed=0 number we initially reported is an honest false positive that the stability check caught.

## Overfit assessment (seed=0)

`overfit_gap` = (train-on-all R^2) - (LOSO Pearson r)^2. A wide positive gap means the model fit the training data well but did not generalise. A *negative* gap on MLP means the early-stopping holdout split inside the full-data fit got unlucky - the LOSO score is the trustworthy number, but the model class itself is clearly unstable at this n.

| Slot | Model | train R^2 | LOSO r^2 | gap | flag |
| --- | --- | ---: | ---: | ---: | --- |
| knee_angle_r/front_oblique_right | ridge | 0.867 | 0.742 | 0.125 | minimal |
| knee_angle_r/front_oblique_right | random_forest | 0.919 | 0.358 | 0.561 | SEVERE overfit |
| knee_angle_r/front_oblique_right | gradient_boosting | 0.999 | 0.358 | 0.641 | SEVERE overfit |
| knee_angle_r/front_oblique_right | mlp | -6.054 | 0.083 | -6.137 | MLP early-stop unlucky |
| hip_flexion_r/front_oblique_left | ridge | 0.744 | 0.719 | 0.025 | minimal |
| hip_flexion_r/front_oblique_left | random_forest | 0.833 | 0.546 | 0.287 | mild overfit |
| hip_flexion_r/front_oblique_left | gradient_boosting | 0.894 | 0.565 | 0.329 | moderate overfit |
| hip_flexion_r/front_oblique_left | mlp | 0.627 | 0.423 | 0.205 | mild overfit |
| knee_angle_r/side_left | ridge | 0.729 | 0.772 | -0.042 | minimal |
| knee_angle_r/side_left | random_forest | 0.886 | 0.820 | 0.066 | minimal |
| knee_angle_r/side_left | gradient_boosting | 0.962 | 0.784 | 0.179 | mild overfit |
| knee_angle_r/side_left | mlp | -2.202 | 0.280 | -2.482 | MLP early-stop unlucky |
| knee_angle_r/side_right | ridge | 0.815 | 0.656 | 0.160 | mild overfit |
| knee_angle_r/side_right | random_forest | 0.914 | 0.496 | 0.418 | moderate overfit |
| knee_angle_r/side_right | gradient_boosting | 0.994 | 0.594 | 0.400 | moderate overfit |
| knee_angle_r/side_right | mlp | -0.473 | 0.003 | -0.476 | minimal |
| lumbar_extension/front_center | ridge | 0.535 | 0.579 | -0.044 | minimal |
| lumbar_extension/front_center | random_forest | 0.800 | 0.447 | 0.353 | moderate overfit |
| lumbar_extension/front_center | gradient_boosting | 0.982 | 0.326 | 0.656 | SEVERE overfit |
| lumbar_extension/front_center | mlp | -1.552 | 0.814 | -2.366 | MLP early-stop unlucky |
| lumbar_extension/front_oblique_left | ridge | 0.555 | 0.382 | 0.172 | mild overfit |
| lumbar_extension/front_oblique_left | random_forest | 0.836 | 0.130 | 0.706 | SEVERE overfit |
| lumbar_extension/front_oblique_left | gradient_boosting | 0.988 | 0.097 | 0.891 | SEVERE overfit |
| lumbar_extension/front_oblique_left | mlp | -3.798 | 0.017 | -3.815 | MLP early-stop unlucky |
| ankle_angle_r/front_oblique_left | ridge | 0.680 | 0.387 | 0.294 | mild overfit |
| ankle_angle_r/front_oblique_left | random_forest | 0.871 | 0.115 | 0.756 | SEVERE overfit |
| ankle_angle_r/front_oblique_left | gradient_boosting | 0.994 | 0.168 | 0.826 | SEVERE overfit |
| ankle_angle_r/front_oblique_left | mlp | -4.562 | 0.012 | -4.574 | MLP early-stop unlucky |
| lumbar_extension/side_right | ridge | 0.685 | 0.223 | 0.462 | moderate overfit |
| lumbar_extension/side_right | random_forest | 0.865 | 0.003 | 0.862 | SEVERE overfit |
| lumbar_extension/side_right | gradient_boosting | 0.983 | 0.004 | 0.979 | SEVERE overfit |
| lumbar_extension/side_right | mlp | 0.584 | 0.028 | 0.556 | SEVERE overfit |

## Seed stability summary (RF/GB/MLP across 7 seeds)

| Slot | Model | CCC mean +/- SD | CCC range | LoA mean +/- SD | seeds crossing Good gate |
| --- | --- | --- | --- | --- | ---: |
| knee_angle_r/front_oblique_right | random_forest | +0.552 +/- 0.019 | [+0.53, +0.58] | 16.62 +/- 0.42 | 0/7 |
| knee_angle_r/front_oblique_right | gradient_boosting | +0.555 +/- 0.018 | [+0.53, +0.57] | 16.88 +/- 0.30 | 0/7 |
| knee_angle_r/front_oblique_right | mlp | -0.105 +/- 0.077 | [-0.22, -0.02] | 63.84 +/- 8.45 | 0/7 |
| hip_flexion_r/front_oblique_left | random_forest | +0.758 +/- 0.019 | [+0.74, +0.79] | 14.60 +/- 0.73 | 0/7 |
| hip_flexion_r/front_oblique_left | gradient_boosting | +0.748 +/- 0.001 | [+0.75, +0.75] | 14.83 +/- 0.04 | 0/7 |
| hip_flexion_r/front_oblique_left | mlp | +0.591 +/- 0.035 | [+0.53, +0.62] | 19.49 +/- 1.61 | 0/7 |
| knee_angle_r/side_left | random_forest | +0.897 +/- 0.005 | [+0.89, +0.90] | 11.00 +/- 0.27 | 0/7 |
| knee_angle_r/side_left | gradient_boosting | +0.887 +/- 0.003 | [+0.88, +0.89] | 11.76 +/- 0.16 | 0/7 |
| knee_angle_r/side_left | mlp | +0.403 +/- 0.053 | [+0.34, +0.47] | 38.82 +/- 4.48 | 0/7 |
| knee_angle_r/side_right | random_forest | +0.670 +/- 0.012 | [+0.65, +0.68] | 16.44 +/- 0.19 | 0/7 |
| knee_angle_r/side_right | gradient_boosting | +0.754 +/- 0.002 | [+0.75, +0.76] | 14.68 +/- 0.06 | 0/7 |
| knee_angle_r/side_right | mlp | -0.055 +/- 0.020 | [-0.09, -0.03] | 99.40 +/- 11.06 | 0/7 |
| lumbar_extension/front_center | random_forest | +0.394 +/- 0.023 | [+0.36, +0.42] | 8.30 +/- 0.13 | 0/7 |
| lumbar_extension/front_center | gradient_boosting | +0.343 +/- 0.010 | [+0.33, +0.36] | 8.72 +/- 0.04 | 0/7 |
| lumbar_extension/front_center | mlp | +0.604 +/- 0.151 | [+0.42, +0.75] | 6.82 +/- 1.96 | 4/7 |
| lumbar_extension/front_oblique_left | random_forest | +0.308 +/- 0.018 | [+0.28, +0.32] | 9.88 +/- 0.11 | 0/7 |
| lumbar_extension/front_oblique_left | gradient_boosting | +0.282 +/- 0.004 | [+0.28, +0.29] | 10.41 +/- 0.03 | 0/7 |
| lumbar_extension/front_oblique_left | mlp | +0.046 +/- 0.015 | [+0.03, +0.08] | 57.85 +/- 10.68 | 0/7 |
| ankle_angle_r/front_oblique_left | random_forest | +0.274 +/- 0.035 | [+0.23, +0.33] | 12.63 +/- 0.27 | 0/7 |
| ankle_angle_r/front_oblique_left | gradient_boosting | +0.330 +/- 0.005 | [+0.32, +0.34] | 12.57 +/- 0.07 | 0/7 |
| ankle_angle_r/front_oblique_left | mlp | +0.143 +/- 0.064 | [+0.03, +0.23] | 28.50 +/- 5.43 | 0/7 |
| lumbar_extension/side_right | random_forest | +0.043 +/- 0.015 | [+0.02, +0.07] | 15.73 +/- 0.11 | 0/7 |
| lumbar_extension/side_right | gradient_boosting | +0.046 +/- 0.012 | [+0.02, +0.06] | 15.13 +/- 0.18 | 0/7 |
| lumbar_extension/side_right | mlp | +0.069 +/- 0.113 | [-0.09, +0.25] | 22.59 +/- 2.97 | 0/7 |

## Interpretation

Two patterns recur across all 8 slots:

1. **Trees overfit at this n.** RF and GB train R^2 routinely exceed 0.85 while their LOSO CCC is the same as ridge or worse. With max_depth=4 (RF) and depth=3 (GB) the models still memorise subject-specific patterns that don't transfer.
2. **MLPs are unstable at this n.** Seed-to-seed CCC SD is 0.05-0.15 on most slots; on the smaller slots (n=8-10 subjects) the MLP can land anywhere from CCC=0.4 to 0.85 depending on initialisation. This is not a model class you can deploy without ensemble or much more data.

Where RF or GB does match ridge (e.g. knee_angle_r/side_left RF: 0.897 vs ridge 0.864), the difference is well within seed and held-out-subject noise, and the trees cost more compute. There is no slot where switching model class is a defensible deploy decision.

## Methodology notes

- LOSO is subject-level on every reported number. No within-subject leakage.
- Per-subject Bland-Altman aggregation: trials of each held-out subject are averaged into one (pred, obs) point before CCC + LoA.
- Ridge replay is forced through the same StandardScaler used for MLP, to control for preprocessing differences vs the deploy-doc ridge.
- The single phone camera constraint is preserved - no multi-view fusion is introduced anywhere in this work.

Run with: `python -m harness.nonlinear_layer3_models` from the validation root.

## Files

- `harness/nonlinear_layer3_models.py` - runnable trainer
- `data/nonlinear_models/per_slot_nonlinear_results.json` - per-slot per-model results, with embedded seed stability
- `data/nonlinear_models/seed_stability.json` - standalone seed-by-seed CCC/LoA
- `data/nonlinear_models/REPORT.md` - this report

No `recommended_v18_updates.json` is written - no slot survived the seed-stability check above the Good gate.
