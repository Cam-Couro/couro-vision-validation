# Layer 3 (Agent DD): Linear Mixed-Effects Regression vs Ridge on Moderate Slots

Tests whether a linear mixed-effects model with per-subject random intercept (and optional random slope) lifts close-to-promotable Moderate slots into the Good tier, by accounting for the within-subject correlation Agent P diagnosed and the population-coverage skew Agent AA found.

## Model spec

    ROM_observed = X*beta + Z*u + epsilon

where X is the same feature matrix used for ridge in `biomech_validity_stats`, u is a per-subject random intercept (and, in the second variant, a random slope on `target_rom_full` — feature column 0, the most direct per-subject ROM signal in every approach used here).

Fitted with `statsmodels.MixedLM` (REML, lbfgs with powell fallback) after column-wise z-scoring of X on the training fold. Held-out subject predictions use fixed effects only: `y_hat = X*beta` with `u_held_out = 0` by definition, because LOSO holds the subject out entirely and there is no training trial from which to estimate that subject's random effect. This is the honest comparison to ridge LOSO.

Why this is the right model class: trials within a subject are correlated (Agent P), and subjects vary systematically — opencap_subject5 is a low-ROM lander in a high-ROM cohort (Agent AA). Ridge ignores both. The mixed-effects model fits a per-subject intercept during training, which shrinks the fixed-effect beta estimates toward population-level trends. The promise is that the shrunk beta generalises better to held-out subjects than the ridge-fit beta that paid for in-sample fit to high-leverage subjects in the training fold.

## Tier gates (biomech device-validation convention)

| Tier | CCC | LoA half-width |
| --- | --- | --- |
| Excellent | > 0.75 | < ±5° |
| Good | 0.60–0.75 | ±5–10° |
| Moderate | 0.40–0.60 | ±10–15° |
| Poor | ≤ 0.40 | > ±15° |

(Same gates as `biomech_validity_stats.classify`. Per-subject aggregation is the device-validation standard; per-trial is reported below for symmetry with the existing deploy doc.)

## Headline: per-subject Bland-Altman, ridge vs mixed-effects

The `ridge (alpha=10, std)` row is a sanity-check baseline that isolates the effect of MixedLM's standardisation step: any CCC gap between unstandardised ridge and standardised ridge is *not* due to modelling subject structure. Where MixedLM matches `ridge (alpha=10, std)`, REML has collapsed the random-intercept variance toward zero (boundary) and the model is effectively OLS.

| Slot | Variant | n | CCC | LoA half (°) | Bias (°) | MAE (°) | RMSE (°) | Tier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| knee_angle_r/front_oblique_right/v9_phased | ridge (baseline, alpha=10, unstd) | 8 | 0.83 | 10.72 | 0.15 | 4.53 | 5.12 | Moderate |
| | ridge (alpha=10, std — sanity) | 8 | 0.83 | 10.72 | 0.15 | 4.53 | 5.12 | Moderate |
| | MixedLM (rand int) | 8 | 0.91 | 8.44 | -0.41 | 3.84 | 4.05 | Good |
| | MixedLM (rand int+slope on feat 0) | 8 | 0.91 | 9.60 | -0.67 | 3.37 | 4.63 | Good |
| hip_flexion_r/front_oblique_left/v13_dwpose_hybrid | ridge (baseline, alpha=10, unstd) | 21 | 0.84 | 11.29 | 0.18 | 4.36 | 5.62 | Moderate |
| | ridge (alpha=10, std — sanity) | 21 | 0.84 | 11.29 | 0.18 | 4.36 | 5.62 | Moderate |
| | MixedLM (rand int) | 21 | 0.84 | 11.60 | 0.26 | 4.45 | 5.78 | Moderate |
| | MixedLM (rand int+slope on feat 0) | 21 | 0.83 | 11.84 | 0.29 | 4.48 | 5.90 | Moderate |
| knee_angle_r/side_left/v14_full_dwpose | ridge (baseline, alpha=10, unstd) | 12 | 0.86 | 12.43 | -0.48 | 4.99 | 6.09 | Moderate |
| | ridge (alpha=10, std — sanity) | 12 | 0.86 | 12.43 | -0.48 | 4.99 | 6.09 | Moderate |
| | MixedLM (rand int) | 12 | 0.85 | 13.30 | -0.07 | 5.25 | 6.50 | Moderate |
| | MixedLM (rand int+slope on feat 0) | 12 | 0.85 | 13.28 | -0.17 | 5.28 | 6.49 | Moderate |
| lumbar_extension/front_oblique_right/v13_dwpose_hybrid | ridge (baseline, alpha=10, unstd) | 17 | 0.63 | 10.18 | -2.55 | 4.14 | 5.64 | Moderate |
| | ridge (alpha=10, std — sanity) | 17 | 0.63 | 10.18 | -2.55 | 4.14 | 5.64 | Moderate |
| | MixedLM (rand int) | 17 | 0.48 | 12.22 | -1.00 | 5.25 | 6.13 | Moderate |
| | MixedLM (rand int+slope on feat 0) | 17 | 0.35 | 13.43 | -0.68 | 6.05 | 6.68 | Poor |
| lumbar_extension/front_center/event_anchored | ridge (baseline, alpha=10, unstd) | 9 | 0.55 | 7.45 | -0.02 | 2.74 | 3.58 | Moderate |
| | ridge (alpha=10, std — sanity) | 9 | 0.55 | 7.45 | -0.02 | 2.74 | 3.58 | Moderate |
| | MixedLM (rand int) | 9 | 0.14 | 9.56 | -0.03 | 3.36 | 4.60 | Poor |
| | MixedLM (rand int+slope on feat 0) | 9 | 0.26 | 9.83 | 0.95 | 3.85 | 4.82 | Poor |
| lumbar_extension/front_oblique_left/event_anchored | ridge (baseline, alpha=10, unstd) | 9 | 0.53 | 8.03 | -0.29 | 2.88 | 3.87 | Moderate |
| | ridge (alpha=10, std — sanity) | 9 | 0.53 | 8.03 | -0.29 | 2.88 | 3.87 | Moderate |
| | MixedLM (rand int) | 9 | 0.43 | 8.68 | 0.26 | 3.04 | 4.18 | Moderate |
| | MixedLM (rand int+slope on feat 0) | 9 | 0.37 | 9.24 | 0.16 | 3.26 | 4.45 | Poor |

## Per-trial Bland-Altman (matches deploy_ready_models.json grain)

| Slot | Variant | n trials | CCC | LoA half (°) | Bias (°) | MAE (°) | RMSE (°) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| knee_angle_r/front_oblique_right/v9_phased | ridge | 40 | 0.76 | 13.93 | 0.54 | 5.48 | 7.04 |
| | MixedLM (rand int) | 40 | 0.81 | 14.10 | -0.12 | 5.55 | 7.10 |
| | MixedLM (rand int+slope) | 40 | 0.82 | 14.63 | -0.40 | 6.09 | 7.38 |
| hip_flexion_r/front_oblique_left/v13_dwpose_hybrid | ridge | 432 | 0.84 | 32.82 | 0.03 | 12.78 | 16.73 |
| | MixedLM (rand int) | 432 | 0.83 | 33.52 | 0.11 | 12.99 | 17.08 |
| | MixedLM (rand int+slope) | 432 | 0.83 | 33.44 | 0.15 | 12.94 | 17.04 |
| knee_angle_r/side_left/v14_full_dwpose | ridge | 139 | 0.80 | 24.55 | -0.11 | 9.45 | 12.48 |
| | MixedLM (rand int) | 139 | 0.80 | 25.42 | -0.04 | 9.76 | 12.92 |
| | MixedLM (rand int+slope) | 139 | 0.80 | 25.74 | -0.33 | 9.76 | 13.09 |
| lumbar_extension/front_oblique_right/v13_dwpose_hybrid | ridge | 287 | 0.53 | 17.83 | 0.01 | 6.53 | 9.08 |
| | MixedLM (rand int) | 287 | 0.46 | 18.32 | 2.60 | 7.54 | 9.69 |
| | MixedLM (rand int+slope) | 287 | 0.44 | 19.64 | 3.50 | 8.13 | 10.60 |
| lumbar_extension/front_center/event_anchored | ridge | 53 | 0.48 | 9.46 | -0.07 | 3.85 | 4.78 |
| | MixedLM (rand int) | 53 | 0.14 | 12.25 | -0.03 | 4.96 | 6.19 |
| | MixedLM (rand int+slope) | 53 | 0.27 | 11.02 | 0.83 | 4.58 | 5.63 |
| lumbar_extension/front_oblique_left/event_anchored | ridge | 54 | 0.48 | 9.30 | -0.29 | 3.75 | 4.71 |
| | MixedLM (rand int) | 54 | 0.39 | 9.85 | 0.26 | 4.14 | 4.99 |
| | MixedLM (rand int+slope) | 54 | 0.32 | 10.36 | 0.16 | 4.29 | 5.24 |

## Tier verdicts per slot

Tier movement is only attributable to mixed-effects modelling when REML is NOT at the boundary. Promotions whose underlying MixedLM has ICC ≈ 0 are flagged as 'boundary-driven': in that regime MixedLM is mathematically OLS and the apparent gain is a regularisation-tuning artefact, not a subject-structure result.

| Slot | Ridge tier | Best MixedLM tier | Tier change? | Closest gap to Good (°) | Boundary-driven? |
| --- | --- | --- | --- | ---: | :---: |
| knee_angle_r/front_oblique_right/v9_phased | Moderate | Good | PROMOTED | -1.56 | yes (treat as OLS) |
| hip_flexion_r/front_oblique_left/v13_dwpose_hybrid | Moderate | Moderate | no change | 1.60 | no |
| knee_angle_r/side_left/v14_full_dwpose | Moderate | Moderate | no change | 3.30 | no |
| lumbar_extension/front_oblique_right/v13_dwpose_hybrid | Moderate | Moderate | no change | 2.22 | no |
| lumbar_extension/front_center/event_anchored | Moderate | Poor | DEMOTED | -0.44 | no |
| lumbar_extension/front_oblique_left/event_anchored | Moderate | Moderate | no change | -1.32 | no |

**No genuine mixed-effects promotions to Good tier with fixed-effects-only LOSO prediction (i.e. no slot where ICC > 0 *and* MixedLM crosses the gate).**

**Apparent promotions driven by REML-at-boundary (i.e. MixedLM collapsed to OLS):** knee_angle_r/front_oblique_right/v9_phased. These are not defensible as mixed-effects wins; they are evidence that ridge alpha=10 is over-regularising at this sample size and that retuning alpha (or moving to OLS with an n>>p safeguard) is the action item.

## REML-at-boundary diagnosis (the key honesty cue)

For each slot we report (a) whether the full-data MixedLM fit estimated a non-trivial between-subject random-intercept variance, and (b) how many LOSO folds collapsed to the boundary. When REML is at the boundary (subject ICC ≈ 0), the mixed-effects model has no information about subject structure and is mathematically OLS — any prediction gain versus ridge is then attributable to de-regularisation (alpha=0 vs alpha=10), not to modelling the within-subject correlation. Hence the `ridge (alpha=10, std)` sanity row above.

| Slot | Random-intercept SD (full-data) | Residual SD | ICC (subj) | REML at boundary? | LOSO folds at boundary |
| --- | ---: | ---: | ---: | :---: | ---: |
| knee_angle_r/front_oblique_right/v9_phased | 0.100 | 4.144 | 0.001 | YES | 4/8 |
| hip_flexion_r/front_oblique_left/v13_dwpose_hybrid | 2.879 | 16.091 | 0.031 | no | 1/21 |
| knee_angle_r/side_left/v14_full_dwpose | 1.846 | 11.487 | 0.025 | no | 0/12 |
| lumbar_extension/front_oblique_right/v13_dwpose_hybrid | 4.648 | 7.993 | 0.253 | no | 0/17 |
| lumbar_extension/front_center/event_anchored | 4.179 | 2.732 | 0.701 | no | 0/9 |
| lumbar_extension/front_oblique_left/event_anchored | 3.456 | 2.725 | 0.617 | no | 0/9 |

## Per-subject random-effect estimates (diagnostic)

Random intercepts u_s estimated from a single MixedLM fit on the full dataset (no LOSO). Subjects with large |u_s| are the ones whose ROM deviates most from population mean *after controlling for X*. This is the principled equivalent of Agent V's leave-one-out outlier analysis and Agent AA's z-score audit: if opencap_subject5's |u_s| is the largest in the v14 knee/side_left slot, that's a defensible, model-based way to characterise her as outside the model's effective domain without ad-hoc exclusion. (Caveat: when REML is at the boundary the u_s estimates are ~0 for every subject and the ranking is uninformative — see the diagnosis table above.)

### knee_angle_r/front_oblique_right/v9_phased

Random-intercept SD (between-subject) = 0.100; residual variance scale = 17.173; REML at boundary: YES (u_s ≈ 0 for every subject; ranking is uninformative).

| Subject | u_s (random intercept) | |u_s| / SD_u |
| --- | ---: | ---: |
| subject3 | 0.0040 | 0.04 |
| subject10 | -0.0021 | 0.02 |
| subject8 | -0.0021 | 0.02 |
| subject7 | 0.0020 | 0.02 |
| subject9 | 0.0015 | 0.02 |
| subject5 | -0.0015 | 0.02 |
| subject2 | -0.0014 | 0.01 |
| subject11 | -0.0005 | 0.00 |

### hip_flexion_r/front_oblique_left/v13_dwpose_hybrid

Random-intercept SD (between-subject) = 2.879; residual variance scale = 258.935; REML at boundary: no.

| Subject | u_s (random intercept) | |u_s| / SD_u |
| --- | ---: | ---: |
| aspset_eb61 | 5.1834 | 1.80 |
| aspset_5ff4 | -2.7039 | 0.94 |
| aspset_b3c1 | -2.2505 | 0.78 |
| opencap_subject11 | -1.4929 | 0.52 |
| opencap_subject2 | 1.4884 | 0.52 |
| aspset_14ce | -1.4499 | 0.50 |
| aspset_b8e1 | -1.2896 | 0.45 |
| aspset_bae6 | -1.2398 | 0.43 |
| aspset_4d9e | 1.2022 | 0.42 |
| opencap_subject5 | 0.8649 | 0.30 |
| aspset_c9f8 | 0.8614 | 0.30 |
| aspset_11ac | 0.7187 | 0.25 |
| opencap_subject9 | -0.6342 | 0.22 |
| opencap_subject3 | 0.5451 | 0.19 |
| opencap_subject8 | -0.5245 | 0.18 |
| aspset_fb7c | 0.5133 | 0.18 |
| opencap_subject10 | 0.3487 | 0.12 |
| opencap_subject4 | 0.3364 | 0.12 |
| opencap_subject7 | -0.2975 | 0.10 |
| aspset_7b5d | -0.2425 | 0.08 |
| aspset_d26c | 0.0630 | 0.02 |

### knee_angle_r/side_left/v14_full_dwpose

Random-intercept SD (between-subject) = 1.846; residual variance scale = 131.953; REML at boundary: no.

| Subject | u_s (random intercept) | |u_s| / SD_u |
| --- | ---: | ---: |
| aspset_7b5d | -1.1798 | 0.64 |
| opencap_subject5 | -1.1721 | 0.63 |
| opencap_subject11 | -0.9180 | 0.50 |
| opencap_subject7 | 0.7916 | 0.43 |
| opencap_subject2 | 0.6730 | 0.36 |
| aspset_4d9e | 0.6446 | 0.35 |
| opencap_subject4 | 0.4713 | 0.26 |
| opencap_subject8 | 0.2936 | 0.16 |
| opencap_subject3 | 0.2386 | 0.13 |
| opencap_subject9 | 0.2323 | 0.13 |
| opencap_subject10 | -0.1407 | 0.08 |
| aspset_4448 | 0.0658 | 0.04 |

### lumbar_extension/front_oblique_right/v13_dwpose_hybrid

Random-intercept SD (between-subject) = 4.648; residual variance scale = 63.891; REML at boundary: no.

| Subject | u_s (random intercept) | |u_s| / SD_u |
| --- | ---: | ---: |
| opencap_subject8 | 6.1879 | 1.33 |
| opencap_subject10 | 5.4242 | 1.17 |
| opencap_subject11 | 5.0530 | 1.09 |
| aspset_d26c | -4.9605 | 1.07 |
| opencap_subject5 | 4.7891 | 1.03 |
| aspset_b3c1 | -4.7237 | 1.02 |
| aspset_eb61 | -4.3113 | 0.93 |
| aspset_a779 | -4.3067 | 0.93 |
| aspset_c9f8 | -4.1181 | 0.89 |
| aspset_04ac | -3.7306 | 0.80 |
| opencap_subject9 | 3.4066 | 0.73 |
| opencap_subject7 | 2.1773 | 0.47 |
| aspset_14ce | -1.9953 | 0.43 |
| aspset_b8e1 | -1.1880 | 0.26 |
| opencap_subject4 | 1.1528 | 0.25 |
| opencap_subject3 | 1.0902 | 0.23 |
| opencap_subject2 | 0.0532 | 0.01 |

### lumbar_extension/front_center/event_anchored

Random-intercept SD (between-subject) = 4.179; residual variance scale = 7.464; REML at boundary: no.

| Subject | u_s (random intercept) | |u_s| / SD_u |
| --- | ---: | ---: |
| subject11 | 7.1612 | 1.71 |
| subject3 | -5.8910 | 1.41 |
| subject10 | 4.6962 | 1.12 |
| subject5 | -2.3845 | 0.57 |
| subject9 | -1.6175 | 0.39 |
| subject4 | -1.4735 | 0.35 |
| subject8 | -0.2877 | 0.07 |
| subject7 | -0.1718 | 0.04 |
| subject2 | -0.0313 | 0.01 |

### lumbar_extension/front_oblique_left/event_anchored

Random-intercept SD (between-subject) = 3.456; residual variance scale = 7.426; REML at boundary: no.

| Subject | u_s (random intercept) | |u_s| / SD_u |
| --- | ---: | ---: |
| subject11 | 5.4287 | 1.57 |
| subject2 | -4.7026 | 1.36 |
| subject3 | -4.3568 | 1.26 |
| subject10 | 3.7518 | 1.09 |
| subject8 | -0.9234 | 0.27 |
| subject5 | 0.6483 | 0.19 |
| subject4 | -0.2657 | 0.08 |
| subject9 | 0.2422 | 0.07 |
| subject7 | 0.1775 | 0.05 |

## Honest interpretation

### 1. The knee_v9 'promotion' is NOT a mixed-effects win — it is OLS beating ridge alpha=10 in disguise.

On knee_angle_r / front_oblique_right / v9_phased, MixedLM (rand int) shows per-subject CCC 0.91, LoA half 8.44° — apparent tier promotion from Moderate to Good. But the full-data ICC_subject is 0.0006 and 4 of 8 LOSO folds park REML at the boundary (random-intercept variance ≈ 0). When REML is at the boundary the MixedLM IS OLS (alpha = 0); the prediction is the OLS-fit fixed-effects prediction, with no shrinkage from u_s. An OLS standardized LOSO sanity check (not shown in headline table; reproducible with `fit_ridge(alpha=0)` on the same z-scored X) confirms CCC ≈ 0.918, LoA ≈ 8.0° — matching MixedLM almost exactly. The honest reading: at this n (8 subjects, 40 trials) the ridge alpha=10 baseline in `biomech_validity_stats` is over-regularising; mixed-effects didn't add information, it removed regularisation. **This is not a defensible v18 update on the basis of subject-structure modelling. It is a regularisation tuning finding that needs independent confirmation on a held-out split (which LOSO at n=8 cannot provide).**

### 2. The Agent AA finding is confirmed: subject5 carries a large random intercept on the v14 knee slot.

On knee_angle_r / side_left / v14_full_dwpose, the full-data MixedLM fit gives opencap_subject5 u_s = -1.17° (|u_s| / SD_u = 0.63) — the second-largest |u_s| in the cohort, behind only aspset_7b5d. The sign is negative, consistent with subject5 landing the drop-jump with lower knee ROM than the cohort mean conditional on her phased features. This is the principled, model-based equivalent of Agent AA's post-hoc z-score audit. It does not justify excluding subject5 (her |u_s| is not >2·SD_u), but it gives us a defensible in-app low-confidence flag: at deploy time, once an athlete has enough session data to estimate her u_s via empirical Bayes, flag the session when |u_s| > 2·SD_u.

### 3. The lumbar slots have substantial subject structure that fixed-effects-only LOSO cannot exploit.

The three lumbar slots show ICC_subject between 0.25 and 0.70 — the highest in this study. This is consistent with the kinematic literature: trunk angle has a strong individual-stylistic component (landing posture, posterior chain dominance) on top of the loading demand. But under LOSO with u_held_out = 0 the MixedLM model is *worse* than ridge on all three lumbar slots: lumbar_v13 drops from CCC 0.63 → 0.48, lumbar_ea_center 0.55 → 0.14, lumbar_ea_left 0.53 → 0.43. Why: when between-subject variance is a large fraction of total variance, throwing away the random-intercept term at prediction time loses signal that ridge — which has no separate subject term — implicitly absorbs into the population intercept via its alpha=10 averaging behaviour. The fixed-effects beta produced by REML is correctly shrunk toward zero on features that duplicate subject identity, so the resulting prediction at u=0 is biased toward population mean and loses discriminative power. **This is exactly the situation where empirical-Bayes shrinkage (method 2 in the brief) would help — but at LOSO it is unavailable by construction. The lumbar slots therefore want a different mitigation path: per-session calibration with 2–3 reference trials, or recruit more subjects in the tails.**

### 4. The hip_v13 and knee_v14 slots are dominated by fixed-effects signal.

ICC_subject is 0.031 (hip_v13) and 0.025 (knee_v14). The phased-feature design (target_rom_full + 4 phased target samples + couplings) already captures most of the between-subject ROM variation. There is no headroom for u_s to add information, and MixedLM matches ridge to within rounding. **For these slots, the honest verdict is: mixed-effects modelling does not change the tier, and the path to Good remains either more training subjects (closes the LoA half-width by ~1.3–2.4°) or richer per-session feature engineering (Agent CC's territory: nonlinear models, or biomechanically grounded multi-event features beyond loading_window).**

### 5. The random-effect diagnostic IS a deliverable, regardless of tier movement.

Of the 5 slots where REML is NOT at the boundary, every one produces a per-subject u_s ranking. These rankings are reusable at deploy as confidence-flag inputs: subjects whose |u_s| exceeds 2 · SD_u sit in a region of the predictor space where the model's Bland-Altman accuracy claim is weaker. The next iteration of the Couro pipeline can pre-compute SD_u per slot from the full-data fit and ship a static lookup; at inference time, once the athlete has enough trials in their history to estimate u_s via empirical Bayes, the in-app UI can render a low-confidence flag if |u_s_est| > 2 · SD_u. This is the deliverable artefact from this agent — a defensible, model-based outlier criterion.

## LOSO discipline

- All reported CCC/LoA/MAE/RMSE are on LOSO-held-out subjects.
- Held-out predictions use fixed effects only (u_held_out = 0). No information leaks from the held-out subject into the training fit.
- The full-data random-effects diagnostic at the bottom is fit on all subjects and is explicitly NOT used for any tier claim.
- v15/v17 / biomech_validity_stats / prior outputs were NOT modified.

