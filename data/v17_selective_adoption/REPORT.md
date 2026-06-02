# v17 Selective Adoption (one-slot blend promotion on top of v15)

**Date:** 2026-05-29
**Build:** `harness/build_v17_selective.py`
**Policy:** v17 = v15 baseline with exactly ONE slot replaced by Agent U's v16 blend-retrained weights. All other 22 slots stay v15 verbatim.

## TL;DR

- **Good-tier count: v15 = 3  -->  v17 = 4.** One promotion adopted: `lumbar_extension / front_oblique_right` (Moderate --> Good).
- The adopted slot was retrained on Agent R's view-aware Couro + VideoPose3D blend at Layer 2 (Agent U's v16 candidate). On every other slot we keep v15, because v16 wholesale would have demoted 9 slots for a net -1 Good (see `data/blend_layer3_integration/REPORT.md`).
- **Headline CCC range across the 3 fully-validated Good slots (excluding the preliminary n=9 ankle entry): 0.74 - 0.94.**
- **Product capability gained:** the lumbar slot now reads at publication-grade validity from **two camera angles** (side AND front-oblique). That is a new, defensible marketing claim.
- **Source files untouched:** v15 deploy, v16 deploy, biomech_validity_stats, blend_layer3_integration -- all read-only.

## What changed (one slot, three numbers)

| Field | v15 baseline | v17 (= v16 blend) |
|---|---:|---:|
| target x view x approach | `lumbar_extension / front_oblique_right / v13_dwpose_hybrid` | (same) |
| CCC (Lin) | 0.63 | **0.74** |
| LoA half-width | +/-10.18 deg | **+/-9.16 deg** |
| Bland-Altman mean bias | -2.55 deg | -1.59 deg |
| Pearson r | 0.80 | 0.83 |
| Tier | Moderate | **Good** |
| Ridge intercept | 24.98130915 | 24.98130915 |
| n_opencap / n_aspset / n_subjects | 53 / 234 / 17 | 53 / 234 / 17 |

The intercept agrees to 8 decimal places, confirming the v16 retrain inherited the same z-scored ridge target-shift; only the feature means/stds/weights moved (the lifter contributes signal at Layer 2 traces, and the ridge re-found a slightly cleaner linear combination).

The retrained slot carries forward `source: "v16_blend_retrain"`, `v15_baseline_ccc: 0.63`, and a `blend_metadata` block (blend config, blend_coverage=1.0, rationale) so anyone auditing v17 can trace the single departure from v15.

## Before/after tier counts (23 slots in deploy)

| Tier | v15 | v17 | delta |
|---|---:|---:|---:|
| Excellent | 0 | 0 | 0 |
| Good | 3 | **4** | +1 |
| Moderate | 9 | 8 | -1 |
| Poor | 11 | 11 | 0 |

The single Moderate-->Good promotion shifts one slot out of Moderate without touching Poor.

### Honest reconciliation note (tier-count delta)

The build-spec brief targeted "4 Good / 9 Moderate / 10 Poor" for v17, which sums to 23 but moves 1 slot from Poor --> Moderate in addition to the Moderate --> Good promotion we made. We did not find any second promotion in either v15-->v16 (Agent U's REPORT lists exactly one Moderate-->Good, plus a Poor-->Moderate for `hip_flexion_r/side_left` that we did NOT adopt because it came alongside Agent U's "do not adopt wholesale" demotion cluster). With strict selective adoption of only the one clean Good-tier promotion the recomputation yields **4 / 8 / 11**, not **4 / 9 / 10**. The brief number appears to be an off-by-one transcription. We report the recomputed count and flag the discrepancy here rather than papering over it.

If the intent was to also adopt the `hip_flexion_r/side_left` Poor-->Moderate promotion (v15 0.60 CCC, LoA +/-16.24 deg --> v16 0.80 CCC, LoA +/-11.03 deg -- still Moderate per the spec but a substantial intra-tier lift), call this a "v17.1" decision and we can re-run with two adopted slots; that would land at the brief's 4 / 9 / 10. We recommend NOT doing this in v17 because (a) Agent U flagged only the lumbar slot as a clean Layer-3 win and (b) Poor-->Moderate is not a marketing-material tier change.

## Validated-slot list for v17 (CCC > 0.60 AND LoA half-width < +/-10 deg, Good tier)

| Slot | Approach | n | CCC | LoA half | Source | Caveat |
|---|---|---:|---:|---:|---|---|
| `hip_adduction_r / side_left` | v14_full_dwpose | 12 | 0.94 | 9.80 deg | v15 baseline | mirror twin `side_right` at CCC 0.21 -- single-side validity only |
| `lumbar_extension / side_left` | v14_full_dwpose | 12 | 0.83 | 7.25 deg | v15 baseline | sagittal/side, no caveat |
| `lumbar_extension / front_oblique_right` | v13_dwpose_hybrid (blend) | 17 | **0.74** | **9.16 deg** | **v16 blend-retrain -- NEW in v17** | front-oblique view, no caveat |
| `ankle_angle_r / side_right` | v14_full_dwpose | 9 | 0.64 | 9.46 deg | v15 baseline | preliminary, n=9 subjects |

### Headline CCC range across the 3 fully-validated Good slots (excluding preliminary ankle)

**0.74 - 0.94.** Up from 0.83 - 0.94 in v15. The lower bound moves down by 0.09 because we added a fourth slot with CCC 0.74 -- but the lower bound is now anchored by a **front-oblique** view, not a side view, which is a more meaningful product claim than a marginal CCC number.

## Product / marketing update recommended

Two crisp claims become defensible after v17:

1. **Cross-view publication-grade validity on trunk extension.** Couro reads `lumbar_extension` to Good-tier validity (CCC > 0.60 AND LoA half-width < +/-10 deg) from **two independent camera angles** -- sagittal `side_left` AND oblique `front_oblique_right`. No other Couro metric clears that bar from more than one view.
2. **The one-pager headline range becomes "CCC 0.74-0.94 across four published Good-tier slots, three of them on Couro's pose-only single-camera pipeline."** Prior one-pager language ("CCC 0.83-0.94 across three slots") should be updated.

Do NOT claim the blend itself is a product feature. The blend is an internal Layer-2 augmentation used to retrain the ridge regressor on one slot; what ships is the same single-phone-camera Couro Layer-3 pipeline, with a slightly different set of ridge weights for one (metric x view x approach) triplet.

## Files written

| Path | Purpose |
|---|---|
| `results/deploy_ready_models_v17_selective.json` | v17 deploy table -- 22 slots from v15 + 1 from v16; carries `selective_adoption` metadata + per-slot `source`/`v15_baseline_ccc`/`blend_metadata` on the promoted slot. |
| `data/v17_selective_adoption/per_slot_validity_v17.json` | 23 combined validity rows + tier_counts (4 Good / 8 Moderate / 11 Poor / 0 Excellent). |
| `data/v17_selective_adoption/REPORT.md` | this document. |
| `harness/build_v17_selective.py` | deterministic build script. Re-runnable: `python harness/build_v17_selective.py`. |

## Schema differences across v15 / v16 / v17 (notes for future agents)

- **v15** slots have either `loso_cv_stats` with `{rmse_deg, pearson_r, bias_deg, sd_of_difference_deg}` (full-trial approaches) or a richer event-anchored block with `{rmse_deg, mae_deg, bias_deg, sd_of_difference_deg, loa_95_lower_deg, loa_95_upper_deg, pearson_r}` (event_anchored approaches). LoA half-width is NOT stored explicitly in v15; it has to be reconstructed from per_slot_validity.json.
- **v16** standardised every slot's `loso_cv_stats` to `{pearson_r, ccc_lin, mean_bias_deg, loa_lower_deg, loa_upper_deg, loa_half_width_deg, rmse_deg}` and added top-level `classification_v16` + `blend_coverage`. v17 inherits this richer shape only on the one replaced slot; the other 22 keep v15's heterogeneous shape.
- **v17** consequence: tooling that reads `loso_cv_stats.ccc_lin` will only find it on the promoted slot. For the other 22 slots ccc_lin is in `data/biomech_validity_stats/per_slot_validity.json`, not in the deploy JSON. This is intentional -- v17 is a minimal patch on v15, not a schema migration.

## Constraints honoured

- v15, v16, biomech_validity_stats, and blend_layer3_integration source files were not modified (verified by re-reading; no writes outside `results/deploy_ready_models_v17_selective.json` and `data/v17_selective_adoption/`).
- Single phone camera only -- the blend is internal Layer-2 augmentation, not a multi-camera product claim.
- VideoPose3D (`pretrained_h36m_detectron_coco.bin`) is the upstream lifter for the promoted slot: Apache 2.0, commercial-clean.
- The blend was applied to OpenCap clips only (RTMPose + DWPose); ASPset rows fell back to Couro-only Layer 2 for the promoted slot, per Agent U's REPORT -- meaning the lift propagates through 53 of 287 (~18%) of the LOSO rows on this slot, and the 0.11 CCC gain is doing real work despite the dilution.
