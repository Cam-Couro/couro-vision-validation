# v43 Selective Oracle + L3 Ensemble (Agent SS)

**Date:** 2026-06-04
**Build:** v42 reader pool + per-slot ensembles (ccc_weighted, equal_weight, top3). Per-slot fallback to v42 if ensemble CCC regresses by more than 0.02 or tier is no better.

**Verdict:** **12 validated Good-tier slots** (v42 was 12, delta +0). Tier 1 (CCC >= 0.79) count: **14** (v42 was 9, delta +0).

## 1. Readers with per-clip dumps

Loaded: v17, v23, v26, v31

## 2. Tier counts vs v42

| Tier | v42 | v43 | Delta |
| --- | ---: | ---: | ---: |
| Excellent | 0 | 0 | +0 |
| Good | 12 | 12 | +0 |
| Moderate | 5 | 5 | +0 |
| Poor | 6 | 6 | +0 |
| Tier 1 (CCC >= 0.79) | 14 | 14 | +0 |

## 3. Ensemble usage

- N slots using ensemble: **2**
- N slots falling back to v42 single reader: **21**
- Ensemble flavour usage: {'equal_weight': 1, 'top3': 1}

## 4. Reader distribution in v43

| Reader | Slots |
| --- | ---: |
| v17 | 4 |
| v31 | 3 |
| v23 | 3 |
| v26 | 2 |
| v30 | 2 |
| v24 | 2 |
| v27 | 1 |
| ensemble:equal_weight(v17+v23+v26+v31) | 1 |
| v20 | 1 |
| v38 | 1 |
| ensemble:top3(v17+v23+v26+v31) | 1 |
| v37 | 1 |
| v33 | 1 |

## 5. Category A target slots (LoA-limited)

| Slot | v42 tier | v42 CCC | v42 LoA | v43 tier | v43 CCC | v43 LoA | Flavour | Promoted? |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | :-: |
| knee_angle_r|front_oblique_left | Moderate | 0.928 | 10.77 | Moderate | 0.928 | 10.77 | - | no |
| knee_angle_r|side_right | Moderate | 0.813 | 12.92 | Moderate | 0.834 | 12.43 | top3 | YES |
| hip_flexion_r|front_oblique_left | Moderate | 0.843 | 11.29 | Moderate | 0.843 | 11.29 | - | no |
| hip_adduction_r|front_oblique_right | Moderate | 0.889 | 13.80 | Moderate | 0.889 | 13.80 | - | no |

## 6. Tier 2 supplementary slots

| Slot | v42 tier | v42 CCC | v42 LoA | v43 tier | v43 CCC | v43 LoA | Flavour | Promoted? |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | :-: |
| ankle_angle_r|front_oblique_right | Good | 0.751 | 8.03 | Good | 0.751 | 8.03 | - | no |
| hip_adduction_r|front_oblique_left | Good | 0.690 | 3.29 | Good | 0.690 | 3.29 | - | no |
| ankle_angle_r|side_right | Good | 0.644 | 9.46 | Good | 0.644 | 9.46 | - | no |

## 7. Promotions vs v42

**No tier promotions vs v42.**

## 8. Demotions vs v42

**No demotions vs v42 (per-slot fallback rule enforced).**

## 9. Coverage honesty

Readers without per-clip dumps (skipped from ensemble pool but still available via v42 fallback): see ``loaded_readers`` vs the full v42 pool (16 readers). The ensemble was built on the intersection of available dumps; per-slot fallback to v42 preserves the no-regression guarantee for slots that v42 picked from un-dumped readers (e.g. v20-only slot fallback).

