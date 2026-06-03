# v36 Selective Oracle with LoA-then-CCC Tie-Break

**Date:** 2026-06-02
**Build:** Agent PP, Lever 1 -- v35 reader pool with one selection rule fix: within the Moderate tier, when every top-tier candidate has CCC >= 0.79 (LoA-limited band), tie-break on **lowest LoA then highest CCC** instead of CCC then LoA. No new training.

**Verdict:** **11 Good slots** (v35 was 11). Tier 1 (CCC >= 0.79): **14** (v35 was 14).

## Tier counts vs v35

| Tier | v35 | v36 | Delta |
| --- | ---: | ---: | ---: |
| Excellent | 0 | 0 | +0 |
| Good | 11 | 11 | +0 |
| Moderate | 6 | 6 | +0 |
| Poor | 6 | 6 | +0 |
| Tier 1 (CCC >= 0.79) | 14 | 14 | +0 |

## Tie-break shifts vs v35

Total slots where the v36 LoA-first rule changed the picked reader: **1**.

| Slot | v35 reader | v35 CCC | v35 LoA/2 | v36 reader | v36 CCC | v36 LoA/2 | v35 -> v36 tier | rule |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| knee_angle_r|side_left | v31 | 0.899 | 10.15 | v29 | 0.895 | 10.02 | Moderate -> Moderate | loa_first |

## Category A target slots

These are the 5 LoA-limited borderlines that motivated the fix. Reminder: even with the tie-break change, none of these slots will be promoted to Good unless one of the candidate readers actually has LoA <= 10.0 deg. The fix only ensures we pick the candidate with the lowest LoA when ties exist.

| Slot | v35 reader | v35 LoA/2 | v36 reader | v36 LoA/2 | Promoted? |
| --- | --- | ---: | --- | ---: | --- |
| knee_angle_r|front_oblique_left | v31 | 10.77 | v31 | 10.77 | no |
| knee_angle_r|side_left | v31 | 10.15 | v29 | 10.02 | no |
| knee_angle_r|side_right | v27 | 12.92 | v27 | 12.92 | no |
| hip_flexion_r|front_oblique_left | v17 | 11.29 | v17 | 11.29 | no |
| hip_adduction_r|front_oblique_right | v30 | 13.80 | v30 | 13.80 | no |

**Category A promotions to Good: 0/5.**

## Reader distribution in v36

| Reader | Slots |
| --- | ---: |
| v17 | 4 |
| v18 | 0 |
| v20 | 1 |
| v23 | 4 |
| v24 | 2 |
| v26 | 2 |
| v27 | 2 |
| v29 | 2 |
| v30 | 2 |
| v31 | 3 |
| v33 | 1 |
| v34 | 0 |

## Honest caveats

- This fix is selection hygiene only. It will rarely cross a tier gate by itself -- it picks the candidate with the tightest LoA among ties in the LoA-limited band, but if no candidate is under +/-10 deg, no Moderate slot is promoted to Good.
- For slots with a single dominant reader, behavior is unchanged.
- The CCC >= 0.79 threshold for LoA-limited classification matches v32 Tier-1 / Agent OO Category A definition.
- Higher tiers (Good, Excellent) are unaffected by this change.
- LOSO discipline is unchanged: outer subject-level LOSO at L3.

