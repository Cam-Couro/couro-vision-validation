# Attacking the 9 failing slots — diagnosis before spend

**Verified from `data/biomech_validity_stats/per_slot_validity.json` and
`data/per_slot_predictions/per_slot_predictions_v44.json`, recomputed in Python. 23 slots, 14 pass
(CCC > 0.60 AND LoA half < ±10°), 9 fail.**

The campaign's closing verdict was "the remaining stuck slots are DATA-limited, not
model-limited — the next gains require cohort expansion." **Three findings below suggest cohort
expansion alone would waste money, and one of them changes what the numbers mean.**

---

## Finding 1 — Cohort size does NOT separate passing from failing slots

| | n slots | median subjects | median trials | median LoA half |
|---|---|---|---|---|
| **Passing** | 14 | 10 | 69 | 10.4° |
| **Failing** | 9 | 10 | 83 | 15.6° |

**Failing slots have MORE trials than passing ones** (83 vs 69), at identical subject counts. Among
the 13 thinnest slots (< 100 trials), **8 of 13 pass**: `hip_adduction_r/front_oblique_left` clears
the bar at ±6.5° on 48 trials, while `hip_adduction_r/side_right` fails at ±19.6° on 77.

So volume is not the discriminator. Buying more of the same data does not predict a fix. This is
the single strongest argument against "expand the cohort" as the headline action.

## Finding 2 — Camera geometry is the discriminator, and the pattern is backwards

For **right**-limb joints, the **left**-side camera beats the right-side camera every time:

| Metric | side_left | side_right | Left better by |
|---|---|---|---|
| `hip_flexion_r` | CCC 0.86 / ±9.2° | CCC 0.58 / ±16.9° | **7.7°** |
| `hip_adduction_r` | CCC 0.95 / ±9.3° | CCC 0.28 / ±15.3° | **6.0°** |
| `knee_angle_r` | CCC 0.90 / ±9.8° | CCC 0.82 / ±13.7° | **3.9°** |
| `ankle_angle_r` | CCC 0.64 / ±9.2° | CCC 0.64 / ±9.5° | 0.3° |

A right-side camera should see a right limb *better* — the near limb is unoccluded. Getting the
opposite, consistently, across three independent joints, is the signature of a **systematic
left/right handling defect**, not sampling noise. `side_right` holds 3 of the 9 failures and
`front_oblique_left` another 3; together that is **6 of 9 failures in two view families**.

More cohort data cannot fix a view-handling defect. It would just measure it more precisely.

## Finding 3 — The decisive one: one cross-dataset subject distorts most slots

The validation pool mixes two public datasets, `opencap_*` and `aspset_*`. Recomputing each slot
with and without the aspset subjects:

| Slot | All subjects | OpenCap only | Verdict |
|---|---|---|---|
| `hip_adduction_r/side_right` | CCC +0.18 / ±17.3° | CCC +0.17 / **±9.2°** | **would clear on LoA** |
| `hip_adduction_r/front_center` | CCC +0.90 / ±11.9° | CCC +0.08 / **±3.9°** | LoA clears, CCC collapses |
| `knee_angle_r/side_right` | CCC **−0.69** / ±37.4° | CCC −0.45 / ±23.3° | structurally broken either way |
| `lumbar_extension/side_right` | CCC −0.61 / ±28.0° | CCC +0.03 / ±11.6° | **passes in deployed table** |

A single subject, `aspset_04ac`, carries a **+48.3° error** on `knee_angle_r/side_right` — five
times any other subject's error, and it alone swings that slot's LoA from ±23° to ±37°.

**This cuts in two directions and I will not report only the flattering one:**

- Several "failures" are partly an outlier-and-pooling artifact, not a capability limit.
- But removing aspset makes several **CCCs collapse** (`hip_adduction_r/front_center` 0.90 → 0.08).
  That is the clearer signal: those high CCCs were being produced by **between-dataset spread**, not
  by tracking individual athletes well. CCC rewards wide range; pooling two cohorts manufactures it.

**So the honest reading is that some of our passing CCCs are flattered by cohort pooling.** That is
a finding about the 14 passing slots, not just the 9 failing ones, and it needs to be resolved
before any of these numbers go in front of an investor.

## Finding 4 — The v44 reader is not the deployed reader

`knee_angle_r/side_right` is CCC **−0.69** under v44, but the deployed table reports **+0.82** via
`v39 (v17 + residual calibration)`. Both are true of different readers. Any per-slot remediation
must name the deployed reader, or it will optimise code that never runs — the same
source-versus-deployed gap LeSean's Finding 1 flagged.

## What recalibration alone buys (measured, not estimated)

Applying scale-and-offset recalibration to each failing slot's per-subject predictions:

- **1 of 9 clears the bar**: `hip_adduction_r/front_center`, CCC 0.90 → 0.95, LoA ±11.9° → **±9.5°**
- 8 of 9 do not improve, and several worsen — rescaling widens LoA when the error is scatter, not bias

Confirmed against bias decomposition: in all 9 failing slots |bias| is small (0.02°–3.6°) relative
to 1.96·SD, so **calibration is nearly exhausted**. That part of the original verdict holds.

---

## Recommended attack order

1. **Resolve the pooling question first — zero cost, one script.** Recompute all 23 slots
   per-dataset and within-dataset. If the 14 passing slots hold up within OpenCap alone, the
   validity table stands. If their CCCs collapse the way `front_center` does, we have been reporting
   between-cohort spread as tracking accuracy. **This gates everything else, including anything
   shown to an investor.**
2. **Chase the left/right view defect — a bug hunt, not a data purchase.** A right camera reading a
   right limb worse than a left camera, across three joints, is almost certainly a handling defect.
   Ceiling if fixed: ~6 of 9 failures sit in `side_right` and `front_oblique_left`.
3. **Set an outlier policy before adding subjects.** `aspset_04ac` moves a slot by 14° of LoA on its
   own. A pre-registered exclusion rule — stated before seeing results, not after — is required so
   exclusions are not selection-after-the-fact. LeSean's point about the calibration participant was
   the same concern.
4. **Pin the deployed reader per slot** so remediation targets what actually runs.
5. **Only then expand the cohort**, targeted at the slots that survive 1–3, with per-dataset power
   stated in advance.

**Cost note:** steps 1–4 are analysis on data already on disk. No GPU, no capture, no spend. Step 5
is the only one that costs money, and it should be the last, not the first.

## Caveats

- `n_subjects` is 9–22 per slot; with correlated camera and limb observations the effective n is
  lower, so every CCC here carries wide uncertainty.
- I recomputed CCC and LoA with standard definitions (Lin's CCC; LoA half = 1.96·SD of
  differences). My reaggregation of the deployed table matched its published 14/23 and Tier-1 10,
  which is the cross-check that the method agrees.
- I did not fetch the pinned SHAs; this is the local working copy of `couro-vision-validation`.
- Finding 3's direction of effect is robust (it reverses sign across slots), but the exact
  per-slot numbers depend on which readers' dumps are used, and only `v44` has per-subject records
  on disk here.

Authors.Hermes
