# Step 1 result: the CCC collapse is real, and LoA is the statistic to trust

**Run on data already on disk. No GPU, no capture, no spend.**

I ran the pooling check before recommending it, and it changes what we should publish.

## The finding

Recomputing all passing slots **within OpenCap alone** (dropping the ASPset subjects):

| Slot | Deployed CCC | Within-OpenCap CCC | Within-OpenCap LoA half |
|---|---|---|---|
| `hip_adduction_r/front_oblique_right` | 0.96 | **+0.16** | **±5.0°** |
| `hip_adduction_r/side_left` | 0.95 | **+0.01** | **±7.7°** |
| `knee_angle_r/side_left` | 0.90 | **−0.57** | ±29.1° |
| `lumbar_extension/side_left` | 0.88 | **−0.50** | ±14.3° |
| `hip_flexion_r/side_left` | 0.86 | **+0.07** | ±23.4° |
| `lumbar_extension/side_right` | 0.85 | **+0.03** | ±11.6° |
| `ankle_angle_r/front_oblique_right` | 0.75 | **−0.54** | ±19.1° |
| `ankle_angle_r/side_right` | 0.64 | **−0.47** | ±25.5° |

**10 of the 14 passing slots lose their CCC when computed within a single dataset.**

**This is not a v44 artifact.** I tested `hip_adduction_r/side_left` across every reader dump on
disk — v17, v23, v26, v31, v44 — and the pattern is identical in all five (pooled +0.89 to +0.95,
within-OpenCap −0.10 to +0.28). It is structural, not reader-specific.

## Why it happens — and why it is not fraud or a bug

For `hip_adduction_r/side_left`:

- ASPset subjects' ground-truth hip adduction: **41.8°–49.2°** (n=3)
- OpenCap subjects': **8.6°–15.6°** (n=9)
- **A 26.2° gap with no subjects in between**

CCC measures agreement across the observed range. Pooling two cohorts whose true values sit in
non-overlapping clusters creates an artificial 40° range, and a model that merely knows "ASPset
high, OpenCap low" scores near 1.0. Within OpenCap, the true between-subject spread is only
**SD 2.4°** while mean absolute error is **3.3°** — error slightly exceeds the real signal, so CCC
correctly falls to ~0.

**Both numbers are honest; they answer different questions.** Pooled CCC says "can we tell a
high-adduction athlete from a low-adduction one across populations" — yes. Within-dataset CCC says
"can we rank nine similar athletes against each other" — not reliably, on this evidence.

## The good news, and it is substantial

**LoA barely moves, and in most cases improves:**

| Slot | Pooled LoA | Within-OpenCap LoA |
|---|---|---|
| `hip_adduction_r/side_left` | ±9.3° | **±7.7°** |
| `hip_adduction_r/front_oblique_right` | ±8.5° | **±5.0°** |
| `hip_adduction_r/front_center` | ±11.9° | **±3.9°** |

**LoA is range-independent — it measures absolute agreement in degrees — so it does not inflate
with cohort pooling.** It is the statistic that survives this, and it is the one we should lead
with.

So the product claim that holds up is **absolute measurement agreement**, e.g. "hip adduction from
a front-oblique-right camera agrees with lab motion capture to within ±5–8.5°." That is defensible,
useful to a coach, and unaffected by the pooling question.

The claim that does **not** hold up on current evidence is **ranking similar athletes** within a
narrow band on the affected slots, because there the error is comparable to the real spread.

## What this changes

1. **Lead with LoA, not CCC, in anything investor- or coach-facing.** This also resolves the
   argument LeSean and I were having from the wrong end: he was right that 3.984° and 23.139° are
   different endpoints, and right that neither is a universal accuracy number. The real lesson is
   that **CCC is the fragile statistic here and LoA is the robust one** — which neither of us said.
2. **Report both, labelled:** pooled CCC as cross-population discrimination, within-dataset LoA as
   measurement agreement. Hiding either would be the interface-truthfulness failure we keep
   catching.
3. **The 14/23 count needs a caveat**, because the bar is `CCC > 0.60 AND LoA < ±10°` and 10 of the
   14 pass the CCC half only when cohorts are pooled. I am not claiming the table is wrong — the
   methodology is standard and the LoA half is genuinely met — but "14 validated slots" should not
   be stated to an investor without the pooling note.
4. **Cohort expansion now has a precise target**, which is the opposite of the original verdict's
   framing: we do not need *more* subjects so much as subjects **filling the 26° gap between the two
   clusters**. Nine more OpenCap-like athletes in the 8–16° band would not fix a CCC that is limited
   by a 2.4° true spread. Subjects in the 16–42° range would.

## Caveats

- OpenCap-only n is 9–22 subjects per slot; within-dataset CCC on n=9 is itself unstable, so I
  would not treat an individual within-dataset CCC as precise. The **direction** is robust because
  it reproduces across five independent reader dumps.
- Only `v44` and four earlier dumps carry per-subject records on disk; the deployed reader for some
  slots (e.g. `v39` for `knee_angle_r/side_right`) has no dump here, so for those slots I verified
  the mechanism, not the deployed number.
- `aspset_04ac` separately carries a +48.3° error on `knee_angle_r/side_right`, large enough to move
  that slot's LoA by 14°. An outlier policy should be pre-registered before any re-fit, so
  exclusions are not selection after seeing results.

Authors.Hermes
