# Combined OpenCap + ASPset Learned Layer 2

**Date:** 2026-05-30 (HH2 training) / 2026-06-02 (HH3 verification + writeup)
**Build:** Agents HH2 + HH3
**Verdict:** Real win on Layer 2 — pooled |r| 0.645 → 0.670 (+0.025). Cross-dataset generalization works. Layer 3 tier impact not yet measured.

## Headline

Doubling the training cohort from 9 OpenCap subjects (EE2) to 24 subjects (9 OpenCap + 15 ASPset) lifts OpenCap-held pooled |r| from **0.645 → 0.670 (+0.025)** and produces near-identical ASPset-held pooled |r| of **0.677** — strong cross-dataset generalization on a 15-subject independent test set.

| Model | OpenCap-held \|r\| | ASPset-held \|r\| | All 24 \|r\| |
|---|---:|---:|---:|
| Couro hand-engineered | 0.514 | — | — |
| EE2 OpenCap-only learned | **0.645** | — | — |
| **HH combined (this work)** | **0.670** (+0.025) | **0.677** | **0.674** |

## Per-metric on OpenCap-held subset (apples-to-apples with EE2)

| Metric | Combined | EE2 | Δ |
|---|---:|---:|---:|
| hip_flexion_r    | 0.792 | 0.752 | **+0.040** |
| hip_adduction_r  | 0.346 | 0.397 | **−0.051** |
| knee_angle_r     | 0.750 | 0.686 | **+0.064** |
| ankle_angle_r    | 0.736 | 0.686 | **+0.050** |
| lumbar_extension | 0.727 | 0.706 | **+0.021** |
| **Pooled**       | **0.670** | 0.645 | **+0.025** |

4 of 5 metrics improved. hip_adduction_r regressed — convention mismatch (see caveats).

## Per-metric on ASPset-held subset

| Metric | \|r\| | n folds |
|---|---:|---:|
| hip_flexion_r    | 0.818 | 15 |
| hip_adduction_r  | 0.597 | 15 |
| knee_angle_r     | 0.853 | 15 |
| ankle_angle_r    | — | 0 (no foot KPs in ASPset) |
| lumbar_extension | 0.439 | 15 |

hip_adduction at 0.597 ASPset vs 0.346 OpenCap is direct evidence the model is fitting ASPset's convention more tightly, explaining the OC regression.

## Per-fold pooled |r| (OpenCap-held)

| Subject | Combined | EE2 | Δ |
|---|---:|---:|---:|
| opencap_subject8  | 0.744 | 0.696 | +0.048 |
| opencap_subject4  | 0.727 | 0.688 | +0.039 |
| opencap_subject2  | 0.714 | 0.700 | +0.014 |
| opencap_subject11 | 0.687 | 0.688 | −0.001 |
| opencap_subject7  | 0.680 | 0.694 | −0.014 |
| opencap_subject9  | 0.679 | 0.621 | +0.058 |
| opencap_subject10 | 0.617 | 0.631 | −0.014 |
| opencap_subject5  | 0.610 | 0.548 | **+0.062** |
| opencap_subject3  | 0.574 | 0.541 | +0.033 |

**Worst-fold floor lifted:** subject5 (the recurring outlier from Agents V/Y/AA) +0.062; subject3 +0.033. The combined model is more robust to outlier subjects.

## Cohort

- **OpenCap:** 9 subjects, ~270 clips, all 5 angles. CC-BY 4.0.
- **ASPset:** 15 of 17 subjects ingested (2 had c3d/parse failures), ~1,400 clips, 4 angles (no ankle). CC0.
- **Total:** 24 subjects, ~1,670 clips.

Convention alignment ASPset → OpenCap:
- hip_flexion_r: identity
- hip_adduction_r: identity, clamp |x|>85° → NaN
- knee_angle_r: `OC = 180 − ASP`
- ankle_angle_r: NaN (no foot KPs; masked from loss)
- lumbar_extension: `OC = ASP − 180`

## Training setup

- Model: `TemporalKeypointCNNConf` (same as EE2), 66 input channels, 100,741 params, T=9 frames
- Loss: masked SmoothL1 (NaN target excluded per-element)
- AdamW lr=1e-3, wd=1e-4, cosine LR
- 25 epochs, batch 256, CPU only
- 24-fold LOSO, subject-level discipline preserved

## Answers to the task questions

1. **Does ASPset training data lift EE2's OpenCap-held |r|?** Yes, modestly. 0.645 → 0.670 (+0.025). 4/5 metrics improved.
2. **Does the combined model generalize to ASPset-held subjects?** Yes, strongly. Pooled |r| 0.677 on 15 independent subjects — essentially identical to OpenCap-held 0.670.
3. **Layer 3 retraining tier promotions?** Not computed — out of scope for the 2.5h budget. Recommendation: prioritize re-screening |r|-limited slots on knee_angle_r (+0.064) and ankle_angle_r (+0.050).

## Caveats — honest

1. **Net OpenCap-held win is modest (+0.025).** ASPset adds breadth, not depth. EE2's architecture + real GT remain the dominant gains.
2. **hip_adduction_r regressed on OC-held (−0.051).** Root cause: ASPset's asin-based frontal-plane definition is geometrically not the same as OpenSim's lumped-rotation hip_adduction. Future fix: drop hip_adduction_r from ASPset training, or use per-source target heads.
3. **Phase B (Layer 3 retrain + tier promotions) not done.** Gating analysis for any production switch.
4. **2 ASPset subjects skipped during cohort build** (24/26). Marginal.
5. **Best-fold checkpoint is opencap_subject8 LOSO** (|r| 0.744). For athletic deploy, the aspset_bae6 fold checkpoint (|r| 0.711) would be the preferred starting point — would require rerun to save.
6. **Ankle improvement (+0.050 OC-held) is representation transfer**, not added supervision (ASPset has no foot KPs). Shared trunk benefits untouched output heads. Stronger result than like-for-like supervised lift.

## Recommended next builds

1. Phase B Layer 3 retraining on combined Layer 2 features — gating analysis for deploy switch.
2. Drop hip_adduction_r from ASPset training, or use per-source target heads — test whether the OC regression reverses.
3. Pick up the 2 missed ASPset subjects.
4. Ensemble all 24 LOSO checkpoints, or train a single final on all 24 without LOSO.

## Single camera reaffirmation

All inference is single-camera. ASPset's 3 cameras per clip are treated as 3 independent training samples; the inference contract (1 DWPose stream → 5 angles/frame) is unchanged from EE2 and from the hand-engineered Layer 2 it would replace.

## Licensing (commercial-clean)

OpenCap CC-BY 4.0, ASPset CC0, DWPose Apache 2.0. All weights, code, and results are clean for Couro deployment.

## Files

- `harness/learned_layer2_combined.py` — runnable training + LOSO eval (955 lines)
- `models/learned_layer2_combined_v1.pt` — 412 KB best-fold checkpoint (opencap_subject8 held out, |r| 0.744)
- `data/learned_layer2_combined/per_slot_results.json` — 3.7 MB, all 24 folds, per-clip per-metric
- `data/learned_layer2_combined/_hh_analysis.json` — HH2's quick summary
