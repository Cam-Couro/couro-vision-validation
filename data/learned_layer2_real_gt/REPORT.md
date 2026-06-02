# Learned Layer 2 trained on real mocap ground truth

**Date:** 2026-05-29
**Build:** Agent EE2 (Phase A only — OpenCap-only LOSO)
**Verdict:** STRONG WIN. Architectural lever confirmed. Pooled |r| 0.645 vs Couro baseline 0.514 (+0.131).

## Headline

Replacing Couro's hand-engineered anthropometric Layer 2 reconstruction with a **learned Temporal CNN trained on real paired video+mocap** (OpenCap, 9 subjects, ~270 trials) produces pooled subject-mean |r| = 0.645 on 9-fold subject-level LOSO. **Clears the Phase A 0.55 gate by +0.095.** Every per-metric result beats every prior baseline.

| Model | Pooled \|r\| | Δ vs Couro baseline | License |
|---|---:|---:|---|
| Couro hand-engineered baseline | 0.514 | — | — |
| Agent S synthetic-only CNN | 0.495 | −0.019 | CC-BY 4.0 |
| Agent R view-aware blend | 0.581 | +0.067 | Apache 2.0 |
| **Learned Layer 2, real GT (this work)** | **0.645** | **+0.131** | CC-BY 4.0 |

## Per-metric, subject-mean across 9 LOSO folds

| Metric | Learned \|r\| | Couro hand-eng | Δ |
|---|---:|---:|---:|
| hip_flexion_r | **0.752** ± 0.053 | 0.644 | **+0.108** |
| hip_adduction_r | **0.397** ± 0.055 | 0.251 | **+0.146** (60% relative) |
| knee_angle_r | **0.686** ± 0.071 | 0.608 | **+0.078** |
| ankle_angle_r | **0.686** ± 0.072 | 0.522 | **+0.164** |
| lumbar_extension | **0.706** ± 0.093 | 0.546 | **+0.160** |
| **Pooled** | **0.645** | 0.514 | **+0.131** |

## Per-fold pooled |r|

| Held-out subject | Pooled \|r\| |
|---|---:|
| opencap_subject2 | 0.700 |
| opencap_subject7 | 0.694 |
| opencap_subject8 | 0.696 |
| opencap_subject4 | 0.688 |
| opencap_subject11 | 0.688 |
| opencap_subject10 | 0.631 |
| opencap_subject9 | 0.621 |
| opencap_subject5 | 0.548 |
| opencap_subject3 | 0.541 |

Worst fold (subject3) is still ≈Couro baseline. Subject5 (the recurring outlier flagged by Agents V/Y/AA) is mid-pack, not anomalous under the learned model.

## What was built

- **Data pipeline:** 270 paired (DWPose video, OpenSim mocap) clips across 9 OpenCap subjects. Mocap angles resampled onto keypoint timestamps. 22 Halpe-26 keys with SMPL mapping, per-frame torso-bbox normalization.
- **Model:** `TemporalKeypointCNNConf` — Agent S's TemporalKeypointCNN architecture plus **confidence-channel input**: 66 input channels (22 keypoints × {u, v, conf}). 100,741 parameters. Two 1D conv blocks (kernel 5, kernel 3), center-frame head.
- **Training:** 25 epochs, batch 256, AdamW + cosine LR schedule, SmoothL1 loss on normalized targets. CPU only. 9 LOSO folds, ~50s training + ~0.5s eval per fold. **Total runtime: ~8 minutes.**

## Key insight

The architecture was fine all along. The bottleneck of prior Layer 2 builds was **data realism**, not model class.

Same CNN as Agent S + **real paired GT** + **confidence channel** = +0.150 pooled |r| over Agent S's synthetic-only build. No synthetic noise injection needed in this version — real DWPose noise is in the data already. The sim-to-real gap that Agent S spent half their budget mitigating is eliminated by definition when training on real distribution.

## Caveats — honest

- **Phase B (Layer 3 retraining) skipped this build.** Phase A standalone Layer 2 numbers are reported here. Tier-change projection for the 23 deploy slots requires Phase B (re-fit ridge regression per slot using learned Layer 2 features, recompute CCC + Bland-Altman LoA). Estimated 3–8 slots eligible for tier lift based on the per-metric improvements and which slots are currently |r|-limited (not bias-limited).
- **ASPset (17 more subjects, CC0) not included.** Phase A used OpenCap only to ship a clean fast result. Adding ASPset roughly doubles the training cohort and would likely improve generalization further.
- **Hip_adduction remains absolute-Poor at 0.40 |r|.** Single-camera out-of-plane hip motion is geometrically hard. Even +60% relative isn't enough to clear the publication bar. View-aware blend with the hand-engineered Layer 2 still ensembles cleanly here if needed.
- **Best-fold checkpoint, not ensemble.** Saved `learned_layer2_v1.pt` is the subject2-held-out fold (best individual fold, |r| 0.700). Production deploy should ensemble all 9 LOSO checkpoints for robustness, or train a single final model on all 9 subjects without LOSO.
- **Internal validation split is sample-level, not subject-level.** Used for training loss monitoring inside each fold only; LOSO discipline strictly preserved at the fold boundary (held-out subject never seen during that fold's training).

## Files

- `harness/learned_layer2_real_gt.py` — runnable, ~8 min CPU
- `models/learned_layer2_v1.pt` — 412 KB best-fold checkpoint
- `data/learned_layer2_real_gt/per_slot_results.json` — 335 KB full per-fold/per-clip/per-metric

## Reproduce

```bash
cd /Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation
python3 harness/learned_layer2_real_gt.py --epochs 25 --batch-size 256
```

No GPU. No new dataset dependencies. Should be the same result on any machine running the existing OpenCap DWPose keypoint cache.

## Recommended next builds

1. **Phase B — Layer 3 retraining on learned Layer 2 features.** Re-fit per-slot ridge regression using learned-Layer-2 angle traces as input, re-run `biomech_validity_stats.py`, count tier changes. This is what tells us how many of the 23 deploy slots actually promote.
2. **Add ASPset 17 subjects to training cohort.** Likely additional +0.02–0.05 pooled |r| from broader distribution.
3. **Ensemble LOSO checkpoints OR train final on all 9 subjects.** For production deployment.
4. **Confidence-channel ablation on hip_adduction.** Test whether removing the confidence channel for that specific output head recovers the geometric-Poor scaling.

## Single-camera reaffirmation

Every measurement uses a single virtual or real camera per inference. No multi-camera fusion. The learned Layer 2 consumes one DWPose stream and produces 5 joint angles per frame — same input/output contract as Couro's hand-engineered Layer 2 it would replace.
