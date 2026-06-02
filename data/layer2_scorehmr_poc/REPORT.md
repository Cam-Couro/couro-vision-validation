# VPoser-prior SMPL Layer 2 POC

**Date:** 2026-05-28
**Verdict:** Promising hypothesis confirmation, doesn't beat Couro. Needs full 4D-Humans on GPU + license review before shipping.

## Model picked

**VPoser v1** (lithiumice/vposer_v1_0, HuggingFace, 2.7 MB). Used as a learned pose prior on top of Agent K's naive SMPL fit. **License:** MPI VPoser license — non-commercial / research only. Same restriction as SMPL, 4D-Humans, ScoreHMR.

Why not 4D-Humans / ScoreHMR directly: checkpoints are 2.7-3.3 GB, depend on PyTorch3D + Detectron2 which don't install on Mac CPU. VPoser isolates the exact variable we care about: does a learned pose prior fix depth-ambiguity?

## Headline

| Model | Pooled \|r\| | Δ vs Couro |
|---|---|---|
| Couro Layer 2 baseline | **0.514** | — |
| Naive SMPL (Agent K) | 0.358 | −0.156 |
| **VPoser-prior SMPL** | **0.450** | **−0.065** |

Pretrained prior closes ~58% of naive-fit gap (+0.091 over naive), still −0.065 below Couro overall.

## Per-metric

| Metric | Couro | Naive | VPoser | Δ vs naive | Δ vs Couro |
|---|---|---|---|---|---|
| hip_flexion_r | 0.644 | 0.420 | **0.708** | **+0.289** | **+0.064** |
| knee_angle_r | 0.608 | 0.443 | 0.568 | +0.125 | −0.040 |
| ankle_angle_r | 0.522 | 0.518 | 0.522 | +0.004 | 0.000 |
| lumbar_extension | 0.546 | 0.247 | 0.292 | +0.045 | −0.254 |
| hip_adduction_r | 0.251 | 0.162 | 0.157 | −0.005 | −0.093 |

**VPoser beats Couro on hip_flexion_r** (+0.064) — first single SMPL-pipeline win. Ties on ankle. Loses on Z-axis-broken metrics (lumbar, hip adduction) per Saad's softball audit.

Per-clip: VPoser beats Couro on 3/8 clips — all side/rear views where Couro's 2D signal is weakest and a learned prior provides most lift.

## Latency

- 15 ms/frame, 2.2 s/clip mean, 2.7 s max (Mac CPU, no MPS)
- ~1.6× naive fit; real-time plausible for typical sessions

## Verdict

Hypothesis confirmed: pretrained priors DO beat naive fits (+0.091 pooled). They do NOT beat Couro baseline overall (−0.065).

Best reading: half the naive deficit was Adam optimization artifact (VPoser fixes); other half is real depth ambiguity no single-camera method can fix on Z-axis metrics.

**Don't ship. Worth more investigation on GPU with full 4D-Humans / HMR2.0** — those add IMAGE features (not just 2D keypoints) which could push remaining gap. But:
- **LICENSE BLOCKER**: VPoser/SMPL/4D-Humans all non-commercial. Need permissively-licensed prior for production.
- Z-axis metrics likely hard ceiling for any single-camera method.

## Files

- `harness/scorehmr_layer2.py` — runnable script
- `per_clip_r.json` — raw per-clip results
- `models/vposer_v1/TR00_E096.pt` — 2.7 MB checkpoint
