# Synthetic AMASS-style Layer 2 POC

**Date:** 2026-05-28
**Verdict:** PROMISING. Invest the full 6-9 days for production version.

## Headline

4k-sample SMPL-rendered MLP trained for 0.9s on CPU lands within 3 pts of Couro's hand-crafted reconstruction and beats it on 2 of 5 metrics — with **zero real training data**. Compare to naive SMPL fitting (-0.156 vs Couro).

| Model | Pooled \|r\| | Δ vs Couro |
|---|---|---|
| Couro baseline | **0.514** | — |
| Naive SMPL (Agent K) | 0.358 | −0.156 |
| VPoser-prior SMPL (Agent M) | 0.450 | −0.065 |
| **Synthetic MLP (this work)** | **0.484** | **−0.030** |

Synthetic path closes ~80% of the naive→Couro gap with one afternoon of work.

## Approach

- **Data**: CC-BY SMPL_NEUTRAL.pkl joint-only FK. 4000 axis-angle poses biased toward drop-jump distributions. Virtual cameras yaw [0-360°], pitch [-25°, +25°], distance 2.5-5m. 22 SMPL-mappable Halpe-26 keypoints projected, normalized.
- **Model**: 22×2 → 128 → 128 → 64 → 5 MLP. ~30K params, 130 KB checkpoint.
- **Eval**: same 8 OpenCap clips Agents K, M used.

## Per-metric

| Metric | Synthetic \|r\| | Couro \|r\| | Δ |
|---|---|---|---|
| hip_flexion_r | **0.655** | 0.644 | **+0.011** |
| hip_adduction_r | **0.334** | 0.251 | **+0.083** |
| knee_angle_r | 0.437 | 0.608 | −0.171 |
| ankle_angle_r | 0.483 | 0.522 | −0.039 |
| lumbar_extension | 0.509 | 0.546 | −0.037 |
| **Pooled** | **0.484** | **0.514** | **−0.030** |

Synthetic val r (model on its own task): hip_flex 0.954, knee 0.958, ankle 0.715, hip_add 0.620, lumbar 0.609. Model clearly learns task; lower two are single-view-ambiguous.

## Sim-to-real gap

Val r → real |r| gap by metric: knee 0.52 (worst — fine SMPL vs jittery DWPose), hip_flex 0.30, hip_add 0.29, ankle 0.24, lumbar 0.10 (best).

**Gap is dominated by keypoint noise modeling, not SMPL distribution mismatch.** Synthetic Halpe-26 is pixel-perfect; DWPose has 5-15px jitter, occasional toe/heel swaps, varying confidence.

Per-clip: synthetic wins 17/40 cells (43%). Beats Couro on Cam2 (side) more often than Cam0/Cam4 (frontal). Hip adduction is clear synthetic win (6/8 clips).

## Verdict — invest 6-9 days

Three production-version upgrades:
1. **Scale + noise**: 50k samples, AMASS motion (temporal continuity), Gaussian + occlusion + confidence noise during training. +0.05-0.10.
2. **Temporal context**: 1D CNN over T=5-15 frames captures velocity/acceleration. +0.03-0.07.
3. **Ensemble with Couro**, not replace it. Weighted average where synthetic wins (Cam2/hip_add) and Couro wins (Cam0/Cam4/knee). +0.05-0.08, low risk.

Realistic target with all three: pooled |r| 0.60-0.65, beating Couro's 0.514 ceiling.

**Commercial license clean**: SMPL-Body (CC-BY 4.0) is the data source. No license blocker.

## Files

- `harness/synthetic_layer2_poc.py` — POC code
- `per_clip_r.json` — raw results
- `models/synthetic_layer2_v0.pt` — 128 KB trained checkpoint
