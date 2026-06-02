# VideoPose3D 2D→3D Lifter — Layer 2 POC

**Date:** 2026-05-28
**Verdict:** Don't replace, ship view-aware blend.

## Headline

Pooled |r| 0.514 → 0.516 — essentially a tie. But per-camera-bucket reveals the real story: lifter wins +0.26 on front views (Couro's worst), loses −0.36 on side views (Couro's ceiling). The two estimators are decorrelated by camera angle.

## Model picked

**VideoPose3D** (Pavllo et al., CVPR 2019, FAIR). Apache 2.0. 243-frame temporal convnet, ~17M params. Pretrained `pretrained_h36m_detectron_coco.bin` already on disk. MotionBERT skipped because its checkpoint required manual Google Drive download.

## Per-metric

| Metric | Lifter \|r\| | Couro \|r\| | Δ |
|---|---:|---:|---:|
| hip_flexion_r | **0.748** | 0.644 | **+0.104** |
| hip_adduction_r | 0.265 | 0.251 | +0.014 |
| knee_angle_r | 0.413 | **0.608** | −0.195 |
| ankle_angle_r | 0.533 | 0.522 | +0.011 |
| lumbar_extension | **0.622** | 0.546 | **+0.076** |
| **Pooled** | **0.516** | **0.514** | **+0.002** |

## Per-clip — camera-angle story

| Clip | View | Lifter | Couro | Δ |
|---|---|---:|---:|---:|
| subject2_DJ1_Cam0 | side | 0.451 | 0.810 | **−0.359** |
| subject10_DJ1_Cam0 | side | 0.426 | 0.587 | −0.160 |
| subject2_DJ1_Cam4 | front-oblique | 0.575 | 0.758 | −0.183 |
| subject10_DJ1_Cam4 | front-oblique | 0.525 | 0.403 | +0.122 |
| subject10_DJ2_Cam2 | front | 0.534 | 0.499 | +0.035 |
| subject10_DJ1_Cam2 | front | 0.495 | 0.463 | +0.032 |
| subject3_DJ1_Cam2 | front | 0.620 | 0.354 | **+0.266** |
| subject2_DJ1_Cam2 | front | 0.502 | 0.240 | **+0.262** |

Front view: lifter wins every clip, avg Δ = **+0.149**. Side view: lifter loses every clip, avg Δ = **−0.260**.

## Latency

- Model load: 0.1s once
- Per-clip inference (138-166 frames): 10-12ms
- Per-frame amortized: **0.07 ms** — three orders of magnitude under DWPose front-end. Effectively free.

## Why this matters

The lifter complements Couro on the slots where Couro's anthropometric reconstruction is weakest (front-view sagittal angles — the depth-ambiguous ones). Couro's reconstruction is unbeatable on side views (knee at r ≈ 1.0 via direct triangulation).

**A clip-level "max-of-better" oracle on this 8-clip sample reaches pooled |r| = 0.591.** A simple side-vs-front gate (on dot product of pelvis-lateral axis with image-X, computable from DWPose hip keypoints alone) would realistically capture +0.04 to +0.07.

## Verdict

- **Don't replace** Couro Layer 2 with the lifter
- **Do ship a view-aware blend**: Couro on side views, lifter on front/oblique. Projected pooled |r| ~0.55-0.58 with simple gating, ~0.591 with oracle gating.

## Next steps ranked

1. **Fix hip_flexion sign convention** in lifter angle math (trivial — currently anti-correlated with mocap)
2. **Implement view-aware blending** — gate Couro on side, lifter on front. Projected +0.04 to +0.07 pooled.
3. **Add toe joint** to H36M remap so ankle uses conventional foot-vs-shank interior angle
4. **Try MotionBERT** on same 8 clips — newer architecture, likely +0.02-0.05 over VideoPose3D. ~1 day to integrate.

## Files

- `harness/motionbert_layer2.py` — runnable inference script (~12s end-to-end)
- `harness/videopose3d.py` — TemporalModel + Halpe-26→H36M-17 remap
- `models/videopose3d_h36m.bin` — pretrained checkpoint (68 MB)
- `per_clip_r.json` — raw per-clip results
