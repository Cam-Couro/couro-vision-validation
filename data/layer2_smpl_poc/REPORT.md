# SMPL Single-Camera Body Model POC — Layer 2 Impact

**Date:** 2026-05-28
**Verdict:** SHELVE — naive SMPL fitting drops Layer 2 |r| from 0.514 → 0.358 (−0.156). Couro's existing anthropometric reconstruction is stronger on this slice.

## Approach

Per-clip Adam gradient descent fitting SMPL pose (24×3 axis-angle), root translation (T×3), and scalar body scale to minimize confidence-weighted 2D reprojection error of DWPose Halpe-26 vs SMPL 24-joint kinematic-tree FK. Joint-only FK (no mesh, no LBS, no shape betas).

Loss = reproj + 5.0·temporal_smoothness + 0.05·pose_L2 + 0.5·(scale−1)².

**Tested on 8 OpenCap drop-jump clips, 40 (clip × metric) pairs.**

## Headline numbers

| Metric | SMPL \|r\| | Couro \|r\| | Δ |
|---|---:|---:|---:|
| hip_flexion_r | 0.420 | 0.644 | −0.224 |
| hip_adduction_r | 0.162 | 0.251 | −0.089 |
| knee_angle_r | 0.443 | 0.608 | −0.165 |
| ankle_angle_r | 0.518 | 0.522 | −0.004 |
| lumbar_extension | 0.247 | 0.546 | −0.299 |
| **Pooled** | **0.358** | **0.514** | **−0.156** |

## Latency

- Per-clip fit: 1.1–1.6 s on CPU, 400 Adam iters
- Per-frame amortized: ~8–12 ms during fit
- Deploy-time FK only: ~0.1 ms/frame
- Engineering effort: ~3 days for naive fitter, ~1 week for SMPLify-grade

## Why SMPL didn't help here

1. **2D reprojection is depth-ambiguous.** Reproj converges to <2% of image width but many 3D poses produce the same 2D shadow. SMPL bone-length prior alone is too weak to break the null space when the camera is roughly orthogonal to the sagittal plane.
2. **No pose prior.** Real fitters use VPoser/GMM/diffusion priors over plausible human pose; our L2 zero-mean prior lets the optimizer settle into anatomically odd poses that fit 2D equally well.
3. **No coarse-to-fine schedule** (orient → torso → limbs). Single-shot Adam misses SMPLify scaffolding.
4. **Couro is already strong on this slice.** The anthropometric reconstruction encodes a subset of what SMPL would encode (bone-length foreshortening) and is convention-tuned to OpenSim. Replacing it with a generic SMPL prior gives up that tuning.

## Notable: where Couro is already at the ceiling

- subject2_Cam0: Couro hip_flex r=0.99, knee r=1.00, lumbar r=0.92. There is no headroom on this clip. SMPL gives 0.00, 0.91, 0.38.
- This shows the pooled |r| 0.54 is dragged down by hard slots (depth-ambiguous front views), not by easy slots being underexploited.

## Honest verdict

Shelve naive SMPL fitting. A SMPLify-grade fitter with VPoser priors could plausibly lift hip_flex/knee by +0.05–0.10 on side-view clips, but probably still leaves pooled |r| under 0.65 because front-view sagittal angles remain depth-ambiguous regardless of model.

## Alternative Layer 2 levers (ranked by ROI)

1. **Fix convention sign-flips in Couro's GT loaders.** Cost 1 day. Lift on signed r is large; on |r| baseline 0. Pure reporting cleanup.
2. **AMASS-synthetic training data → 2D-to-angle regressor.** Cost 6–9 days. Highest ceiling: +0.05–0.15. SMPL stays in the pipeline — as data generator, not as fitter.
3. **Pretrained 4D-Humans / ScoreHMR via HuggingFace.** Cost 1 day. Lift: +0.05–0.10 on side-view; ~0 on front-view. Worth trying with a GPU.
4. **Per-view-bucket angle formulae / view-aware reporting.** Cost 1–2 days. Lift: +0.05–0.10 pooled.
5. **Temporal Kalman smoothing on angle trace** — already in production per April 2026 autoresearch (signal processing ceiling at 6.39° handcrafted limit). No new lift available.

## Notes for future work

**Model acquisition (saves a day):** SMPL-Body normally needs academic registration at `smpl.is.tue.mpg.de`. The CC-BY 4.0 `SMPL_NEUTRAL.pkl` (247 MB) is mirrored at `https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl`. It pickles with chumpy types that break on numpy 2.x — workaround is to monkey-patch `np.bool/np.int/np.float/...` before `pickle.load(..., encoding="latin1")`. Joint-only FK doesn't need chumpy itself.

**The synthetic AMASS pipeline (Agent C work) remains the most promising Layer 2 lever** — but uses SMPL as a data generator, not as a fitter. That's a separate week of work.

## Files

- `harness/smpl_layer2_poc.py` — working POC code
- `per_clip_r.json` — raw per-clip results (8 clips × 5 metrics)
- `models/smpl/SMPL_NEUTRAL.pkl` — 247 MB SMPL model, CC-BY 4.0 (saved for future use)
