# Higher-Resolution DWPose Test — Layer 2 Impact

**Date:** 2026-05-28
**Verdict:** SHELVE — lever is dead at gate 1.

## TL;DR

DWPose-L ONNX has **fixed** spatial input shape (384×288). Only `batch` is dynamic. Can't run at higher resolution without re-exporting from PyTorch — explicitly out of scope.

Moreover, even if it worked, **higher input resolution would address the wrong bottleneck**: Layer 1 is already at r=0.98 (essentially perfect keypoint lockstep), and the Layer 2 |r|=0.54 ceiling is dominated by 2D projection ambiguity (single-camera depth recovery), not keypoint precision.

## ONNX input shape probe

`onnxruntime.InferenceSession.get_inputs()[0].shape` returns `['batch', 3, 384, 288]` — H and W are baked literal ints, not symbolic `dim_param` strings.

| Input (H×W) | Result |
|---|---|
| 384×288 (baseline) | OK |
| 480×352 | INVALID_ARGUMENT |
| 576×432 | INVALID_ARGUMENT |
| 768×576 | INVALID_ARGUMENT |

This is structural — DWPose's SimCC head has MatMul weight tensors tied to `W*split_ratio=576` and `H*split_ratio=768`. Changing input size requires re-exporting from PyTorch with new head weights.

## Latency baseline at 384×288 (Mac M-series, batch=1)

| Provider | ms/frame | fps |
|---|---|---|
| CoreML EP | 12.6 ms | ~80 fps |
| CPU only | 118 ms | ~8.5 fps |

CoreML EP supports 263/306 ops (5 partitions). A 1.5× spatial scale would be ~2.25× compute (~28 ms CoreML) — inside 30 fps, breaks 60 fps target, no pipeline headroom on phones.

## Why this doesn't help even if it worked

Layer 1 r = 0.98 (Agent G). Layer 2 |r| = 0.54 (Agent H). The gap is **NOT from imprecise keypoints** — it's from single-camera depth ambiguity (2D→3D reconstruction noise). Sharper keypoints don't recover Z information that isn't in the image.

This is the same structural constraint Saad's softball audit identified: stride / leg-block / landing are Z-axis quantities, fundamentally unrecoverable from single-camera 2D regardless of pixel precision.

## Where Layer-2 effort should go instead

Per Agent L's suggestion:
1. SMPL body model fitting (different 2D→3D approach with anatomical prior) — Agent K is testing
2. Confidence + posture filtering at frame level (handle weak-slot variance) — Agent J is testing
3. Drop/down-weight Z-dominated metrics in single-camera rear view (deployment rule, not algorithm)
4. Per-camera-angle target calibration (memory note — rear-view scoring needs its own calibration)

## Conditional revisit criteria

Only if all three hold:
1. Official DWPose-L higher-resolution checkpoint ships without retraining
2. Mobile inference of that variant stays <20 ms on iPhone 15 Pro CoreML
3. Evidence that sub-pixel precision (not geometry) is what's actually capping Layer 2

None hold today. Shelve.

## Artifacts

- `measurements.json` — raw probe results
- `latency_baseline.json` — per-frame timing at 384×288
- `harness/test_dwpose_dynamic.py` — probe script
- `harness/test_dwpose_latency.py` — latency benchmark
- `harness/dwpose_higher_res.py` — NOT produced (gate 1 blocked)

## Time used

~15 min (gate-1 kill).
