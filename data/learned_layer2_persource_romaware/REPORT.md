# Per-Source Heads Learned Layer 2 — MM-B (per-source heads + ROM-aware) (Agent MM, Phase A)

**Date:** 2026-06-02
**Build:** Agent MM Phase A — mm_b variant

## What this is

HH2's own recommended fix for the OpenCap/ASPset convention mismatch on `hip_adduction_r` (and, by extension, `lumbar_extension`). HH2's REPORT, verbatim:

> hip_adduction_r regressed on OC-held (−0.051). Root cause: ASPset's asin-based frontal-plane definition is geometrically not the same as OpenSim's lumped-rotation hip_adduction. Future fix: drop hip_adduction_r from ASPset training, or use per-source target heads.

LL chose the 'drop' route. On the bias-limited slot `hip_adduction_r / front_oblique_left` (v20 CCC 0.69, LoA ±3.29° — the tightest LoA in the entire deploy table), v24 collapsed to CCC −0.77, confirming that dropping supervision alone is not the right medicine. MM tries the alternative: **separate output heads per source** so each head sees a clean target signal in its own convention.

## Architecture (TemporalKeypointCNNConfPerSource)

  * **Backbone**: same 2-layer 1D conv over (T=9 frames × 22 keypoints × 3 channels = 66 input channels) used by HH2 / LL. Hidden=128. Identical to `TemporalKeypointCNNConf` from `harness.learned_layer2_combined`.
  * **Shared head** → 5 outputs (hip_flexion_r, hip_adduction_r, knee_angle_r, ankle_angle_r, lumbar_extension). This head is **deployed** — produces the OpenCap-convention angle vector consumed by Layer 3.
  * **ASPset head** → 2 outputs (hip_adduction_r_aspset, lumbar_extension_aspset). This head is **discarded at deploy**; it exists only to absorb the ASPset gradient on the two convention-mismatched metrics so it doesn't pollute the shared head.

Total params: 109,127 (+8.3K over HH2's 100,741 from the second head).

## Loss routing

Each training sample carries a `source` flag (`opencap` or `aspset`).

  * Shared head outputs **hip_flexion_r, knee_angle_r, ankle_angle_r**: masked SmoothL1 against the convention-aligned target. Both sources train these. (ankle_angle_r ASPset target is NaN by construction → masked.)
  * Shared head outputs **hip_adduction_r, lumbar_extension**:
      * OpenCap sample: loss against shared head columns.
      * ASPset sample: shared head columns NaN-masked → no gradient on these two outputs.
  * ASPset head outputs (the 2 mismatched metrics only):
      * OpenCap sample: NaN target → no gradient.
      * ASPset sample: masked SmoothL1 against the ASPset-convention target.

Net: ASPset's hip_adduction_r/lumbar_extension gradient flows ONLY into the ASPset head, never into the shared (deployed) head. The shared head sees ONLY OpenCap supervision for these two metrics. The backbone still receives a gradient from every sample (via shared metrics + ASPset head metrics).

## Training recipe (mm_b)

  * **Loss**: masked per-frame SmoothL1 + lam=1.0 * extrema (peak + min) summed across the two heads (LL-style ROM-aware loss).
  * **Optimizer**: AdamW lr=1e-3, weight_decay=1e-4, cosine LR.
  * **Step**: clips_per_step=4 (LL-style full-clip mini-batches), required by the per-(clip, metric) extrema computation.
  * **Epochs**: 25.
  * **CPU only.**

## Cohort (same as HH2/LL)

  * **OpenCap**: 9 subjects, ~270 clips, all 5 angles.
  * **ASPset**: 15 of 17 subjects ingested (2 c3d/parse failures, same as HH2), 1,409 clips, 4 angles.
  * **Total**: 24 subjects, ~1,679 clips, single-camera DWPose.

Convention alignment ASPset → OpenCap (carried from HH2; the per-source head simply prevents the ASPset-converted value from polluting the shared head):

  * `hip_flexion_r`: identity
  * `hip_adduction_r`: identity; routed to ASPset head for ASPset samples
  * `knee_angle_r`: `OC = 180 − ASP`
  * `ankle_angle_r`: NaN (no foot KPs; masked)
  * `lumbar_extension`: `OC = ASP − 180`; routed to ASPset head for ASPset samples

## All-data L2 checkpoint (used by v27 Phase B)

  * **Training**: `mm_b_persource_*_alldata_v1`
  * **Checkpoint**: `models/learned_layer2_persource_romaware_alldata_v1.pt`
  * **LOSO discipline**: `ALL_DATA_NO_LOSO_AT_L2` — no LOSO at L2; this is the cached L2 used by v27 Phase B L3 ridge re-fit.

## Phase A 24-fold LOSO eval

**Not run this build cycle.** Same compute trade-off as the LL build: prioritized the all-data L2 → Phase B → v28 oracle path to answer Cameron's floor-lift question within the time budget. The harness supports Phase A LOSO via `python3 -m harness.learned_layer2_combined_persource --variant mm_b`.

## What mm_b actually delivered (Phase B preview)

See `data/layer3_retrain_persource_romaware/REPORT.md` for the full v27 Phase B tier table. The MM-brief headline question (did the floor lift on the 2 target supplementary slots?) is answered in `data/v28_selective_oracle/REPORT.md`.

## Honest caveats

1. **Per-source heads are tested only via the all-data L2 + L3 ridge re-fit (Phase B).** No 24-fold LOSO Phase A pooled |r| was computed this build cycle.
2. **The shared head's gradient on hip_adduction_r and lumbar_extension is now driven by 9 OpenCap subjects only.** This is by design — clean convention — but it narrows the training distribution for those two outputs. The other 3 outputs still see 24 subjects.
3. **ASPset head is dead weight at inference.** It exists only as a gradient sink during training. Adds 8.3K parameters to the checkpoint.
4. **No invented numbers.** All metrics shown in the v28 report were computed on the cached all-data MM L2 checkpoint driving the v27 Phase B Layer 3 ridge re-fit.

## Files

  * `harness/learned_layer2_combined_persource.py` — MM trainer (architecture + MM-A + MM-B variants)
  * `harness/layer3_retrain_on_persource_romaware_l2.py` — mm_b Phase B driver
  * `models/learned_layer2_persource_romaware_alldata_v1.pt` — all-data mm_b L2 checkpoint
  * `data/layer3_retrain_persource_romaware/per_slot_validity_v27.json` — per-slot v27 validity stats (LOSO at L3 only)
  * `data/v28_selective_oracle/REPORT.md` — v28 oracle narrative (the floor-lift verdict)
