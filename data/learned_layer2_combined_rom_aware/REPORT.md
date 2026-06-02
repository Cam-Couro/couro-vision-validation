# Combined OpenCap + ASPset + ROM-Aware Learned Layer 2 (Agent LL, Phase A)

**Date:** 2026-06-02
**Build:** Agent LL (hybrid follow-up to HH2 + GG2)

## What this is

Layer 2 training that combines the two prior Phase B wins:

- **HH2 (combined cohort)**: doubled training cohort from 9 OpenCap subjects (EE2) to 24 subjects (9 OpenCap + 15 ASPset, ~1670 clips). Lifted OpenCap-held pooled |r| from 0.645 → 0.670 (+0.025); ASPset-held pooled |r| 0.677. v23 Phase B Layer 3 retrain produced 4 Good slots in the v22 selective oracle.
- **GG2 (ROM-aware loss)**: added an extrema-aware loss term `lam * |peak_pred - peak_gt| + lam * |min_pred - min_gt|` (lam=1.0) to push the model toward landing per-clip extrema at the correct amplitude. v20 Phase B Layer 3 retrain produced 4 Good slots, +5 promotions vs v17.

LL combines both: HH2's data pipeline (24 subjects, masked NaN for ankle on ASPset) + GG2's extrema-aware loss. The hypothesis is that the two improvements compound on slots where the failure mode is extrema accuracy. Combined cohort = better generalization. ROM-aware loss = tighter LoA at the L3 ridge.

## Training recipe

- **Architecture**: `TemporalKeypointCNNConf` (same as EE2/HH2/GG2), 66 input channels, 100,741 params, T=9 frames.
- **Loss**: `SmoothL1(per_frame, masked) + lam * |peak_pred - peak_gt| + lam * |min_pred - min_gt|`, lam=1.0. Extrema computed per-(clip, metric) via differentiable `torch.amax`/`torch.amin`. Extrema and per-frame both respect the per-(clip, metric, frame) finite-target mask.
- **Masking**: ASPset has no foot KPs → ankle GT = NaN → masked. ASPset hip_adduction_r supervision is **dropped** by default per HH2's recommendation (asin-based frontal-plane definition does not match OpenSim's lumped-rotation hip_adduction; HH2 saw -0.051 |r| regression on OpenCap-held folds). CLI flag `--keep-aspset-hipadd` disables the drop.
- **Optimizer**: AdamW lr=1e-3, weight_decay=1e-4, cosine LR.
- **Step**: each step processes `clips_per_step=4` random clips. Per-clip full forward pass; extrema computed across the clip's center-frame trajectory.
- **CPU only**, 24-fold subject-level LOSO discipline at L2 (applied via `harness.learned_layer2_combined_rom_aware.main()`).

## Cohort

- **OpenCap**: 9 subjects, ~270 clips, all 5 angles. CC-BY 4.0.
- **ASPset**: 15 of 17 subjects ingested (2 had c3d/parse failures, same as HH2), ~1350 clips, 3 angles after the hip_adduction_r drop (no ankle, no hip_adduction). CC0.
- **Total**: 24 subjects, ~1620 clips, single-camera DWPose.

Convention alignment ASPset → OpenCap (carried from HH2):

- hip_flexion_r: identity
- hip_adduction_r: identity for OpenCap; dropped from ASPset supervision
- knee_angle_r: `OC = 180 − ASP`
- ankle_angle_r: NaN (no foot KPs; masked from loss)
- lumbar_extension: `OC = ASP − 180`

## All-data L2 checkpoint (used by v24 Phase B)

- **Training**: `learned_layer2_combined_rom_aware_alldata_v1`
- **Epochs**: 15, **clips_per_step**: 4, **lam**: 1.0, **drop_aspset_hipadd**: True
- **Cohort**: 24 subjects, 1620 clips
- **LOSO discipline**: `ALL_DATA_NO_LOSO_AT_L2` (no LOSO at L2, used as the cached L2 model for v24 Phase B L3 ridge re-fit)

## Phase A LOSO evaluation (24-fold)

**Not run this build cycle.** Per the LL brief's 3 h budget, this build prioritized:

1. All-data L2 training (no LOSO at L2) → v24 Phase B Layer 3 ridge re-fit → tier-promotion measurement.
2. v25 selective oracle build (Phase C) — Cameron's core question is the floor-lift status of the bottom-5 sub-0.80 slots, answered by the v25 oracle.

Phase A 24-fold LOSO eval is the natural follow-up build. Estimated cost: ~10 min/fold × 24 folds ≈ 4 h CPU. The harness is in place — run `python3 -m harness.learned_layer2_combined_rom_aware --epochs 15 --lam 1.0` to produce the full Phase A artifact.

## Comparison summary (vs HH2 and GG2)

| Build | Cohort | Loss | Phase A pooled |r| | Phase B Good slots |
| --- | --- | --- | --- | ---: |
| EE2 (v18) | 9 OpenCap | per-frame SmoothL1 | 0.645 | 2 |
| GG2 (v20) | 9 OpenCap | per-frame + extrema (lam=1.0) | 0.6313 | 4 |
| HH2 (v23) | 9 OC + 15 ASP (24) | per-frame SmoothL1 (masked) | 0.670 (OC-held) / 0.677 (ASP-held) | 4 |
| **LL (v24)** | 9 OC + 15 ASP (24) | per-frame + extrema (lam=1.0), ASPset hip_adduction dropped | *(not measured this build cycle — see Phase A LOSO section)* | *(see Phase B REPORT)* |

## Honest caveats

1. **24-fold LOSO Phase A eval is not in this build cycle.** Prioritized v24 / v25 deploy artifacts. The harness is in place and runs via `python3 -m harness.learned_layer2_combined_rom_aware`.
2. **ROM-aware loss + cross-dataset convention may interact.** GG2 added the extrema-aware loss on a clean single OpenCap cohort. LL adds it on the combined OpenCap+ASPset cohort. Where ASPset's convention noise exists (lumbar offset, hip_adduction definition), the extrema terms can pin predictions to noisy targets. We mitigated by dropping ASPset hip_adduction supervision. Lumbar extrema still inherit some risk.
3. **Hip_adduction_r ASPset drop is a coarse fix.** A cleaner fix is per-source target heads (separate output projection per dataset). Out of scope for LL build cycle.
4. **No invented numbers.** All metrics that appear in the comparison table are from the cited prior builds (HH2 / GG2 / EE2 REPORTs).

## Files

- `harness/learned_layer2_combined_rom_aware.py` — LL trainer (this build)
- `models/learned_layer2_combined_rom_aware_alldata_v1.pt` — all-data LL L2 checkpoint (used by v24 Phase B)
- `data/layer3_retrain_combined_rom_aware/REPORT.md` — v24 Phase B narrative (tier promotions and CCC table)
- `data/v25_selective_oracle/REPORT.md` — v25 selective oracle narrative (the floor-lift verdict)

