# Couro Vision — Multiview Validation

Validation of Couro's single-phone-camera biomechanical CV pipeline against gold-standard motion capture.

## What's in this repo

| Directory | What it contains |
|---|---|
| `docs/` | Validation documents (v2, v2.1, v2.2), accuracy one-pager, slot breakdown, mental model |
| `harness/` | Runnable training, inference, and validation scripts |
| `results/` | Deploy table candidates (v15, v17, v18, v19) — model weights + per-slot metadata |
| `data/biomech_validity_stats/` | Per-slot CCC + Bland-Altman LoA reference (the canonical validity table) |
| `data/*/REPORT.md` | Per-build narratives and findings |
| `models/` | Trained Layer 2 checkpoints (pretrained weights for DWPose/VideoPose3D excluded; download from source) |

## Current state (as of 2026-06-12 / v49 selective oracle)

**14 validated deploy slots** clearing the standard biomechanics validity bar (Lin's CCC > 0.60 AND Bland-Altman 95% LoA half-width < ±10°) — **unchanged vs v47**. **10 slots at Tier 1 (CCC ≥ 0.79 AND Good)** — corrected count (prior builds' `tier1` field over-counted by including LoA-failing slots; the consolidated artifacts are authoritative). v49 is the **final modeling lever** of the campaign: a **Layer 1 two-architecture detector ensemble** (Agent VV2, "v48" candidate) — confidence-weighted average of **DWPose-L + RTMPose-Halpe26** keypoints on the same single-camera frame, the strongest available attack on the detector-side left/right asymmetry. Smoke check confirmed the two detectors' Halpe-26 keypoints align directly (no remap, no L/R swap). **Result: v48 is adopted in 0 of 23 slots — clean negative.** On the geometry-reader probe it tightened LoA on side-camera slots (e.g. mirror twin `hip_adduction_r/side_right` LoA 13.5° → 8.4°) and lifted some CCCs (knee/side-left 0.58 → 0.77), but never beat the existing deployed reader on any slot, and the mirror-twin CCC stayed ~0.22 (Poor). Two independent detectors could not crack what one flipped detector (v46) already maxed out. Latency 142 ms/frame (two detectors). **Campaign-closing verdict: the remaining stuck slots are DATA-limited, not model-limited** — every architectural lever across all four layers (L1 detector, L2 angle reconstruction, L3 ROM, calibration, ensembling, 3D lifting) is now exhausted. The next gains require cohort expansion, not more modeling. See `data/v49_selective_oracle/REPORT.md` and `consolidated_metrics_v49.json` (source of truth).

---

### Historical: v47 (Layer 1 flip-TTA, Agent UU)

flip-TTA is the deploy reader for **2 of 23 slots**, and added the Good slot:
- `ankle_angle_r / side_left` **Poor → Good** (CCC 0.375 → 0.638, LoA 12.2° → 9.2°) — DWPose was under-reading the left ankle; flip-TTA corrected it.

**The mirror-twin verdict is nuanced.** flip-TTA was launched to fix `hip_adduction_r / side_right` (stuck at 0.27). On the transparent geometry reader it lifted the side_right CCC 0.198 → 0.272 toward its side_left sibling (0.281 → 0.509) and tightened LoA ~5°, confirming a real detector asymmetry — but **not enough to promote it to Good** (stays Poor). So the asymmetry is partly detector-side (flip-TTA helps) but the deploy-grade gap on that slot is deeper. flip-TTA cost: **246 ms/frame** (2 detector passes, CPU) — exactly double single-pass L1; reserve for the slots it improves. See `data/v47_selective_oracle/REPORT.md`.

Two PP levers shipped:

- **Lever 1 (v36) — LoA-then-CCC tie-break in the LoA-limited Moderate band.** Hygiene fix to v35's CCC-first tie-break inside the Moderate tier when all top-tier candidates have CCC ≥ 0.79 (LoA is the binding constraint). One shift: `knee_angle_r / side_left` v31 → v29 (LoA 10.15 → 10.02). Still Moderate, doesn't cross the gate alone — that took calibration. Rule preserved in v40.
- **Lever 2 (v37/v38/v39) — Nested-LOSO residual calibration.** Per slot × reader, fit linear calibration `pred_cal = a*pred + b` on pseudo-residuals from inner-LOSO over training subjects only; apply to outer held-out subject. Per-slot fallback to uncalibrated if calibration inflates LoA. Cracks the LoA wall on **1 of 5** Category A targets (`knee_angle_r / side_left` via v38). On average across all 23 slots × 3 readers, calibration tightens LoA on ~5/23 slots and is neutral or worse on the rest — the per-slot fallback handles regressions cleanly.

See `data/v35_selective_oracle/REPORT.md` for the v35 delta vs v32 (Agent OO extrema-aware L3). **v35 Good slot count is unchanged from v32 (11);** the extrema-aware lever did NOT crack the LoA wall on Category A targets (knee/side_left stuck at 10.15° vs the 10.0° gate). Tier 1 count picked up +1 via v33 (extrema-aware L3) on `lumbar_extension / front_oblique_right` (CCC 0.79→0.80). Detailed history of v32's wins vs v28 is in `data/v32_selective_oracle/REPORT.md`. v32 had added three readers to the v28 pool: v29 (mirror-flip Layer 2 + ridge L3), v30 (v23 L2 + learned L3), v31 (mirror-flip L2 + learned L3). Two slot promotions came in v32 vs v28:

- `hip_adduction_r / front_oblique_right`: Poor → Moderate via v30 (CCC 0.84 → 0.89, LoA tightened 15.74 → 13.80°).
- `lumbar_extension / front_center`: Moderate → **Good** via v31 (CCC 0.55 → 0.80, LoA 8.48 → 5.95°). Surprise win — front_center was previously the broken view family.

Mirror-flip augmentation did NOT lift the Category B mirror-twin asymmetric slots (hip_adduction_r/side_right, hip_flexion_r/side_right, ankle_angle_r/side_left, ankle_angle_r/front_oblique_left all unchanged). Learned L3 did NOT promote any Category A LoA-limited knee/hip_flexion borderlines to Good, but did tighten LoA on 3 of 5 slots (knee_angle_r FOL/side_left LoA below 11° for the first time).

### v40 deploy slot table (12 Good slots ordered by CCC)



| Slot | CCC | LoA half | Reader |
|---|---:|---:|---|
| hip_adduction_r / side_left | 0.94 | ±9.32° | v29 mirror-flip Layer 2 (NN) |
| knee_angle_r / front_oblique_right | 0.89 | ±8.05° | v24 combined-cohort + ROM-aware learned Layer 2 (LL) |
| lumbar_extension / side_left | 0.88 | ±6.42° | v23 combined-cohort learned Layer 2 |
| hip_flexion_r / side_left | 0.86 | ±9.24° | v26 per-source heads + per-frame SmoothL1 (MM-A) |
| lumbar_extension / side_right | 0.85 | ±7.03° | v23 combined-cohort learned Layer 2 |
| lumbar_extension / front_oblique_left | 0.82 | ±5.18° | v26 per-source heads + per-frame SmoothL1 (MM-A) |
| **lumbar_extension / front_oblique_right** | **0.80** | **±8.80°** | **v33 extrema-aware learned Layer 3 (OO)** |
| lumbar_extension / front_center | 0.80 | ±5.95° | v31 mirror-flip Layer 2 + learned Layer 3 (NN) |
| **knee_angle_r / side_left** | **0.90** | **±9.78°** | **v38 mirror-flip L2 + ridge L3 + nested-LOSO calibration (PP)** |
| ankle_angle_r / front_oblique_right | 0.73 | ±8.08° | v23 combined-cohort learned Layer 2 |
| hip_adduction_r / front_oblique_left | 0.69 | ±3.29° | v20 ROM-aware learned Layer 2 |
| ankle_angle_r / side_right | 0.64 | ±9.46° | v17 hand-engineered |

**Range: CCC 0.64–0.94, single phone camera.** v40 selective oracle: **12 Good slots (+1 vs v35); 14 Tier 1 slots at CCC ≥ 0.79 (unchanged vs v35, +1 vs v32, +7 vs v28)**. Reader distribution: v17=4, v20=1, v23=3, v24=2, v26=2, v27=2, v30=2, v31=3, v33=1, v37=1, v38=1, v39=1. Three calibrated readers ship at v40. **One Category A LoA-limited slot promoted to Good** (knee_angle_r/side_left via v38 calibration). Historical: v32 added `lumbar_extension / front_center` (Moderate → Good via v31) and bumped `hip_adduction_r / front_oblique_right` from Poor to Moderate via v30 (CCC 0.84 → 0.89).

> **Mirror-flip + learned Layer 3 verdict (NN build):** Mirror flip Layer 2 augmentation did NOT lift the Category B mirror-twin asymmetric slots (hip_adduction_r/side_right, hip_flexion_r/side_right, ankle_angle_r/side_left, ankle_angle_r/front_oblique_left — all unchanged within ±0.01 CCC). Learned Layer 3 (TinyMLP replacing ridge per slot, with ridge fallback if learned underperforms) did NOT promote any Category A LoA-limited knee/hip_flexion borderlines to Good, but it DID tighten LoA on 3 of 5 borderline slots (knee_angle_r/FOL: LoA 11.83→10.77; knee_angle_r/side_left: 11.79→10.15; hip_adduction_r/FOR: 15.74→13.80). See `data/v32_selective_oracle/REPORT.md` for the slot-by-slot verdict.

> **Extrema-aware Layer 3 verdict (OO build):** Two-head TinyMLP at L3 (pred_max + pred_min, with loss = SmoothL1(ROM) + 0.5·SmoothL1(max) + 0.5·SmoothL1(min)) **did NOT crack the LoA wall** on any Category A target. Across the 5 Cat A slots, extrema-aware L3 produced LoA equal-to or worse-than the v32 winner on every slot (e.g. knee/side_left v32 v31 ridge LoA 10.15° → v33 extrema 11.79°, v34 extrema 13.48°). v33 still earned 1 reader slot in v35 (lumbar_extension/front_oblique_right, CCC 0.79→0.80, +1 Tier 1). With per-slot n=9–22 LOSO inner folds and two output heads, the model overfits the extrema signal before improving LoA. Negative result: max/min supervision alone is not enough at this dataset size. See `data/v35_selective_oracle/REPORT.md`.

## Headline build cycles

- **2026-05-28 build cycle:** Demoted 2 broken front_center slots (Agent Q); added view-aware blend (+0.067 Layer 2, Agent R); synthetic AMASS pipeline POC (Agent S); rear-view path documented.
- **2026-05-29 build cycle:** Selective adoption of blend-retrained lumbar slot (Agent X); rear-view synthetic validation (Agent W, pooled \|r\| 0.74); learned Layer 2 trained on real OpenCap mocap GT (Agent EE2, pooled \|r\| 0.645 frame-level); ensemble of hand-engineered + learned Layer 2 (Agent JJ, the v19 deploy candidate).
- **2026-06-02 build cycle:** ROM-aware learned Layer 2 (Agent GG2) → v20 deploy candidate; v21 selective oracle across v17/v18/v20 reaches 7 Good slots; combined-cohort OpenCap+ASPset learned Layer 2 (Agent HH2, pooled OpenCap-held \|r\| 0.670, +0.025 vs EE2 OpenCap-only baseline); v23 Phase B Layer 3 retrain on HH2's combined L2 (Agent KK) lifts the v22 selective oracle to **8 Good slots** (CCC 0.64–0.94). Trunk extension now validated from 4 camera angles; ankle dorsi/plantarflexion from 2.
- **2026-06-02 LL build:** LL combined-cohort + ROM-aware learned Layer 2 (Agent LL) → v24 Phase B Layer 3 retrain; v25 selective oracle adds v24 to the reader pool. Verdict: **9 Good slots** (+1 vs v22), new Good slot is `knee_angle_r / front_oblique_right` at CCC 0.887 via v24. **Floor lift to ≥0.80 was NOT achieved** on any of the 5 sub-0.80 slots Cameron called out. See `data/v25_selective_oracle/REPORT.md`.
- **2026-06-02 MM build:** Per-source heads learned Layer 2 (Agent MM) addresses HH2's own recommended fix for the OpenCap/ASPset hip_adduction_r convention mismatch. Two variants: v26 (MM-A, per-frame SmoothL1) and v27 (MM-B, ROM-aware). v28 selective oracle adds both to the v25 pool. **Verdict: 10 Good slots (+1 vs v25); 7 Tier 1 slots at CCC ≥ 0.79 (+2 vs v25).** `lumbar_extension / front_oblique_left` lifted from 0.71 to **0.82** via v26 (the only target slot that crossed Tier 1). `hip_adduction_r / front_oblique_left` did NOT lift — convention mismatch was not the bottleneck on the bias-limited slot. Bonus: `hip_flexion_r / side_left` unexpectedly promoted Moderate → Good via v26 at CCC 0.858. See `data/v28_selective_oracle/REPORT.md`.
- **2026-06-02 NN build (this work):** Two combined improvements to attack the remaining gaps. (1) **Mirror-flip Layer 2 augmentation** — flip every clip horizontally + swap L/R keypoint pairs + swap _r/_l target labels, doubling effective coverage. Aimed at Category B (mirror-twin asymmetric) slots. (2) **Per-slot learned Layer 3 (TinyMLP, hidden=32, dropout 0.2, AdamW lr=1e-2 wd=1e-3, 200 epochs w/ early stopping)** replacing ridge regression. Aimed at Category A (LoA-limited knee/hip_flexion borderlines). Per-slot fallback to ridge if learned underperforms by >0.05 CCC keeps no-regression guarantee. v29 = mirror-flip L2 + ridge L3; v30 = v23 L2 + learned L3; v31 = mirror-flip L2 + learned L3. v32 selective oracle adds all three to the v28 pool. **Verdict: 11 Good slots (+1 vs v28); 13 Tier 1 slots at CCC ≥ 0.79 (+6 vs v28).** The new Good slot is `lumbar_extension / front_center` at CCC 0.80 via v31 — a surprise win on a previously-broken view. `hip_adduction_r / front_oblique_right` lifted Poor → Moderate via v30 (CCC 0.84 → 0.89). Category B (mirror-twin) slots did NOT lift (all unchanged within ±0.01 CCC). Category A (LoA-limited) borderlines did NOT promote to Good but DID see LoA tightening on 3 of 5 slots. See `data/v32_selective_oracle/REPORT.md`.

## Deploy table candidates

| File | Use |
|---|---|
| `results/deploy_ready_models.json` | v14 baseline (production reference) |
| `results/deploy_ready_models_v15.json` | v14 minus 2 broken front_center slots |
| `results/deploy_ready_models_v17_selective.json` | v15 + view-aware blend retrain for lumbar/front_oblique_right |
| `results/deploy_ready_models_v18_learned_l2.json` | Full v17 + EE2 OpenCap-only learned Layer 2 (mixed results — see data/layer3_retrain_learned_l2/REPORT.md) |
| `results/deploy_ready_models_v19_ensemble.json` | Per-slot oracle-best between v17 and v18 (superseded by v21/v22) |
| `results/deploy_ready_models_v20_rom_aware.json` | Full v17 + GG2 ROM-aware OpenCap-only learned Layer 2 |
| `results/deploy_ready_models_v21_selective.json` | Per-slot oracle-best across v17/v18/v20 (7 Good slots) |
| `results/deploy_ready_models_v23_combined_l2.json` | Full v17 + v23 combined-cohort (OpenCap+ASPset) learned Layer 2 (Agent KK build, see data/layer3_retrain_combined_l2/REPORT.md) |
| `results/deploy_ready_models_v22_selective.json` | **Recommended:** per-slot oracle-best across v17/v18/v20/v23 (**8 Good slots**, +1 vs v21) |
| `results/deploy_ready_models_v24_combined_rom_aware.json` | v17 base + v24 LL combined-cohort + ROM-aware learned-L2 ridge re-fit (Agent LL build) |
| `results/deploy_ready_models_v25_selective.json` | Per-slot oracle-best across v17/v18/v20/v23/v24 (**9 Good slots**, +1 vs v22). |
| `results/deploy_ready_models_v26_persource_perframe.json` | v17 base + v26 MM-A per-source heads + per-frame SmoothL1 learned-L2 ridge re-fit (Agent MM build) |
| `results/deploy_ready_models_v27_persource_romaware.json` | v17 base + v27 MM-B per-source heads + ROM-aware learned-L2 ridge re-fit (Agent MM build) |
| `results/deploy_ready_models_v28_selective.json` | Per-slot oracle-best across v17/v18/v20/v23/v24/v26/v27 (**10 Good slots**, +1 vs v25; 7 Tier 1 slots at CCC ≥ 0.79, +2 vs v25). Superseded by v32. |
| `results/deploy_ready_models_v29_mirrorflip.json` | v17 base + v29 NN mirror-flip per-source per-frame learned-L2 + ridge L3 (Agent NN, Phase 1). |
| `results/deploy_ready_models_v30_learned_l3.json` | v17 base + v23 HH2 combined L2 + per-slot **learned Layer 3** (TinyMLP) with ridge fallback (Agent NN, Phase 2). |
| `results/deploy_ready_models_v31_mirrorflip_learned_l3.json` | v17 base + v29 mirror-flip L2 + per-slot learned Layer 3 with ridge fallback (Agent NN, Phase 2). |
| `results/deploy_ready_models_v32_selective.json` | Per-slot oracle-best across v17/v18/v20/v23/v24/v26/v27/v29/v30/v31 (**11 Good slots**, +1 vs v28; 13 Tier 1 slots at CCC ≥ 0.79, +6 vs v28). New Good slot: `lumbar_extension/front_center` at CCC 0.80 via v31. New Moderate via v30: `hip_adduction_r/front_oblique_right` at CCC 0.89. Superseded by v40. |
| `results/deploy_ready_models_v33_extrema_l3.json` | v17 base + v23 L2 + per-slot extrema-aware learned L3 (Agent OO, two-head pred_max + pred_min). |
| `results/deploy_ready_models_v34_mirrorflip_extrema_l3.json` | v17 base + v29 mirror-flip L2 + extrema-aware learned L3 (Agent OO). |
| `results/deploy_ready_models_v35_selective.json` | Per-slot oracle-best across v17/v18/v20/v23/v24/v26/v27/v29/v30/v31/v33/v34 (**11 Good slots, 14 Tier 1**). Adds v33/v34 to pool; no net Good promotion. Superseded by v40. |
| `results/deploy_ready_models_v36_selective.json` | Same reader pool as v35 with LoA-then-CCC tie-break inside the LoA-limited Moderate band (Agent PP, Lever 1). Selection hygiene only — no net tier change. Rule preserved in v40. |
| `results/deploy_ready_models_v37_v23_calibrated.json` | v17 base + v23 L2 + ridge L3 + nested-LOSO residual calibration with per-slot fallback to uncalibrated (Agent PP, Lever 2). |
| `results/deploy_ready_models_v38_v31_calibrated.json` | v17 base + v29 mirror-flip L2 + ridge L3 + nested-LOSO residual calibration with per-slot fallback (Agent PP, Lever 2). |
| `results/deploy_ready_models_v39_v17_calibrated.json` | v17 base + hand-engineered L2 + ridge L3 + nested-LOSO residual calibration with per-slot fallback (Agent PP, Lever 2). |
| `results/deploy_ready_models_v40_selective.json` | Per-slot oracle-best across the v35 reader pool + v37/v38/v39 calibrated readers, with the v36 LoA-then-CCC tie-break (**12 Good slots**, +1 vs v35; 14 Tier 1 slots at CCC ≥ 0.79, unchanged). New Good slot: `knee_angle_r / side_left` via v38 at LoA 9.78° / CCC 0.903 — first Category A LoA-wall crossing. Superseded by v42. |
| `results/deploy_ready_models_v41_v20_calibrated.json` | v17 base + v20 (GG2 ROM-aware OpenCap-only L2) + ridge L3 + nested-LOSO residual calibration with per-slot fallback to uncalibrated v20 (Agent QQ). |
| `results/deploy_ready_models_v42_selective.json` | **Recommended:** per-slot oracle-best across the v40 reader pool + v41 (v20 + cal) (**12 Good slots, 14 Tier 1** — unchanged vs v40). v41 did not promote `hip_adduction_r / front_oblique_left` to Tier 1: calibration collapsed CCC from 0.690 to 0.213 on the n=9 OpenCap subject pool, per-slot fallback held v20 uncalibrated. Confirms calibration is not the right lever for v20-based slots. |

## Methodology — what "validated" means here

Three statistical measures per slot:
- **Pearson r** — frame-by-frame agreement between prediction and mocap
- **Lin's Concordance Correlation Coefficient (CCC)** — Pearson r penalized for bias and scale mismatch; the standard biomech validity statistic
- **Bland-Altman 95% Limits of Agreement (LoA)** — range within which 95% of (prediction − ground truth) differences fall

Validity tier thresholds:
- **Good:** CCC > 0.60 AND LoA half-width < ±10°
- **Moderate:** CCC > 0.40 AND LoA half-width < ±15°
- **Poor:** otherwise

All numbers computed via Leave-One-Subject-Out cross-validation. See `harness/biomech_validity_stats.py`.

## Data dependencies (not included — download from sources)

| Dataset | Subjects | Use | License | Source |
|---|---|---|---|---|
| OpenCap LabValidation | 9 | L3 LOSO | CC-BY 4.0 | https://simtk.org/projects/opencap-lab |
| ASPset-510 | 17 | L3 LOSO | CC0 | https://github.com/anibali/aspset-510 |
| Fukuchi RBDS | 28 | L4 calibration | CC-BY | https://figshare.com/articles/dataset/4543435 |
| MPI-INF-3DHP | 8 | L1 keypoint r (academic only) | Non-commercial | https://vcai.mpi-inf.mpg.de/3dhp-dataset/ |

## Reproduce key results

```bash
# Validity stats baseline
python3 -m harness.biomech_validity_stats

# Learned Layer 2 training (real OpenCap GT)
python3 -m harness.learned_layer2_real_gt --epochs 25 --batch-size 256

# View-aware blend evaluation
python3 -m harness.view_aware_blend

# Rear-view synthetic validation
python3 -m harness.generate_rear_view_validation
```

## License

Code: MIT. Validation documents: CC-BY 4.0. See LICENSE files in subdirectories where applicable.
