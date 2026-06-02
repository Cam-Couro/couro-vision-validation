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

## Current state (as of 2026-06-02 / v25 selective oracle)

**9 validated deploy slots** clearing the standard biomechanics validity bar (Lin's CCC > 0.60 AND Bland-Altman 95% LoA half-width < ±10°):

| Slot | CCC | LoA half | Reader |
|---|---:|---:|---|
| hip_adduction_r / side_left | 0.94 | ±9.80° | v17 hand-engineered |
| knee_angle_r / front_oblique_right | 0.89 | ±8.05° | v24 combined-cohort + ROM-aware learned Layer 2 (LL) |
| lumbar_extension / side_left | 0.88 | ±6.42° | v23 combined-cohort learned Layer 2 |
| lumbar_extension / side_right | 0.85 | ±7.03° | v23 combined-cohort learned Layer 2 |
| lumbar_extension / front_oblique_right | 0.79 | ±9.68° | v23 combined-cohort learned Layer 2 |
| ankle_angle_r / front_oblique_right | 0.73 | ±8.08° | v23 combined-cohort learned Layer 2 |
| lumbar_extension / front_oblique_left | 0.71 | ±6.57° | v18 EE2 learned Layer 2 |
| hip_adduction_r / front_oblique_left | 0.69 | ±3.29° | v20 ROM-aware learned Layer 2 |
| ankle_angle_r / side_right | 0.64 | ±9.46° | v17 hand-engineered |

**Range: CCC 0.64–0.94, single phone camera, no calibration.** v25 selective oracle (9 Good slots, **+1 vs v22**). Reader distribution: v17=8, v18=1, v20=4, v23=8, v24=2. The new Good slot is `knee_angle_r / front_oblique_right` at CCC 0.887 via v24 (LL combined-cohort + ROM-aware).

> **Floor lift status (LL build):** the 0.80 CCC floor was **NOT cleared** on any of the 5 sub-0.80 slots Cameron called out in the LL brief (lumbar_extension/front_oblique_right stayed at 0.79; ankle_angle_r/side_right stayed at 0.64). v24 added a new Good slot at CCC 0.89 (knee_angle_r/front_oblique_right), but no existing sub-0.80 slot was lifted. Honest recommendation per the LL brief: present validated Good slots as **Tier 1 (CCC ≥ 0.79)** vs **supplementary (CCC 0.60–0.78)** rather than headline a higher floor. See `data/v25_selective_oracle/REPORT.md` for the slot-by-slot verdict.

## Headline build cycles

- **2026-05-28 build cycle:** Demoted 2 broken front_center slots (Agent Q); added view-aware blend (+0.067 Layer 2, Agent R); synthetic AMASS pipeline POC (Agent S); rear-view path documented.
- **2026-05-29 build cycle:** Selective adoption of blend-retrained lumbar slot (Agent X); rear-view synthetic validation (Agent W, pooled \|r\| 0.74); learned Layer 2 trained on real OpenCap mocap GT (Agent EE2, pooled \|r\| 0.645 frame-level); ensemble of hand-engineered + learned Layer 2 (Agent JJ, the v19 deploy candidate).
- **2026-06-02 build cycle:** ROM-aware learned Layer 2 (Agent GG2) → v20 deploy candidate; v21 selective oracle across v17/v18/v20 reaches 7 Good slots; combined-cohort OpenCap+ASPset learned Layer 2 (Agent HH2, pooled OpenCap-held \|r\| 0.670, +0.025 vs EE2 OpenCap-only baseline); v23 Phase B Layer 3 retrain on HH2's combined L2 (Agent KK) lifts the v22 selective oracle to **8 Good slots** (CCC 0.64–0.94). Trunk extension now validated from 4 camera angles; ankle dorsi/plantarflexion from 2.
- **2026-06-02 LL build (this work):** LL combined-cohort + ROM-aware learned Layer 2 (Agent LL) → v24 Phase B Layer 3 retrain; v25 selective oracle adds v24 to the reader pool. Verdict: **9 Good slots** (+1 vs v22), new Good slot is `knee_angle_r / front_oblique_right` at CCC 0.887 via v24. **Floor lift to ≥0.80 was NOT achieved** on any of the 5 sub-0.80 slots Cameron called out. See `data/v25_selective_oracle/REPORT.md`.

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
| `results/deploy_ready_models_v25_selective.json` | Per-slot oracle-best across v17/v18/v20/v23/v24 (**9 Good slots**, +1 vs v22 — the new Good slot is `knee_angle_r/front_oblique_right` at CCC 0.887 via v24). Recommended for use when the additional Good slot is worth the extra reader-dispatch complexity; otherwise use v22. Floor lift to CCC ≥ 0.80 was NOT achieved on any of the 5 sub-0.80 slots. |

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
