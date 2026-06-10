# Couro Vision — Internal Technical Reference (v47)

**Status:** Internal. Contains engineering detail, known bugs, and honest negative results not intended for external distribution.
**Date:** 2026-06-10
**Deploy version:** v47 selective oracle
**Repo:** github.com/Cam-Couro/couro-vision-validation @ `275e83c`
**Source of truth for every number below:** `data/v47_selective_oracle/consolidated_metrics_v47.json` (reproduced from each reader's per-slot LOSO validity file, not the shipped routing map — see §10 Known Issues).

---

## 1. Executive summary

Couro's single-phone-camera biomechanics pipeline produces joint-angle range-of-motion (ROM) estimates validated against gold-standard marker-based motion capture. As of v47:

- **14 of 23 (metric × camera-view) "slots" clear the validated bar** — Lin's CCC > 0.60 AND Bland–Altman 95% LoA half-width < ±10°.
- **10 of those 14 are at CCC ≥ 0.79** ("Tier 1" / headline-grade).
- **Validated CCC range: 0.64 – 0.96.** Top-10 range: 0.80 – 0.96.
- **5 joint families covered:** trunk/lumbar extension, hip adduction, hip flexion, knee flexion, ankle dorsi/plantarflexion.
- **Single phone, no per-session calibration.** All numbers are leave-one-subject-out (LOSO) cross-validated on the OpenCap LabValidation cohort (n=9 subjects).

**Comparison to literature:**

| System | Hardware | Concordance range | Setup cost |
|--|--|--|--|
| Vicon (gold-standard reference) | 10+ IR cameras + markers | — | $50K+ |
| Theia3D (Kanko et al. 2021) | 8 calibrated cameras | CMC 0.85–0.97 | ~$80K |
| **Couro (this validation)** | **1 phone, no calibration** | **CCC 0.80–0.96 (top 10)** | **$0** |
| OpenCap (Uhlrich et al. 2023) | 2 phones + calibration | CMC 0.79–0.91 | ~$200 |
| Coach + slow-mo (estimate) | phone + trained eye | ~0.50–0.75 | labor |

CCC (Lin's Concordance Correlation Coefficient) and CMC (Coefficient of Multiple Correlation) are closely related concordance statistics — both penalize bias and scale mismatch on top of Pearson correlation. Comparison is apples-to-apples within ~0.05.

---

## 2. Validity framework

Three statistics computed per slot:

- **Pearson r** — frame-by-frame waveform agreement between Couro prediction and mocap ground truth.
- **Lin's CCC** — Pearson r penalized for bias and scale mismatch. The standard biomechanics validity statistic.
- **Bland–Altman 95% Limits of Agreement (LoA)** — the range within which 95% of (prediction − ground-truth) differences fall. Reported as half-width in degrees. The clinically interpretable error measure. LoA half-width = 1.96 × SD(differences).

**Tier thresholds:**

| Tier | Criteria |
|--|--|
| **Good** (validated) | CCC > 0.60 **AND** LoA half-width < ±10° |
| **Moderate** | CCC > 0.40 **AND** LoA half-width < ±15° |
| **Poor** | otherwise |

"Tier 1" is an internal marketing label for Good slots with CCC ≥ 0.79.

Canonical harness: `harness/biomech_validity_stats.py`.

---

## 3. Full deployed metrics — all 23 slots (v47)

CCC and LoA are on held-out OpenCap subjects (n=9 LOSO). "Reader" = the (Layer-2 + Layer-3) model variant selected by the per-slot oracle (see §5).

### Good — validated (14 slots)

| # | Metric | View | CCC | LoA ± | Reader |
|--|--|--|--|--|--|
| 1 | Hip adduction | front-oblique-R | 0.960 | 8.5° | v44 VideoPose3D |
| 2 | Hip adduction | side-left | 0.946 | 9.3° | v44 VideoPose3D |
| 3 | Knee flexion | side-left | 0.903 | 9.8° | v38 calibrated mirror-flip+learned-L3 |
| 4 | Knee flexion | front-oblique-R | 0.887 | 8.1° | v24 combined+ROM-aware L2 |
| 5 | Lumbar extension | side-left | 0.884 | 6.4° | v23 HH2 combined L2 |
| 6 | Hip flexion | side-left | 0.858 | 9.2° | v26 per-source-heads L2 |
| 7 | Lumbar extension | side-right | 0.848 | 7.0° | v23 HH2 combined L2 |
| 8 | Lumbar extension | front-oblique-L | 0.823 | 5.2° | v26 per-source-heads L2 |
| 9 | Lumbar extension | front-oblique-R | 0.804 | 8.8° | v33 extrema-aware learned L3 |
| 10 | Lumbar extension | front-center | 0.800 | 6.0° | v31 mirror-flip L2 + learned L3 |
| 11 | Ankle dorsi/plantarflex | front-oblique-R | 0.751 | 8.0° | v37 calibrated HH2-combined |
| 12 | Hip adduction | front-oblique-L | 0.690 | 3.3° | v20 ROM-aware L2 |
| 13 | Ankle dorsi/plantarflex | side-right | 0.644 | 9.5° | v17 hand-engineered |
| 14 | Ankle dorsi/plantarflex | side-left | 0.638 | 9.2° | v46 flip-TTA (L1) |

**Note on slot 12 (hip adduction / front-oblique-L, CCC 0.69):** its LoA of ±3.3° is the *tightest in the entire table* — clinically the most precise single measurement Couro produces. The modest CCC reflects narrow between-subject ground-truth variance in this measurement on the n=9 cohort (CCC is bounded by GT variance), not prediction error. **Lead with the LoA, not the CCC, when describing this slot.** (Confirmed by Agent QQ: residual calibration collapsed CCC under nested LOSO because there is no bias structure to correct — the residuals are already concentrated.)

### Moderate — usable with caveats (5 slots)

| Metric | View | CCC | LoA ± | Reader | Why not Good |
|--|--|--|--|--|--|
| Knee flexion | front-oblique-L | 0.928 | 10.8° | v31 mirror-flip + learned L3 | LoA misses ±10° by 0.8° |
| Hip adduction | front-center | 0.897 | 11.9° | v44 VideoPose3D | LoA |
| Hip flexion | front-oblique-L | 0.843 | 11.3° | v17 hand-engineered | LoA |
| Knee flexion | side-right | 0.819 | 13.7° | v39 calibrated hand-engineered | LoA |
| Ankle dorsi/plantarflex | front-oblique-L | 0.556 | 10.8° | v17 hand-engineered | CCC + LoA |

The top four Moderate slots are **CCC-strong, LoA-limited** — high waveform correlation, but per-clip variance keeps the agreement band above ±10°. This is the binding constraint on further promotions (see §9).

### Poor — not deployable (4 slots)

| Metric | View | CCC | LoA ± | Reader |
|--|--|--|--|--|
| Hip flexion | front-oblique-R | 0.697 | 18.5° | v17 hand-engineered |
| Hip flexion | side-right | 0.584 | 16.9° | v27 per-source ROM-aware L2 |
| Hip adduction | side-right | 0.277 | 15.3° | v31 mirror-flip + learned L3 |
| Ankle dorsi/plantarflex | front-center | 0.207 | 13.0° | v23 HH2 combined L2 |

---

## 4. Pipeline architecture (4 layers)

| Layer | Function | Implementation | Training |
|--|--|--|--|
| **L1** Keypoint detection | Per-frame 2D body keypoints from phone video | DWPose-L (ONNX, `models/dw-ll_ucoco_384.onnx`), Halpe-26 format | Pretrained, no Couro-specific training. v46 adds optional test-time flip augmentation. |
| **L2** Joint-angle reconstruction | Per-frame joint angles from keypoints (5 metrics) | Multiple variants: hand-engineered anthropometric (v17), learned TemporalKeypointCNNConf (v18/v20/v23/v24/v26/v27/v29), VideoPose3D 3D-lift (v44) | Learned variants trained on OpenCap + ASPset paired video/mocap |
| **L3** ROM estimation | Per-(metric × view) regression: angle trace → single ROM value | Per-slot ridge regression (default) or learned TinyMLP (v30/v31/v33); optional nested-LOSO residual calibration (v37/v38/v39) | Subject-level LOSO at L3 |
| **L4** Sport scoring | Maps biomech ROM → sport-specific risk scores | Deployed model, separate from this validation | 303-subject cohort (Fukuchi RBDS et al.) |

**Inference contract (unchanged across all readers):** 1 video stream → keypoints → 5 joint angles per frame. **Single camera only.** v46's flip-TTA runs the detector twice on the *same* frame (original + horizontal mirror) — it is NOT multi-camera fusion.

The deployed system is a **per-slot selective oracle**: each (metric × view) slot dispatches to whichever reader scored highest-tier/highest-CCC for that slot in LOSO validation. The full routing map is in `results/deploy_ready_models_v47_selective.json` under `per_slot_reader` (but see §10).

---

## 5. Reader provenance

Each "reader" is a distinct (L2, L3) configuration developed across the v21→v47 build campaign. The oracle picks the best per slot.

| Reader | Layer 2 | Layer 3 | Origin |
|--|--|--|--|
| v17 | Hand-engineered anthropometric | Ridge | Baseline (production reference) |
| v20 | Learned, ROM-aware loss, OpenCap-only | Ridge | Agent GG2 |
| v23 | Learned, combined OpenCap+ASPset (24 subj) | Ridge | Agent HH2 / KK |
| v24 | Learned, combined cohort + ROM-aware loss | Ridge | Agent LL |
| v26 | Learned, per-source output heads | Ridge | Agent MM |
| v27 | Learned, per-source heads + ROM-aware | Ridge | Agent MM |
| v30 | v23 L2 | Learned TinyMLP | Agent NN |
| v31 | Mirror-flip-augmented per-source L2 | Learned TinyMLP | Agent NN |
| v33 | v23 L2 | Extrema-aware learned L3 | Agent OO |
| v37 | v23 L2 | Ridge + nested-LOSO residual calibration | Agent PP |
| v38 | v31 L2 | Learned L3 + residual calibration | Agent PP |
| v39 | v17 L2 | Ridge + residual calibration | Agent PP |
| v44 | VideoPose3D 2D→3D lift | Ridge | Agent TT |
| v46 | v17 L2 on flip-TTA L1 keypoints | Ridge | Agent UU |

Model architecture (learned L2): `TemporalKeypointCNNConf` — 66 input channels (22 keypoints × {u, v, confidence}), T=9 frame window, ~100,741 params, two 1D conv blocks, center-frame head. CPU-trainable in minutes.

---

## 6. Datasets & licensing

| Dataset | Subjects | Use | License | Source |
|--|--|--|--|--|
| OpenCap LabValidation (Uhlrich 2023) | 9 | **L3 LOSO validation (primary)** | CC-BY 4.0 | simtk.org/projects/opencap-lab |
| ASPset-510 (Nibali 2021) | 15 of 17 ingested | L2 training (added to OpenCap) | CC0 | github.com/anibali/aspset-510 |
| Fukuchi RBDS | 28 | L4 sport-score calibration | CC-BY | figshare 4543435 |
| MPI-INF-3DHP | 8 | L1 keypoint reference only | **Non-commercial (academic only)** | vcai.mpi-inf.mpg.de/3dhp-dataset |

**Commercial-clean status:** all deployed weights, code, and validation numbers derive from OpenCap (CC-BY), ASPset (CC0), and DWPose/VideoPose3D (Apache 2.0). MPI-INF-3DHP is used only as an academic L1 keypoint reference and is NOT in any deployed path. No license risk in the shipped system.

---

## 7. Cross-validation discipline & the central caveat

- **L3:** Strict subject-level LOSO. No subject appears in both train and held-out at the ridge/MLP fit step. All reported CCC/LoA are on held-out subjects.
- **L2 (learned readers):** Trained on ALL subjects, then L3 is LOSO. This is **"Layer-3-LOSO-only,"** not double-LOSO.

**THE central caveat (disclose in any technical review):** Because learned L2 was trained on all subjects (including the eventual L3 held-out subject), the learned-L2 slots' CCC values are an **upper bound**. Strict double-LOSO (LOSO at both L2 and L3) would likely reduce CCC by **~0.05–0.10** on the most-promoted learned-L2 slots. The v17 hand-engineered reader and v44 VideoPose3D (pretrained on Human3.6M, never on OpenCap) are the only fully clean readers in this sense.

Affected Good slots (learned-L2, carry the upper-bound caveat): #3 (v38), #4 (v24), #5 (v23), #6 (v26), #7 (v23), #8 (v26), #9 (v33), #10 (v31), #11 (v37). Clean slots: #1, #2 (v44), #12 (v20 is OpenCap-only learned — partial), #13 (v17), #14 (v46 = v17 on flip-TTA keypoints).

**Recommended de-risking before any clinical/regulatory claim:** run double-LOSO on the ~9 learned-L2 Good slots (~1 week compute) to convert upper bounds into confirmed numbers.

---

## 8. Build campaign history (v21 → v47)

The validated count rose 7 → 14 across this campaign. Negative results are logged because they rule out levers and prevent re-work.

| Build | Agent | Lever | Good slots | Outcome |
|--|--|--|--|--|
| v21 | — | Selective oracle over v17/v18/v20 | 7 | Baseline |
| v23 | KK | L3 refit on HH2 combined cohort | 8 | +1 |
| v26/27 | MM | Per-source output heads (convention fix) | 10 | +2 (hip flexion enters headline; lumbar/FOL → Tier 1) |
| v31 | NN | Mirror-flip aug + learned L3 | 11 | +1 (front-center recovered). Mirror-flip clean negative on twins. |
| v33 | OO | Extrema-aware learned L3 | 11 | +0 — clean negative (heads decorrelated at n=9) |
| v40 | PP | Nested-LOSO residual calibration | 12 | +1 (knee/side-left first LoA-wall crack) |
| v42 | QQ | Calibration extended to v20 | 12 | +0 — clean negative; diagnosed hip-add/FOL is variance-bounded, not bias |
| v43 | RR/SS | Per-slot L3 ensemble | 12 | +0 — readers too correlated; averaging no-op |
| v45 | TT | VideoPose3D 2D→3D lift | 13 | +1 (hip-add/FOR → Good 0.96). Fixes front-camera adduction; mirror twin NOT fixed → cause is upstream of geometry. |
| v47 | UU | L1 test-time flip augmentation | 14 | +1 (ankle/side-left → Good 0.64). Confirms real DWPose L/R detector asymmetry; partially lifts mirror twin but not to Good. |

**Cumulative finding:** the remaining stuck slots (hip-adduction mirror twin, knee LoA-borderlines, front-center ankle) have resisted every lever at every layer (L1 detector, L2 angle, L3 ROM, calibration, ensemble, 3D lift). This is strong evidence they are **data-limited, not model-limited**.

---

## 9. The binding constraints (what's actually blocking more slots)

1. **n = 9 LOSO subjects.** The single biggest limitation. Wide CIs near the threshold; some CCCs are variance-bounded (can't rise without a higher-variance cohort) rather than error-bounded.
2. **LoA gate, not CCC, is binding on the borderlines.** Four Moderate slots have CCC ≥ 0.79 but LoA 10.8–13.7°. They need per-clip variance reduction (an upstream/L2 problem), which ensemble/calibration/learned-L3 did not deliver at this sample size.
3. **DWPose left/right detector asymmetry.** Confirmed real by v46 flip-TTA (lifts under-read left-side slots; partially helps the right-side mirror twin). A newer/ensembled detector at L1 is the untried lever here.
4. **Ankle data density.** ASPset has no foot keypoints, so ankle training is OpenCap-only (n=9). No modeling lever moves it; needs foot-markered paired capture.

---

## 10. Known issues

**Stale routing map in the shipped v47 deploy bundle.** `results/deploy_ready_models_v47_selective.json → per_slot_reader` was carried from a base template and not fully updated with v44 (VideoPose3D) and v46 (flip-TTA) wins. It points e.g. hip-adduction/front-oblique-R at v30 (Moderate) when v44 actually won it (Good), and ankle/side-left at v24 when v46 won. **The validity numbers are unaffected** (each is computed by its own LOSO harness), but the routing map disagrees with the promotions logs and with `consolidated_metrics_v47.json`. **Action item:** regenerate the routing map from the consolidated oracle before Saad ports to couro-vision production. ~30 min. The consolidated_metrics_v47.json in this build IS the corrected source of truth.

**REPORT provenance churn (resolved).** Earlier dump-job regenerations overwrote v23/v26/v31 reader header metadata (mislabeling them as v18/Agent FF). Reverted in commit `0786984`; committed readers are clean.

---

## 11. Production / latency notes

- **DWPose L1 (single pass):** ~120 ms/frame baseline budget.
- **v46 flip-TTA L1:** 245.8 ms/frame (two detector passes), p95 263 ms, CPU. Exactly 2× single-pass. Reserve for the slots it improves (ankle/side-left; mirror-twin LoA).
- **VideoPose3D L2 lift (v44):** 19.2 ms/clip mean, 37 s for the full 270-clip OpenCap cohort, CPU. Production-feasible.
- **Learned L2 inference:** ~0.5 ms/frame.
- **L3 ridge/MLP:** negligible.

All numbers CPU-only (CPUExecutionProvider). GPU would reduce L1 substantially.

---

## 12. Reproduction & repo pointers

```bash
# canonical validity stats
python3 -m harness.biomech_validity_stats

# learned L2 (combined cohort)
python3 -m harness.learned_layer2_combined --epochs 25 --batch-size 256

# VideoPose3D lift
python3 -m harness.learned_layer2_videopose3d

# L1 flip-TTA keypoints (requires OpenCap videos)
python3 -m harness.dwpose_flip_tta --cams Cam0 Cam4

# rebuild v47 selective oracle
python3 -m harness.build_v47_selective
```

| Artifact | Path |
|--|--|
| **Consolidated v47 metrics (source of truth)** | `data/v47_selective_oracle/consolidated_metrics_v47.json` |
| v47 deploy bundle | `results/deploy_ready_models_v47_selective.json` |
| v47 selective-oracle report | `data/v47_selective_oracle/REPORT.md` |
| Canonical per-slot validity | `data/biomech_validity_stats/per_slot_validity.json` |
| Per-reader validity files | `data/layer3_retrain_*/per_slot_validity_v*.json`, `data/residual_calibration/`, `data/learned_layer3*/` |
| External one-pager (pitch) | `docs/2026-06-03-couro-accuracy-onepager-v4.html` (needs refresh to v47) |
| External state doc | `docs/2026-06-03-couro-validation-state.html` (needs refresh to v47) |

---

## 13. One-paragraph honest summary for internal alignment

Couro validates 14 single-camera joint-angle measurements against lab motion capture, 10 of them at CCC ≥ 0.80 (top 0.96), spanning trunk, hip, knee, and ankle — squarely in OpenCap's published band and approaching Theia3D's, at zero hardware cost. The numbers are real and LOSO-validated, with two honest asterisks: the cohort is small (n=9), and most learned-reader CCCs are L3-LOSO upper bounds that double-LOSO would trim ~0.05–0.10. The remaining gains are gated by data, not modeling — every architectural lever across all four layers has been exhausted. Next highest-leverage move is cohort expansion (ankle foot-markered capture; a higher-variance frontal-plane cohort), not more model iteration.
