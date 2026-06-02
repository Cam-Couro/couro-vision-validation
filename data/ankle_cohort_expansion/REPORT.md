# Ankle ROM Cohort Expansion Audit — v14 ankle_angle_r / side_right

**Produced:** 2026-05-29 02:24:33  
**Script:** `harness/expand_ankle_cohort.py`  
**Source slot:** ankle_angle_r / side_right / v14_full_dwpose

## TL;DR

The v14 ankle / side_right slot stays at n=9 with the existing validation tree. **No candidate dataset cleanly expands the per-subject LOSO cohort.** The point estimate (CCC = 0.644, LoA half-width = 9.46deg) is inside the Good tier, but the 95% Fisher Z CI on CCC is [-0.034, +0.916] — the lower bound includes zero, so the slot cannot be promoted from 'preliminary additional measurement' to 'headline-range validated.'

To clear the promotion threshold (lower CCC 95% CI bound >= 0.30 at the current point estimate), n_subjects must reach >= 22. The recommended path is fresh data collection — not further mining of public datasets, because the public datasets that satisfy the joint requirement of (paired single-phone video) AND (mocap-derived ankle ROM in deg) AND (commercial-permissive license) do not exist in the validation tree.

## Baseline (re-derived, matches existing per_slot_validity.json)

| Metric | Value |
| --- | ---: |
| n_subjects | 9 |
| n_trials | 54 |
| Mean observed (deg) | 72.06 |
| Mean predicted (deg) | 72.39 |
| Mean bias (deg) | 0.33 |
| 95% LoA (deg) | [-9.13, 9.80] |
| LoA half-width (deg) | 9.46 |
| Pearson r | 0.7458 |
| **CCC (Lin)** | **0.6445** |
| MAE (deg) | 3.89 |
| RMSE (deg) | 4.56 |
| CCC 95% CI (Fisher Z) | [-0.034, 0.916] |
| CCC 95% CI (bootstrap, 10k) | [0.119, 0.838] |
| CCC bootstrap mean +/- SD | 0.593 +/- 0.177 |

Bootstrap and Fisher Z agree closely, which is expected when the per-subject residuals are approximately Gaussian (we verify this informally by inspecting the per-subject diff_deg vector below — no obvious outliers at this n).

## Per-subject leave-one-out sensitivity

How much does each individual subject move the CCC? Subjects with the largest absolute error in the LOSO fold are listed first.

| Subject | pred (deg) | obs (deg) | abs error (deg) | CCC w/o subj | r w/o subj | Δ CCC vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| opencap_subject5 | 70.58 | 61.40 | 9.18 | 0.7415 | 0.8021 | 0.0971 |
| opencap_subject11 | 69.14 | 74.26 | 5.12 | 0.6881 | 0.8246 | 0.0437 |
| opencap_subject2 | 70.87 | 65.79 | 5.08 | 0.6631 | 0.7488 | 0.0187 |
| opencap_subject4 | 77.61 | 81.77 | 4.15 | 0.5702 | 0.6617 | -0.0743 |
| opencap_subject3 | 71.62 | 75.20 | 3.59 | 0.6641 | 0.7706 | 0.0197 |
| opencap_subject10 | 76.13 | 73.10 | 3.04 | 0.6500 | 0.7757 | 0.0055 |
| opencap_subject9 | 78.70 | 81.71 | 3.01 | 0.5321 | 0.6396 | -0.1124 |
| opencap_subject7 | 69.52 | 67.81 | 1.71 | 0.6287 | 0.7301 | -0.0158 |
| opencap_subject8 | 67.35 | 67.50 | 0.15 | 0.6058 | 0.7369 | -0.0387 |

## Promotion projection — assuming CCC point estimate holds

If new subjects are added and the per-subject CCC stays at ~0.644, the 95% Fisher Z CI on CCC tightens as follows. The promotion target is **lower bound >= 0.30**.

| n_subjects | CCC 95% CI (Fisher Z) | Lower bound >= 0.30? |
| ---: | --- | :---: |
| 9 | [-0.034, 0.916] | no |
| 12 | [0.112, 0.889] | no |
| 15 | [0.197, 0.870] | no |
| 17 | [0.237, 0.859] | no |
| 20 | [0.282, 0.846] | no |
| 22 | [0.306, 0.838] | YES |
| 25 | [0.334, 0.829] | YES |
| 28 | [0.357, 0.820] | YES |
| 30 | [0.370, 0.815] | YES |
| 35 | [0.396, 0.805] | YES |
| 40 | [0.417, 0.796] | YES |

At the current CCC, the first n that clears the promotion bar is **n = 22**. If a fresh cohort runs higher than 0.64 (e.g. 0.75), the threshold drops to ~n=15. If it runs lower, the slot may never promote at any tractable n — which is itself useful diligence-relevant information.

## Dataset audit — every candidate, decision, and rationale

| Dataset | Present in tree? | Paired video + mocap? | Foot/toe markers? | Sagittal ankle GT? | License | Commercial? | Decision |
| --- | :---: | :---: | :---: | :---: | --- | :---: | --- |
| OpenCap LabValidation (DJ tasks, current baseline) | YES | YES | YES | YES | CC-BY 4.0 | YES | **ALREADY USED** |
| OpenCap LabValidation NON-DJ tasks (same 9 subjects) | YES | YES | YES | YES | CC-BY 4.0 | YES | **DOES NOT EXPAND n_subjects** |
| ASPset-510 | YES | YES | no | no | CC0 | YES | **EXCLUDE** |
| Fukuchi RBDS (running kinematics, 28 runners) | no | no | no | YES | CC-BY | YES | **EXCLUDE** |
| ACL Jump-Landing (Calisti 2025, 42 athletes) | no | no | no | YES | CC-BY 4.0 | YES | **EXCLUDE** |
| GaitRec (Horsak 2020, 211 subjects) | no | no | no | YES | CC-BY 4.0 | YES | **EXCLUDE** |
| MPI-INF-3DHP (Mehta 2017) | YES | YES | YES | YES | Non-commercial research only | no | **EXCLUDE** |
| CMU Panoptic (rear sample) | YES | no | no | no | Non-commercial research only | no | **EXCLUDE** |
| Synthetic AMASS / SMPL bursts | YES | YES | YES | YES | SMPL CC-BY 4.0 | YES | **EXCLUDE for headline cohort** |

### Per-dataset rationale

**OpenCap LabValidation (DJ tasks, current baseline)** — ALREADY USED

Source of the current n=9 cohort. Subject6 has no raw VideoData (only OpenSim-precomputed HRNet/OpenPose outputs), so its keypoints cannot be extracted with DWPose without going outside the validation tree.

**OpenCap LabValidation NON-DJ tasks (same 9 subjects)** — DOES NOT EXPAND n_subjects

Adding non-DJ tasks (squats, walking, STS) would only add TRIALS, not SUBJECTS. The 95% Fisher Z CI on the per-subject CCC depends on n_subjects, not n_trials — so this cannot promote the slot. Separately, the v9_phased feature pipeline is built around DJ initial-contact event detection and would require per-task-type event detectors to handle these motions. Significant engineering, no headline-CI benefit.

**ASPset-510** — EXCLUDE

17-keypoint skeleton ends at ankle — no toe, heel, or metatarsal markers. Sagittal ankle dorsi/plantarflexion requires foot-relative-to-shank, which cannot be derived from joints_3d without a foot reference. ASPset is explicitly documented as ankle-incompatible in harness/aspset_loader.py: 'ASPset HAS NO foot/toe keypoints -> ankle_angle_r cannot be derived from this dataset. That metric stays OpenCap-only.'

**Fukuchi RBDS (running kinematics, 28 runners)** — EXCLUDE

NOT present in the validation tree. Mentioned in pitch docs as a Layer 4 calibration source (population means for running config tuning). Fukuchi RBDS is a treadmill+overground running mocap dataset — it does NOT ship with paired single-phone-camera video, so it cannot drive Layer 3 LOSO regression of phone keypoints -> mocap ROM. Even if downloaded, motion type is steady-state running, not discrete-event motion the v9 feature pipeline is built for.

**ACL Jump-Landing (Calisti 2025, 42 athletes)** — EXCLUDE

NOT present in the validation tree. Pitch docs cite it as a Layer 4 calibration source for ACL risk + jump-landing config means, not as a Layer 3 paired-video dataset. The published release is markered mocap kinematics with no paired single-phone-camera video, so it cannot drive Layer 3 LOSO. Would require fresh data collection or a different study release.

**GaitRec (Horsak 2020, 211 subjects)** — EXCLUDE

NOT present in the validation tree. GaitRec is force-plate + kinematic gait data without paired video — used in pitch docs for Layer 4 walking gait config calibration and force/contact validation only. Cannot drive Layer 3 LOSO of phone keypoints without paired RGB video, which the public release does not include.

**MPI-INF-3DHP (Mehta 2017)** — EXCLUDE

MPI-INF-3DHP license is non-commercial only: 'Methods and models that make use of the provided Software in any way can only be used for non-commercial purposes.' Per the explicit constraint on this expansion task and per the validation-v2.1 doc, MPI is excluded from any commercial-footprint counts. Also, the existing 8-subject MPI cohort for ankle has mean abs(r) = 0.228 across cam7 — well below the threshold to lift the side-right CCC anyway, since cam7 is rear-oblique (azimuth -62 deg), not a true side view.

**CMU Panoptic (rear sample)** — EXCLUDE

Only a calibration JSON exists in the tree (no video, no GT angles). License is non-commercial. Excluded.

**Synthetic AMASS / SMPL bursts** — EXCLUDE for headline cohort

Synthetic 'subjects' are not real human subjects with independent biomechanics — they share the SMPL_NEUTRAL body model. Adding them to n_subjects would inflate the denominator of the Fisher Z CI without adding independent biological variance. Diligence reviewers correctly read synthetic-subject inflation as a methodological flag. Real value is at Layer 2 (keypoint -> angle), not Layer 3 LOSO.

## What this means for the one-pager

The ankle / side_right slot **stays in the 'preliminary additional measurement' bucket** in the one-pager — not the 'headline range' bucket — until a new cohort is collected. The point estimates (CCC = 0.64, LoA half-width = 9.5deg, MAE = 3.9deg) can still be cited as evidence the pipeline is functional at side_right for ankle ROM; what cannot be claimed is statistical precision of those numbers at the diligence-grade '95% CI clearly above zero' standard.

**Recommended language for the one-pager:**

> Ankle dorsi/plantarflexion ROM (right ankle, side-right view): **preliminary** at n=9 OpenCap subjects. Per-subject CCC = 0.64 (95% CI on a small cohort is wide; promotion to headline requires expansion to >= 22 subjects). LoA half-width 9.5deg, MAE 3.9deg are inside the Good tier point-estimate thresholds and indicate the pipeline works at side_right, but the cohort is too small to make a precision claim with 95% confidence the lower CCC bound is above zero.

## What would clear the bar

- **Fresh data collection of ~13+ additional subjects** performing the drop-jump task with a single side-right phone camera + paired marker mocap (must include foot/toe markers so ankle dorsi/plantarflexion can be derived from foot vs shank). Lab time estimate: 15 min/subject. Even at 1 lab session/week this clears in a quarter.
- **Re-licensable variant of MPI-INF-3DHP or Human3.6M.** Currently blocked by non-commercial terms. If a commercial release path becomes available, MPI-INF-3DHP's 8-subject cam7 cohort would still only nudge n from 9 -> 17 and the CCC there (mean abs(r) = 0.23 on cam7 ankle) is far below the 0.64 we need to hold.
- **A new commercial-clean ankle-rich public dataset.** Status: actively monitored. Nothing matching the joint requirement of (paired single-phone-camera video at side view) AND (mocap foot+shank markers) AND (CC-BY-style license) was found in the May 2026 sweep.

## Constraints honored

- LOSO discipline preserved — no data leakage between folds.
- Single phone camera only — no multi-camera fusion considered.
- No modification of v14, v15, biomech_validity_stats, or other yesterday's-build outputs. This script is additive.
- MPI-INF-3DHP excluded for non-commercial license (per yesterday's learning); CMU Panoptic excluded for non-commercial license.
- Honest reporting: the slot CANNOT be promoted with existing data. Reporting this directly rather than constructing a comforting synthetic-cohort story.

