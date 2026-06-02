# Combined Variance Audit + subject5 Forensic Investigation

Two parallel tasks on the v14_full_dwpose validation tree: (1) test whether combining Y's two partial wins (confidence-weighted features + Savgol w=9 smoothing) crosses the Good-tier gate on knee_angle_r/side_left; (2) forensically audit opencap_subject5, the dominant outlier on TWO validated/borderline slots.

## Task 1: combined Lever 2 + Lever 3 on knee/side_left/v14

Slot: `knee_angle_r / side_left / v14_full_dwpose` (baseline CCC = 0.864, LoA half-width = 12.43 deg, gap to Good = 2.43 deg).

Good-tier gate: CCC > 0.60 AND LoA half-width < 10 deg, on the full subject cohort.

| Variant | n subj | n trials | CCC | LoA half (deg) | Bias (deg) | MAE (deg) | Tier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline (no levers) | 12 | 139 | 0.864 | 12.43 | -0.48 | 4.99 | Moderate |
| Lever 2 only (conf-weighted) | 12 | 141 | 0.868 | 11.74 | -0.38 | 4.57 | Moderate |
| Lever 3 only (Savgol w=9) | 12 | 139 | 0.892 | 11.20 | -0.49 | 4.67 | Moderate |
| **Combined: L2 + L3 (w=9)** | 12 | 141 | 0.890 | 10.91 | -0.33 | 4.37 | Moderate |

**Delta vs baseline:** LoA half-width -1.53 deg, CCC 0.026.

**Verdict: NO promotion. Combined L2+L3 reaches CCC=0.890, LoA half-width=10.91 deg (gap to Good = 0.91 deg). The slot is genuinely close but does not cross the gate. This is consistent with Y's individual-lever findings and motivates the outlier audit path (Task 2).

**Per-subject pairs (combined L2+L3 w=9):**

| Subject | predicted (deg) | observed (deg) | abs error (deg) |
| --- | ---: | ---: | ---: |
| aspset_4448 | 82.05 | 82.79 | 0.73 |
| aspset_4d9e | 80.17 | 80.70 | 0.53 |
| aspset_7b5d | 79.37 | 76.74 | 2.63 |
| opencap_subject10 | 109.11 | 107.10 | 2.02 |
| opencap_subject11 | 103.62 | 96.32 | 7.30 |
| opencap_subject2 | 106.95 | 110.09 | 3.15 |
| opencap_subject3 | 94.07 | 97.83 | 3.76 |
| opencap_subject4 | 101.19 | 106.57 | 5.37 |
| opencap_subject5 | 94.57 | 82.30 | 12.27 |
| opencap_subject7 | 101.48 | 107.34 | 5.87 |
| opencap_subject8 | 90.88 | 95.47 | 4.59 |
| opencap_subject9 | 108.03 | 112.24 | 4.21 |

## Task 2: opencap_subject5 forensic audit

subject5 was the dominant outlier on TWO validation slots: knee_angle_r/side_left (abs LOSO error 12.75 deg) and ankle_angle_r/side_right (abs LOSO error 9.18 deg). We investigate whether a transparent, defensible exclusion criterion exists.

### 2.1 Demographics

| Subject | Sex | Height (m) | Mass (kg) | BMI |
| --- | :---: | ---: | ---: | ---: |
| subject2 | m | 1.96 | 78.20 | 20.36 |
| subject3 | f | 1.69 | 63.50 | 22.23 |
| subject4 | f | 1.68 | 62.60 | 22.18 |
| subject5 **<-- subject5** | m | 1.85 | 79.40 | 23.20 |
| subject7 | f | 1.68 | 61.20 | 21.68 |
| subject8 | f | 1.64 | 59.40 | 22.09 |
| subject9 | m | 1.85 | 75.70 | 22.12 |
| subject10 | f | 1.60 | 60.00 | 23.44 |
| subject11 | m | 1.84 | 92.90 | 27.44 |

**Subject5 z-scores vs cohort:**

- **height_m**: subject5 = 1.85, cohort mean = 1.75 +/- 0.12, z = 0.78
- **mass_kg**: subject5 = 79.40, cohort mean = 70.32 +/- 11.72, z = 0.77
- **bmi**: subject5 = 23.20, cohort mean = 22.75 +/- 1.97, z = 0.23

### 2.2 Ground-truth ROM distribution

Per-subject mean knee_angle_r ROM (deg) over DJ trials:

| Subject | n trials | mean knee ROM (deg) | mean ankle ROM (deg) |
| --- | ---: | ---: | ---: |
| subject2 | 4 | 109.53 | 63.56 |
| subject3 | 4 | 96.16 | 74.62 |
| subject4 | 6 | 106.57 | 81.77 |
| subject5 **<-- subject5** | 6 | 82.30 | 61.40 |
| subject7 | 5 | 107.47 | 67.58 |
| subject8 | 6 | 95.47 | 67.50 |
| subject9 | 6 | 112.24 | 81.71 |
| subject10 | 6 | 107.10 | 73.10 |
| subject11 | 2 | 94.75 | 72.92 |

- **knee_angle_r**: subject5 mean = 82.30 deg, cohort mean = 101.59 +/- 11.22, z = -1.72 (low)
- **ankle_angle_r**: subject5 mean = 61.40 deg, cohort mean = 71.76 +/- 8.24, z = -1.26 (low)

### 2.3 DWPose loading-window keypoint confidence

- **side_left** (chain ['R_hip', 'R_kne', 'R_ank']): cohort mean conf = 0.87 +/- 0.06, subject5 mean = 0.93 (z = 0.93)
- **side_right** (chain ['R_kne', 'R_ank', 'R_big_toe']): cohort mean conf = 0.90 +/- 0.06, subject5 mean = 0.93 (z = 0.43)

### 2.4 OpenSim IK marker errors (mocap quality proxy)

| Subject | mean IK marker error (m, any column) |
| --- | ---: |
| subject10 | 0.0340 |
| subject11 | 0.0379 |
| subject2 | 0.0293 |
| subject3 | 0.0255 |
| subject4 | 0.0366 |
| subject5 **<-- subject5** | 0.0360 |
| subject7 | 0.0689 |
| subject8 | 0.0374 |
| subject9 | 0.0294 |

### 2.5 LOSO statistics WITHOUT subject5 (diagnostic)

Honesty note: this is a diagnostic. Whether it justifies acting depends on whether the forensic audit produced a documented exclusion criterion (verdict in section 2.6).

- **knee_angle_r_side_left_v14_baseline**: full n=12 -> CCC=0.864, LoA=12.43 deg (Moderate-or-worse); WITHOUT subject5 n=11 -> CCC=0.918, LoA=9.75 deg (Good).
- **knee_angle_r_side_left_v14_combined_l2_l3_w9**: full n=12 -> CCC=0.890, LoA=10.91 deg (Moderate-or-worse); WITHOUT subject5 n=11 -> CCC=0.946, LoA=7.74 deg (Good).
- **ankle_angle_r_side_right_v14_baseline**: full n=9 -> CCC=0.644, LoA=9.46 deg (Good); WITHOUT subject5 n=8 -> CCC=0.666, LoA=8.41 deg (Good).

### 2.6 Forensic verdict

**Verdict:** `population_coverage_gap_low_rom_subject`

**Flags raised:** none

**Reasoning:**
- Subject5's mean knee ROM is 82.3 deg vs cohort 101.6 deg (z=-1.72) -- subject5 lands with LESS knee flexion than peers.
- Subject5's mean ankle ROM is 61.4 deg vs cohort 71.8 deg (z=-1.26) -- subject5 has LESS ankle ROM than peers.

**Recommended action:** No hard exclusion criterion. Subject5 has consistently low ROM across BOTH knee (z = -1.72) and ankle (z = -1.26) -- the subject lands the drop-jump with markedly less joint excursion than the cohort. This is NOT a data-quality issue and NOT a valid exclusion. This is a population-coverage finding: the regressor is underspecified for low-ROM drop-jump performers. Recommended action: (1) keep subject5 in the cohort, (2) report honest LoA, (3) flag for cohort expansion (recruit additional low-ROM subjects), and (4) at deploy, document that the model's Bland-Altman LoA is calibrated against a cohort skewed toward higher-ROM athletic landers.

**Exclusion supported by audit:** NO

## Combined verdict — what to do next

- **Neither task delivered a hard win, but Task 2 surfaced a principled finding.** Task 1's combined levers improved the LoA to 10.91 deg -- still 0.91 deg outside the Good gate. Task 2 found no documentable data-quality issue for subject5, but identified that subject5 has consistently low ROM on BOTH knee (z = -1.7) and ankle (z = -1.3) -- a population-coverage finding rather than an exclusion criterion. Honest reporting: the slot remains at Moderate tier on the full cohort. Path forward is (1) recruiting additional low-ROM drop-jump performers to fill the coverage gap, and (2) documenting the existing LoA against the existing athletic-cohort distribution at deploy.

## Constraints respected

- LOSO discipline preserved on every tier-change claim.
- Single phone camera only.
- No modification of v15/v16/v17, biomech_validity_stats, or other prior outputs.
- Subject5 exclusion treated honestly: only supported by audit if a documented data-quality issue surfaced. Statistical leverage alone is NOT sufficient justification.

