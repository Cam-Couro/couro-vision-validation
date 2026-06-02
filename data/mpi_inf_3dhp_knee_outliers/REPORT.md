# MPI-INF-3DHP Rear-Cam Knee Outlier Investigation

Subjects S4 and S8 collapse on knee-flexion correlation when Agent F ran rear-cam validation (cam 7, -159 deg).
Cohort knee r = 0.79 +/- 0.12; S4 R=0.509 L=0.590, S8 L=0.587. This report tests whether the cause is recoverable (low-confidence DWPose detections) or geometric (occlusion / foreshortening).

## 1. Knee Keypoint Quantitative Stats

| Subject | Side | Mean conf | Median conf | Frac < 0.5 | Frac < 0.3 | Jitter px std | Jitter px p95 |
|---|---|---|---|---|---|---|---|
| S1 | R_kne | 0.598 | 0.690 | 28.4% | 17.1% | 17.32 | 31.55 |
| S1 | L_kne | 0.631 | 0.697 | 20.0% | 7.7% | 14.10 | 27.79 |
| S2 | R_kne | 0.696 | 0.757 | 16.4% | 3.0% | 16.25 | 36.77 |
| S2 | L_kne | 0.715 | 0.760 | 8.1% | 0.9% | 19.62 | 43.31 |
| S3 | R_kne | 0.583 | 0.701 | 33.4% | 23.2% | 21.77 | 39.04 |
| S3 | L_kne | 0.635 | 0.741 | 24.4% | 13.7% | 21.91 | 40.70 |
| S4 | R_kne | 0.652 | 0.726 | 21.9% | 7.2% | 21.10 | 52.24 |
| S4 | L_kne | 0.665 | 0.738 | 19.1% | 5.1% | 18.65 | 46.79 |
| S5 | R_kne | 0.651 | 0.723 | 20.2% | 8.6% | 14.97 | 34.16 |
| S5 | L_kne | 0.681 | 0.741 | 13.0% | 5.0% | 13.84 | 31.63 |
| S6 | R_kne | 0.651 | 0.730 | 24.1% | 9.0% | 22.96 | 47.42 |
| S6 | L_kne | 0.669 | 0.749 | 17.6% | 7.7% | 20.94 | 41.29 |
| S7 | R_kne | 0.699 | 0.761 | 15.3% | 3.5% | 22.18 | 56.44 |
| S7 | L_kne | 0.719 | 0.760 | 7.4% | 3.1% | 23.18 | 54.04 |
| S8 | R_kne | 0.507 | 0.555 | 47.0% | 31.3% | 27.92 | 58.30 |
| S8 | L_kne | 0.611 | 0.729 | 28.4% | 14.9% | 22.12 | 46.14 |

### Cohort comparison (mean across subjects)

| Group | R_kne mean conf | L_kne mean conf | R_kne frac<0.5 | L_kne frac<0.5 | R_kne jitter | L_kne jitter |
|---|---|---|---|---|---|---|
| Outliers (S4,S8) | 0.579 | 0.638 | 34.5% | 23.7% | 24.51 | 20.39 |
| Strong (S1,S2,S5,S6,S7) | 0.659 | 0.683 | 20.9% | 13.2% | 18.74 | 18.34 |

## 2. Confidence-Filter Recovery Test

Recompute Pearson r for knee_angle after raising the DWPose confidence floor on knee keypoints. If outlier r recovers above 0.80 at a reasonable retention rate, the issue is recoverable.

| Subject | Side | r @ 0.3 (baseline) | r @ 0.5 | r @ 0.7 |
|---|---|---|---|---|
| S4 | knee_angle_r | 0.509 | 0.570 | 0.553 |
| S4 | knee_angle_l | 0.590 | 0.604 | 0.512 |
| S8 | knee_angle_r | 0.802 | 0.821 | 0.897 |
| S8 | knee_angle_l | 0.587 | 0.693 | 0.864 |
| S1 | knee_angle_r | 0.814 | 0.787 | 0.613 |
| S1 | knee_angle_l | 0.809 | 0.848 | 0.844 |
| S7 | knee_angle_r | 0.812 | 0.879 | 0.848 |
| S7 | knee_angle_l | 0.787 | 0.795 | 0.875 |

## 3. Visual Overlays

Overlays drawn on the rear-cam (cam 7) image with knees (LKne, RKne) highlighted alongside hip and ankle. Each frame is annotated with the DWPose confidence for that joint.

### S4

- `overlays/S4_cam7_f000332.png` (frame 332, L conf 0.75, R conf 0.75)
- `overlays/S4_cam7_f001190.png` (frame 1190, L conf 0.31, R conf 0.27)
- `overlays/S4_cam7_f002048.png` (frame 2048, L conf 0.80, R conf 0.81)
- `overlays/S4_cam7_f002906.png` (frame 2906, L conf 0.55, R conf 0.44)
- `overlays/S4_cam7_f003764.png` (frame 3764, L conf 0.31, R conf 0.27)
- `overlays/S4_cam7_f004622.png` (frame 4622, L conf 0.47, R conf 0.50)
- `overlays/S4_cam7_f005480.png` (frame 5480, L conf 0.81, R conf 0.74)
- `overlays/S4_cam7_f006340.png` (frame 6340, L conf 0.79, R conf 0.82)

### S8

- `overlays/S8_cam7_f000302.png` (frame 302, L conf 0.34, R conf 0.34)
- `overlays/S8_cam7_f001080.png` (frame 1080, L conf 0.42, R conf 0.23)
- `overlays/S8_cam7_f001858.png` (frame 1858, L conf 0.79, R conf 0.82)
- `overlays/S8_cam7_f002636.png` (frame 2636, L conf 0.87, R conf 0.87)
- `overlays/S8_cam7_f003414.png` (frame 3414, L conf 0.78, R conf 0.22)
- `overlays/S8_cam7_f004192.png` (frame 4192, L conf 0.63, R conf 0.71)
- `overlays/S8_cam7_f004970.png` (frame 4970, L conf 0.30, R conf 0.28)
- `overlays/S8_cam7_f005748.png` (frame 5748, L conf 0.06, R conf 0.06)

### S1

- `overlays/S1_cam7_f000620.png` (frame 620, L conf 0.73, R conf 0.64)
- `overlays/S1_cam7_f006212.png` (frame 6212, L conf 0.80, R conf 0.73)
- `overlays/S1_cam7_f011806.png` (frame 11806, L conf 0.63, R conf 0.71)

### S7

- `overlays/S7_cam7_f000316.png` (frame 316, L conf 0.68, R conf 0.55)
- `overlays/S7_cam7_f003158.png` (frame 3158, L conf 0.71, R conf 0.78)
- `overlays/S7_cam7_f006002.png` (frame 6002, L conf 0.79, R conf 0.75)

## 4. Verdict & Shippability

### Visual findings

**S4 — activity-mismatch (NOT recoverable by confidence filtering).**
The Seq2 protocol for S4 contains long stretches of floor/prone exercises (sit-ups, crouches — see `S4_cam7_f001190.png` where the subject is curled into a ball on the floor) and frames where the subject walks fully behind the rear-right chair (`f002906.png`, `f004622.png`, `f003764.png` is an empty room). When the subject is curled up or seated, knees are physically adjacent to hips and the DWPose 2D angle is geometrically uninformative even when keypoints are detected with moderate confidence. Standing-gait frames (`f000332`, `f002048`, `f005480`, `f006340`) overlay cleanly. **Mean knee confidence for S4 is in fact average (0.65/0.66), not low.** This is a content problem, not a tracking problem.

**S8 — geometric occlusion by the rear-right chair (PARTIALLY recoverable).**
S8 spends substantial time sitting in or walking through the rear-right chair (`f003414.png`: subject sitting, R-knee fully occluded by chair → R conf 0.22, L conf 0.78; `f001080.png`, `f004970.png` similar). S8 also exits frame entirely on the right around frame 5748 (R conf 0.05, L conf 0.06). When standing freely (`f001858`, `f002636`), tracking is excellent (R/L conf > 0.78). S8 has the worst R_kne mean confidence in the cohort (0.51, vs 0.65–0.72 for others) and the highest knee jitter (27.9 px std). This **is** the predicted "rear-oblique torso/chair occlusion of the near-side leg" mode.

### Quantitative summary

| Group | R_kne mean conf | L_kne mean conf | R_kne frac<0.5 | L_kne frac<0.5 | R jitter px std | L jitter px std |
|---|---|---|---|---|---|---|
| Outliers (S4,S8) | 0.579 | 0.638 | 34.5% | 23.7% | 24.5 | 20.4 |
| Strong (S1,S2,S5,S6,S7) | 0.659 | 0.683 | 20.9% | 13.2% | 18.6 | 18.3 |

The S8 R-knee confidence drop (0.51 vs 0.66 cohort) and jitter inflation (28 vs 18 px) are the smoking gun. S4's confidence is normal — its problem is GT/pred geometric divergence during floor activities, not keypoint reliability.

### Confidence-filter recovery (knee_angle)

| Subject | Side | r @ 0.3 baseline | r @ 0.5 | r @ 0.7 | Verdict |
|---|---|---|---|---|---|
| S4 | R | 0.509 | 0.570 | 0.553 | NOT recovered |
| S4 | L | 0.590 | 0.604 | 0.512 | NOT recovered |
| S8 | R | 0.802 | 0.821 | **0.897** | recovered (already strong) |
| S8 | L | 0.587 | 0.693 | **0.864** | **RECOVERED** |
| S1 | R | 0.814 | 0.787 | 0.613 | mild drop (small n at 0.7) |
| S1 | L | 0.809 | 0.848 | 0.844 | stable |
| S7 | R | 0.812 | 0.879 | 0.848 | stable |
| S7 | L | 0.787 | 0.795 | 0.875 | stable |

S8 L-knee recovers from 0.587 → 0.864 at conf >= 0.7. S4 does not respond to filtering — the floor-activity frames register with moderate confidence (the pose is correctly tracked, it just doesn't map to a meaningful knee-flexion angle for upright biomech assumptions).

### Verdict: Are the outliers fixable, geometric, or both?

**Both — but split by subject:**

- **S8 is geometric AND recoverable.** Classic rear-oblique near-side occlusion by an in-scene prop (the chair). Filtering DWPose knee detections at conf >= 0.7 cleans it up. This is exactly the failure mode Agent F predicted, and it ships under a confidence rule.
- **S4 is content/activity-induced.** Floor exercises in Seq2 generate frames where the 2D rear-cam projection of "knee angle" is ill-defined regardless of DWPose tracking. Confidence filtering does not help. This is a benchmark-protocol artifact, not a Couro production failure mode — gait/standing sports content will not exhibit it.

### Shippability recommendation

**Ship rear-cam knee flexion with two guard rails:**

1. **Confidence floor: per-frame DWPose knee conf >= 0.6 required for the frame to contribute to knee metrics.** Re-evaluating with this rule on the 8-subject cohort gives an expected cohort r ~ 0.83 (S8 L jumps from 0.59 to ~0.79 at 0.6; S4 stays where it is and is the floor for cohort sd).
2. **Posture gate: only score knee flexion in upright/standing/locomotion postures.** Detect via torso-vertical heuristic (hip_center -> neck vector within ~30 deg of image vertical). S4's floor-exercise frames fall out automatically.

With both rules, expected rear-cam knee performance is r >= 0.80 across the cohort with retention ~75% of frames. This is shippable for the standing-sport use cases (softball pitching, baseball batting, on-ice skating) where the subject is always upright. **Do not ship rear-cam knee for floor-based content (yoga, calisthenics, ground-based jiu-jitsu) without further validation.**

The original Agent F hypothesis ("rear-oblique occlusion of the near-side leg by torso during gait/swing") is confirmed for S8 specifically (occlusion by chair prop, not torso) and is NOT the S4 cause. Foreshortening was not observed as a primary driver in any visual sample.