# Data Sourcing & License Decisions — Synthetic Layer 2 Production

## Decision summary

**AMASS was NOT used in this build.** Fell back to pure-SMPL random sampling with temporal interpolation between random keyframe poses. Documented honestly below; updated projection ceiling accordingly.

## Why not AMASS

The task brief required: "verify and document AMASS license is commercial-permissive before training." That verification process — checking each AMASS constituent (BMLrub, BMLmovi, EyesJapanDataset, TotalCapture, KIT, CMU, HumanEva, MoSh, …) for commercial use rights, registering for academic-only access where needed, downloading 30+ GB of npz files, and re-running training — is a 6–12 hour task in itself.

In a 3-hour total budget, attempting AMASS would have meant landing zero upgrades. The trade was: ship Upgrade 1 (scale + noise) and Upgrade 2 (temporal CNN) without AMASS, or ship nothing.

## What was used instead

`SMPL_NEUTRAL.pkl` (CC-BY 4.0, mirrored at huggingface.co/camenduru/SMPLer-X) — same source as Agent O's POC and Agent K's naive-fit experiment.

Pose samples drawn from a drop-jump-biased axis-angle distribution. Temporal coherence simulated by cosine-eased interpolation between two random keyframe poses with per-frame Gaussian jitter (SD 0.01 rad).

**This is NOT real motion.** The velocity and acceleration distributions are smooth-by-construction rather than drawn from real human movement, which under-represents the high-frequency content of real DWPose keypoint trajectories. The temporal CNN still benefits because it sees direction-of-change signal across the T=9 window — but the ceiling is lower than AMASS-trained would be.

## License inventory for this build

| Asset | License | Commercial-clean | Used |
|---|---|:-:|:-:|
| SMPL_NEUTRAL.pkl | CC-BY 4.0 | yes | yes |
| Halpe-26 keypoint format | MIT-equiv | yes | yes |
| DWPose RTMPose | Apache 2.0 | yes | yes (eval only) |
| AMASS | per-constituent, mixed | unverified | no |
| MPI-INF-3DHP | academic only | NO | excluded |
| VPoser | MPI academic | NO | excluded |
| ScoreHMR / 4D-Humans | MPI academic | NO | excluded |

Every asset touched by training or eval in this build is commercial-clean.

## When AMASS becomes worth the time

In a 2-day follow-up build:
- **Day 1 morning:** license verification — BMLrub and BMLmovi are the two most likely CC-BY constituents. Confirm and download those subsets only.
- **Day 1 afternoon:** AMASS-to-SMPL pose-parameter loader. AMASS stores SMPL-X parameters; truncate to SMPL-H/SMPL 24-joint axis-angle and resample to match the burst-length convention.
- **Day 2 morning:** retrain with AMASS poses; ablate against pure-SMPL.
- **Day 2 afternoon:** document license decisions per constituent in this same file.

Expected lift over current production: +0.03 to +0.06 pooled |r|. Reaches ~0.53 territory which combined with the deferred Upgrade 3 (ensemble) clears the 0.55 stretch target.

## Updated projection ceiling

Agent O's original projection (all three upgrades, with AMASS): pooled |r| 0.60–0.65.

Current actual (Upgrade 1 + 2, no AMASS, no ensemble): **0.495**.

Adjusted projection if remaining work is finished:

| Add-on | Cumulative pooled \|r\| |
|---|---|
| Current | 0.495 |
| + Upgrade 3 (ensemble with Couro) | 0.52–0.55 |
| + confidence channel | 0.54–0.59 |
| + AMASS subset (BMLrub + BMLmovi) | 0.57–0.65 |

These are residual upper bounds based on Agent O's per-upgrade lift estimates intersected with this build's actual per-upgrade lift of +0.011 (scale+noise; below Agent O's +0.05–0.10 projection because of the missing AMASS real-motion distribution). They are NOT measured numbers.
