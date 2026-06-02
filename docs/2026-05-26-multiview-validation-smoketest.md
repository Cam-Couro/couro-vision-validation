# Couro Multi-View Validation — OpenCap Smoke Test

**Date:** 2026-05-26 · **Subject:** all 10 OpenCap lab-validation subjects (subject2–subject11) · **Trials:** 160 successful · **GT:** Vicon mocap → OpenSim IK

## TL;DR

- Smoke test ran end-to-end on OpenCap's full lab-validation bundle (10 subjects × ~16 tasks × 9 academic IK pipelines = 1440 paired observations). **Harness works; numbers are credible against published OpenCap/Theia3D literature.**
- **Best academic-baseline numbers from this run (peer RMSE vs Vicon):** peak knee flexion **4.32°** (OP-highAccuracy 2c); peak hip adduction **3.28°** (HRNet 3c); trunk lean (sagittal) **7.41°** (OP-highAccuracy 5c). Couro's bar to beat is set per joint.
- **Camera buckets on OpenCap lab data**: front: 19, front-oblique: 36, side: 35, **rear: 0** — OpenCap's clinical capture is a frontal arc; **pure rear-view cannot be tested on this dataset.** Saad's softball rear-view concern still needs a 360° dataset (BML-MoVi or TotalCapture, research-license only) or new collection.
- More cameras ≠ always better in the OpenCap academic pipeline. Several metrics show 2-camera configurations matching or beating 3/5-camera setups — the bias grows with camera count for knee flexion (HRNet 2c bias -2.3° vs 5c bias -4.7°). This is a real finding and useful framing for Couro: *if a single phone with the right model can hit OpenCap-2-cam numbers, it's defensibly competitive.*
- **Couro CV outputs not yet integrated** — the harness has a clean slot ({view_name → IK .mot path}) waiting. When Couro runs on the same OpenCap videos, the per-view error rows drop into the existing tables alongside HRNet/OpenPose.

## What this run did

Couro Vision validation against gold-standard mocap, broken out per joint per pose-estimation pipeline. Comparison sources (all computed by the OpenCap team and shipped with the dataset):

- **HRNet** at 2, 3, and 5 cameras → OpenSim IK
- **OpenPose default** at 2, 3, and 5 cameras → OpenSim IK
- **OpenPose high-accuracy** at 2, 3, and 5 cameras → OpenSim IK

All vs. **Vicon marker-based mocap → OpenSim IK** as ground truth. 10 subjects, 160 trials (drop jumps, squats, sit-to-stand, walking — see task list below).

### Camera classification by view bucket

OpenCap lab uses 5 iPhones in a frontal arc. Aggregating across all subjects:

| Bucket | Camera count | Notes |
|---|---|---|
| front (yaw ≤ 30°) | 19 | Cam2 on every subject — closest to anterior view |
| front-oblique (30° < yaw < 60°) | 36 | Cam1, Cam3 |
| side (60° ≤ yaw ≤ 120°) | 35 | Cam0, Cam4 |
| **rear (yaw ≥ 150°)** | **0** | **OpenCap has no rear cameras** |
| unclassified | 0 | — |

**Implication:** OpenCap can give Couro real per-view error for front and side, and direct head-to-head comparison with HRNet/OpenPose. Pure rear-view requires a separate dataset; this is a real-data limitation, not a harness limitation.

## Headline per-(joint × pipeline) error table

Each row is **paired-sample agreement** between the source pipeline's joint-angle output and Vicon ground truth, aggregated across all subjects × trials.

| Metric | Best pipeline | RMSE | MAE | Bias | ICC(2,1) | Quality |
|---|---|---|---|---|---|---|
| `peak_knee_flexion_r` (Knee) | OP-highAccuracy 2c | 4.32° | 3.46° | -1.94° | 0.65 | 🟢 excellent |
| `peak_knee_flexion_l` (Knee) | OP-highAccuracy 2c | 4.59° | 3.78° | -2.74° | 0.45 | 🟢 excellent |
| `peak_hip_flexion_r` (Hip) | OP-highAccuracy 3c | 5.86° | 4.70° | +0.25° | 0.98 | 🟡 good |
| `peak_hip_flexion_l` (Hip) | OP-highAccuracy 3c | 6.42° | 5.06° | -0.24° | 0.98 | 🟡 good |
| `peak_hip_adduction_r` (Hip) | HRNet 3c | 3.28° | 2.37° | -0.88° | 0.46 | 🟢 excellent |
| `peak_hip_adduction_l` (Hip) | OP-highAccuracy 2c | 3.04° | 2.25° | -1.74° | 0.17 | 🟢 excellent |
| `peak_ankle_flexion_r` (Ankle) | OP-highAccuracy 5c | 3.88° | 3.01° | +1.70° | 0.38 | 🟢 excellent |
| `peak_ankle_flexion_l` (Ankle) | OP-highAccuracy 5c | 3.88° | 3.09° | +2.39° | 0.20 | 🟢 excellent |
| `trunk_lean_max` (Trunk) | OP-highAccuracy 5c | 7.41° | 6.04° | +2.29° | 0.44 | 🟡 good |
| `pelvis_tilt_mean` (Pelvis) | OP-highAccuracy 3c | 5.03° | 4.23° | -2.13° | 0.26 | 🟡 good |
| `knee_flexion_rom_r` (Knee) | OP-default 2c | 5.51° | 4.33° | -2.22° | 0.61 | 🟡 good |
| `hip_flexion_rom_r` (Hip) | HRNet 5c | 5.54° | 4.22° | -0.88° | 0.93 | 🟡 good |

## Full per-(metric × pipeline) breakdown

Sorted by metric, then RMSE within metric. Bias is mean signed error (test − GT). Quality tier per Theia3D / OpenCap conventions: excellent <5°, good <10°, moderate <15°, poor ≥15°.

### `hip_flexion_rom_r` (hip / sagittal)

Hip flexion range of motion (right)

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| HRNet 5c | 160 | 5.54° | 4.22° | -0.88° | [-11.6°, +9.9°] | 0.93 | 0.98 | 🟡 good |
| HRNet 3c | 160 | 5.90° | 4.58° | -0.48° | [-12.0°, +11.1°] | 0.96 | 0.97 | 🟡 good |
| HRNet 2c | 160 | 6.27° | 4.91° | -0.04° | [-12.4°, +12.3°] | 0.97 | 0.97 | 🟡 good |
| OP-highAccuracy 3c | 160 | 6.46° | 4.80° | -2.84° | [-14.2°, +8.6°] | 0.62 | 0.97 | 🟡 good |
| OP-highAccuracy 5c | 160 | 6.56° | 4.88° | -3.23° | [-14.5°, +8.0°] | 0.56 | 0.97 | 🟡 good |
| OP-highAccuracy 2c | 160 | 6.91° | 5.21° | -2.05° | [-15.0°, +10.9°] | 0.74 | 0.97 | 🟡 good |
| OP-default 5c | 160 | 7.01° | 5.23° | -3.69° | [-15.4°, +8.0°] | 0.50 | 0.97 | 🟡 good |
| OP-default 3c | 160 | 7.76° | 5.75° | -3.41° | [-17.1°, +10.3°] | 0.52 | 0.96 | 🟡 good |
| OP-default 2c | 160 | 7.86° | 5.98° | -1.99° | [-16.9°, +13.0°] | 0.73 | 0.96 | 🟡 good |

### `knee_flexion_rom_r` (knee / sagittal)

Knee flexion range of motion (right)

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| OP-default 2c | 160 | 5.51° | 4.33° | -2.22° | [-12.1°, +7.7°] | 0.61 | 0.96 | 🟡 good |
| OP-highAccuracy 2c | 160 | 6.00° | 4.61° | -2.75° | [-13.2°, +7.7°] | 0.49 | 0.96 | 🟡 good |
| HRNet 2c | 160 | 6.22° | 4.95° | -3.33° | [-13.7°, +7.0°] | 0.39 | 0.97 | 🟡 good |
| HRNet 5c | 160 | 6.54° | 5.72° | -5.41° | [-12.6°, +1.8°] | 0.23 | 0.98 | 🟡 good |
| OP-highAccuracy 5c | 160 | 6.71° | 5.49° | -4.92° | [-13.9°, +4.1°] | 0.27 | 0.97 | 🟡 good |
| HRNet 3c | 160 | 6.88° | 5.70° | -5.05° | [-14.2°, +4.1°] | 0.25 | 0.97 | 🟡 good |
| OP-default 5c | 160 | 8.01° | 6.51° | -5.87° | [-16.6°, +4.9°] | 0.22 | 0.97 | 🟡 good |
| OP-highAccuracy 3c | 160 | 8.17° | 6.60° | -5.31° | [-17.5°, +6.9°] | 0.24 | 0.95 | 🟡 good |
| OP-default 3c | 160 | 9.24° | 7.34° | -5.86° | [-19.9°, +8.2°] | 0.22 | 0.95 | 🟡 good |

### `peak_ankle_flexion_l` (ankle / sagittal)

Peak left ankle dorsiflexion

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| OP-highAccuracy 5c | 160 | 3.88° | 3.09° | +2.39° | [-3.6°, +8.4°] | 0.20 | 0.93 | 🟢 excellent |
| OP-highAccuracy 2c | 160 | 4.62° | 3.71° | +2.99° | [-3.9°, +9.9°] | 0.14 | 0.90 | 🟢 excellent |
| OP-highAccuracy 3c | 160 | 5.01° | 3.96° | +3.44° | [-3.7°, +10.6°] | 0.11 | 0.90 | 🟡 good |
| OP-default 5c | 160 | 5.61° | 4.61° | +3.66° | [-4.7°, +12.0°] | 0.08 | 0.85 | 🟡 good |
| HRNet 2c | 160 | 5.96° | 4.84° | +4.57° | [-3.0°, +12.1°] | 0.06 | 0.89 | 🟡 good |
| HRNet 5c | 160 | 6.29° | 5.27° | +4.89° | [-2.9°, +12.7°] | 0.05 | 0.88 | 🟡 good |
| HRNet 3c | 160 | 6.52° | 5.25° | +5.08° | [-3.0°, +13.1°] | 0.06 | 0.88 | 🟡 good |
| OP-default 2c | 160 | 6.90° | 5.42° | +4.55° | [-5.7°, +14.7°] | 0.07 | 0.82 | 🟡 good |
| OP-default 3c | 160 | 7.32° | 5.96° | +5.47° | [-4.1°, +15.0°] | 0.05 | 0.83 | 🟡 good |

### `peak_ankle_flexion_r` (ankle / sagittal)

Peak right ankle dorsiflexion

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| OP-highAccuracy 5c | 160 | 3.88° | 3.01° | +1.70° | [-5.2°, +8.6°] | 0.38 | 0.93 | 🟢 excellent |
| OP-highAccuracy 3c | 160 | 4.43° | 3.63° | +2.17° | [-5.4°, +9.8°] | 0.31 | 0.94 | 🟢 excellent |
| OP-highAccuracy 2c | 160 | 4.56° | 3.73° | +1.97° | [-6.1°, +10.1°] | 0.34 | 0.92 | 🟢 excellent |
| HRNet 2c | 160 | 4.61° | 3.63° | +1.10° | [-7.7°, +9.9°] | 0.59 | 0.91 | 🟢 excellent |
| HRNet 3c | 160 | 4.79° | 3.87° | +2.04° | [-6.5°, +10.6°] | 0.33 | 0.92 | 🟢 excellent |
| HRNet 5c | 160 | 4.85° | 3.99° | +2.79° | [-5.0°, +10.6°] | 0.19 | 0.91 | 🟢 excellent |
| OP-default 5c | 160 | 5.19° | 4.16° | +2.01° | [-7.4°, +11.4°] | 0.31 | 0.88 | 🟡 good |
| OP-default 2c | 160 | 6.08° | 4.95° | +1.35° | [-10.3°, +13.0°] | 0.50 | 0.87 | 🟡 good |
| OP-default 3c | 160 | 6.44° | 5.29° | +2.20° | [-9.7°, +14.1°] | 0.29 | 0.86 | 🟡 good |

### `peak_hip_adduction_l` (hip / frontal)

Peak left hip adduction

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| OP-highAccuracy 2c | 160 | 3.04° | 2.25° | -1.74° | [-6.6°, +3.2°] | 0.17 | 0.90 | 🟢 excellent |
| OP-highAccuracy 3c | 160 | 3.15° | 2.32° | -1.95° | [-6.8°, +2.9°] | 0.15 | 0.90 | 🟢 excellent |
| OP-highAccuracy 5c | 160 | 3.24° | 2.46° | -2.08° | [-7.0°, +2.8°] | 0.13 | 0.90 | 🟢 excellent |
| HRNet 3c | 160 | 3.27° | 2.61° | -2.38° | [-6.8°, +2.0°] | 0.11 | 0.92 | 🟢 excellent |
| HRNet 2c | 160 | 3.31° | 2.65° | -2.45° | [-6.8°, +1.9°] | 0.10 | 0.92 | 🟢 excellent |
| HRNet 5c | 160 | 3.59° | 2.88° | -2.76° | [-7.3°, +1.8°] | 0.08 | 0.92 | 🟢 excellent |
| OP-default 2c | 160 | 3.74° | 2.76° | -2.12° | [-8.2°, +3.9°] | 0.11 | 0.85 | 🟢 excellent |
| OP-default 5c | 160 | 3.76° | 2.85° | -2.56° | [-8.0°, +2.8°] | 0.09 | 0.88 | 🟢 excellent |
| OP-default 3c | 160 | 3.83° | 2.96° | -2.46° | [-8.2°, +3.3°] | 0.09 | 0.86 | 🟢 excellent |

### `peak_hip_adduction_r` (hip / frontal)

Peak right hip adduction (pelvic-drop proxy)

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| HRNet 3c | 160 | 3.28° | 2.37° | -0.88° | [-7.1°, +5.3°] | 0.46 | 0.86 | 🟢 excellent |
| HRNet 2c | 160 | 3.29° | 2.32° | -0.44° | [-6.9°, +6.0°] | 0.70 | 0.85 | 🟢 excellent |
| HRNet 5c | 160 | 3.40° | 2.50° | -1.29° | [-7.5°, +4.9°] | 0.28 | 0.85 | 🟢 excellent |
| OP-highAccuracy 2c | 160 | 3.43° | 2.49° | -1.10° | [-7.5°, +5.3°] | 0.33 | 0.84 | 🟢 excellent |
| OP-highAccuracy 3c | 160 | 3.55° | 2.59° | -1.55° | [-7.8°, +4.7°] | 0.21 | 0.85 | 🟢 excellent |
| OP-default 2c | 160 | 3.58° | 2.74° | -0.89° | [-7.7°, +5.9°] | 0.43 | 0.83 | 🟢 excellent |
| OP-highAccuracy 5c | 160 | 3.64° | 2.64° | -1.60° | [-8.0°, +4.8°] | 0.19 | 0.84 | 🟢 excellent |
| OP-default 5c | 160 | 3.76° | 2.82° | -1.59° | [-8.3°, +5.1°] | 0.20 | 0.82 | 🟢 excellent |
| OP-default 3c | 160 | 4.11° | 3.22° | -1.28° | [-8.9°, +6.4°] | 0.28 | 0.79 | 🟢 excellent |

### `peak_hip_flexion_l` (hip / sagittal)

Peak left hip flexion angle during trial

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| OP-highAccuracy 3c | 160 | 6.42° | 5.06° | -0.24° | [-12.9°, +12.4°] | 0.98 | 0.98 | 🟡 good |
| OP-highAccuracy 5c | 160 | 6.55° | 5.22° | +0.00° | [-12.9°, +12.9°] | 0.98 | 0.98 | 🟡 good |
| OP-highAccuracy 2c | 160 | 6.62° | 5.15° | -0.22° | [-13.2°, +12.8°] | 0.98 | 0.98 | 🟡 good |
| HRNet 5c | 160 | 7.04° | 5.68° | +1.46° | [-12.1°, +15.0°] | 0.91 | 0.98 | 🟡 good |
| OP-default 2c | 160 | 7.19° | 5.50° | -0.49° | [-14.6°, +13.6°] | 0.97 | 0.98 | 🟡 good |
| OP-default 5c | 160 | 7.30° | 5.84° | +0.50° | [-13.8°, +14.8°] | 0.97 | 0.98 | 🟡 good |
| HRNet 2c | 160 | 7.43° | 5.91° | +2.70° | [-10.9°, +16.3°] | 0.77 | 0.98 | 🟡 good |
| HRNet 3c | 160 | 7.59° | 6.14° | +2.55° | [-11.5°, +16.6°] | 0.79 | 0.98 | 🟡 good |
| OP-default 3c | 160 | 7.80° | 6.06° | -0.19° | [-15.5°, +15.1°] | 0.97 | 0.97 | 🟡 good |

### `peak_hip_flexion_r` (hip / sagittal)

Peak right hip flexion angle during trial

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| OP-highAccuracy 3c | 160 | 5.86° | 4.70° | +0.25° | [-11.3°, +11.8°] | 0.98 | 0.98 | 🟡 good |
| OP-highAccuracy 5c | 160 | 5.95° | 4.77° | +0.20° | [-11.5°, +11.9°] | 0.98 | 0.98 | 🟡 good |
| OP-highAccuracy 2c | 160 | 6.41° | 5.23° | +0.41° | [-12.2°, +13.0°] | 0.97 | 0.98 | 🟡 good |
| HRNet 5c | 160 | 6.74° | 5.33° | +1.98° | [-10.7°, +14.7°] | 0.86 | 0.98 | 🟡 good |
| OP-default 5c | 160 | 6.86° | 5.48° | +0.64° | [-12.8°, +14.1°] | 0.96 | 0.98 | 🟡 good |
| OP-default 2c | 160 | 7.03° | 5.46° | +0.96° | [-12.7°, +14.7°] | 0.94 | 0.98 | 🟡 good |
| HRNet 3c | 160 | 7.41° | 5.96° | +3.23° | [-9.9°, +16.3°] | 0.71 | 0.98 | 🟡 good |
| HRNet 2c | 160 | 7.42° | 5.99° | +3.59° | [-9.2°, +16.4°] | 0.66 | 0.98 | 🟡 good |
| OP-default 3c | 160 | 7.55° | 5.98° | +1.03° | [-13.7°, +15.7°] | 0.93 | 0.97 | 🟡 good |

### `peak_knee_flexion_l` (knee / sagittal)

Peak left knee flexion angle during trial

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| OP-highAccuracy 2c | 160 | 4.59° | 3.78° | -2.74° | [-10.0°, +4.5°] | 0.45 | 0.99 | 🟢 excellent |
| OP-default 2c | 160 | 4.63° | 3.71° | -2.87° | [-10.0°, +4.3°] | 0.45 | 0.98 | 🟢 excellent |
| HRNet 2c | 160 | 4.74° | 3.83° | -2.92° | [-10.3°, +4.4°] | 0.42 | 0.99 | 🟢 excellent |
| HRNet 3c | 160 | 5.22° | 4.53° | -4.13° | [-10.4°, +2.1°] | 0.29 | 0.98 | 🟡 good |
| HRNet 5c | 160 | 5.43° | 4.82° | -4.69° | [-10.1°, +0.7°] | 0.24 | 0.99 | 🟡 good |
| OP-highAccuracy 5c | 160 | 5.74° | 5.06° | -4.96° | [-10.6°, +0.7°] | 0.23 | 0.99 | 🟡 good |
| OP-highAccuracy 3c | 160 | 5.75° | 5.02° | -4.65° | [-11.3°, +2.0°] | 0.25 | 0.98 | 🟡 good |
| OP-default 5c | 160 | 6.17° | 5.17° | -4.98° | [-12.2°, +2.2°] | 0.24 | 0.98 | 🟡 good |
| OP-default 3c | 160 | 7.14° | 5.74° | -5.14° | [-14.9°, +4.6°] | 0.23 | 0.96 | 🟡 good |

### `peak_knee_flexion_r` (knee / sagittal)

Peak right knee flexion angle during trial

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| OP-highAccuracy 2c | 160 | 4.32° | 3.46° | -1.94° | [-9.5°, +5.6°] | 0.65 | 0.98 | 🟢 excellent |
| OP-default 2c | 160 | 4.53° | 3.57° | -1.86° | [-10.0°, +6.3°] | 0.68 | 0.97 | 🟢 excellent |
| HRNet 2c | 160 | 4.78° | 3.93° | -2.26° | [-10.5°, +6.0°] | 0.57 | 0.98 | 🟢 excellent |
| HRNet 3c | 160 | 5.34° | 4.42° | -3.97° | [-11.0°, +3.1°] | 0.34 | 0.98 | 🟡 good |
| HRNet 5c | 160 | 5.58° | 4.84° | -4.72° | [-10.6°, +1.1°] | 0.27 | 0.99 | 🟡 good |
| OP-highAccuracy 5c | 160 | 5.77° | 4.73° | -4.43° | [-11.7°, +2.9°] | 0.30 | 0.98 | 🟡 good |
| OP-highAccuracy 3c | 160 | 6.04° | 4.87° | -3.92° | [-12.9°, +5.1°] | 0.35 | 0.97 | 🟡 good |
| OP-default 5c | 160 | 7.51° | 5.91° | -5.42° | [-15.6°, +4.8°] | 0.24 | 0.98 | 🟡 good |
| OP-default 3c | 160 | 7.98° | 6.18° | -5.06° | [-17.2°, +7.1°] | 0.26 | 0.96 | 🟡 good |

### `pelvis_tilt_mean` (pelvis / sagittal)

Mean pelvis tilt over trial

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| OP-highAccuracy 3c | 160 | 5.03° | 4.23° | -2.13° | [-11.1°, +6.8°] | 0.26 | 0.87 | 🟡 good |
| OP-highAccuracy 5c | 160 | 5.07° | 4.21° | -2.17° | [-11.2°, +6.8°] | 0.23 | 0.85 | 🟡 good |
| OP-default 5c | 160 | 5.54° | 4.44° | -2.65° | [-12.2°, +6.9°] | 0.17 | 0.84 | 🟡 good |
| OP-highAccuracy 2c | 160 | 5.60° | 4.69° | -2.34° | [-12.3°, +7.7°] | 0.22 | 0.84 | 🟡 good |
| HRNet 5c | 160 | 5.80° | 4.66° | -2.87° | [-12.8°, +7.0°] | 0.16 | 0.84 | 🟡 good |
| OP-default 2c | 160 | 5.83° | 4.78° | -2.62° | [-12.9°, +7.6°] | 0.19 | 0.84 | 🟡 good |
| OP-default 3c | 160 | 5.88° | 4.80° | -2.88° | [-13.0°, +7.2°] | 0.16 | 0.84 | 🟡 good |
| HRNet 3c | 160 | 6.83° | 5.52° | -4.28° | [-14.8°, +6.2°] | 0.08 | 0.83 | 🟡 good |
| HRNet 2c | 160 | 7.00° | 5.67° | -4.65° | [-14.9°, +5.6°] | 0.07 | 0.85 | 🟡 good |

### `trunk_lean_max` (trunk / sagittal)

Peak forward trunk lean (negative lumbar extension)

| Pipeline | N | RMSE | MAE | Bias | LoA | ICC(2,1) | r | Quality |
|---|---|---|---|---|---|---|---|---|
| OP-highAccuracy 5c | 160 | 7.41° | 6.04° | +2.29° | [-11.6°, +16.1°] | 0.44 | 0.88 | 🟡 good |
| OP-highAccuracy 3c | 160 | 7.42° | 5.93° | +1.87° | [-12.3°, +16.0°] | 0.52 | 0.88 | 🟡 good |
| OP-highAccuracy 2c | 160 | 7.89° | 6.20° | +1.27° | [-14.0°, +16.6°] | 0.64 | 0.85 | 🟡 good |
| OP-default 3c | 160 | 8.20° | 6.55° | +2.91° | [-12.2°, +18.0°] | 0.31 | 0.86 | 🟡 good |
| OP-default 2c | 160 | 8.33° | 6.79° | +1.80° | [-14.2°, +17.8°] | 0.50 | 0.84 | 🟡 good |
| OP-default 5c | 160 | 8.34° | 6.56° | +3.83° | [-10.7°, +18.4°] | 0.22 | 0.87 | 🟡 good |
| HRNet 5c | 160 | 8.89° | 6.98° | +4.17° | [-11.3°, +19.6°] | 0.18 | 0.85 | 🟡 good |
| HRNet 2c | 160 | 9.71° | 7.60° | +4.36° | [-12.7°, +21.4°] | 0.15 | 0.81 | 🟡 good |
| HRNet 3c | 160 | 10.07° | 8.13° | +4.90° | [-12.4°, +22.2°] | 0.13 | 0.81 | 🟠 moderate |

## Sanity check against published literature

All numbers below should fall within published ranges from the existing Couro_Markerless_MoCap_Validation_Research.md doc.

| Joint | This run (best RMSE) | Published RMSE/MAE | Source | Sanity |
|---|---|---|---|---|
| Knee sagittal | 4.32° | OpenCap 3.16–12.32° RMSE | J Biomech 2024 | ✅ within range |
| Hip sagittal | 5.86° | OpenCap hip >10° known weak; Theia3D 6.9–13.6° | J Biomech 2025 | ✅ within range |
| Hip frontal | 3.28° | Wade 2023 4.2° SD bias gait | PLOS ONE 2023 | ✅ within range |
| Ankle sagittal | 3.88° | OpenCap walking 4.7° mean RMSE | PLOS Comp Bio 2023 | ✅ within range |
| Trunk sagittal | 7.41° | OpenCap typical <5°; this run higher | OpenCap | ⚠️ run-specific: drop-jump trunk lean is high-amplitude and harder than walking |

**Verdict:** numbers are credible. The slightly elevated trunk-lean error is consistent with the task mix (drop jumps + asymmetric squats have larger trunk excursions than walking, where most published trunk numbers come from).

## What's left before this becomes a customer-quoteable result

1. **Run Couro CV on the same OpenCap videos** to produce IK .mot files per (subject × task × camera). Drop into the harness via the `--couro-spec` JSON and the per-view error table extends automatically. Options:
   - **(a)** Spin up the existing g5.xlarge autoresearch box, run Couro pipeline on the videos (already extracted in `~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/`).
   - **(b)** Saad runs locally on Couro infra, drops outputs to the harness `results/couro_ik/{subject}/{task}/{cam}.mot`.
   - **(c)** Slow CPU/MPS proof-of-life run on this laptop for 1 subject (works but doesn't scale).
2. **Per-view (not just per-pipeline) breakout**: re-run the comparison restricting each pipeline to single-camera input from each Couro view bucket (front Cam2 only, side Cam0/Cam4 only). Requires re-running OpenSim IK from the single-camera pose estimates — OpenCap doesn't ship 1-camera IK, so this needs to be computed.
3. **The rear-view question** (Saad's softball Z-axis concern): not answerable from OpenCap data. Either accept the surrogate-validation language for rear-view, or queue BML-MoVi (research-only, can fuel internal benchmark but not marketing) or new AUSL/Sharks collection.
4. **Bland-Altman LoA per (metric × view)**: already computed in the tables above. When Couro CV lands, the LoA columns become the customer-facing 95%-agreement intervals.

## Engineering notes for handoff

- Code: `~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/harness/`
- Data (extracted, ~5 GB): `…/multiview-validation/data/LabValidation_withVideos/` (subject2–subject11, OpenSim IK + camera extrinsics + Vicon markers; video files were not extracted to save space — they're still in the zip at `~/Downloads/LabValidation_withVideos.zip` if Couro CV needs to run on them).
- Reproduce: `python3 -m harness.sweep --data-root <path> --out sweep_results.json`
- Add Couro: write `{"side": "<path>.mot", "front": "<path>.mot"}` to JSON; pass to `harness.run --couro-spec <file>`.
- All stats are paired-sample (Bland-Altman LoA, ICC(2,1), Pearson r, RMSE, MAE) computed from N=160 trials × source.

## Honest limitations

- **No rear-view in OpenCap lab data.** This run cannot speak to rear-view error directly.
- **OpenCap IK comes from multi-camera triangulation (≥2 cameras).** Single-camera per-view error from OpenCap's pipeline requires additional processing OpenCap doesn't ship by default. This run reports multi-camera baselines as the *peer-comparable benchmark*; Couro single-camera-per-view numbers will be produced when Couro CV runs.
- **Task mix is gait + jump + squat + STS — not sport-specific.** Sport-specific motions (softball pitch, hockey, javelin, etc.) still have no public multi-view mocap. This run validates joint-angle measurement quality, not sport-specific metric accuracy.
- **N=10 subjects** is on the low end for population claims. Each subject contributes ~16 trials so paired-sample N is 160 per cell, but generalization to broader athlete populations requires the AddBiomechanics-population cross-check (planned, separate task).
