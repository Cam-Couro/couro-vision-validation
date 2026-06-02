"""Layer 2 Synthetic-Data POC: train a small MLP on SMPL-rendered 2D
keypoints + 3D-derived joint angles, then evaluate on real OpenCap DWPose
clips and compare to Couro's anthropometric reconstruction.

Pipeline:
  1. Synthetic data generation
     - Load SMPL_NEUTRAL.pkl (joint-only FK, no LBS).
     - Sample N axis-angle poses with anatomically reasonable distributions
       (bias toward drop-jump-like crouches and squats).
     - For each sampled pose, place a virtual camera at a random
       (azimuth, elevation, distance) drawn from a coverage of side / rear /
       front 3/4 views (matches the OpenCap Cam0..Cam4 deployment envelope).
     - Forward-kinematic the SMPL 24-joint chain (Rodrigues + parent
       transforms).
     - Project to Halpe-26 pixels using the same HALPE26_TO_SMPL map as
       smpl_layer2_poc.py.
     - Compute the 5 Couro joint angles using smpl_joints_to_metrics from
       smpl_layer2_poc.py (so synthetic-GT and Couro-eval definitions match).
     - Save (kp_xy_normalized [52], angles_deg [5]) pairs.

  2. Train a small MLP per-frame regressor (52 -> 128 -> 64 -> 5).

  3. Evaluate on the same 8 OpenCap drop-jump clips used by smpl_layer2_poc.

Outputs:
  models/synthetic_layer2_v0.pt
  data/layer2_synthetic_poc/per_clip_r.json
  data/layer2_synthetic_poc/REPORT.md
"""

from __future__ import annotations

import builtins
import json
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

# numpy 2.x compatibility shims so chumpy-pickled SMPL_NEUTRAL.pkl loads.
for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from harness.couro_keypoints import load_couro_output, keypoints_to_motion_data
    from harness.parsers import parse_mot
    from harness.smpl_layer2_poc import (
        HALPE26_INDEX,
        HALPE26_TO_SMPL,
        SmplKinematics,
        axis_angle_to_rotmat,
        load_smpl_kinematics,
        smpl_forward,
        smpl_joints_to_metrics,
    )
else:
    from .couro_keypoints import load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .smpl_layer2_poc import (
        HALPE26_INDEX,
        HALPE26_TO_SMPL,
        SmplKinematics,
        axis_angle_to_rotmat,
        load_smpl_kinematics,
        smpl_forward,
        smpl_joints_to_metrics,
    )


SMPL_MODEL_PATH: Final[Path] = REPO_ROOT / "models" / "smpl" / "SMPL_NEUTRAL.pkl"
DATA_ROOT: Final[Path] = REPO_ROOT / "data"
LAB: Final[Path] = DATA_ROOT / "LabValidation_withVideos"
OPENCAP_KP: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"
OUT_DIR: Final[Path] = DATA_ROOT / "layer2_synthetic_poc"
MODEL_OUT: Final[Path] = REPO_ROOT / "models" / "synthetic_layer2_v0.pt"

DEPLOY_METRICS: Final[tuple[str, ...]] = (
    "hip_flexion_r",
    "hip_adduction_r",
    "knee_angle_r",
    "ankle_angle_r",
    "lumbar_extension",
)

# Subset of Halpe-26 names that have a SMPL target joint. Frozen so the MLP
# input vector ordering is deterministic.
HALPE_KEYS_WITH_SMPL: Final[tuple[str, ...]] = tuple(
    name for name, smpl_idx in HALPE26_TO_SMPL.items() if smpl_idx is not None
)
N_KEYS: Final[int] = len(HALPE_KEYS_WITH_SMPL)


# -------------------------------------------------------------------------
# 1. Synthetic data generator.
# -------------------------------------------------------------------------

# SMPL 24-joint indices used for biased sampling (drop-jump-like crouches).
SMPL_J = {
    "pelvis": 0, "L_hip": 1, "R_hip": 2, "spine1": 3,
    "L_knee": 4, "R_knee": 5, "spine2": 6, "L_ank": 7, "R_ank": 8,
    "spine3": 9, "L_foot": 10, "R_foot": 11, "neck": 12,
    "L_collar": 13, "R_collar": 14, "head": 15,
    "L_sho": 16, "R_sho": 17, "L_elb": 18, "R_elb": 19,
    "L_wri": 20, "R_wri": 21, "L_hand": 22, "R_hand": 23,
}


@dataclass(frozen=True)
class SyntheticSample:
    """One synthetic frame: 2D keypoints + GT angles + camera params."""

    kp_xy_norm: np.ndarray  # (N_KEYS, 2) normalized to [-1, 1]-ish
    angles_deg: np.ndarray  # (5,) hip_flexion_r, hip_adduction_r, knee, ankle, lumbar
    cam_yaw_deg: float


def _sample_pose() -> np.ndarray:
    """Return (24, 3) axis-angle pose, weighted toward drop-jump-like flexions.

    50%: crouch / drop-jump pose (knee + hip flexed)
    30%: mid-flexion (running / landing)
    20%: near-rest standing
    """
    pose = np.zeros((24, 3), dtype=np.float64)
    pose += np.random.normal(0.0, 0.05, size=(24, 3))  # small global jitter

    r = np.random.rand()
    if r < 0.5:
        # Crouched: hip flex ~30-90 deg, knee flex ~40-110 deg, ankle dorsiflex ~5-25 deg
        hip_flex = np.deg2rad(np.random.uniform(30.0, 90.0))
        knee_flex = np.deg2rad(np.random.uniform(40.0, 110.0))
        ank_dorsi = np.deg2rad(np.random.uniform(5.0, 25.0))
    elif r < 0.8:
        hip_flex = np.deg2rad(np.random.uniform(10.0, 45.0))
        knee_flex = np.deg2rad(np.random.uniform(15.0, 60.0))
        ank_dorsi = np.deg2rad(np.random.uniform(-5.0, 15.0))
    else:
        hip_flex = np.deg2rad(np.random.uniform(0.0, 15.0))
        knee_flex = np.deg2rad(np.random.uniform(0.0, 20.0))
        ank_dorsi = np.deg2rad(np.random.uniform(-5.0, 10.0))

    # Add bilateral asymmetry (small)
    asym = np.random.normal(0.0, np.deg2rad(8.0))

    # SMPL axis-angle convention: hip flexion ~ rotation about local X (lateral).
    # The SMPL rest skeleton has thighs hanging down (-Y). Rotating about +X
    # swings the thigh forward (drop-jump squat).
    pose[SMPL_J["L_hip"], 0] = hip_flex + asym
    pose[SMPL_J["R_hip"], 0] = hip_flex - asym
    # Knees flex about local X (counter-rotation moves shank backward toward butt).
    pose[SMPL_J["L_knee"], 0] = -knee_flex
    pose[SMPL_J["R_knee"], 0] = -knee_flex
    # Ankle dorsiflexion: tilt foot up
    pose[SMPL_J["L_ank"], 0] = -ank_dorsi
    pose[SMPL_J["R_ank"], 0] = -ank_dorsi

    # Lumbar / spine forward lean (split across spine1..spine3)
    lumbar = np.deg2rad(np.random.uniform(0.0, 35.0))
    pose[SMPL_J["spine1"], 0] = lumbar * 0.3
    pose[SMPL_J["spine2"], 0] = lumbar * 0.4
    pose[SMPL_J["spine3"], 0] = lumbar * 0.3

    # Hip adduction (sample modest range)
    add = np.deg2rad(np.random.uniform(-15.0, 15.0))
    pose[SMPL_J["L_hip"], 2] += -add
    pose[SMPL_J["R_hip"], 2] += +add

    # Slight global orientation jitter (subject not always facing perfectly forward)
    pose[0, 1] = np.random.uniform(-0.3, 0.3)  # yaw
    pose[0, 0] = np.random.uniform(-0.1, 0.1)  # mild pitch
    pose[0, 2] = np.random.uniform(-0.1, 0.1)  # mild roll

    return pose


def _sample_camera() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a virtual camera covering the OpenCap deployment envelope.

    OpenCap rig is 5 cameras roughly evenly spaced around the subject; the
    drop-jump videos are typically 1280x720 portrait-rotated frames at ~60fps.
    We sample yaw in [0, 360), pitch in [-25, 25] deg, distance in [2.5, 5.0] m.

    Returns: (R world->cam, t world->cam, K intrinsics, yaw_deg).
    """
    yaw = np.random.uniform(0.0, 360.0)
    pitch = np.random.uniform(-25.0, 25.0)
    dist = np.random.uniform(2.5, 5.0)

    yaw_r = np.deg2rad(yaw)
    pitch_r = np.deg2rad(pitch)

    # Camera position in world (look at subject at ~1m height).
    look_at = np.array([0.0, 1.0, 0.0])
    cam_pos = look_at + dist * np.array([
        np.cos(pitch_r) * np.sin(yaw_r),
        np.sin(pitch_r),
        np.cos(pitch_r) * np.cos(yaw_r),
    ])

    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # OpenCV camera convention: rows of R are (right, -up, forward) in world coords.
    R = np.stack([right, -up, forward], axis=0)
    t = -R @ cam_pos

    # Intrinsics ~ typical phone wide lens, image 720x1280 (portrait).
    fx = fy = np.random.uniform(900.0, 1300.0)
    width, height = 720, 1280
    K = np.array([[fx, 0.0, width / 2.0],
                  [0.0, fy, height / 2.0],
                  [0.0, 0.0, 1.0]])
    return R, t, K, np.array([yaw, pitch, fx, width, height])


def _generate_synthetic_batch(
    kin: SmplKinematics, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (kp_xy_norm[N, N_KEYS, 2], angles[N, 5], yaws[N]).

    Forward-kinematics is run in batches of 1 frame (synthetic FK is cheap;
    sampling poses dominates).
    """
    kp_buf = np.zeros((n_samples, N_KEYS, 2), dtype=np.float32)
    angle_buf = np.zeros((n_samples, len(DEPLOY_METRICS)), dtype=np.float32)
    yaw_buf = np.zeros((n_samples,), dtype=np.float32)

    halpe_idx_list = [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL]
    smpl_idx_list = [HALPE26_TO_SMPL[name] for name in HALPE_KEYS_WITH_SMPL]
    smpl_idx_arr = np.array(smpl_idx_list, dtype=np.int64)

    with torch.no_grad():
        for i in range(n_samples):
            pose = _sample_pose()
            R, t, K, params = _sample_camera()
            yaw_buf[i] = float(params[0])

            pose_t = torch.from_numpy(pose).unsqueeze(0)  # (1, 24, 3)
            trans_t = torch.zeros(1, 3, dtype=torch.float64)
            scale_t = torch.tensor([1.0], dtype=torch.float64)
            joints = smpl_forward(pose_t, trans_t, scale_t, kin)  # (1, 24, 3)
            joints_np = joints[0].cpu().numpy()

            # Compute GT angles for the 5 deploy metrics using shared helper.
            metrics = smpl_joints_to_metrics(joints_np[None])  # (T=1, 24, 3)
            angle_buf[i, 0] = metrics["hip_flexion_r"][0]
            angle_buf[i, 1] = metrics["hip_adduction_r"][0]
            angle_buf[i, 2] = metrics["knee_angle_r"][0]
            angle_buf[i, 3] = metrics["ankle_angle_r"][0]
            angle_buf[i, 4] = metrics["lumbar_extension"][0]

            # Project Halpe-26-with-SMPL keypoints to 2D.
            kp3d = joints_np[smpl_idx_arr]  # (N_KEYS, 3)
            cam_pts = (R @ kp3d.T).T + t
            z = np.where(np.abs(cam_pts[:, 2]) < 1e-3, 1e-3, cam_pts[:, 2])
            u = K[0, 0] * cam_pts[:, 0] / z + K[0, 2]
            v = K[1, 1] * cam_pts[:, 1] / z + K[1, 2]
            uv = np.stack([u, v], axis=1)

            # Normalize per-frame: subtract pelvis (always present), divide by
            # pose bbox diagonal. This is what the model will see at deploy
            # time, where image resolution and subject position vary.
            kp_buf[i] = _normalize_kp(uv).astype(np.float32)

    return kp_buf, angle_buf, yaw_buf


def _normalize_kp(uv: np.ndarray) -> np.ndarray:
    """Per-frame keypoint normalization.

    uv: (N_KEYS, 2). Output: (N_KEYS, 2) zero-mean, unit pose-scale.
    """
    center = uv.mean(axis=0, keepdims=True)
    shift = uv - center
    scale = np.linalg.norm(shift, axis=1).max() + 1e-6
    return shift / scale


# -------------------------------------------------------------------------
# 2. MLP model.
# -------------------------------------------------------------------------


class KeypointMLP(nn.Module):
    """Per-frame MLP: Halpe-26-subset normalized xy (n_keys * 2) -> 5 angles."""

    def __init__(self, n_keys: int, n_out: int = 5, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_keys * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


def train_mlp(
    kp: np.ndarray,
    angles: np.ndarray,
    *,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> tuple[KeypointMLP, dict]:
    """Train the keypoint MLP on synthetic data. Returns model + training log."""
    rng = np.random.default_rng(0)
    n = kp.shape[0]
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    x_train = torch.from_numpy(kp[train_idx])
    y_train = torch.from_numpy(angles[train_idx])
    x_val = torch.from_numpy(kp[val_idx])
    y_val = torch.from_numpy(angles[val_idx])

    # Normalize targets to zero mean / unit std for stable training.
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp(min=1e-3)
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    model = KeypointMLP(N_KEYS, n_out=len(DEPLOY_METRICS), hidden=128)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    log = {"train": [], "val": [], "val_r": []}
    n_train = x_train.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        tot = 0.0
        nb = 0
        for s in range(0, n_train, batch_size):
            ii = perm[s : s + batch_size]
            opt.zero_grad()
            pred = model(x_train[ii])
            loss = loss_fn(pred, y_train_n[ii])
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        train_loss = tot / max(nb, 1)

        model.eval()
        with torch.no_grad():
            v_pred_n = model(x_val)
            v_loss = float(loss_fn(v_pred_n, y_val_n).item())
            v_pred = v_pred_n * y_std + y_mean
            # per-metric Pearson r on val set.
            r_per = []
            for k in range(y_val.shape[1]):
                a = v_pred[:, k].numpy()
                b = y_val[:, k].numpy()
                if np.var(a) < 1e-6 or np.var(b) < 1e-6:
                    r_per.append(float("nan"))
                else:
                    r_per.append(float(np.corrcoef(a, b)[0, 1]))
        log["train"].append(train_loss)
        log["val"].append(v_loss)
        log["val_r"].append(r_per)

        if ep % 10 == 0 or ep == epochs - 1:
            print(
                f"  ep {ep:3d}  train {train_loss:.4f}  val {v_loss:.4f}  "
                f"val r {[f'{r:+.2f}' for r in r_per]}"
            )

    # Persist normalization with the model for deploy use.
    model.y_mean = y_mean.squeeze(0).numpy()  # type: ignore[attr-defined]
    model.y_std = y_std.squeeze(0).numpy()    # type: ignore[attr-defined]
    return model, log


# -------------------------------------------------------------------------
# 3. Evaluate on OpenCap clips.
# -------------------------------------------------------------------------


def _resample_to_common(
    pred_t: np.ndarray, pred_y: np.ndarray,
    gt_t: np.ndarray, gt_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_finite = np.isfinite(pred_t) & np.isfinite(pred_y)
    gt_finite = np.isfinite(gt_t) & np.isfinite(gt_y)
    if pred_finite.sum() < 2 or gt_finite.sum() < 2:
        return np.array([]), np.array([])
    pt = pred_t[pred_finite]; py = pred_y[pred_finite]
    gt_x = gt_t[gt_finite]; gy = gt_y[gt_finite]
    lo = max(pt[0], gt_x[0]); hi = min(pt[-1], gt_x[-1])
    if hi - lo < 0.05:
        return np.array([]), np.array([])
    mask = (gt_x >= lo) & (gt_x <= hi)
    if mask.sum() < 2:
        return np.array([]), np.array([])
    return np.interp(gt_x[mask], pt, py), gy[mask]


def _pearson_r(a: np.ndarray, b: np.ndarray, min_pairs: int = 30) -> float:
    if a.size < min_pairs:
        return float("nan")
    if np.var(a) < 1e-9 or np.var(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_opencap_clip(kp_path: Path) -> dict:
    with open(kp_path) as f:
        d = json.load(f)
    seq = d["keypoints_sequence"]
    T = len(seq)
    kp = np.zeros((T, 26, 2), dtype=np.float64)
    conf = np.zeros((T, 26), dtype=np.float64)
    t_ms = np.zeros(T, dtype=np.float64)
    for i, fr in enumerate(seq):
        kp[i] = np.asarray(fr["keypoints"], dtype=np.float64)
        conf[i] = np.asarray(fr["keypoint_scores"], dtype=np.float64)
        t_ms[i] = fr["timestamp_ms"]
    return {
        "kp": kp,
        "conf": conf,
        "t_ms": t_ms,
        "width": d["video_metadata"]["width"],
        "height": d["video_metadata"]["height"],
        "fps": d["video_metadata"]["fps"],
    }


def predict_clip(
    model: KeypointMLP, kp_full: np.ndarray
) -> np.ndarray:
    """Run the MLP on a full Halpe-26 clip (T, 26, 2).

    Returns (T, 5) angle array (denormalized).
    """
    # Extract the SMPL-mappable subset.
    halpe_idx = [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL]
    sub = kp_full[:, halpe_idx, :]  # (T, N_KEYS, 2)
    # Normalize each frame.
    norm = np.zeros_like(sub, dtype=np.float32)
    for i in range(sub.shape[0]):
        norm[i] = _normalize_kp(sub[i]).astype(np.float32)
    x = torch.from_numpy(norm)
    model.eval()
    with torch.no_grad():
        p_n = model(x).numpy()
    y_mean = model.y_mean  # type: ignore[attr-defined]
    y_std = model.y_std    # type: ignore[attr-defined]
    return p_n * y_std + y_mean


def process_opencap_clip(
    kp_path: Path, model: KeypointMLP
) -> dict:
    base = kp_path.stem
    parts = base.split("_")
    subject = parts[0]; cam_name = parts[-1]
    trial = "_".join(parts[1:-1])
    mot_path = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{trial}.mot"
    if not mot_path.exists():
        return {"clip": base, "skip": "no mot"}

    clip = load_opencap_clip(kp_path)
    meta_path = LAB / subject / "sessionMetadata.yaml"
    height_m = float(yaml.safe_load(meta_path.read_text())["height_m"])
    cam_pkl = next((LAB / subject / "VideoData").glob(
        f"Session*/{cam_name}/cameraIntrinsicsExtrinsics.pickle"
    ))

    # Synthetic-trained MLP predictions: per-frame.
    pred = predict_clip(model, clip["kp"])  # (T, 5)
    pred_t = clip["t_ms"] / 1000.0

    # GT.
    gt = parse_mot(mot_path)
    gt_t = gt.time

    # Couro baseline on this clip.
    series = load_couro_output(kp_path)
    try:
        couro_md = keypoints_to_motion_data(
            series, subject_height_m=height_m, camera_pickle=cam_pkl
        )
    except Exception:
        couro_md = None

    results = {}
    for k, metric in enumerate(DEPLOY_METRICS):
        if not gt.has(metric):
            continue
        gt_y = gt.column(metric)
        # Synthetic-MLP vs GT.
        syn_pred = pred[:, k]
        sm_resamp, gy = _resample_to_common(pred_t, syn_pred, gt_t, gt_y)
        syn_r = _pearson_r(sm_resamp, gy)
        syn_abs_r = abs(syn_r) if np.isfinite(syn_r) else float("nan")
        syn_mae = float(np.mean(np.abs(sm_resamp - gy))) if np.isfinite(syn_r) else float("nan")

        # Couro vs GT.
        couro_r = float("nan")
        couro_abs_r = float("nan")
        couro_mae = float("nan")
        if couro_md is not None and couro_md.has(metric):
            cp_resamp, gy2 = _resample_to_common(
                couro_md.time, couro_md.column(metric), gt_t, gt_y
            )
            couro_r = _pearson_r(cp_resamp, gy2)
            if np.isfinite(couro_r):
                couro_abs_r = abs(couro_r)
                couro_mae = float(np.mean(np.abs(cp_resamp - gy2)))

        results[metric] = {
            "syn_r": syn_r,
            "syn_abs_r": syn_abs_r,
            "syn_mae": syn_mae,
            "couro_r": couro_r,
            "couro_abs_r": couro_abs_r,
            "couro_mae": couro_mae,
        }

    return {
        "clip": base,
        "subject": subject,
        "trial": trial,
        "cam": cam_name,
        "n_frames": int(clip["kp"].shape[0]),
        "metrics": results,
    }


# -------------------------------------------------------------------------
# Main driver.
# -------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SMPL_MODEL_PATH.exists():
        raise SystemExit(f"Missing SMPL model file: {SMPL_MODEL_PATH}")

    print("Loading SMPL kinematics...")
    kin = load_smpl_kinematics()

    np.random.seed(0)
    torch.manual_seed(0)

    n_samples = 4000
    print(f"\nGenerating {n_samples} synthetic samples...")
    t0 = time.time()
    kp, angles, yaws = _generate_synthetic_batch(kin, n_samples)
    gen_seconds = time.time() - t0
    print(f"  done in {gen_seconds:.1f}s")
    print(f"  kp shape: {kp.shape}  angles shape: {angles.shape}")
    print(f"  angle stats (deg):")
    for k, name in enumerate(DEPLOY_METRICS):
        print(
            f"    {name:20s} mean={angles[:, k].mean():+7.2f}  "
            f"std={angles[:, k].std():6.2f}  "
            f"range=[{angles[:, k].min():+7.2f}, {angles[:, k].max():+7.2f}]"
        )

    print("\nTraining MLP on synthetic data (80/20 split)...")
    t0 = time.time()
    model, log = train_mlp(kp, angles, epochs=80, lr=1e-3, batch_size=256)
    train_seconds = time.time() - t0
    final_val_r = log["val_r"][-1]
    print(f"  done in {train_seconds:.1f}s")
    print(f"  final val per-metric r: {final_val_r}")

    # Save model.
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "y_mean": model.y_mean,  # type: ignore[attr-defined]
        "y_std": model.y_std,    # type: ignore[attr-defined]
        "n_keys": N_KEYS,
        "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
        "metrics": list(DEPLOY_METRICS),
    }, MODEL_OUT)
    print(f"  saved model to {MODEL_OUT}")

    # OpenCap evaluation.
    test_clips = [
        ("subject10", "DJ1", "Cam0"),
        ("subject10", "DJ1", "Cam2"),
        ("subject10", "DJ1", "Cam4"),
        ("subject10", "DJ2", "Cam2"),
        ("subject2",  "DJ1", "Cam0"),
        ("subject2",  "DJ1", "Cam2"),
        ("subject2",  "DJ1", "Cam4"),
        ("subject3",  "DJ1", "Cam2"),
    ]
    kp_paths = []
    for subject, trial, cam in test_clips:
        p = OPENCAP_KP / f"{subject}_{trial}_{cam}.json"
        if p.exists():
            kp_paths.append(p)
        else:
            print(f"  skip (no kp file): {p.name}")

    print(f"\nEvaluating on {len(kp_paths)} OpenCap clips...")
    out = {
        "version": "1.0",
        "n_samples": n_samples,
        "synthetic_val_r": final_val_r,
        "metrics": list(DEPLOY_METRICS),
        "clips": [],
    }
    for i, kp_path in enumerate(kp_paths):
        print(f"\n[{i+1}/{len(kp_paths)}] {kp_path.name}")
        try:
            res = process_opencap_clip(kp_path, model)
        except Exception as e:
            print("  FAILED:", e)
            traceback.print_exc()
            continue
        if "skip" in res:
            print(f"  skip: {res['skip']}")
            continue
        out["clips"].append(res)
        print(f"  T={res['n_frames']}")
        for metric, m in res["metrics"].items():
            print(
                f"    {metric:20s} syn r={m['syn_r']:+.2f} (|r|={m['syn_abs_r']:+.2f})  "
                f"couro r={m['couro_r']:+.2f} (|r|={m['couro_abs_r']:+.2f})"
            )

    # Aggregate.
    agg = {}
    for metric in DEPLOY_METRICS:
        syn_rs, couro_rs = [], []
        for c in out["clips"]:
            m = c["metrics"].get(metric)
            if m is None:
                continue
            if np.isfinite(m["syn_abs_r"]): syn_rs.append(m["syn_abs_r"])
            if np.isfinite(m["couro_abs_r"]): couro_rs.append(m["couro_abs_r"])
        agg[metric] = {
            "n_syn": len(syn_rs),
            "n_couro": len(couro_rs),
            "syn_mean_abs_r": float(np.mean(syn_rs)) if syn_rs else float("nan"),
            "couro_mean_abs_r": float(np.mean(couro_rs)) if couro_rs else float("nan"),
        }
    pooled_syn, pooled_couro = [], []
    for c in out["clips"]:
        for m in c["metrics"].values():
            if np.isfinite(m["syn_abs_r"]): pooled_syn.append(m["syn_abs_r"])
            if np.isfinite(m["couro_abs_r"]): pooled_couro.append(m["couro_abs_r"])
    overall = {
        "n_syn": len(pooled_syn),
        "n_couro": len(pooled_couro),
        "syn_mean_abs_r": float(np.mean(pooled_syn)) if pooled_syn else float("nan"),
        "couro_mean_abs_r": float(np.mean(pooled_couro)) if pooled_couro else float("nan"),
    }
    out["aggregate"] = {"per_metric": agg, "overall": overall}

    out_path = OUT_DIR / "per_clip_r.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    print("\nAggregate (mean |r| over test clips):")
    print(f"  pooled SYN   |r| = {overall['syn_mean_abs_r']:.3f} (n={overall['n_syn']})")
    print(f"  pooled COURO |r| = {overall['couro_mean_abs_r']:.3f} (n={overall['n_couro']})")
    for metric, a in agg.items():
        print(
            f"  {metric:20s}  syn={a['syn_mean_abs_r']:.3f} (n={a['n_syn']})  "
            f"couro={a['couro_mean_abs_r']:.3f} (n={a['n_couro']})"
        )


if __name__ == "__main__":
    main()
