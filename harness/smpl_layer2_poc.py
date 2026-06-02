"""SMPL Layer-2 POC fitter.

Per-clip SMPL fit: optimize 24-joint SMPL kinematic tree (axis-angle pose +
global translation, isotropic body-scale) to minimize 2D reprojection error
of DWPose Halpe-26 keypoints, then compute the 5 Couro joint angles from the
fitted 3D joints and compare to OpenSim mocap GT.

Approach: option (b) "temporal SMPL fitter via gradient descent" from the
spec. We bypass full SMPL LBS + vertex skinning (which needs chumpy) and use
only:
  - the rest-pose joint positions `J` (24, 3) from SMPL_NEUTRAL.pkl
  - the kintree (parent-of-child) 24-vector

Forward kinematics: given axis-angle local rotations per joint (24, 3) and a
global root translation (3,) plus a scalar body scale (1,), produce world
joint positions (24, 3) per frame. The kinematic tree is rooted at pelvis (0).

Optimization (per clip):
  - Variables: pose (T, 24, 3), trans (T, 3), scale (1,) per clip,
    shared across frames.
  - Loss: weighted L1 of (project(fk(pose, trans, scale)) - dwpose_xy),
    where weights are DWPose confidence scores.
  - Temporal prior: L2 of pose finite differences (encourages smoothness).
  - Pose prior: L2 of pose magnitude (encourages near-rest pose).

Outputs: per-clip Pearson r per metric vs OpenSim GT, written to
`data/layer2_smpl_poc/per_clip_r.json`.

This is a proof-of-concept; it deliberately uses joint-only FK (no mesh, no
shape betas) because the comparison is angles-only and SMPL's joint
hierarchy already encodes the inductive bias we care about: anatomically
plausible bone lengths + parent-child constraints across time.
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

# numpy compat shims so chumpy-pickled SMPL_NEUTRAL.pkl loads.
for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from harness.couro_keypoints import load_couro_output, keypoints_to_motion_data
    from harness.parsers import parse_mot
else:
    from .couro_keypoints import load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot


SMPL_MODEL_PATH: Final[Path] = REPO_ROOT / "models" / "smpl" / "SMPL_NEUTRAL.pkl"
DATA_ROOT: Final[Path] = REPO_ROOT / "data"
LAB: Final[Path] = DATA_ROOT / "LabValidation_withVideos"
OPENCAP_KP: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"
OUT_DIR: Final[Path] = DATA_ROOT / "layer2_smpl_poc"

DEPLOY_METRICS: Final[tuple[str, ...]] = (
    "hip_flexion_r",
    "hip_adduction_r",
    "knee_angle_r",
    "ankle_angle_r",
    "lumbar_extension",
)

# Halpe-26 ordering used in DWPose keypoints jsons.
HALPE26_NAMES: Final[tuple[str, ...]] = (
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sho", "R_sho", "L_elb", "R_elb", "L_wri", "R_wri",
    "L_hip", "R_hip", "L_kne", "R_kne", "L_ank", "R_ank",
    "head_top", "neck", "hip_center",
    "L_big_toe", "R_big_toe", "L_small_toe", "R_small_toe",
    "L_heel", "R_heel",
)
HALPE26_INDEX: Final[dict[str, int]] = {n: i for i, n in enumerate(HALPE26_NAMES)}

# Halpe26 -> SMPL 24-joint index. None means no SMPL joint to map.
# SMPL 24 joints:
# 0 pelvis, 1/2 L/R hip, 3 spine1, 4/5 L/R knee, 6 spine2, 7/8 L/R ankle,
# 9 spine3, 10/11 L/R foot, 12 neck, 13/14 L/R collar, 15 head,
# 16/17 L/R shoulder, 18/19 L/R elbow, 20/21 L/R wrist, 22/23 L/R hand.
HALPE26_TO_SMPL: Final[dict[str, int | None]] = {
    "nose": 15,           # use head joint as nose proxy
    "L_eye": None,
    "R_eye": None,
    "L_ear": None,
    "R_ear": None,
    "L_sho": 16,
    "R_sho": 17,
    "L_elb": 18,
    "R_elb": 19,
    "L_wri": 20,
    "R_wri": 21,
    "L_hip": 1,
    "R_hip": 2,
    "L_kne": 4,
    "R_kne": 5,
    "L_ank": 7,
    "R_ank": 8,
    "head_top": 15,
    "neck": 12,
    "hip_center": 0,
    "L_big_toe": 10,      # SMPL foot joint as toe proxy
    "R_big_toe": 11,
    "L_small_toe": 10,
    "R_small_toe": 11,
    "L_heel": 7,          # ankle as heel proxy
    "R_heel": 8,
}


# -------------------------------------------------------------------------
# SMPL model loader (joint-only, no mesh).
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class SmplKinematics:
    """Joint-only SMPL kinematics: rest joints + parent table."""
    rest_joints: torch.Tensor    # (24, 3) float64
    parents: torch.Tensor        # (24,) int64; parents[0] == -1


def load_smpl_kinematics(model_path: Path = SMPL_MODEL_PATH) -> SmplKinematics:
    with open(model_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    J = np.asarray(d["J"], dtype=np.float64)               # (24, 3)
    parents = np.asarray(d["kintree_table"][0], dtype=np.int64)  # (24,)
    parents[parents > 1_000_000] = -1  # uint -> -1 for root
    return SmplKinematics(
        rest_joints=torch.from_numpy(J),
        parents=torch.from_numpy(parents),
    )


# -------------------------------------------------------------------------
# Forward kinematics: axis-angle -> 3D joint positions.
# -------------------------------------------------------------------------

def axis_angle_to_rotmat(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula. axis_angle: (..., 3) -> (..., 3, 3).

    Differentiable, vectorized over leading dims.
    """
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-9)
    k = axis_angle / theta
    K = torch.zeros(*axis_angle.shape[:-1], 3, 3, dtype=axis_angle.dtype, device=axis_angle.device)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] =  k[..., 0]
    I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(K)
    sin = torch.sin(theta).unsqueeze(-1)
    cos = torch.cos(theta).unsqueeze(-1)
    return I + sin * K + (1 - cos) * (K @ K)


def smpl_forward(
    pose: torch.Tensor,            # (T, 24, 3) axis-angle
    trans: torch.Tensor,           # (T, 3) global translation
    scale: torch.Tensor,           # (1,) isotropic body scale (m)
    kin: SmplKinematics,
) -> torch.Tensor:
    """Joint-only SMPL FK. Returns (T, 24, 3) world-space joint positions.

    Rest joints (kin.rest_joints) are scaled by `scale` first. Each joint's
    world rotation = parent_world_rot @ local_rot(axis_angle). Each joint's
    world position = parent_world_pos + parent_world_rot @ (rest_offset).
    """
    T = pose.shape[0]
    rest = kin.rest_joints.to(pose.dtype).to(pose.device) * scale  # (24, 3)
    parents = kin.parents.to(pose.device)
    R_local = axis_angle_to_rotmat(pose)            # (T, 24, 3, 3)

    # Build per-joint world rotation + position by autograd-safe list accumulation.
    R_world_list: list[torch.Tensor] = [None] * 24
    pos_list: list[torch.Tensor] = [None] * 24
    R_world_list[0] = R_local[:, 0]
    pos_list[0] = trans

    for j in range(1, 24):
        p = int(parents[j].item())
        offset = rest[j] - rest[p]                  # (3,)
        # World rotation of joint j: parent_world_rot @ local_rot[j].
        R_world_list[j] = R_world_list[p] @ R_local[:, j]
        pos_list[j] = pos_list[p] + torch.einsum('tij,j->ti', R_world_list[p], offset)

    return torch.stack(pos_list, dim=1)              # (T, 24, 3)


# -------------------------------------------------------------------------
# Camera projection (matches OpenCap pickle: K, R_cam (world->cam), t_cam).
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraExtrinsics:
    K: torch.Tensor       # (3, 3)
    R: torch.Tensor       # (3, 3) world->cam
    t: torch.Tensor       # (3,)  world->cam translation (mm in OpenCap pickle)
    width: int
    height: int


def load_opencap_camera(
    pickle_path: Path, image_width: int, image_height: int
) -> CameraExtrinsics:
    with open(pickle_path, "rb") as f:
        d = pickle.load(f)
    K = np.asarray(d["intrinsicMat"], dtype=np.float64).copy()
    stored = np.asarray(d.get("imageSize", [[image_width], [image_height]])).ravel()
    stored_w, stored_h = float(stored[0]), float(stored[1])
    K[0, 0] *= image_width / stored_w
    K[0, 2] *= image_width / stored_w
    K[1, 1] *= image_height / stored_h
    K[1, 2] *= image_height / stored_h
    R = np.asarray(d["rotation"], dtype=np.float64).copy()
    t = np.asarray(d["translation"], dtype=np.float64).ravel() / 1000.0  # mm -> m
    return CameraExtrinsics(
        K=torch.from_numpy(K),
        R=torch.from_numpy(R),
        t=torch.from_numpy(t),
        width=image_width,
        height=image_height,
    )


def project(points_world: torch.Tensor, cam: CameraExtrinsics) -> torch.Tensor:
    """points_world (..., 3) -> pixel (..., 2). All points in m."""
    R = cam.R.to(points_world.dtype).to(points_world.device)
    t = cam.t.to(points_world.dtype).to(points_world.device)
    K = cam.K.to(points_world.dtype).to(points_world.device)
    cam_pts = torch.einsum('ij,...j->...i', R, points_world) + t
    Z = cam_pts[..., 2:3].clamp(min=1e-3)
    proj = cam_pts[..., :2] / Z
    uv = proj * torch.diagonal(K)[:2] + K[:2, 2]
    return uv


# -------------------------------------------------------------------------
# Joint-angle extraction from SMPL-fitted joints.
# Matches Couro deploy metric definitions used in compute_layer2_angle_r.py.
# -------------------------------------------------------------------------

def smpl_joints_to_metrics(joints: np.ndarray) -> dict[str, np.ndarray]:
    """joints: (T, 24, 3). Returns dict[metric_name -> (T,) deg array].

    Definitions chosen to match OpenSim conventions where possible:
    - hip_flexion_r: thigh-vs-pelvis-down angle (R side, joints 2 -> 5).
    - hip_adduction_r: lateral component of right thigh in pelvis frame.
    - knee_angle_r: 180 - thigh^shank angle on right (joints 2-5-8).
    - ankle_angle_r: shank-vs-foot angle on right (joints 5-8-11), with
      neutral standing ~= 0 (dorsiflexion positive).
    - lumbar_extension: spine3 (idx 9) vs world up, signed by forward lean.
    """
    pelvis = joints[:, 0]
    r_hip, r_knee, r_ank, r_foot = joints[:, 2], joints[:, 5], joints[:, 8], joints[:, 11]
    spine3 = joints[:, 9]
    l_hip = joints[:, 1]

    pelvis_axis = pelvis - 0.5 * (l_hip + r_hip)
    # Define a pelvis-down direction using hips and spine: down = -(spine3 - pelvis_center)
    pelvis_center = 0.5 * (l_hip + r_hip)
    down = -(spine3 - pelvis_center)
    down /= (np.linalg.norm(down, axis=1, keepdims=True) + 1e-9)

    thigh = r_knee - r_hip
    shank = r_ank - r_knee
    footv = r_foot - r_ank

    def _ang(a, b):
        na = np.linalg.norm(a, axis=1) + 1e-9
        nb = np.linalg.norm(b, axis=1) + 1e-9
        cos = np.clip(np.einsum('ij,ij->i', a, b) / (na * nb), -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    # Hip flexion: angle between thigh and pelvis-down.
    hip_flex = _ang(thigh, down)
    # Sign: positive when knee in front of pelvis (forward lean of thigh).
    # Use world Z (camera forward varies, but world +Z roughly forward in SMPL frame).
    # We instead compute the projection of thigh onto a plane normal to pelvis lateral.
    # For waveform correlation, sign is recovered by the GT loader if there's a flip.

    # Knee flexion: 180 - interior angle.
    knee_flex = 180.0 - _ang(thigh, shank)

    # Ankle dorsiflexion: 90 - angle between -shank and foot vector.
    ankle_dorsi = 90.0 - _ang(-shank, footv)

    # Hip adduction: lateral angle in pelvis frame.
    # Pelvis lateral axis from L_hip - R_hip.
    lat = l_hip - r_hip
    lat /= (np.linalg.norm(lat, axis=1, keepdims=True) + 1e-9)
    # Project thigh onto pelvis frontal plane (drop component along down).
    thigh_unit = thigh / (np.linalg.norm(thigh, axis=1, keepdims=True) + 1e-9)
    thigh_lat = np.einsum('ij,ij->i', thigh_unit, lat)
    thigh_down = np.einsum('ij,ij->i', thigh_unit, down)
    # Adduction angle: arctan2 of lateral over down. Inward (toward midline) = positive.
    hip_add = np.degrees(np.arctan2(thigh_lat, thigh_down))

    # Lumbar extension: spine3 vs world-up (y-up in SMPL).
    world_up = np.array([0.0, 1.0, 0.0])
    spine = spine3 - pelvis_center
    spine_unit = spine / (np.linalg.norm(spine, axis=1, keepdims=True) + 1e-9)
    # Forward lean angle in degrees, signed by z-component (negative = forward in SMPL).
    cosang = np.clip(spine_unit @ world_up, -1.0, 1.0)
    lumbar = np.degrees(np.arccos(cosang))
    # Sign: forward lean positive.
    lumbar = lumbar * np.sign(-spine_unit[:, 2] + 1e-9)

    return {
        "hip_flexion_r": hip_flex,
        "hip_adduction_r": hip_add,
        "knee_angle_r": knee_flex,
        "ankle_angle_r": ankle_dorsi,
        "lumbar_extension": lumbar,
    }


# -------------------------------------------------------------------------
# Clip fitter.
# -------------------------------------------------------------------------

@dataclass
class FitResult:
    joints_world: np.ndarray   # (T, 24, 3)
    final_loss: float
    reproj_mae_px: float


def fit_smpl_to_clip(
    kp2d: np.ndarray,        # (T, 26, 2)
    conf: np.ndarray,         # (T, 26)
    cam: CameraExtrinsics,
    kin: SmplKinematics,
    *,
    n_iters: int = 200,
    lr: float = 0.03,
    pose_smooth_weight: float = 5.0,
    pose_l2_weight: float = 0.05,
    init_scale: float = 1.0,
    device: str = "cpu",
    verbose: bool = False,
) -> FitResult:
    """Fit SMPL pose+trans+scale to a sequence of DWPose keypoints."""
    T = kp2d.shape[0]
    torch_dtype = torch.float64

    # Build SMPL idx -> Halpe26 idx mapping (only joints with a SMPL target).
    pairs: list[tuple[int, int]] = []
    for hname, smpl_idx in HALPE26_TO_SMPL.items():
        if smpl_idx is None:
            continue
        pairs.append((HALPE26_INDEX[hname], smpl_idx))
    halpe_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    smpl_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    kp_t = torch.from_numpy(kp2d[:, halpe_idx, :]).to(torch_dtype).to(device)   # (T, K, 2)
    w_t = torch.from_numpy(conf[:, halpe_idx]).to(torch_dtype).to(device)        # (T, K)
    # Drop low-conf samples.
    w_t = torch.where(w_t > 0.3, w_t, torch.zeros_like(w_t))

    # Variables.
    pose = torch.zeros(T, 24, 3, dtype=torch_dtype, device=device, requires_grad=True)
    # Initial root translation: place subject roughly in front of camera using
    # camera extrinsics, ~depth 3m.
    # Use the inverse of cam.R @ p + t = (0, 0, 3); world p = R^T @ ([0,0,3] - t)
    initial_world_pos = (cam.R.T @ (torch.tensor([0.0, 0.0, 3.0], dtype=torch_dtype) - cam.t)).numpy()
    trans = torch.tensor(np.tile(initial_world_pos, (T, 1)),
                         dtype=torch_dtype, device=device, requires_grad=True)
    scale = torch.tensor([init_scale], dtype=torch_dtype, device=device, requires_grad=True)

    opt = torch.optim.Adam([pose, trans, scale], lr=lr)

    last_loss = 0.0
    last_reproj = 0.0

    for it in range(n_iters):
        opt.zero_grad()
        joints = smpl_forward(pose, trans, scale, kin)          # (T, 24, 3)
        # Project subset matching halpe26.
        target_joints = joints[:, smpl_idx, :]                  # (T, K, 3)
        uv = project(target_joints, cam)                        # (T, K, 2)
        diff = uv - kp_t                                        # (T, K, 2)
        # Pixel residual, weighted by confidence.
        residual = torch.sqrt((diff ** 2).sum(-1) + 1e-6)       # (T, K)
        reproj_loss = (w_t * residual).sum() / (w_t.sum().clamp(min=1.0))

        # Temporal smoothness on pose.
        if T > 1:
            dpose = pose[1:] - pose[:-1]
            smooth = (dpose ** 2).sum(dim=-1).mean()
        else:
            smooth = torch.tensor(0.0, dtype=torch_dtype, device=device)
        # Pose magnitude prior (encourage near-rest).
        pose_l2 = (pose ** 2).sum(dim=-1).mean()
        # Scale prior (keep near 1.0).
        scale_prior = (scale - 1.0) ** 2

        loss = reproj_loss + pose_smooth_weight * smooth + pose_l2_weight * pose_l2 + 0.5 * scale_prior.sum()
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        last_reproj = float(reproj_loss.item())
        if verbose and (it % 25 == 0 or it == n_iters - 1):
            print(f"    it={it:3d} loss={last_loss:.3f}  reproj_px={last_reproj:.2f}  scale={float(scale):.3f}")

    with torch.no_grad():
        joints_final = smpl_forward(pose, trans, scale, kin).cpu().numpy()
    return FitResult(
        joints_world=joints_final,
        final_loss=last_loss,
        reproj_mae_px=last_reproj,
    )


# -------------------------------------------------------------------------
# Driver: load clip, fit, score.
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


def process_opencap_clip(
    kp_path: Path, kin: SmplKinematics,
    *, n_iters: int, lr: float,
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
    cam = load_opencap_camera(cam_pkl, clip["width"], clip["height"])

    t0 = time.time()
    fit = fit_smpl_to_clip(
        clip["kp"], clip["conf"], cam, kin,
        n_iters=n_iters, lr=lr,
        init_scale=height_m / 1.65,    # SMPL neutral rest = ~1.65 m tall
        verbose=False,
    )
    fit_seconds = time.time() - t0

    # SMPL-derived per-frame metrics.
    smpl_metrics = smpl_joints_to_metrics(fit.joints_world)

    # GT.
    gt = parse_mot(mot_path)
    gt_t = gt.time

    # Couro current baseline on this clip (for apples-to-apples).
    series = load_couro_output(kp_path)
    try:
        couro_md = keypoints_to_motion_data(
            series, subject_height_m=height_m, camera_pickle=cam_pkl
        )
    except Exception as e:
        couro_md = None

    smpl_t = clip["t_ms"] / 1000.0
    results = {}
    for metric in DEPLOY_METRICS:
        if not gt.has(metric):
            continue
        gt_y = gt.column(metric)
        # SMPL vs GT.
        smpl_pred = smpl_metrics.get(metric)
        smpl_r = float("nan")
        smpl_abs_r = float("nan")
        smpl_mae = float("nan")
        if smpl_pred is not None:
            sm_resamp, gy = _resample_to_common(smpl_t, smpl_pred, gt_t, gt_y)
            smpl_r = _pearson_r(sm_resamp, gy)
            if np.isfinite(smpl_r):
                smpl_abs_r = abs(smpl_r)
                smpl_mae = float(np.mean(np.abs(sm_resamp - gy)))
        # Couro current vs GT.
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
            "smpl_r": smpl_r,
            "smpl_abs_r": smpl_abs_r,
            "smpl_mae": smpl_mae,
            "couro_r": couro_r,
            "couro_abs_r": couro_abs_r,
            "couro_mae": couro_mae,
        }

    return {
        "clip": base,
        "subject": subject,
        "trial": trial,
        "cam": cam_name,
        "fit_seconds": fit_seconds,
        "fit_loss": fit.final_loss,
        "fit_reproj_px": fit.reproj_mae_px,
        "n_frames": int(clip["kp"].shape[0]),
        "metrics": results,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SMPL_MODEL_PATH.exists():
        raise SystemExit(f"Missing SMPL model file: {SMPL_MODEL_PATH}")
    kin = load_smpl_kinematics()
    print(f"Loaded SMPL kinematics (rest joints {tuple(kin.rest_joints.shape)}).")

    # Pick a handful of OpenCap DJ clips spanning views.
    # Use subject10 and subject2 for variety; one of each main cam view.
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

    print(f"\nProcessing {len(kp_paths)} OpenCap clips with SMPL fit:")
    out = {"version": "1.0", "n_iters": 400, "lr": 0.1, "clips": []}
    n_iters = 400
    lr = 0.1
    for i, kp_path in enumerate(kp_paths):
        print(f"\n[{i+1}/{len(kp_paths)}] {kp_path.name}")
        try:
            res = process_opencap_clip(kp_path, kin, n_iters=n_iters, lr=lr)
        except Exception as e:
            print("  FAILED:", e)
            traceback.print_exc()
            continue
        if "skip" in res:
            print(f"  skip: {res['skip']}")
            continue
        out["clips"].append(res)
        print(f"  fit {res['fit_seconds']:.1f}s  reproj={res['fit_reproj_px']:.2f}px  T={res['n_frames']}")
        for metric, m in res["metrics"].items():
            print(f"    {metric:20s}  smpl r={m['smpl_r']:+.2f} (|r|={m['smpl_abs_r']:+.2f})  "
                  f"couro r={m['couro_r']:+.2f} (|r|={m['couro_abs_r']:+.2f})")

    # Aggregate.
    agg = {}
    for metric in DEPLOY_METRICS:
        smpl_rs = []; couro_rs = []
        for c in out["clips"]:
            m = c["metrics"].get(metric)
            if m is None: continue
            if np.isfinite(m["smpl_abs_r"]): smpl_rs.append(m["smpl_abs_r"])
            if np.isfinite(m["couro_abs_r"]): couro_rs.append(m["couro_abs_r"])
        agg[metric] = {
            "n_smpl": len(smpl_rs),
            "n_couro": len(couro_rs),
            "smpl_mean_abs_r": float(np.mean(smpl_rs)) if smpl_rs else float("nan"),
            "couro_mean_abs_r": float(np.mean(couro_rs)) if couro_rs else float("nan"),
        }
    pooled_smpl = []; pooled_couro = []
    for c in out["clips"]:
        for m in c["metrics"].values():
            if np.isfinite(m["smpl_abs_r"]): pooled_smpl.append(m["smpl_abs_r"])
            if np.isfinite(m["couro_abs_r"]): pooled_couro.append(m["couro_abs_r"])
    overall = {
        "n_smpl": len(pooled_smpl),
        "n_couro": len(pooled_couro),
        "smpl_mean_abs_r": float(np.mean(pooled_smpl)) if pooled_smpl else float("nan"),
        "couro_mean_abs_r": float(np.mean(pooled_couro)) if pooled_couro else float("nan"),
    }
    out["aggregate"] = {"per_metric": agg, "overall": overall}

    out_path = OUT_DIR / "per_clip_r.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    print("\nAggregate (mean |r| over test clips):")
    print(f"  pooled SMPL  |r| = {overall['smpl_mean_abs_r']:.3f} (n={overall['n_smpl']})")
    print(f"  pooled Couro |r| = {overall['couro_mean_abs_r']:.3f} (n={overall['n_couro']})")
    for metric, a in agg.items():
        print(f"  {metric:20s}  smpl={a['smpl_mean_abs_r']:.3f} (n={a['n_smpl']})  "
              f"couro={a['couro_mean_abs_r']:.3f} (n={a['n_couro']})")


if __name__ == "__main__":
    main()
