"""Layer-2 POC: VPoser-prior SMPL fit (proxy for ScoreHMR / 4D-Humans).

Why VPoser instead of ScoreHMR or 4D-Humans:
- 4D-Humans / HMR2.0 ckpts are ~2.7GB and require PyTorch3D + Detectron2,
  neither of which install cleanly on Mac CPU in our 2-3 hr budget.
- ScoreHMR data bundles are 3GB+ and pull in the same 4D-Humans backbone.
- VPoser is the *exact same learned pose prior* that ScoreHMR and SMPLify-X
  use to constrain SMPL poses to the anatomically plausible manifold. It is
  2.7MB, downloads in seconds, has no academic registration wall, and slots
  directly into Agent K's existing fitting harness so the comparison
  isolates the variable we actually want to test: does adding a learned
  pose prior fix the depth ambiguity in the naive Adam fit?

Pipeline:
  - Same SMPL kinematics, same DWPose keypoints, same camera projection,
    same metric extraction as Agent K's `smpl_layer2_poc.py`.
  - Instead of optimizing 24 free axis-angle vectors per frame, we
    parameterize body joints 1..21 through a 32D latent z and decode via
    VPoser's pretrained decoder. Root (joint 0) and hand joints (22, 23)
    are kept free.
  - Loss: weighted L1 reprojection + temporal smoothness on z + small L2
    on z (KL-like regularizer toward N(0,I)) + scale prior.

VPoser weights: lithiumice/vposer_v1_0 (HuggingFace mirror of the original
nghorbani/MPI VPoser v1 from SMPLify-X). License: research/non-commercial
(MPI VPoser license). Pretrained ckpt only, no training here.

Outputs:
  - data/layer2_scorehmr_poc/per_clip_r.json
  - data/layer2_scorehmr_poc/REPORT.md
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
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from harness.couro_keypoints import load_couro_output, keypoints_to_motion_data
    from harness.parsers import parse_mot
    from harness.smpl_layer2_poc import (
        SMPL_MODEL_PATH,
        DATA_ROOT,
        LAB,
        OPENCAP_KP,
        DEPLOY_METRICS,
        HALPE26_INDEX,
        HALPE26_TO_SMPL,
        SmplKinematics,
        load_smpl_kinematics,
        axis_angle_to_rotmat,
        smpl_forward,
        CameraExtrinsics,
        load_opencap_camera,
        project,
        smpl_joints_to_metrics,
        load_opencap_clip,
        _resample_to_common,
        _pearson_r,
    )
else:
    from .couro_keypoints import load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .smpl_layer2_poc import (
        SMPL_MODEL_PATH,
        DATA_ROOT,
        LAB,
        OPENCAP_KP,
        DEPLOY_METRICS,
        HALPE26_INDEX,
        HALPE26_TO_SMPL,
        SmplKinematics,
        load_smpl_kinematics,
        axis_angle_to_rotmat,
        smpl_forward,
        CameraExtrinsics,
        load_opencap_camera,
        project,
        smpl_joints_to_metrics,
        load_opencap_clip,
        _resample_to_common,
        _pearson_r,
    )

VPOSER_CKPT: Final[Path] = REPO_ROOT / "models" / "vposer_v1" / "TR00_E096.pt"
OUT_DIR: Final[Path] = DATA_ROOT / "layer2_scorehmr_poc"


# -------------------------------------------------------------------------
# VPoser decoder (MPI nghorbani/vposer_v1_0 architecture).
# We only need the decoder path: latent z (B, 32) -> body pose (B, 21, 3, 3).
# -------------------------------------------------------------------------

class ContinuousRotReprDecoder(nn.Module):
    """Decode 6D continuous rotation representation -> 3x3 rotmat."""

    def forward(self, module_input: torch.Tensor) -> torch.Tensor:
        reshaped = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped[:, :, 0], dim=1)
        dot = (b1 * reshaped[:, :, 1]).sum(dim=1, keepdim=True)
        b2 = F.normalize(reshaped[:, :, 1] - dot * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


class VPoserDecoder(nn.Module):
    """Decoder half of the MPI VPoser v1 VAE.

    Maps latent z in R^32 to 21 SMPL body joint rotations as 3x3 matrices.
    Standalone: no torchgeometry dependency, we implement matrot2aa ourselves.
    """

    NUM_JOINTS: Final[int] = 21
    LATENT_D: Final[int] = 32
    NUM_NEURONS: Final[int] = 512

    def __init__(self) -> None:
        super().__init__()
        # Decoder layers — names must match the checkpoint keys.
        self.bodyprior_dec_fc1 = nn.Linear(self.LATENT_D, self.NUM_NEURONS)
        self.bodyprior_dec_fc2 = nn.Linear(self.NUM_NEURONS, self.NUM_NEURONS)
        self.bodyprior_dec_out = nn.Linear(self.NUM_NEURONS, self.NUM_JOINTS * 6)
        self.rot_decoder = ContinuousRotReprDecoder()
        # Dropout is identity in eval mode but must exist for state_dict compat
        # (it has no parameters so we skip it).

    def decode_rotmat(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 32) -> rotmats (B, 21, 3, 3)."""
        x = F.leaky_relu(self.bodyprior_dec_fc1(z), negative_slope=0.2)
        x = F.leaky_relu(self.bodyprior_dec_fc2(x), negative_slope=0.2)
        x = self.bodyprior_dec_out(x)
        # x: (B, 21*6); pass through 6D rotation decoder
        rotmats = self.rot_decoder(x)  # (B*21, 3, 3)
        return rotmats.view(-1, self.NUM_JOINTS, 3, 3)


def load_vposer(ckpt_path: Path = VPOSER_CKPT) -> VPoserDecoder:
    """Load VPoser decoder weights from the lithiumice/vposer_v1_0 checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    decoder = VPoserDecoder()
    decoder_state = {
        k: v for k, v in sd.items()
        if k.startswith("bodyprior_dec_") or k.startswith("rot_decoder.")
    }
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
    if missing:
        # The rot_decoder has no params, so we expect zero missing keys.
        # Anything else is a real load problem.
        real_missing = [m for m in missing if "rot_decoder" not in m]
        if real_missing:
            raise RuntimeError(f"VPoser decoder missing keys: {real_missing}")
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder


# -------------------------------------------------------------------------
# Rotmat -> axis-angle.
# -------------------------------------------------------------------------

def rotmat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """R: (..., 3, 3) -> axis-angle (..., 3). Differentiable, batch-safe.

    Uses the standard log-map: theta = acos((tr(R) - 1) / 2); axis = unskew.
    """
    # Numerical guard on trace.
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=1e-6)
    # Skew-symmetric part: w = (R - R^T) / 2; axis = unskew(w) / sin(theta)
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    coef = (0.5 * theta / sin_theta).unsqueeze(-1)
    return torch.stack([rx, ry, rz], dim=-1) * coef


# -------------------------------------------------------------------------
# VPoser-prior SMPL fitter.
# -------------------------------------------------------------------------

@dataclass
class FitResult:
    joints_world: np.ndarray   # (T, 24, 3)
    final_loss: float
    reproj_mae_px: float


def fit_vposer_smpl_to_clip(
    kp2d: np.ndarray,        # (T, 26, 2)
    conf: np.ndarray,        # (T, 26)
    cam: CameraExtrinsics,
    kin: SmplKinematics,
    vposer: VPoserDecoder,
    *,
    n_iters: int = 400,
    lr: float = 0.05,
    z_smooth_weight: float = 2.0,
    z_l2_weight: float = 0.05,
    root_smooth_weight: float = 5.0,
    init_scale: float = 1.0,
    device: str = "cpu",
    verbose: bool = False,
) -> FitResult:
    """Fit SMPL through VPoser latent z + free root/hand poses.

    Variables (per clip):
      - z         : (T, 32)  VPoser latent for body joints 1..21
      - root_pose : (T, 3)   axis-angle for joint 0 (pelvis), free
      - hand_pose : (T, 2, 3) axis-angle for joints 22, 23, free
      - trans     : (T, 3)   world translation
      - scale     : (1,)     isotropic body scale
    """
    T = kp2d.shape[0]
    torch_dtype = torch.float64

    pairs = [
        (HALPE26_INDEX[hname], smpl_idx)
        for hname, smpl_idx in HALPE26_TO_SMPL.items()
        if smpl_idx is not None
    ]
    halpe_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    smpl_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    kp_t = torch.from_numpy(kp2d[:, halpe_idx, :]).to(torch_dtype).to(device)
    w_t = torch.from_numpy(conf[:, halpe_idx]).to(torch_dtype).to(device)
    w_t = torch.where(w_t > 0.3, w_t, torch.zeros_like(w_t))

    vposer = vposer.to(torch_dtype).to(device)

    # Variables.
    z = torch.zeros(T, VPoserDecoder.LATENT_D, dtype=torch_dtype,
                    device=device, requires_grad=True)
    root_pose = torch.zeros(T, 3, dtype=torch_dtype, device=device,
                            requires_grad=True)
    hand_pose = torch.zeros(T, 2, 3, dtype=torch_dtype, device=device,
                            requires_grad=True)

    initial_world_pos = (
        cam.R.T @ (torch.tensor([0.0, 0.0, 3.0], dtype=torch_dtype) - cam.t)
    ).numpy()
    trans = torch.tensor(np.tile(initial_world_pos, (T, 1)),
                         dtype=torch_dtype, device=device, requires_grad=True)
    scale = torch.tensor([init_scale], dtype=torch_dtype, device=device,
                         requires_grad=True)

    opt = torch.optim.Adam([z, root_pose, hand_pose, trans, scale], lr=lr)

    last_loss = 0.0
    last_reproj = 0.0

    for it in range(n_iters):
        opt.zero_grad()
        # Decode body pose through VPoser.
        rotmats = vposer.decode_rotmat(z)             # (T, 21, 3, 3)
        body_aa = rotmat_to_axis_angle(rotmats)        # (T, 21, 3)

        # Assemble full 24-joint pose.
        pose_list = [root_pose.unsqueeze(1)]            # (T, 1, 3)
        pose_list.append(body_aa)                       # (T, 21, 3)
        pose_list.append(hand_pose)                     # (T, 2, 3)
        pose = torch.cat(pose_list, dim=1)              # (T, 24, 3)

        joints = smpl_forward(pose, trans, scale, kin)  # (T, 24, 3)
        target_joints = joints[:, smpl_idx, :]          # (T, K, 3)
        uv = project(target_joints, cam)
        diff = uv - kp_t
        residual = torch.sqrt((diff ** 2).sum(-1) + 1e-6)
        reproj_loss = (w_t * residual).sum() / (w_t.sum().clamp(min=1.0))

        # Temporal smoothness on z and root.
        if T > 1:
            dz = z[1:] - z[:-1]
            z_smooth = (dz ** 2).sum(dim=-1).mean()
            droot = root_pose[1:] - root_pose[:-1]
            root_smooth = (droot ** 2).sum(dim=-1).mean()
        else:
            z_smooth = torch.tensor(0.0, dtype=torch_dtype, device=device)
            root_smooth = torch.tensor(0.0, dtype=torch_dtype, device=device)

        # KL-like prior: latent toward N(0, I).
        z_prior = (z ** 2).sum(dim=-1).mean()

        # Scale prior.
        scale_prior = (scale - 1.0) ** 2

        loss = (
            reproj_loss
            + z_smooth_weight * z_smooth
            + z_l2_weight * z_prior
            + root_smooth_weight * root_smooth
            + 0.5 * scale_prior.sum()
        )
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        last_reproj = float(reproj_loss.item())
        if verbose and (it % 50 == 0 or it == n_iters - 1):
            print(
                f"    it={it:3d} loss={last_loss:.3f}  reproj_px={last_reproj:.2f}  "
                f"scale={float(scale):.3f}"
            )

    with torch.no_grad():
        rotmats = vposer.decode_rotmat(z)
        body_aa = rotmat_to_axis_angle(rotmats)
        pose = torch.cat(
            [root_pose.unsqueeze(1), body_aa, hand_pose], dim=1
        )
        joints_final = smpl_forward(pose, trans, scale, kin).cpu().numpy()

    return FitResult(
        joints_world=joints_final,
        final_loss=last_loss,
        reproj_mae_px=last_reproj,
    )


# -------------------------------------------------------------------------
# Driver: process one clip.
# -------------------------------------------------------------------------

def process_opencap_clip(
    kp_path: Path,
    kin: SmplKinematics,
    vposer: VPoserDecoder,
    naive_smpl_lookup: dict,
    *,
    n_iters: int,
    lr: float,
) -> dict:
    base = kp_path.stem
    parts = base.split("_")
    subject = parts[0]
    cam_name = parts[-1]
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
    fit = fit_vposer_smpl_to_clip(
        clip["kp"], clip["conf"], cam, kin, vposer,
        n_iters=n_iters, lr=lr,
        init_scale=height_m / 1.65,
        verbose=False,
    )
    fit_seconds = time.time() - t0
    per_frame_ms = 1000.0 * fit_seconds / max(1, clip["kp"].shape[0])

    vposer_metrics = smpl_joints_to_metrics(fit.joints_world)

    gt = parse_mot(mot_path)
    gt_t = gt.time

    series = load_couro_output(kp_path)
    try:
        couro_md = keypoints_to_motion_data(
            series, subject_height_m=height_m, camera_pickle=cam_pkl
        )
    except Exception:
        couro_md = None

    pred_t = clip["t_ms"] / 1000.0
    results = {}
    naive_entry = naive_smpl_lookup.get(base, {}).get("metrics", {})
    for metric in DEPLOY_METRICS:
        if not gt.has(metric):
            continue
        gt_y = gt.column(metric)

        vp_pred = vposer_metrics.get(metric)
        vp_r = float("nan")
        vp_abs_r = float("nan")
        vp_mae = float("nan")
        if vp_pred is not None:
            vp_resamp, gy = _resample_to_common(pred_t, vp_pred, gt_t, gt_y)
            vp_r = _pearson_r(vp_resamp, gy)
            if np.isfinite(vp_r):
                vp_abs_r = abs(vp_r)
                vp_mae = float(np.mean(np.abs(vp_resamp - gy)))

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

        naive = naive_entry.get(metric, {})
        results[metric] = {
            "vposer_smpl_r": vp_r,
            "vposer_smpl_abs_r": vp_abs_r,
            "vposer_smpl_mae": vp_mae,
            "naive_smpl_r": naive.get("smpl_r", float("nan")),
            "naive_smpl_abs_r": naive.get("smpl_abs_r", float("nan")),
            "naive_smpl_mae": naive.get("smpl_mae", float("nan")),
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
        "fit_seconds": fit_seconds,
        "per_frame_ms": per_frame_ms,
        "fit_loss": fit.final_loss,
        "fit_reproj_px": fit.reproj_mae_px,
        "metrics": results,
    }


def _load_naive_lookup(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    return {c["clip"]: c for c in raw.get("clips", [])}


# -------------------------------------------------------------------------
# Main.
# -------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SMPL_MODEL_PATH.exists():
        raise SystemExit(f"Missing SMPL model file: {SMPL_MODEL_PATH}")
    if not VPOSER_CKPT.exists():
        raise SystemExit(
            f"Missing VPoser checkpoint: {VPOSER_CKPT}. "
            f"Download from https://huggingface.co/lithiumice/vposer_v1_0"
        )

    kin = load_smpl_kinematics()
    print(f"Loaded SMPL kinematics (rest joints {tuple(kin.rest_joints.shape)}).")

    vposer = load_vposer()
    print(f"Loaded VPoser v1 decoder from {VPOSER_CKPT.name}")

    naive_lookup = _load_naive_lookup(DATA_ROOT / "layer2_smpl_poc" / "per_clip_r.json")
    print(f"Loaded naive-SMPL baseline for {len(naive_lookup)} clips.")

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

    n_iters = 400
    lr = 0.05
    out = {
        "version": "1.0",
        "model": "VPoser-prior SMPL (lithiumice/vposer_v1_0)",
        "n_iters": n_iters,
        "lr": lr,
        "clips": [],
    }
    print(f"\nProcessing {len(kp_paths)} OpenCap clips with VPoser-prior SMPL fit:")
    for i, kp_path in enumerate(kp_paths):
        print(f"\n[{i+1}/{len(kp_paths)}] {kp_path.name}")
        try:
            res = process_opencap_clip(
                kp_path, kin, vposer, naive_lookup,
                n_iters=n_iters, lr=lr,
            )
        except Exception as e:
            print("  FAILED:", e)
            traceback.print_exc()
            continue
        if "skip" in res:
            print(f"  skip: {res['skip']}")
            continue
        out["clips"].append(res)
        print(
            f"  fit {res['fit_seconds']:.1f}s "
            f"({res['per_frame_ms']:.0f}ms/frame)  "
            f"reproj={res['fit_reproj_px']:.2f}px  T={res['n_frames']}"
        )
        for metric, m in res["metrics"].items():
            print(
                f"    {metric:20s}  "
                f"vp r={m['vposer_smpl_r']:+.2f} (|r|={m['vposer_smpl_abs_r']:+.2f})  "
                f"naive r={m['naive_smpl_r']:+.2f} (|r|={m['naive_smpl_abs_r']:+.2f})  "
                f"couro r={m['couro_r']:+.2f} (|r|={m['couro_abs_r']:+.2f})"
            )

    # Aggregate per-metric and pooled means.
    agg = {}
    for metric in DEPLOY_METRICS:
        vp_rs, naive_rs, couro_rs = [], [], []
        for c in out["clips"]:
            m = c["metrics"].get(metric)
            if m is None:
                continue
            if np.isfinite(m["vposer_smpl_abs_r"]):
                vp_rs.append(m["vposer_smpl_abs_r"])
            if np.isfinite(m["naive_smpl_abs_r"]):
                naive_rs.append(m["naive_smpl_abs_r"])
            if np.isfinite(m["couro_abs_r"]):
                couro_rs.append(m["couro_abs_r"])
        agg[metric] = {
            "n_vposer": len(vp_rs),
            "n_naive": len(naive_rs),
            "n_couro": len(couro_rs),
            "vposer_mean_abs_r": float(np.mean(vp_rs)) if vp_rs else float("nan"),
            "naive_mean_abs_r": float(np.mean(naive_rs)) if naive_rs else float("nan"),
            "couro_mean_abs_r": float(np.mean(couro_rs)) if couro_rs else float("nan"),
        }

    pooled_vp, pooled_naive, pooled_couro = [], [], []
    for c in out["clips"]:
        for m in c["metrics"].values():
            if np.isfinite(m["vposer_smpl_abs_r"]):
                pooled_vp.append(m["vposer_smpl_abs_r"])
            if np.isfinite(m["naive_smpl_abs_r"]):
                pooled_naive.append(m["naive_smpl_abs_r"])
            if np.isfinite(m["couro_abs_r"]):
                pooled_couro.append(m["couro_abs_r"])
    overall = {
        "n_vposer": len(pooled_vp),
        "n_naive": len(pooled_naive),
        "n_couro": len(pooled_couro),
        "vposer_mean_abs_r": float(np.mean(pooled_vp)) if pooled_vp else float("nan"),
        "naive_mean_abs_r": float(np.mean(pooled_naive)) if pooled_naive else float("nan"),
        "couro_mean_abs_r": float(np.mean(pooled_couro)) if pooled_couro else float("nan"),
    }

    latencies = [c["per_frame_ms"] for c in out["clips"] if "per_frame_ms" in c]
    fit_secs = [c["fit_seconds"] for c in out["clips"] if "fit_seconds" in c]
    out["aggregate"] = {
        "per_metric": agg,
        "overall": overall,
        "latency": {
            "per_frame_ms_mean": float(np.mean(latencies)) if latencies else float("nan"),
            "per_clip_sec_mean": float(np.mean(fit_secs)) if fit_secs else float("nan"),
            "per_clip_sec_max": float(np.max(fit_secs)) if fit_secs else float("nan"),
        },
    }

    out_path = OUT_DIR / "per_clip_r.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    print("\nAggregate (mean |r| over test clips):")
    print(f"  pooled VPoser SMPL |r| = {overall['vposer_mean_abs_r']:.3f} (n={overall['n_vposer']})")
    print(f"  pooled naive  SMPL |r| = {overall['naive_mean_abs_r']:.3f} (n={overall['n_naive']})")
    print(f"  pooled Couro       |r| = {overall['couro_mean_abs_r']:.3f} (n={overall['n_couro']})")
    print()
    for metric, a in agg.items():
        print(
            f"  {metric:20s}  "
            f"vp={a['vposer_mean_abs_r']:.3f} (n={a['n_vposer']})  "
            f"naive={a['naive_mean_abs_r']:.3f} (n={a['n_naive']})  "
            f"couro={a['couro_mean_abs_r']:.3f} (n={a['n_couro']})"
        )
    print()
    lat = out["aggregate"]["latency"]
    print(
        f"  latency: per-frame {lat['per_frame_ms_mean']:.0f}ms  "
        f"per-clip {lat['per_clip_sec_mean']:.1f}s avg / "
        f"{lat['per_clip_sec_max']:.1f}s max"
    )


if __name__ == "__main__":
    main()
