"""Learned Layer 2 — combined OpenCap + ASPset training cohort.

Agent HH build. Extends Agent EE2 (OpenCap-only Temporal CNN) by adding
ASPset-510's 17 subjects of paired DWPose video + mocap c3d ground truth.

Cohort:
  - OpenCap: 9 subjects, ~270 clips, 5 angles (hip_flexion_r, hip_adduction_r,
    knee_angle_r, ankle_angle_r, lumbar_extension).
  - ASPset:  17 subjects, ~1,410 clips, 4 angles (no ankle — ASPset has no
    foot keypoints). 3 camera views per clip (left/mid/right).

Convention alignment (ASPset -> OpenCap):
  - hip_flexion_r:   identity (both define ~0=neutral, +=flexed thigh)
  - hip_adduction_r: identity (sign roughly aligned; magnitude differs but
    relative variation is correlated). Clamp |x| > 90 to filter saturated.
  - knee_angle_r:    aspset uses 180=straight; OpenCap uses 0=straight.
                     aligned = aspset - 180 (still inverted!), so we use
                     aligned = 180 - aspset_raw, but ASPset already does
                     180 - angle_between in joint_angles_from_aspset, so the
                     raw aspset knee_angle_r needs OpenCap = 180 - aspset_raw.
                     **Verified: ASPset 73-179 vs OpenCap 0-110. Use
                     OpenCap_equiv = 180 - ASPset_knee.**
  - lumbar_extension: ASPset ~127-180 (180=neutral); OpenCap ~-60 to 0
                     (0=neutral, negative=flexed). Align: aligned =
                     aspset_lumbar - 180  (range -53 to 0, matches OpenCap).
  - ankle_angle_r:   NOT available in ASPset; target = NaN, masked from loss.

Training:
  - Same TemporalKeypointCNNConf architecture (66-channel input, 5 outputs).
  - Per-target masked SmoothL1 loss (NaN target -> excluded from gradient).
  - LOSO across 26 subjects total.
  - 25 epochs, batch 256, AdamW + cosine LR.

Evaluation:
  - Per-fold pooled |r| (frame-level Pearson per held-out clip, averaged).
  - Per-source breakdown: OpenCap-held-out folds vs ASPset-held-out folds.
  - Comparison vs EE2's OpenCap-only baseline (0.645 pooled).

ANTI-STALL: prints progress every clip-group, every fold, every epoch.
"""
from __future__ import annotations

import builtins
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Final

import numpy as np

# numpy 2.x compat
for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from harness.parsers import parse_mot
    from harness.smpl_layer2_poc import HALPE26_INDEX, HALPE26_TO_SMPL
    from harness.aspset_loader import load_clip as aspset_load_clip
    from harness.aspset_loader import joint_angles_from_aspset
else:
    from .parsers import parse_mot
    from .smpl_layer2_poc import HALPE26_INDEX, HALPE26_TO_SMPL
    from .aspset_loader import load_clip as aspset_load_clip
    from .aspset_loader import joint_angles_from_aspset


# -------------------------------------------------------------------------
# Paths and constants.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
LAB: Final[Path] = DATA_ROOT / "LabValidation_withVideos"
OPENCAP_KP: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"
ASPSET_KP: Final[Path] = DATA_ROOT / "aspset510_dwpose_keypoints"
ASPSET_C3D: Final[Path] = DATA_ROOT / "aspset510" / "ASPset-510" / "trainval" / "joints_3d"
OUT_DIR: Final[Path] = DATA_ROOT / "learned_layer2_combined"
MODEL_OUT: Final[Path] = REPO_ROOT / "models" / "learned_layer2_combined_v1.pt"

DEPLOY_METRICS: Final[tuple[str, ...]] = (
    "hip_flexion_r",
    "hip_adduction_r",
    "knee_angle_r",
    "ankle_angle_r",
    "lumbar_extension",
)

# ASPset can supply GT for these (index into DEPLOY_METRICS).
ASPSET_TARGET_AVAILABLE: Final[tuple[bool, ...]] = (
    True,   # hip_flexion_r
    True,   # hip_adduction_r
    True,   # knee_angle_r
    False,  # ankle_angle_r   <-- ASPset has no foot keypoints
    True,   # lumbar_extension
)

HALPE_KEYS_WITH_SMPL: Final[tuple[str, ...]] = tuple(
    name for name, smpl_idx in HALPE26_TO_SMPL.items() if smpl_idx is not None
)
N_KEYS: Final[int] = len(HALPE_KEYS_WITH_SMPL)  # 22
TEMPORAL_T: Final[int] = 9

# -------------------------------------------------------------------------
# Progress helper.
# -------------------------------------------------------------------------

_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[HH t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Shared keypoint normalization (per-frame zero-mean, unit pose-scale).
# -------------------------------------------------------------------------

def _normalize_kp(uv: np.ndarray) -> np.ndarray:
    center = uv.mean(axis=0, keepdims=True)
    shift = uv - center
    scale = np.linalg.norm(shift, axis=1).max() + 1e-6
    return shift / scale


# -------------------------------------------------------------------------
# OpenCap clip loader (same as EE2).
# -------------------------------------------------------------------------

def _load_opencap_keypoints(kp_path: Path) -> dict:
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
        "fps": d["video_metadata"]["fps"],
    }


def _resample_gt(gt_t: np.ndarray, gt_y: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    finite = np.isfinite(gt_t) & np.isfinite(gt_y)
    if finite.sum() < 2:
        return np.full(target_t.shape, np.nan)
    return np.interp(target_t, gt_t[finite], gt_y[finite],
                     left=np.nan, right=np.nan)


def build_opencap_clip(kp_path: Path, halpe_idx_arr: np.ndarray) -> dict | None:
    base = kp_path.stem
    parts = base.split("_")
    subject = parts[0]
    cam_name = parts[-1]
    trial = "_".join(parts[1:-1])
    mot_path = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{trial}.mot"
    if not mot_path.exists():
        return None
    try:
        clip = _load_opencap_keypoints(kp_path)
    except Exception:
        return None
    try:
        gt = parse_mot(mot_path)
    except Exception:
        return None

    t_sec = clip["t_ms"] / 1000.0
    n_frames = t_sec.shape[0]
    angles = np.full((n_frames, len(DEPLOY_METRICS)), np.nan, dtype=np.float32)
    for k, metric in enumerate(DEPLOY_METRICS):
        if not gt.has(metric):
            continue
        gt_y = gt.column(metric)
        angles[:, k] = _resample_gt(gt.time, gt_y, t_sec).astype(np.float32)

    if not np.any(np.isfinite(angles)):
        return None

    kp_sub = clip["kp"][:, halpe_idx_arr, :]
    conf_sub = clip["conf"][:, halpe_idx_arr]
    kp_norm = np.zeros_like(kp_sub, dtype=np.float32)
    for i in range(n_frames):
        kp_norm[i] = _normalize_kp(kp_sub[i]).astype(np.float32)
    return {
        "source": "opencap",
        "subject": f"opencap_{subject}",
        "cam": cam_name,
        "trial": trial,
        "clip": base,
        "kp": kp_norm,
        "conf": conf_sub.astype(np.float32),
        "t_sec": t_sec.astype(np.float32),
        "angles": angles,
    }


# -------------------------------------------------------------------------
# ASPset clip loader.
# -------------------------------------------------------------------------

def _load_aspset_dwpose(kp_path: Path) -> dict:
    """Load ASPset DWPose JSON (same schema as OpenCap DWPose)."""
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
        "fps": d["video_metadata"]["fps"],
    }


def _align_aspset_angles(aspset_angles: dict[str, np.ndarray], n_frames: int) -> np.ndarray:
    """Convert ASPset angles to OpenCap convention. Returns (n_frames, 5)
    with NaN for unavailable (ankle_angle_r) and frames out of range."""
    out = np.full((n_frames, len(DEPLOY_METRICS)), np.nan, dtype=np.float32)

    def _fit(arr: np.ndarray) -> np.ndarray:
        if arr is None:
            return np.full(n_frames, np.nan, dtype=np.float32)
        if arr.shape[0] == n_frames:
            return arr.astype(np.float32)
        # Linear-interp to align lengths (rarely needed; ASPset c3d & DWPose
        # are both 50 Hz on same video, but counts may differ by 1-2).
        src = np.linspace(0, 1, arr.shape[0])
        tgt = np.linspace(0, 1, n_frames)
        return np.interp(tgt, src, arr).astype(np.float32)

    # hip_flexion_r: identity
    out[:, 0] = _fit(aspset_angles.get("hip_flexion_r"))
    # hip_adduction_r: identity, but clamp saturated values (asin near ±90)
    hip_add = aspset_angles.get("hip_adduction_r")
    if hip_add is not None:
        hip_add = hip_add.copy()
        hip_add[np.abs(hip_add) > 85] = np.nan  # saturated
        out[:, 1] = _fit(hip_add)
    # knee_angle_r: ASPset uses 180=straight, OpenCap uses 0=straight
    knee = aspset_angles.get("knee_angle_r")
    if knee is not None:
        out[:, 2] = _fit(180.0 - knee.astype(np.float32))
    # ankle_angle_r: NaN (ASPset doesn't have foot keypoints)
    # already NaN by initialization
    # lumbar_extension: ASPset ~127-180 (180=neutral), OpenCap ~-60 to 0
    lumbar = aspset_angles.get("lumbar_extension")
    if lumbar is not None:
        out[:, 4] = _fit(lumbar.astype(np.float32) - 180.0)
    return out


def build_aspset_clip(kp_path: Path, halpe_idx_arr: np.ndarray) -> dict | None:
    """Load one ASPset clip: DWPose keypoints + ASPset c3d GT angles."""
    base = kp_path.stem
    # filename: "<subject>-<clip>-<view>.json"
    parts = base.split("-")
    if len(parts) != 3:
        return None
    subject, clip_id, view = parts
    c3d_path = ASPSET_C3D / subject / f"{subject}-{clip_id}.c3d"
    if not c3d_path.exists():
        return None

    try:
        clip = _load_aspset_dwpose(kp_path)
    except Exception:
        return None

    try:
        c3d_clip = aspset_load_clip(c3d_path)
        angles_raw = joint_angles_from_aspset(c3d_clip)
    except Exception:
        return None

    n_frames_dw = clip["kp"].shape[0]
    angles = _align_aspset_angles(angles_raw, n_frames_dw)

    # If after alignment no targets are finite, skip.
    if not np.any(np.isfinite(angles)):
        return None

    kp_sub = clip["kp"][:, halpe_idx_arr, :]
    conf_sub = clip["conf"][:, halpe_idx_arr]
    kp_norm = np.zeros_like(kp_sub, dtype=np.float32)
    for i in range(n_frames_dw):
        kp_norm[i] = _normalize_kp(kp_sub[i]).astype(np.float32)

    t_sec = (clip["t_ms"] / 1000.0).astype(np.float32)
    return {
        "source": "aspset",
        "subject": f"aspset_{subject}",
        "cam": view,
        "trial": clip_id,
        "clip": base,
        "kp": kp_norm,
        "conf": conf_sub.astype(np.float32),
        "t_sec": t_sec,
        "angles": angles,
    }


# -------------------------------------------------------------------------
# Cohort construction.
# -------------------------------------------------------------------------

def build_combined_cohort() -> dict[str, list[dict]]:
    halpe_idx_arr = np.array(
        [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL], dtype=np.int64
    )

    cohort: dict[str, list[dict]] = {}

    # ------ OpenCap ------
    oc_files = sorted(OPENCAP_KP.glob("*.json"))
    log(f"OpenCap: discovered {len(oc_files)} DWPose JSONs")
    n_kept = 0
    n_skip = 0
    last_print = time.time()
    for i, kp_path in enumerate(oc_files):
        if (i % 25 == 0) or (time.time() - last_print > 20.0):
            log(f"  OpenCap scan {i+1}/{len(oc_files)} kept={n_kept} skip={n_skip}")
            last_print = time.time()
        rec = build_opencap_clip(kp_path, halpe_idx_arr)
        if rec is None:
            n_skip += 1
            continue
        cohort.setdefault(rec["subject"], []).append(rec)
        n_kept += 1
    log(f"OpenCap: kept={n_kept} skip={n_skip}")

    # ------ ASPset ------
    asp_files = sorted(ASPSET_KP.glob("*.json"))
    log(f"ASPset: discovered {len(asp_files)} DWPose JSONs")
    n_kept_asp = 0
    n_skip_asp = 0
    last_print = time.time()
    for i, kp_path in enumerate(asp_files):
        if (i % 25 == 0) or (time.time() - last_print > 20.0):
            log(f"  ASPset scan {i+1}/{len(asp_files)} kept={n_kept_asp} skip={n_skip_asp}")
            last_print = time.time()
        rec = build_aspset_clip(kp_path, halpe_idx_arr)
        if rec is None:
            n_skip_asp += 1
            continue
        cohort.setdefault(rec["subject"], []).append(rec)
        n_kept_asp += 1
    log(f"ASPset: kept={n_kept_asp} skip={n_skip_asp}")

    # Summary
    log(f"=== combined cohort: {len(cohort)} subjects ===")
    n_oc_subj = sum(1 for s in cohort if s.startswith("opencap_"))
    n_asp_subj = sum(1 for s in cohort if s.startswith("aspset_"))
    n_oc_clips = sum(len(r) for s, r in cohort.items() if s.startswith("opencap_"))
    n_asp_clips = sum(len(r) for s, r in cohort.items() if s.startswith("aspset_"))
    log(f"  OpenCap: {n_oc_subj} subjects, {n_oc_clips} clips")
    log(f"  ASPset:  {n_asp_subj} subjects, {n_asp_clips} clips")

    # Per-target GT histograms
    log("--- combined GT angle ranges (deg) ---")
    for k, metric in enumerate(DEPLOY_METRICS):
        vals = []
        for subj, recs in cohort.items():
            for r in recs:
                col = r["angles"][:, k]
                vals.extend(col[np.isfinite(col)].tolist())
        if not vals:
            log(f"  {metric:20s}  NO DATA")
            continue
        a = np.array(vals)
        log(
            f"  {metric:20s}  n={len(a):>7d}  "
            f"min={a.min():+7.2f}  max={a.max():+7.2f}  "
            f"mean={a.mean():+7.2f}  std={a.std():+6.2f}"
        )
    return cohort


# -------------------------------------------------------------------------
# Model (same as EE2).
# -------------------------------------------------------------------------

class TemporalKeypointCNNConf(nn.Module):
    def __init__(
        self,
        n_keys: int = N_KEYS,
        n_out: int = len(DEPLOY_METRICS),
        hidden: int = 128,
        t_frames: int = TEMPORAL_T,
    ) -> None:
        super().__init__()
        in_ch = n_keys * 3
        self.t_frames = t_frames
        self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.act = nn.ReLU()
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        center = x.shape[-1] // 2
        return self.head(x[..., center])


# -------------------------------------------------------------------------
# Windowing + dataset stacking.
# -------------------------------------------------------------------------

def build_windows_for_clip(clip: dict, t_frames: int = TEMPORAL_T):
    kp = clip["kp"]
    conf = clip["conf"]
    angles = clip["angles"]
    n_frames = kp.shape[0]
    half = t_frames // 2

    if n_frames < t_frames:
        return (
            np.zeros((0, t_frames, N_KEYS, 3), dtype=np.float32),
            np.zeros((0, len(DEPLOY_METRICS)), dtype=np.float32),
        )

    packed = np.concatenate([kp, conf[..., None]], axis=-1).astype(np.float32)
    pad_front = np.repeat(packed[:1], half, axis=0)
    pad_back = np.repeat(packed[-1:], half, axis=0)
    padded = np.concatenate([pad_front, packed, pad_back], axis=0)
    windows = np.stack(
        [padded[i : i + t_frames] for i in range(n_frames)], axis=0
    )
    return windows, angles.astype(np.float32)


def stack_clips(
    clip_recs: list[dict], stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack windows from many clips. Keep windows with ANY finite target
    (masked per-output loss handles the NaN columns).

    `stride` decimates the frame axis to reduce training set size while
    preserving the data distribution (50 Hz -> 25 Hz at stride=2 still has
    enough temporal redundancy for the 9-frame center-anchored CNN).
    Evaluation always uses stride=1 (per-frame).
    """
    X_parts = []
    Y_parts = []
    for rec in clip_recs:
        windows, angles_center = build_windows_for_clip(rec)
        if windows.shape[0] == 0:
            continue
        if stride > 1:
            windows = windows[::stride]
            angles_center = angles_center[::stride]
        mask = np.any(np.isfinite(angles_center), axis=1)
        if not np.any(mask):
            continue
        X_parts.append(windows[mask])
        Y_parts.append(angles_center[mask])
    if not X_parts:
        return (
            np.zeros((0, TEMPORAL_T, N_KEYS, 3), dtype=np.float32),
            np.zeros((0, len(DEPLOY_METRICS)), dtype=np.float32),
        )
    return np.concatenate(X_parts, axis=0), np.concatenate(Y_parts, axis=0)


# -------------------------------------------------------------------------
# Masked loss.
# -------------------------------------------------------------------------

def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SmoothL1 with per-element masking on NaN targets.

    target NaNs are excluded from gradient (zero contribution).
    """
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0.0
    target_safe = torch.where(mask, target, torch.zeros_like(target))
    diff = pred - target_safe
    abs_diff = diff.abs()
    beta = 1.0
    sq = 0.5 * diff * diff / beta
    lin = abs_diff - 0.5 * beta
    loss = torch.where(abs_diff < beta, sq, lin)
    loss = loss * mask.float()
    return loss.sum() / mask.float().sum().clamp(min=1.0)


# -------------------------------------------------------------------------
# Training one LOSO fold.
# -------------------------------------------------------------------------

def _compute_target_norm(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-target mean/std from finite values only."""
    n_out = Y.shape[1]
    mean = np.zeros((1, n_out), dtype=np.float32)
    std = np.ones((1, n_out), dtype=np.float32)
    for k in range(n_out):
        col = Y[:, k]
        f = np.isfinite(col)
        if f.sum() < 10:
            mean[0, k] = 0.0
            std[0, k] = 1.0
            continue
        mean[0, k] = float(col[f].mean())
        std[0, k] = max(float(col[f].std()), 1e-3)
    return mean, std


def train_one_fold(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    seed: int = 0,
    fold_label: str = "",
) -> tuple[TemporalKeypointCNNConf, dict]:
    rng = np.random.default_rng(seed)
    n_train = X_train.shape[0]
    log(f"  [{fold_label}] training set: N={n_train} windows")
    if n_train < 64:
        raise RuntimeError(f"Training set too small: {n_train}")

    perm = rng.permutation(n_train)
    split = int(0.9 * n_train)
    tr_idx, va_idx = perm[:split], perm[split:]

    y_mean, y_std = _compute_target_norm(Y_train[tr_idx])
    log(f"  [{fold_label}] target mean={y_mean.squeeze(0)}, std={y_std.squeeze(0)}")

    def _flat(x: np.ndarray) -> np.ndarray:
        b, t, k, c = x.shape
        return x.reshape(b, t, k * c)

    # Build val tensors once
    Y_va_raw = Y_train[va_idx]
    Y_va_norm = (Y_va_raw - y_mean) / y_std
    x_va = torch.from_numpy(_flat(X_train[va_idx])).float()
    y_va_t = torch.from_numpy(Y_va_norm.astype(np.float32))

    model = TemporalKeypointCNNConf()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  [{fold_label}] params={n_params:,}  epochs={epochs}  batch={batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    log_hist: dict[str, list] = {"train": [], "val": []}
    t_fold_start = time.time()
    last_print = time.time()
    n_tr = tr_idx.shape[0]
    n_batches_per_epoch = (n_tr + batch_size - 1) // batch_size
    log(
        f"  [{fold_label}] n_tr={n_tr} batches_per_epoch={n_batches_per_epoch}"
    )

    for ep in range(epochs):
        model.train()
        perm_ep = rng.permutation(n_tr)
        tot = 0.0
        nb = 0
        log(f"  [{fold_label}] ep {ep}/{epochs} START fold_t={time.time()-t_fold_start:.0f}s")
        last_print = time.time()
        for batch_idx, s in enumerate(range(0, n_tr, batch_size)):
            ii = tr_idx[perm_ep[s : s + batch_size]]
            x_b = _flat(X_train[ii])
            y_b_raw = Y_train[ii]
            y_b_n = (y_b_raw - y_mean) / y_std
            x_t = torch.from_numpy(x_b).float()
            y_t = torch.from_numpy(y_b_n.astype(np.float32))
            opt.zero_grad()
            pred = model(x_t)
            loss = masked_smooth_l1(pred, y_t)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
            # Print every 5 batches OR every 25s to satisfy watchdog.
            if (batch_idx % 5 == 0) or (time.time() - last_print > 25.0):
                log(
                    f"  [{fold_label}] ep {ep}/{epochs} "
                    f"batch {batch_idx}/{n_batches_per_epoch} "
                    f"loss={loss.item():.4f} fold_t={time.time()-t_fold_start:.0f}s"
                )
                last_print = time.time()
        sched.step()
        train_loss = tot / max(nb, 1)

        model.eval()
        with torch.no_grad():
            v_pred = model(x_va)
            v_loss = float(masked_smooth_l1(v_pred, y_va_t).item())
        log_hist["train"].append(train_loss)
        log_hist["val"].append(v_loss)

        log(
            f"  [{fold_label}] ep {ep}/{epochs} END train {train_loss:.4f}  "
            f"val {v_loss:.4f}  fold_t={time.time()-t_fold_start:.0f}s"
        )
        last_print = time.time()

    model.y_mean = y_mean.squeeze(0)  # type: ignore[attr-defined]
    model.y_std = y_std.squeeze(0)    # type: ignore[attr-defined]
    return model, log_hist


# -------------------------------------------------------------------------
# Inference + evaluation.
# -------------------------------------------------------------------------

def predict_on_clip(model: TemporalKeypointCNNConf, rec: dict):
    windows, _ = build_windows_for_clip(rec)
    n_frames = windows.shape[0]
    if n_frames == 0:
        return np.zeros((0, len(DEPLOY_METRICS)), dtype=np.float32)
    b, t, k, c = windows.shape
    x_t = torch.from_numpy(windows.reshape(b, t, k * c)).float()
    model.eval()
    with torch.no_grad():
        p_n = model(x_t).numpy()
    y_mean = model.y_mean  # type: ignore[attr-defined]
    y_std = model.y_std    # type: ignore[attr-defined]
    return (p_n * y_std + y_mean).astype(np.float32)


def evaluate_fold(model: TemporalKeypointCNNConf, held_clips: list[dict]) -> tuple[dict, list[dict]]:
    per_clip = []
    metric_pooled: dict[str, list[float]] = {m: [] for m in DEPLOY_METRICS}
    rom_records = []

    for rec in held_clips:
        pred = predict_on_clip(model, rec)
        gt = rec["angles"]
        n = min(pred.shape[0], gt.shape[0])
        if n == 0:
            continue
        pred = pred[:n]
        gt = gt[:n]
        clip_metrics = {}
        for k, metric in enumerate(DEPLOY_METRICS):
            a = pred[:, k]
            b = gt[:, k]
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() < 30:
                clip_metrics[metric] = {"r": float("nan"), "abs_r": float("nan"),
                                        "mae": float("nan"), "n_frames": int(finite.sum())}
                continue
            ar = a[finite]
            br = b[finite]
            if np.var(ar) < 1e-9 or np.var(br) < 1e-9:
                r = float("nan")
            else:
                r = float(np.corrcoef(ar, br)[0, 1])
            mae = float(np.mean(np.abs(ar - br)))
            abs_r = abs(r) if np.isfinite(r) else float("nan")
            clip_metrics[metric] = {"r": r, "abs_r": abs_r, "mae": mae,
                                    "n_frames": int(finite.sum())}
            if np.isfinite(abs_r):
                metric_pooled[metric].append(abs_r)
            # ROM
            rom_records.append({
                "clip": rec["clip"], "subject": rec["subject"],
                "source": rec["source"], "metric": metric,
                "pred_rom": float(ar.max() - ar.min()),
                "gt_rom": float(br.max() - br.min()),
            })
        per_clip.append({
            "clip": rec["clip"], "subject": rec["subject"], "source": rec["source"],
            "trial": rec["trial"], "cam": rec["cam"], "n_frames": int(n),
            "metrics": clip_metrics,
        })

    per_metric_means = {}
    all_vals = []
    for m, vals in metric_pooled.items():
        per_metric_means[m] = {
            "n_clips": len(vals),
            "mean_abs_r": float(np.mean(vals)) if vals else float("nan"),
        }
        all_vals.extend(vals)
    pooled = {
        "n": len(all_vals),
        "mean_abs_r": float(np.mean(all_vals)) if all_vals else float("nan"),
    }
    return {
        "per_metric": per_metric_means,
        "pooled": pooled,
        "rom_records": rom_records,
    }, per_clip


# -------------------------------------------------------------------------
# LOSO driver.
# -------------------------------------------------------------------------

def run_loso(
    cohort: dict[str, list[dict]],
    *,
    epochs: int = 25,
    batch_size: int = 256,
    train_stride: int = 1,
) -> dict:
    subjects = sorted(cohort.keys())
    log(f"LOSO over {len(subjects)} subjects (train_stride={train_stride})")

    fold_results = []
    best_model_state = None
    best_y_mean = None
    best_y_std = None
    best_subject = None
    best_pooled = -1.0

    for fold_i, held_out in enumerate(subjects):
        fold_label = f"fold {fold_i+1}/{len(subjects)} held={held_out}"
        log(f"=== {fold_label} ===")
        train_subjects = [s for s in subjects if s != held_out]
        train_clips = [c for s in train_subjects for c in cohort[s]]
        held_clips = cohort[held_out]
        log(f"  train clips: {len(train_clips)}  held clips: {len(held_clips)}")

        X_train, Y_train = stack_clips(train_clips, stride=train_stride)
        log(f"  stacked windows: X={X_train.shape} (train_stride={train_stride})")
        if X_train.shape[0] < 100:
            log(f"  SKIP: not enough train samples")
            continue

        model, _hist = train_one_fold(
            X_train, Y_train,
            epochs=epochs, batch_size=batch_size,
            seed=fold_i, fold_label=fold_label,
        )

        log(f"  evaluating on {held_out} ...")
        agg, per_clip = evaluate_fold(model, held_clips)
        log(
            f"  [{fold_label}] held-out pooled |r| = "
            f"{agg['pooled']['mean_abs_r']:.4f}  (n={agg['pooled']['n']})"
        )
        for m, v in agg["per_metric"].items():
            log(f"    {m:20s}  |r|={v['mean_abs_r']:.3f}  n_clips={v['n_clips']}")

        fold_results.append({
            "held_out": held_out,
            "source": "opencap" if held_out.startswith("opencap_") else "aspset",
            "fold_idx": fold_i,
            "n_train_windows": int(X_train.shape[0]),
            "n_held_clips": len(held_clips),
            "aggregate": agg,
            "per_clip": per_clip,
        })

        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ly_mean = np.asarray(model.y_mean)  # type: ignore[attr-defined]
        ly_std = np.asarray(model.y_std)    # type: ignore[attr-defined]
        pooled_r = agg["pooled"]["mean_abs_r"]
        if np.isfinite(pooled_r) and pooled_r > best_pooled:
            best_pooled = pooled_r
            best_subject = held_out
            best_model_state = last_state
            best_y_mean = ly_mean
            best_y_std = ly_std

        _save_intermediate(fold_results)

    # ----- Aggregate -----
    overall_metric_vals: dict[str, list[float]] = {m: [] for m in DEPLOY_METRICS}
    overall_all = []
    for fr in fold_results:
        for m, v in fr["aggregate"]["per_metric"].items():
            if np.isfinite(v["mean_abs_r"]):
                overall_metric_vals[m].append(v["mean_abs_r"])
        if np.isfinite(fr["aggregate"]["pooled"]["mean_abs_r"]):
            overall_all.append(fr["aggregate"]["pooled"]["mean_abs_r"])

    # Per-source split
    by_source = {"opencap": [], "aspset": []}
    by_source_per_metric: dict[str, dict[str, list[float]]] = {
        src: {m: [] for m in DEPLOY_METRICS} for src in by_source
    }
    for fr in fold_results:
        src = fr["source"]
        p = fr["aggregate"]["pooled"]["mean_abs_r"]
        if np.isfinite(p):
            by_source[src].append(p)
        for m, v in fr["aggregate"]["per_metric"].items():
            if np.isfinite(v["mean_abs_r"]):
                by_source_per_metric[src][m].append(v["mean_abs_r"])

    overall = {
        "per_metric_subject_mean_abs_r": {
            m: {
                "n_folds": len(vals),
                "mean_abs_r": float(np.mean(vals)) if vals else float("nan"),
                "std_abs_r": float(np.std(vals)) if vals else float("nan"),
            } for m, vals in overall_metric_vals.items()
        },
        "pooled_subject_mean_abs_r": {
            "n_folds": len(overall_all),
            "mean_abs_r": float(np.mean(overall_all)) if overall_all else float("nan"),
        },
        "by_source": {
            src: {
                "n_folds": len(vals),
                "pooled_mean_abs_r": float(np.mean(vals)) if vals else float("nan"),
            } for src, vals in by_source.items()
        },
        "by_source_per_metric": {
            src: {
                m: {
                    "n_folds": len(vals),
                    "mean_abs_r": float(np.mean(vals)) if vals else float("nan"),
                } for m, vals in d.items()
            } for src, d in by_source_per_metric.items()
        },
    }
    log(
        f"=== OVERALL pooled |r| = "
        f"{overall['pooled_subject_mean_abs_r']['mean_abs_r']:.4f} "
        f"(folds={overall['pooled_subject_mean_abs_r']['n_folds']}) ==="
    )
    log(
        f"  by-source: OpenCap-held |r|="
        f"{overall['by_source']['opencap']['pooled_mean_abs_r']:.4f} "
        f"(n={overall['by_source']['opencap']['n_folds']}), "
        f"ASPset-held |r|={overall['by_source']['aspset']['pooled_mean_abs_r']:.4f} "
        f"(n={overall['by_source']['aspset']['n_folds']})"
    )

    if best_model_state is not None:
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": best_model_state,
            "y_mean": best_y_mean,
            "y_std": best_y_std,
            "n_keys": N_KEYS,
            "temporal_t": TEMPORAL_T,
            "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
            "metrics": list(DEPLOY_METRICS),
            "model_class": "TemporalKeypointCNNConf",
            "best_loso_subject": best_subject,
            "best_loso_pooled_abs_r": best_pooled,
        }, MODEL_OUT)
        log(f"saved best-fold checkpoint -> {MODEL_OUT} (held={best_subject}, |r|={best_pooled:.3f})")

    return {
        "version": "learned_layer2_combined_v1",
        "epochs": epochs,
        "batch_size": batch_size,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "metrics": list(DEPLOY_METRICS),
        "fold_results": fold_results,
        "overall": overall,
    }


def _save_intermediate(fold_results: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "per_slot_results.json"
    out_path.write_text(json.dumps({
        "version": "learned_layer2_combined_v1",
        "intermediate": True,
        "n_folds_completed": len(fold_results),
        "fold_results": fold_results,
    }, indent=2, default=str))


# -------------------------------------------------------------------------
# Main.
# -------------------------------------------------------------------------

def main(
    epochs: int = 25,
    batch_size: int = 256,
    max_subjects: int | None = None,
    train_stride: int = 1,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== learned_layer2_combined START (Agent HH) ===")
    log(f"OPENCAP_KP={OPENCAP_KP}")
    log(f"ASPSET_KP={ASPSET_KP}")
    log(f"ASPSET_C3D={ASPSET_C3D}")
    log(
        f"epochs={epochs}  batch_size={batch_size}  T={TEMPORAL_T}  "
        f"N_KEYS={N_KEYS}  train_stride={train_stride}"
    )

    cohort = build_combined_cohort()
    if not cohort:
        log("EMPTY COHORT; aborting")
        return

    if max_subjects is not None:
        keep = sorted(cohort.keys())[:max_subjects]
        cohort = {k: cohort[k] for k in keep}
        log(f"DEBUG: limited to {len(cohort)} subjects: {keep}")

    results = run_loso(
        cohort, epochs=epochs, batch_size=batch_size,
        train_stride=train_stride,
    )

    out_path = OUT_DIR / "per_slot_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote final results to {out_path}")
    log("=== learned_layer2_combined DONE ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="DEBUG: limit cohort to first N subjects")
    parser.add_argument("--train-stride", type=int, default=4,
                        help="Decimate training windows by this stride "
                             "(eval always uses stride=1)")
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size,
         max_subjects=args.max_subjects, train_stride=args.train_stride)
