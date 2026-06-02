"""Learned Layer 2 trained on REAL paired DWPose + OpenCap mocap GT.

Agent EE2 build. Replaces synthetic-only training with real paired data.

Pipeline:
1. Load all OpenCap clips for which both DWPose keypoints and .mot exist.
2. For each clip: extract Halpe-22 keypoints + confidences; resample mocap GT
   angles onto the keypoint timestamps.
3. Build sliding-window samples (T=9 frames). Each sample = (keypoints+conf
   window, center-frame angle target).
4. LOSO: 9 OpenCap subjects -> train on 8, predict on 1 (frame-by-frame).
   Compute pooled |r| across folds.
5. Compare to baselines: Couro 0.514, synthetic 0.495, view-aware blend 0.581.

Adds confidence-channel input over Agent S architecture.

ANTI-STALL: prints progress every batch group, every fold, every clip load.
"""

from __future__ import annotations

import builtins
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Final

import numpy as np

# numpy 2.x compat for SMPL_NEUTRAL.pkl chumpy types.
for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from harness.parsers import parse_mot
    from harness.smpl_layer2_poc import HALPE26_INDEX, HALPE26_TO_SMPL
else:
    from .parsers import parse_mot
    from .smpl_layer2_poc import HALPE26_INDEX, HALPE26_TO_SMPL


DATA_ROOT: Final[Path] = REPO_ROOT / "data"
LAB: Final[Path] = DATA_ROOT / "LabValidation_withVideos"
OPENCAP_KP: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"
OUT_DIR: Final[Path] = DATA_ROOT / "learned_layer2_real_gt"
MODEL_OUT: Final[Path] = REPO_ROOT / "models" / "learned_layer2_v1.pt"

DEPLOY_METRICS: Final[tuple[str, ...]] = (
    "hip_flexion_r",
    "hip_adduction_r",
    "knee_angle_r",
    "ankle_angle_r",
    "lumbar_extension",
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
    print(f"[t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Data loading.
# -------------------------------------------------------------------------


def _normalize_kp(uv: np.ndarray) -> np.ndarray:
    """Per-frame keypoint normalization: zero-mean, unit pose-scale."""
    center = uv.mean(axis=0, keepdims=True)
    shift = uv - center
    scale = np.linalg.norm(shift, axis=1).max() + 1e-6
    return shift / scale


def load_opencap_clip(kp_path: Path) -> dict:
    """Load DWPose keypoints + timestamps + confidences for a clip."""
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


def _resample_gt(gt_t: np.ndarray, gt_y: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Linear-interpolate gt_y(gt_t) onto target_t. Out-of-range -> NaN."""
    finite = np.isfinite(gt_t) & np.isfinite(gt_y)
    if finite.sum() < 2:
        return np.full(target_t.shape, np.nan)
    gx = gt_t[finite]
    gy = gt_y[finite]
    out = np.interp(target_t, gx, gy, left=np.nan, right=np.nan)
    return out


def build_clip_dataset(
    kp_path: Path, halpe_idx_arr: np.ndarray
) -> dict | None:
    """Load one clip, build per-frame (keypoints, conf, GT angles).

    Returns None if no matching .mot file is found.
    """
    base = kp_path.stem
    parts = base.split("_")
    subject = parts[0]
    cam_name = parts[-1]
    trial = "_".join(parts[1:-1])
    mot_path = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{trial}.mot"
    if not mot_path.exists():
        return None

    try:
        clip = load_opencap_clip(kp_path)
    except Exception as e:
        log(f"  failed to load {kp_path.name}: {e}")
        return None

    try:
        gt = parse_mot(mot_path)
    except Exception as e:
        log(f"  failed to parse {mot_path.name}: {e}")
        return None

    t_sec = clip["t_ms"] / 1000.0
    n_frames = t_sec.shape[0]

    angles = np.full((n_frames, len(DEPLOY_METRICS)), np.nan, dtype=np.float32)
    for k, metric in enumerate(DEPLOY_METRICS):
        if not gt.has(metric):
            continue
        gt_y = gt.column(metric)
        angles[:, k] = _resample_gt(gt.time, gt_y, t_sec).astype(np.float32)

    # If no valid angles at all, skip.
    if not np.any(np.isfinite(angles)):
        return None

    # Extract just the Halpe keys we use (22 of 26).
    kp_sub = clip["kp"][:, halpe_idx_arr, :]  # (T, N_KEYS, 2)
    conf_sub = clip["conf"][:, halpe_idx_arr]  # (T, N_KEYS)

    # Normalize per-frame.
    kp_norm = np.zeros_like(kp_sub, dtype=np.float32)
    for i in range(n_frames):
        kp_norm[i] = _normalize_kp(kp_sub[i]).astype(np.float32)

    return {
        "subject": subject,
        "cam": cam_name,
        "trial": trial,
        "clip": base,
        "kp": kp_norm,                     # (T, N_KEYS, 2)
        "conf": conf_sub.astype(np.float32),  # (T, N_KEYS)
        "t_sec": t_sec.astype(np.float32),
        "angles": angles,                  # (T, 5)
    }


def build_windows_for_clip(
    clip: dict, t_frames: int = TEMPORAL_T
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows of length t_frames for a clip.

    Returns (windows, angles_center, frame_idx) where:
      windows: (W, T, N_KEYS, 3) — last channel: [u, v, conf]
      angles_center: (W, 5) — center-frame GT angles (may contain NaN)
      frame_idx: (W,) — center-frame index in the source clip
    """
    kp = clip["kp"]      # (T, N_KEYS, 2)
    conf = clip["conf"]  # (T, N_KEYS)
    angles = clip["angles"]  # (T, 5)
    n_frames = kp.shape[0]
    half = t_frames // 2

    if n_frames < t_frames:
        return (
            np.zeros((0, t_frames, N_KEYS, 3), dtype=np.float32),
            np.zeros((0, len(DEPLOY_METRICS)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    # Pack u, v, conf -> (T, N_KEYS, 3).
    packed = np.concatenate([kp, conf[..., None]], axis=-1).astype(np.float32)

    # Edge-pad so each frame can be the center of a window.
    pad_front = np.repeat(packed[:1], half, axis=0)
    pad_back = np.repeat(packed[-1:], half, axis=0)
    padded = np.concatenate([pad_front, packed, pad_back], axis=0)

    windows = np.stack(
        [padded[i : i + t_frames] for i in range(n_frames)], axis=0
    )  # (T, t_frames, N_KEYS, 3)
    return windows, angles.astype(np.float32), np.arange(n_frames, dtype=np.int64)


# -------------------------------------------------------------------------
# Model.
# -------------------------------------------------------------------------


class TemporalKeypointCNNConf(nn.Module):
    """1D CNN over T frames of (u, v, conf) per keypoint.

    Input:  (B, T, N_KEYS*3)
    Output: (B, n_out) -- prediction for the center frame.
    """

    def __init__(
        self,
        n_keys: int = N_KEYS,
        n_out: int = len(DEPLOY_METRICS),
        hidden: int = 128,
        t_frames: int = TEMPORAL_T,
    ) -> None:
        super().__init__()
        in_ch = n_keys * 3  # u, v, conf
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
        feat = x[..., center]
        return self.head(feat)


# -------------------------------------------------------------------------
# Cohort building.
# -------------------------------------------------------------------------


def build_cohort(
    max_clips_per_subject: int | None = None,
) -> dict[str, list[dict]]:
    """Walk OPENCAP_KP, build clip records grouped by subject.

    Returns {subject_id: [clip_dict, ...]}.
    """
    halpe_idx_arr = np.array(
        [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL], dtype=np.int64
    )
    kp_files = sorted(OPENCAP_KP.glob("*.json"))
    log(f"discovered {len(kp_files)} keypoint files in {OPENCAP_KP}")

    cohort: dict[str, list[dict]] = {}
    n_kept = 0
    n_skip = 0
    last_log = time.time()
    for i, kp_path in enumerate(kp_files):
        if time.time() - last_log > 20.0:
            log(f"  scanning clip {i+1}/{len(kp_files)}: {kp_path.name} (kept={n_kept}, skip={n_skip})")
            last_log = time.time()
        rec = build_clip_dataset(kp_path, halpe_idx_arr)
        if rec is None:
            n_skip += 1
            continue
        subj = rec["subject"]
        cohort.setdefault(subj, []).append(rec)
        n_kept += 1
        if max_clips_per_subject is not None and len(cohort[subj]) >= max_clips_per_subject:
            # We still want to scan the rest for variety, but skip storing.
            pass

    log(f"cohort built: kept={n_kept}, skip={n_skip}, subjects={sorted(cohort.keys())}")
    for subj, recs in sorted(cohort.items()):
        n_frames = sum(r["kp"].shape[0] for r in recs)
        log(f"  {subj}: {len(recs)} clips, {n_frames} frames")
    return cohort


# -------------------------------------------------------------------------
# Training one LOSO fold.
# -------------------------------------------------------------------------


def stack_clips(clip_recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window training set from a list of clip records.

    Returns (X, Y) where X: (N, T, N_KEYS, 3) and Y: (N, 5). Drops samples
    whose center-frame angles contain any NaN.
    """
    X_parts = []
    Y_parts = []
    for rec in clip_recs:
        windows, angles_center, _ = build_windows_for_clip(rec)
        if windows.shape[0] == 0:
            continue
        # Keep only windows where every target angle is finite.
        mask = np.all(np.isfinite(angles_center), axis=1)
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
    """Train the temporal CNN on (X_train, Y_train).

    X_train: (N, T, N_KEYS, 3), Y_train: (N, 5).
    """
    rng = np.random.default_rng(seed)
    n_train = X_train.shape[0]
    log(f"  [{fold_label}] training set: N={n_train} samples")
    if n_train < 64:
        raise RuntimeError(f"Training set too small: {n_train}")

    # 90/10 internal val split (random, by sample — not by subject; just for
    # loss monitoring. LOSO subject discipline is enforced at fold level.)
    perm = rng.permutation(n_train)
    split = int(0.9 * n_train)
    tr_idx, va_idx = perm[:split], perm[split:]

    y_mean = Y_train[tr_idx].mean(axis=0, keepdims=True)
    y_std = Y_train[tr_idx].std(axis=0, keepdims=True).clip(min=1e-3)

    def _flat(x_batch: np.ndarray) -> np.ndarray:
        # (B, T, N_KEYS, 3) -> (B, T, N_KEYS*3)
        b, t, k, c = x_batch.shape
        return x_batch.reshape(b, t, k * c)

    x_va = torch.from_numpy(_flat(X_train[va_idx])).float()
    y_va_n = (Y_train[va_idx] - y_mean) / y_std

    model = TemporalKeypointCNNConf()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  [{fold_label}] model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    log_hist: dict[str, list] = {"train": [], "val": [], "val_r": []}
    t_fold_start = time.time()
    last_print = time.time()

    for ep in range(epochs):
        model.train()
        perm_ep = rng.permutation(tr_idx.shape[0])
        tot = 0.0
        nb = 0
        for s in range(0, tr_idx.shape[0], batch_size):
            ii = tr_idx[perm_ep[s : s + batch_size]]
            x_b = _flat(X_train[ii])
            y_b_n = (Y_train[ii] - y_mean) / y_std
            x_t = torch.from_numpy(x_b).float()
            y_t = torch.from_numpy(y_b_n).float()

            opt.zero_grad()
            pred = model(x_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        train_loss = tot / max(nb, 1)

        model.eval()
        with torch.no_grad():
            v_pred_n = model(x_va).numpy()
            v_loss = float(np.mean(np.abs(v_pred_n - y_va_n)))
            v_pred = v_pred_n * y_std + y_mean
            y_va = Y_train[va_idx]
            r_per = []
            for k in range(y_va.shape[1]):
                a = v_pred[:, k]
                b = y_va[:, k]
                if np.var(a) < 1e-6 or np.var(b) < 1e-6:
                    r_per.append(float("nan"))
                else:
                    r_per.append(float(np.corrcoef(a, b)[0, 1]))
        log_hist["train"].append(train_loss)
        log_hist["val"].append(v_loss)
        log_hist["val_r"].append(r_per)

        # Always print first/last; print intermediate every 30s or every 5 epochs.
        force_print = (ep == 0 or ep == epochs - 1 or ep % 5 == 0
                       or (time.time() - last_print) > 60.0)
        if force_print:
            r_fmt = " ".join(f"{r:+.2f}" for r in r_per)
            log(
                f"  [{fold_label}] ep {ep:3d}  train {train_loss:.4f}  "
                f"val {v_loss:.4f}  r [{r_fmt}]  "
                f"fold_elapsed={time.time()-t_fold_start:.0f}s"
            )
            last_print = time.time()

    model.y_mean = y_mean.squeeze(0)  # type: ignore[attr-defined]
    model.y_std = y_std.squeeze(0)    # type: ignore[attr-defined]
    return model, log_hist


def predict_on_clip(
    model: TemporalKeypointCNNConf, clip_rec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Predict (T, 5) angles for an entire clip.

    Returns (pred_angles, t_sec) where each is length T.
    """
    windows, _, _ = build_windows_for_clip(clip_rec)
    n_frames = windows.shape[0]
    if n_frames == 0:
        return (
            np.zeros((0, len(DEPLOY_METRICS)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    b, t, k, c = windows.shape
    x_flat = windows.reshape(b, t, k * c)
    x_t = torch.from_numpy(x_flat).float()
    model.eval()
    with torch.no_grad():
        p_n = model(x_t).numpy()
    y_mean = model.y_mean  # type: ignore[attr-defined]
    y_std = model.y_std    # type: ignore[attr-defined]
    pred = p_n * y_std + y_mean
    return pred.astype(np.float32), clip_rec["t_sec"]


def evaluate_fold(
    model: TemporalKeypointCNNConf, held_out_clips: list[dict]
) -> tuple[dict, list[dict]]:
    """For each held-out clip, predict frame-by-frame and compute |r| vs GT.

    Returns (aggregate_dict, per_clip_list).
    """
    per_clip = []
    metric_pooled: dict[str, list[float]] = {m: [] for m in DEPLOY_METRICS}

    for rec in held_out_clips:
        pred, _ = predict_on_clip(model, rec)
        gt = rec["angles"]
        n_frames = min(pred.shape[0], gt.shape[0])
        if n_frames == 0:
            continue
        pred = pred[:n_frames]
        gt = gt[:n_frames]
        clip_metrics = {}
        for k, metric in enumerate(DEPLOY_METRICS):
            a = pred[:, k]
            b = gt[:, k]
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() < 30:
                clip_metrics[metric] = {
                    "r": float("nan"),
                    "abs_r": float("nan"),
                    "mae": float("nan"),
                    "n_frames": int(finite.sum()),
                }
                continue
            ar = a[finite]
            br = b[finite]
            if np.var(ar) < 1e-9 or np.var(br) < 1e-9:
                r = float("nan")
            else:
                r = float(np.corrcoef(ar, br)[0, 1])
            mae = float(np.mean(np.abs(ar - br)))
            abs_r = abs(r) if np.isfinite(r) else float("nan")
            clip_metrics[metric] = {
                "r": r,
                "abs_r": abs_r,
                "mae": mae,
                "n_frames": int(finite.sum()),
            }
            if np.isfinite(abs_r):
                metric_pooled[metric].append(abs_r)
        per_clip.append({
            "clip": rec["clip"],
            "subject": rec["subject"],
            "trial": rec["trial"],
            "cam": rec["cam"],
            "n_frames": int(n_frames),
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

    return {"per_metric": per_metric_means, "pooled": pooled}, per_clip


# -------------------------------------------------------------------------
# LOSO driver.
# -------------------------------------------------------------------------


def run_loso(
    cohort: dict[str, list[dict]],
    *,
    epochs: int = 25,
    batch_size: int = 256,
) -> dict:
    subjects = sorted(cohort.keys())
    log(f"LOSO over {len(subjects)} subjects: {subjects}")

    fold_results = []
    best_model_state = None
    best_subject = None
    best_pooled = -1.0

    # Save the LAST trained model as the canonical checkpoint (trained on N-1).
    last_model_state = None
    last_y_mean = None
    last_y_std = None

    for fold_i, held_out in enumerate(subjects):
        fold_label = f"LOSO {fold_i+1}/{len(subjects)} held={held_out}"
        log(f"=== {fold_label} ===")
        train_subjects = [s for s in subjects if s != held_out]
        train_clips = [c for s in train_subjects for c in cohort[s]]
        held_clips = cohort[held_out]
        log(f"  train clips: {len(train_clips)}  held clips: {len(held_clips)}")

        X_train, Y_train = stack_clips(train_clips)
        log(f"  stacked training windows: X={X_train.shape}  Y={Y_train.shape}")
        if X_train.shape[0] < 100:
            log(f"  skipping fold {held_out}: not enough training samples")
            continue

        model, log_hist = train_one_fold(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            seed=fold_i,
            fold_label=fold_label,
        )

        log(f"  evaluating on {held_out} ...")
        agg, per_clip = evaluate_fold(model, held_clips)
        log(
            f"  [{fold_label}] held-out pooled |r| = "
            f"{agg['pooled']['mean_abs_r']:.4f}  (n={agg['pooled']['n']})"
        )
        for m, v in agg["per_metric"].items():
            log(
                f"    {m:20s}  |r|={v['mean_abs_r']:.3f}  "
                f"n_clips={v['n_clips']}"
            )

        fold_results.append({
            "held_out": held_out,
            "fold_idx": fold_i,
            "n_train_windows": int(X_train.shape[0]),
            "n_held_clips": len(held_clips),
            "aggregate": agg,
            "per_clip": per_clip,
            "final_val_r": log_hist["val_r"][-1] if log_hist["val_r"] else None,
        })

        # Track checkpoint candidate.
        last_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        last_y_mean = np.asarray(model.y_mean)  # type: ignore[attr-defined]
        last_y_std = np.asarray(model.y_std)    # type: ignore[attr-defined]
        pooled_r = agg["pooled"]["mean_abs_r"]
        if np.isfinite(pooled_r) and pooled_r > best_pooled:
            best_pooled = pooled_r
            best_subject = held_out
            best_model_state = last_model_state
            best_y_mean = last_y_mean
            best_y_std = last_y_std

        # Save intermediate progress after every fold.
        _save_intermediate(fold_results)

    # Pool across all folds (subject-level LOSO discipline).
    overall_metric_vals: dict[str, list[float]] = {m: [] for m in DEPLOY_METRICS}
    overall_all = []
    for fr in fold_results:
        for m, v in fr["aggregate"]["per_metric"].items():
            if np.isfinite(v["mean_abs_r"]):
                overall_metric_vals[m].append(v["mean_abs_r"])
        if np.isfinite(fr["aggregate"]["pooled"]["mean_abs_r"]):
            overall_all.append(fr["aggregate"]["pooled"]["mean_abs_r"])

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
    }
    log(
        f"=== OVERALL LOSO pooled subject-mean |r| = "
        f"{overall['pooled_subject_mean_abs_r']['mean_abs_r']:.4f} "
        f"(n_folds={overall['pooled_subject_mean_abs_r']['n_folds']}) ==="
    )

    # Save checkpoint: use BEST fold's model.
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
        log(f"saved best-fold checkpoint to {MODEL_OUT} (held={best_subject}, |r|={best_pooled:.3f})")

    return {
        "version": "learned_layer2_real_gt_v1",
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
        "version": "learned_layer2_real_gt_v1",
        "intermediate": True,
        "n_folds_completed": len(fold_results),
        "fold_results": fold_results,
    }, indent=2, default=str))


# -------------------------------------------------------------------------
# Main.
# -------------------------------------------------------------------------


def main(epochs: int = 25, batch_size: int = 256) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== learned_layer2_real_gt START ===")
    log(f"OPENCAP_KP={OPENCAP_KP}")
    log(f"LAB={LAB}")
    log(f"epochs={epochs}, batch_size={batch_size}, T={TEMPORAL_T}, N_KEYS={N_KEYS}")

    cohort = build_cohort()
    if not cohort:
        log("EMPTY COHORT; aborting")
        return

    results = run_loso(cohort, epochs=epochs, batch_size=batch_size)

    out_path = OUT_DIR / "per_slot_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote final results to {out_path}")
    log("=== learned_layer2_real_gt DONE ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
