"""Learned Layer 2 — combined OpenCap + ASPset cohort + per-source target heads.

Agent MM build. Implements HH2's own recommended fix: per-source output heads
for the two metrics where OpenCap and ASPset use different angle conventions
(hip_adduction_r, lumbar_extension). HH2 REPORT (verbatim):

    > hip_adduction_r regressed on OC-held (−0.051). Root cause: ASPset's
    > asin-based frontal-plane definition is geometrically not the same as
    > OpenSim's lumped-rotation hip_adduction. Future fix: drop
    > hip_adduction_r from ASPset training, or use per-source target heads.

LL chose the "drop" route. On the bias-limited slot
(hip_adduction_r / front_oblique_left, LoA ±3.3°), the v24 build collapsed
to CCC −0.77, confirming that dropping supervision alone is not the right
medicine. MM tries the alternative: separate output heads per source so
each head sees a clean target signal in its own convention.

# Architecture (per-source heads)

`TemporalKeypointCNNConfPerSource`:

  - Backbone: same 2-layer 1D conv over (T_win=9, 22 keypoints * 3 channels)
    used by HH2. Hidden=128.
  - Heads:
      * shared head -> 5 outputs (hip_flex_r, hip_add_r, knee_r, ankle_r,
        lumbar_ext) trained on OpenCap supervision only for the
        convention-mismatched outputs (hip_add_r, lumbar_ext); on shared
        supervision (OC + ASP) for the rest. This is the head we deploy.
      * aspset head -> 2 outputs (hip_add_r_asp, lumbar_ext_asp) trained
        only on ASPset clips. Lets the backbone pick up the shared
        representation signal from ASPset without polluting the OC head.

The two heads share the full backbone. Forward returns (shared_5d,
aspset_2d). Loss routing per sample/source described below.

# Loss routing

A training sample carries a `source` flag (`opencap` or `aspset`).

  * For shared outputs (hip_flex_r, knee_r, ankle_r): loss against shared
    head with masking. Both sources train these.
  * For shared outputs (hip_add_r, lumbar_ext):
      - OpenCap samples: loss against shared head's columns. These are the
        deployed outputs.
      - ASPset samples: loss against ASPset head's columns. The shared
        head is *not* updated from ASPset on these metrics.
  * For ASPset head outputs (only the 2 mismatched metrics):
      - OpenCap samples: NaN target -> excluded.
      - ASPset samples: loss against the source-converted ASPset target
        (same convention alignment as HH2 / LL).

In other words: ASPset's hip_add_r/lumbar_ext gradient flows ONLY into
the ASPset head, never into the shared (deployed) head. The shared head
sees ONLY OpenCap supervision for these two metrics. The backbone still
receives a gradient from every sample.

# Inference

Inference uses the SHARED 5d head. The ASPset head is discarded at
deploy. The runtime API (single OpenCap-equivalent 5-vector per frame)
is unchanged from EE2/HH2/LL.

# Two loss variants (MM-A and MM-B)

  * MM-A: per-source heads + per-frame masked SmoothL1 (HH2 loss).
    Cheaper to train, matches HH2's regime.
  * MM-B: per-source heads + ROM-aware (per-frame SmoothL1 + lam *
    extrema). Matches LL's loss. The hypothesis is that the
    bias-limited hip_adduction_r/front_oblique_left slot (v20 CCC 0.69,
    LoA ±3.3°) benefits more from extrema pinning than from per-frame
    waveform fidelity.

Both variants share the same architecture; they differ only in the loss
function used during training.

# Training other config

  * 24-fold subject-level LOSO (9 OpenCap + 15 ASPset).
  * 25 epochs (matches HH2/LL default), AdamW lr=1e-3 wd=1e-4, cosine LR.
  * MM-A batch=256 (frame-shuffled batches like HH2).
  * MM-B clips_per_step=4 (full-clip mini-batches like LL, required for
    extrema computation).
  * CPU only.

# Reuses HH2's cohort loader and convention alignment.
"""
from __future__ import annotations

import builtins
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Final

import numpy as np

for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.learned_layer2_combined import (
    DEPLOY_METRICS,
    HALPE_KEYS_WITH_SMPL,
    N_KEYS,
    TEMPORAL_T,
    build_combined_cohort,
    build_windows_for_clip,
)


# -------------------------------------------------------------------------
# Paths and constants.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
OUT_DIR_A: Final[Path] = DATA_ROOT / "learned_layer2_persource_perframe"
OUT_DIR_B: Final[Path] = DATA_ROOT / "learned_layer2_persource_romaware"
MODEL_OUT_A: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_persource_perframe_v1.pt"
)
MODEL_OUT_B: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_persource_romaware_v1.pt"
)
MODEL_OUT_A_ALLDATA: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_persource_perframe_alldata_v1.pt"
)
MODEL_OUT_B_ALLDATA: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_persource_romaware_alldata_v1.pt"
)


HIP_ADD_IDX: Final[int] = DEPLOY_METRICS.index("hip_adduction_r")
LUMBAR_IDX: Final[int] = DEPLOY_METRICS.index("lumbar_extension")
# Indices of the two per-source metrics (within DEPLOY_METRICS).
PER_SOURCE_IDX: Final[tuple[int, int]] = (HIP_ADD_IDX, LUMBAR_IDX)


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[MM t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Model with per-source heads.
# -------------------------------------------------------------------------


class TemporalKeypointCNNConfPerSource(nn.Module):
    """HH2-architecture backbone + shared 5d head + ASPset 2d head.

    Shared head outputs 5 metrics in OpenCap convention. ASPset head
    outputs 2 metrics in ASPset convention (hip_adduction_r,
    lumbar_extension) — the two metrics where the conventions differ.
    """

    def __init__(
        self,
        n_keys: int = N_KEYS,
        n_out_shared: int = len(DEPLOY_METRICS),
        n_out_aspset: int = 2,
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
        self.head_shared = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_out_shared),
        )
        self.head_aspset = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_out_aspset),
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        center = x.shape[-1] // 2
        return x[..., center]

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self._features(x)
        return self.head_shared(feats), self.head_aspset(feats)

    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path: shared 5d head only. Used at deploy."""
        return self.head_shared(self._features(x))


# -------------------------------------------------------------------------
# Routing helper: build (Y_shared, Y_aspset) per-sample mask + target arrays.
# -------------------------------------------------------------------------


def route_targets(
    angles: np.ndarray, source: str
) -> tuple[np.ndarray, np.ndarray]:
    """Given an angle vector (n_metrics,) and a source flag, return:

      y_shared: shape (n_metrics,) — finite values only where this sample
                supervises the SHARED head.
      y_aspset: shape (2,)         — finite values only where this sample
                supervises the ASPset head.

    Routing rules:
      - hip_flexion_r, knee_angle_r, ankle_angle_r: train shared head
        with both sources (carry whatever finite values the sample has).
      - hip_adduction_r, lumbar_extension:
          * opencap sample -> goes to SHARED head; ASPset head target NaN.
          * aspset sample  -> goes to ASPset head; SHARED head target NaN.

    Inputs assumed already convention-aligned (HH2's
    `_align_aspset_angles` ran upstream, so the ASPset hip_add/lumbar
    columns are in ASPset's native convention recoded into the same
    array slot as the OpenCap targets). This function does NOT re-align;
    it only routes which head consumes them.
    """
    y_shared = angles.copy()
    y_aspset = np.full(2, np.nan, dtype=angles.dtype)

    if source == "aspset":
        # ASPset routing: hip_add and lumbar go to ASPset head, NaN out of shared.
        for i_aspset, j_metric in enumerate(PER_SOURCE_IDX):
            y_aspset[i_aspset] = angles[j_metric]
            y_shared[j_metric] = np.nan
    elif source == "opencap":
        # OpenCap routing: shared head sees everything; ASPset head stays NaN.
        pass
    else:
        # Unknown source: keep everything in shared, ASPset head all NaN.
        pass
    return y_shared, y_aspset


# -------------------------------------------------------------------------
# Windowing helpers (reuse HH2's build_windows_for_clip).
# -------------------------------------------------------------------------


def stack_clips_persource(
    clip_recs: list[dict], stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like HH2's stack_clips but also returns y_aspset.

    Returns:
        X:        (N, T_win, N_KEYS, 3)
        Y_shared: (N, n_metrics)
        Y_aspset: (N, 2)
    """
    X_parts: list[np.ndarray] = []
    Ys_parts: list[np.ndarray] = []
    Ya_parts: list[np.ndarray] = []

    for rec in clip_recs:
        windows, angles_center = build_windows_for_clip(rec)
        if windows.shape[0] == 0:
            continue
        if stride > 1:
            windows = windows[::stride]
            angles_center = angles_center[::stride]
        # Route each frame's targets according to clip source.
        source = rec.get("source", "unknown")
        n_frames_clip = angles_center.shape[0]
        y_shared = np.full(
            (n_frames_clip, len(DEPLOY_METRICS)), np.nan, dtype=np.float32
        )
        y_aspset = np.full((n_frames_clip, 2), np.nan, dtype=np.float32)
        for f in range(n_frames_clip):
            ys, ya = route_targets(angles_center[f], source)
            y_shared[f] = ys
            y_aspset[f] = ya
        # Keep windows with ANY finite target in EITHER head.
        any_finite = (
            np.any(np.isfinite(y_shared), axis=1)
            | np.any(np.isfinite(y_aspset), axis=1)
        )
        if not np.any(any_finite):
            continue
        X_parts.append(windows[any_finite])
        Ys_parts.append(y_shared[any_finite])
        Ya_parts.append(y_aspset[any_finite])

    if not X_parts:
        return (
            np.zeros((0, TEMPORAL_T, N_KEYS, 3), dtype=np.float32),
            np.zeros((0, len(DEPLOY_METRICS)), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(Ys_parts, axis=0),
        np.concatenate(Ya_parts, axis=0),
    )


# -------------------------------------------------------------------------
# Masked SmoothL1 loss (same as HH2).
# -------------------------------------------------------------------------


def masked_smooth_l1(
    pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0.0
    target_safe = torch.where(mask, target, torch.zeros_like(target))
    diff = pred - target_safe
    abs_diff = diff.abs()
    sq = 0.5 * diff * diff / beta
    lin = abs_diff - 0.5 * beta
    loss = torch.where(abs_diff < beta, sq, lin)
    loss = loss * mask.float()
    return loss.sum() / mask.float().sum().clamp(min=1.0)


# -------------------------------------------------------------------------
# Per-clip tensor preparation for MM-B (ROM-aware) — mirrors LL.
# -------------------------------------------------------------------------


def _flatten(x: np.ndarray) -> np.ndarray:
    n, t, k, c = x.shape
    return x.reshape(n, t, k * c)


def prepare_clip_tensors_persource(clip_recs: list[dict]) -> list[dict]:
    """Build per-clip torch tensors for MM-B (ROM-aware) training.

    For each clip:
        x:              (T_frames, T_win, K*3)
        y_shared:       (T_frames, n_metrics)
        finite_shared:  (T_frames, n_metrics) bool
        y_aspset:       (T_frames, 2)
        finite_aspset:  (T_frames, 2) bool
    """
    out: list[dict] = []
    for rec in clip_recs:
        windows, angles = build_windows_for_clip(rec)
        if windows.shape[0] == 0:
            continue
        source = rec.get("source", "unknown")
        n_frames_clip = angles.shape[0]
        y_shared = np.full(
            (n_frames_clip, len(DEPLOY_METRICS)), np.nan, dtype=np.float32
        )
        y_aspset = np.full((n_frames_clip, 2), np.nan, dtype=np.float32)
        for f in range(n_frames_clip):
            ys, ya = route_targets(angles[f], source)
            y_shared[f] = ys
            y_aspset[f] = ya
        finite_shared = np.isfinite(y_shared)
        finite_aspset = np.isfinite(y_aspset)
        if not (finite_shared.any() or finite_aspset.any()):
            continue
        x_flat = _flatten(windows)
        out.append({
            "x": torch.from_numpy(x_flat).float(),
            "y_shared": torch.from_numpy(y_shared).float(),
            "finite_shared": torch.from_numpy(finite_shared),
            "y_aspset": torch.from_numpy(y_aspset).float(),
            "finite_aspset": torch.from_numpy(finite_aspset),
            "clip_id": rec["clip"],
            "subject": rec["subject"],
            "trial": rec["trial"],
            "cam": rec["cam"],
            "source": source,
        })
    return out


# -------------------------------------------------------------------------
# Target normalization.
# -------------------------------------------------------------------------


def _compute_target_norm(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_out = Y.shape[1]
    mean = np.zeros((1, n_out), dtype=np.float32)
    std = np.ones((1, n_out), dtype=np.float32)
    for k in range(n_out):
        col = Y[:, k]
        f = np.isfinite(col)
        if f.sum() < 10:
            continue
        mean[0, k] = float(col[f].mean())
        std[0, k] = max(float(col[f].std()), 1e-3)
    return mean, std


# -------------------------------------------------------------------------
# MM-A: per-source heads + per-frame SmoothL1 (HH2-style frame batches).
# -------------------------------------------------------------------------


def train_one_fold_mm_a(
    X_train: np.ndarray,
    Ys_train: np.ndarray,
    Ya_train: np.ndarray,
    *,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    seed: int = 0,
    fold_label: str = "",
) -> tuple[TemporalKeypointCNNConfPerSource, dict]:
    rng = np.random.default_rng(seed)
    n_train = X_train.shape[0]
    log(f"  [{fold_label}] training set: N={n_train} frames")
    if n_train < 64:
        raise RuntimeError(f"Training set too small: {n_train}")

    perm = rng.permutation(n_train)
    split = int(0.9 * n_train)
    tr_idx, va_idx = perm[:split], perm[split:]

    ys_mean, ys_std = _compute_target_norm(Ys_train[tr_idx])
    ya_mean, ya_std = _compute_target_norm(Ya_train[tr_idx])
    log(
        f"  [{fold_label}] shared y_mean={ys_mean.squeeze(0).round(2).tolist()} "
        f"y_std={ys_std.squeeze(0).round(2).tolist()}"
    )
    log(
        f"  [{fold_label}] aspset y_mean={ya_mean.squeeze(0).round(2).tolist()} "
        f"y_std={ya_std.squeeze(0).round(2).tolist()}"
    )

    def _flat(x: np.ndarray) -> np.ndarray:
        b, t, k, c = x.shape
        return x.reshape(b, t, k * c)

    Ys_va_n = (Ys_train[va_idx] - ys_mean) / ys_std
    Ya_va_n = (Ya_train[va_idx] - ya_mean) / ya_std
    x_va = torch.from_numpy(_flat(X_train[va_idx])).float()
    ys_va_t = torch.from_numpy(Ys_va_n.astype(np.float32))
    ya_va_t = torch.from_numpy(Ya_va_n.astype(np.float32))

    model = TemporalKeypointCNNConfPerSource()
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"  [{fold_label}] params={n_params:,} epochs={epochs} "
        f"batch={batch_size}"
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    log_hist: dict[str, list] = {"train": [], "val": []}
    t_fold_start = time.time()
    n_tr = tr_idx.shape[0]
    n_batches_per_epoch = (n_tr + batch_size - 1) // batch_size
    log(
        f"  [{fold_label}] n_tr={n_tr} batches_per_epoch="
        f"{n_batches_per_epoch}"
    )

    for ep in range(epochs):
        model.train()
        perm_ep = rng.permutation(n_tr)
        tot = 0.0
        nb = 0
        log(
            f"  [{fold_label}] ep {ep+1}/{epochs} START "
            f"fold_t={time.time()-t_fold_start:.0f}s"
        )
        last_print = time.time()
        for batch_idx, s in enumerate(range(0, n_tr, batch_size)):
            ii = tr_idx[perm_ep[s: s + batch_size]]
            x_b = _flat(X_train[ii])
            ys_b_raw = Ys_train[ii]
            ya_b_raw = Ya_train[ii]
            ys_b_n = (ys_b_raw - ys_mean) / ys_std
            ya_b_n = (ya_b_raw - ya_mean) / ya_std
            x_t = torch.from_numpy(x_b).float()
            ys_t = torch.from_numpy(ys_b_n.astype(np.float32))
            ya_t = torch.from_numpy(ya_b_n.astype(np.float32))
            opt.zero_grad()
            pred_s, pred_a = model(x_t)
            loss_shared = masked_smooth_l1(pred_s, ys_t)
            loss_aspset = masked_smooth_l1(pred_a, ya_t)
            loss = loss_shared + loss_aspset
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
            if (batch_idx % 25 == 0) or (time.time() - last_print > 20.0):
                log(
                    f"  [{fold_label}] ep {ep+1}/{epochs} "
                    f"batch {batch_idx}/{n_batches_per_epoch} "
                    f"loss={loss.item():.4f} "
                    f"(s={loss_shared.item():.4f} "
                    f"a={loss_aspset.item():.4f})"
                )
                last_print = time.time()
        sched.step()
        train_loss = tot / max(nb, 1)
        model.eval()
        with torch.no_grad():
            v_s, v_a = model(x_va)
            v_loss = float(
                (masked_smooth_l1(v_s, ys_va_t)
                 + masked_smooth_l1(v_a, ya_va_t)).item()
            )
        log_hist["train"].append(train_loss)
        log_hist["val"].append(v_loss)
        log(
            f"  [{fold_label}] ep {ep+1}/{epochs} END "
            f"train {train_loss:.4f} val {v_loss:.4f} "
            f"fold_t={time.time()-t_fold_start:.0f}s"
        )

    model.ys_mean = ys_mean.squeeze(0)  # type: ignore[attr-defined]
    model.ys_std = ys_std.squeeze(0)    # type: ignore[attr-defined]
    model.ya_mean = ya_mean.squeeze(0)  # type: ignore[attr-defined]
    model.ya_std = ya_std.squeeze(0)    # type: ignore[attr-defined]
    return model, log_hist


# -------------------------------------------------------------------------
# MM-B: per-source heads + ROM-aware loss (LL-style clip batches).
# -------------------------------------------------------------------------


def extrema_aware_masked_loss_persource(
    pred_shared_list: list[torch.Tensor],
    pred_aspset_list: list[torch.Tensor],
    gt_shared_list: list[torch.Tensor],
    gt_aspset_list: list[torch.Tensor],
    finite_shared_list: list[torch.Tensor],
    finite_aspset_list: list[torch.Tensor],
    *,
    lam: float = 1.0,
    smoothl1_beta: float = 1.0,
    min_finite_for_extrema: int = 5,
) -> tuple[torch.Tensor, dict]:
    """ROM-aware loss applied to BOTH heads independently."""
    # ---- per-frame loss for each head pooled over batch ----
    def _pool_smoothl1(pred_list, gt_list, mask_list, beta):
        frame_pred_parts = []
        frame_gt_parts = []
        for p, g, mask in zip(pred_list, gt_list, mask_list):
            if mask.any():
                frame_pred_parts.append(p[mask])
                frame_gt_parts.append(g[mask])
        if frame_pred_parts:
            all_pred = torch.cat(frame_pred_parts, dim=0)
            all_gt = torch.cat(frame_gt_parts, dim=0)
            return nn.functional.smooth_l1_loss(
                all_pred, all_gt, beta=beta
            )
        return sum(p.sum() * 0.0 for p in pred_list) \
            if pred_list else torch.zeros((), requires_grad=True)

    frame_shared = _pool_smoothl1(
        pred_shared_list, gt_shared_list, finite_shared_list, smoothl1_beta
    )
    frame_aspset = _pool_smoothl1(
        pred_aspset_list, gt_aspset_list, finite_aspset_list, smoothl1_beta
    )

    def _extrema(pred_list, gt_list, mask_list):
        peak_terms = []
        min_terms = []
        for p, g, mask in zip(pred_list, gt_list, mask_list):
            _T, M = p.shape
            for m in range(M):
                mm = mask[:, m]
                if int(mm.sum().item()) < min_finite_for_extrema:
                    continue
                p_m = p[mm, m]
                g_m = g[mm, m]
                peak_terms.append(torch.abs(p_m.amax() - g_m.amax()))
                min_terms.append(torch.abs(p_m.amin() - g_m.amin()))
        if peak_terms:
            return (
                torch.stack(peak_terms).mean(),
                torch.stack(min_terms).mean(),
                len(peak_terms),
            )
        return (
            torch.zeros((), requires_grad=True),
            torch.zeros((), requires_grad=True),
            0,
        )

    peak_s, min_s, n_s = _extrema(
        pred_shared_list, gt_shared_list, finite_shared_list
    )
    peak_a, min_a, n_a = _extrema(
        pred_aspset_list, gt_aspset_list, finite_aspset_list
    )

    total = (
        frame_shared + lam * peak_s + lam * min_s
        + frame_aspset + lam * peak_a + lam * min_a
    )
    return total, {
        "frame_s": float(frame_shared.detach().item()),
        "peak_s": float(peak_s.detach().item()),
        "min_s": float(min_s.detach().item()),
        "frame_a": float(frame_aspset.detach().item()),
        "peak_a": float(peak_a.detach().item()),
        "min_a": float(min_a.detach().item()),
        "n_clip_extrema_s": n_s,
        "n_clip_extrema_a": n_a,
    }


def _target_norm_from_clips(
    clip_data: list[dict], key_y: str, key_finite: str, n_metrics: int
) -> tuple[np.ndarray, np.ndarray]:
    y_mean = np.zeros(n_metrics, dtype=np.float32)
    y_std = np.ones(n_metrics, dtype=np.float32)
    for m in range(n_metrics):
        vals = []
        for d in clip_data:
            y_np = d[key_y].numpy()
            mm = d[key_finite].numpy()[:, m]
            if mm.any():
                vals.append(y_np[mm, m])
        if vals:
            arr = np.concatenate(vals)
            y_mean[m] = float(arr.mean())
            y_std[m] = max(float(arr.std()), 1e-3)
    return y_mean, y_std


def train_one_fold_mm_b(
    train_clips: list[dict],
    *,
    epochs: int = 25,
    clips_per_step: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lam: float = 1.0,
    seed: int = 0,
    fold_label: str = "",
) -> tuple[TemporalKeypointCNNConfPerSource, dict]:
    rng = np.random.default_rng(seed)
    log(
        f"  [{fold_label}] preparing per-clip tensors from "
        f"{len(train_clips)} clips ..."
    )
    t_prep = time.time()
    clip_data = prepare_clip_tensors_persource(train_clips)
    log(
        f"  [{fold_label}] prepared {len(clip_data)} usable clips "
        f"in {time.time()-t_prep:.0f}s"
    )
    if len(clip_data) < 4:
        raise RuntimeError(f"Too few training clips: {len(clip_data)}")

    ys_mean, ys_std = _target_norm_from_clips(
        clip_data, "y_shared", "finite_shared", len(DEPLOY_METRICS)
    )
    ya_mean, ya_std = _target_norm_from_clips(
        clip_data, "y_aspset", "finite_aspset", 2
    )
    ys_mean_t = torch.from_numpy(ys_mean).float()
    ys_std_t = torch.from_numpy(ys_std).float()
    ya_mean_t = torch.from_numpy(ya_mean).float()
    ya_std_t = torch.from_numpy(ya_std).float()
    log(
        f"  [{fold_label}] shared y_mean={ys_mean.round(2).tolist()} "
        f"y_std={ys_std.round(2).tolist()}"
    )
    log(
        f"  [{fold_label}] aspset y_mean={ya_mean.round(2).tolist()} "
        f"y_std={ya_std.round(2).tolist()}"
    )

    # Pre-normalize.
    for d in clip_data:
        ys_n = (d["y_shared"] - ys_mean_t) / ys_std_t
        ya_n = (d["y_aspset"] - ya_mean_t) / ya_std_t
        d["y_shared_n"] = torch.where(
            d["finite_shared"], ys_n, torch.zeros_like(ys_n)
        )
        d["y_aspset_n"] = torch.where(
            d["finite_aspset"], ya_n, torch.zeros_like(ya_n)
        )

    model = TemporalKeypointCNNConfPerSource()
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"  [{fold_label}] model params={n_params:,} epochs={epochs} "
        f"clips_per_step={clips_per_step} lam={lam}"
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_clips = len(clip_data)
    steps_per_epoch = max(1, n_clips // clips_per_step)
    log_hist: dict[str, list] = {"train_total": []}
    t_fold_start = time.time()

    for ep in range(epochs):
        model.train()
        log(
            f"  [{fold_label}] ep {ep+1}/{epochs} START "
            f"steps={steps_per_epoch} "
            f"fold_elapsed={time.time()-t_fold_start:.0f}s"
        )
        last_print = time.time()
        perm = rng.permutation(n_clips)
        ep_total = 0.0
        n_steps = 0

        for step in range(steps_per_epoch):
            start = step * clips_per_step
            picks = perm[start: start + clips_per_step]
            if len(picks) == 0:
                continue

            opt.zero_grad()
            pred_s_list: list[torch.Tensor] = []
            pred_a_list: list[torch.Tensor] = []
            gt_s_list: list[torch.Tensor] = []
            gt_a_list: list[torch.Tensor] = []
            mask_s_list: list[torch.Tensor] = []
            mask_a_list: list[torch.Tensor] = []
            for ci in picks:
                d = clip_data[ci]
                p_s, p_a = model(d["x"])
                pred_s_list.append(p_s)
                pred_a_list.append(p_a)
                gt_s_list.append(d["y_shared_n"])
                gt_a_list.append(d["y_aspset_n"])
                mask_s_list.append(d["finite_shared"])
                mask_a_list.append(d["finite_aspset"])

            total_loss, comp = extrema_aware_masked_loss_persource(
                pred_s_list, pred_a_list,
                gt_s_list, gt_a_list,
                mask_s_list, mask_a_list,
                lam=lam,
            )
            total_loss.backward()
            opt.step()

            tloss = float(total_loss.detach().item())
            ep_total += tloss
            n_steps += 1

            if (step % 5 == 0) or ((time.time() - last_print) > 15.0):
                log(
                    f"  [{fold_label}] ep {ep+1}/{epochs} step "
                    f"{step+1}/{steps_per_epoch} total={tloss:.4f} "
                    f"frame_s={comp['frame_s']:.3f} "
                    f"peak_s={comp['peak_s']:.3f} "
                    f"frame_a={comp['frame_a']:.3f} "
                    f"peak_a={comp['peak_a']:.3f} "
                    f"nMs={comp['n_clip_extrema_s']} "
                    f"nMa={comp['n_clip_extrema_a']}"
                )
                last_print = time.time()

        sched.step()
        ep_total /= max(n_steps, 1)
        log_hist["train_total"].append(ep_total)
        log(
            f"  [{fold_label}] ep {ep+1}/{epochs} END "
            f"total={ep_total:.4f} "
            f"fold_elapsed={time.time()-t_fold_start:.0f}s"
        )

    model.ys_mean = ys_mean  # type: ignore[attr-defined]
    model.ys_std = ys_std    # type: ignore[attr-defined]
    model.ya_mean = ya_mean  # type: ignore[attr-defined]
    model.ya_std = ya_std    # type: ignore[attr-defined]
    return model, log_hist


# -------------------------------------------------------------------------
# Inference + per-fold evaluation (uses SHARED head only).
# -------------------------------------------------------------------------


def predict_on_clip(
    model: TemporalKeypointCNNConfPerSource, rec: dict
) -> np.ndarray:
    """Predict 5-vector per frame using the SHARED head only."""
    windows, _ = build_windows_for_clip(rec)
    if windows.shape[0] == 0:
        return np.zeros((0, len(DEPLOY_METRICS)), dtype=np.float32)
    b, t, k, c = windows.shape
    x_t = torch.from_numpy(windows.reshape(b, t, k * c)).float()
    model.eval()
    with torch.no_grad():
        p_n = model.forward_shared(x_t).numpy()
    ys_mean = np.asarray(model.ys_mean)  # type: ignore[attr-defined]
    ys_std = np.asarray(model.ys_std)    # type: ignore[attr-defined]
    return (p_n * ys_std + ys_mean).astype(np.float32)


def _ccc(p: np.ndarray, o: np.ndarray) -> float:
    if len(p) < 2:
        return float("nan")
    mp, mo = float(np.mean(p)), float(np.mean(o))
    vp = float(np.var(p, ddof=0))
    vo = float(np.var(o, ddof=0))
    cov = float(np.mean((p - mp) * (o - mo)))
    denom = vp + vo + (mp - mo) ** 2
    if denom < 1e-12:
        return float("nan")
    return 2.0 * cov / denom


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    if np.var(a) < 1e-12 or np.var(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_fold(
    model: TemporalKeypointCNNConfPerSource, held_clips: list[dict]
) -> tuple[dict, list[dict]]:
    """Evaluate held-out clips. NOTE: held-out clips' angles array is
    convention-aligned upstream by HH2's loader. We compare against
    the convention-aligned target (the SHARED-head deploy target).

    Important caveat: for ASPset-held clips, the shared head sees the
    OpenCap convention but the held-out target is in ASPset's
    convention for the per-source metrics. We still compute |r| against
    that target for diagnostic purposes; the ASPset head's predictions
    are *not* used at deploy. The pooled per-source metrics on
    OpenCap-held folds are what matters."""
    per_clip = []
    per_metric_pooled: dict[str, list[float]] = {m: [] for m in DEPLOY_METRICS}
    rom_records: list[dict] = []

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
                clip_metrics[metric] = {
                    "r": float("nan"), "abs_r": float("nan"),
                    "mae": float("nan"), "n_frames": int(finite.sum()),
                }
                continue
            ar = a[finite]
            br = b[finite]
            r = _pearson(ar, br)
            mae = float(np.mean(np.abs(ar - br)))
            abs_r = abs(r) if np.isfinite(r) else float("nan")
            clip_metrics[metric] = {
                "r": r, "abs_r": abs_r, "mae": mae,
                "n_frames": int(finite.sum()),
            }
            if np.isfinite(abs_r):
                per_metric_pooled[metric].append(abs_r)
            rom_records.append({
                "clip": rec["clip"], "subject": rec["subject"],
                "source": rec["source"], "metric": metric,
                "pred_rom": float(ar.max() - ar.min()),
                "gt_rom": float(br.max() - br.min()),
            })
        per_clip.append({
            "clip": rec["clip"], "subject": rec["subject"],
            "source": rec["source"], "trial": rec["trial"],
            "cam": rec["cam"], "n_frames": int(n),
            "metrics": clip_metrics,
        })

    per_metric_means = {}
    all_vals: list[float] = []
    for m, vals in per_metric_pooled.items():
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
# LOSO drivers.
# -------------------------------------------------------------------------


def run_loso_mm_a(
    cohort: dict[str, list[dict]],
    *,
    epochs: int = 25,
    batch_size: int = 256,
    train_stride: int = 1,
) -> dict:
    subjects = sorted(cohort.keys())
    log(
        f"MM-A LOSO over {len(subjects)} subjects "
        f"(train_stride={train_stride})"
    )

    fold_results: list[dict] = []
    best_state = None
    best_ys_mean = None
    best_ys_std = None
    best_ya_mean = None
    best_ya_std = None
    best_subject = None
    best_pooled = -1.0

    for fold_i, held_out in enumerate(subjects):
        fold_label = f"fold {fold_i+1}/{len(subjects)} held={held_out}"
        log(f"=== {fold_label} ===")
        train_subjects = [s for s in subjects if s != held_out]
        train_clips = [c for s in train_subjects for c in cohort[s]]
        held_clips = cohort[held_out]
        log(
            f"  train clips: {len(train_clips)} "
            f"held clips: {len(held_clips)}"
        )

        X_train, Ys_train, Ya_train = stack_clips_persource(
            train_clips, stride=train_stride
        )
        log(
            f"  stacked windows: X={X_train.shape} "
            f"Ys={Ys_train.shape} Ya={Ya_train.shape} "
            f"(train_stride={train_stride})"
        )
        if X_train.shape[0] < 100:
            log("  SKIP: not enough train samples")
            continue

        try:
            model, _hist = train_one_fold_mm_a(
                X_train, Ys_train, Ya_train,
                epochs=epochs, batch_size=batch_size, seed=fold_i,
                fold_label=fold_label,
            )
        except Exception as e:
            log(f"  TRAINING FAILED: {e!r}")
            traceback.print_exc(file=sys.stdout)
            continue

        agg, per_clip = evaluate_fold(model, held_clips)
        log(
            f"  [{fold_label}] held-out pooled |r| = "
            f"{agg['pooled']['mean_abs_r']:.4f} "
            f"(n={agg['pooled']['n']})"
        )
        for m, v in agg["per_metric"].items():
            log(
                f"    {m:20s}  |r|={v['mean_abs_r']:.3f}  "
                f"n_clips={v['n_clips']}"
            )

        fold_results.append({
            "held_out": held_out,
            "source": "opencap" if held_out.startswith("opencap_") else "aspset",
            "fold_idx": fold_i,
            "n_train_windows": int(X_train.shape[0]),
            "n_held_clips": len(held_clips),
            "aggregate": agg,
            "per_clip": per_clip,
        })

        last_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        pooled_r = agg["pooled"]["mean_abs_r"]
        if np.isfinite(pooled_r) and pooled_r > best_pooled:
            best_pooled = pooled_r
            best_subject = held_out
            best_state = last_state
            best_ys_mean = np.asarray(model.ys_mean)  # type: ignore[attr-defined]
            best_ys_std = np.asarray(model.ys_std)    # type: ignore[attr-defined]
            best_ya_mean = np.asarray(model.ya_mean)  # type: ignore[attr-defined]
            best_ya_std = np.asarray(model.ya_std)    # type: ignore[attr-defined]

        _save_intermediate(
            fold_results, OUT_DIR_A, "mm_a_persource_perframe_v1"
        )

    overall = _aggregate_overall(fold_results)
    log(
        f"=== MM-A OVERALL pooled |r| = "
        f"{overall['pooled_subject_mean_abs_r']['mean_abs_r']:.4f} "
        f"(folds={overall['pooled_subject_mean_abs_r']['n_folds']}) ==="
    )

    if best_state is not None:
        MODEL_OUT_A.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": best_state,
            "ys_mean": best_ys_mean,
            "ys_std": best_ys_std,
            "ya_mean": best_ya_mean,
            "ya_std": best_ya_std,
            "n_keys": N_KEYS,
            "temporal_t": TEMPORAL_T,
            "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
            "metrics": list(DEPLOY_METRICS),
            "per_source_idx": list(PER_SOURCE_IDX),
            "model_class": "TemporalKeypointCNNConfPerSource",
            "training": "mm_a_persource_perframe_v1",
            "best_loso_subject": best_subject,
            "best_loso_pooled_abs_r": best_pooled,
        }, MODEL_OUT_A)
        log(
            f"saved best-fold MM-A checkpoint -> {MODEL_OUT_A} "
            f"(held={best_subject} |r|={best_pooled:.3f})"
        )

    return {
        "version": "mm_a_persource_perframe_v1",
        "epochs": epochs,
        "batch_size": batch_size,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "metrics": list(DEPLOY_METRICS),
        "fold_results": fold_results,
        "overall": overall,
    }


def run_loso_mm_b(
    cohort: dict[str, list[dict]],
    *,
    epochs: int = 25,
    clips_per_step: int = 4,
    lam: float = 1.0,
) -> dict:
    subjects = sorted(cohort.keys())
    log(f"MM-B LOSO over {len(subjects)} subjects")

    fold_results: list[dict] = []
    best_state = None
    best_ys_mean = None
    best_ys_std = None
    best_ya_mean = None
    best_ya_std = None
    best_subject = None
    best_pooled = -1.0

    for fold_i, held_out in enumerate(subjects):
        fold_label = f"fold {fold_i+1}/{len(subjects)} held={held_out}"
        log(f"=== {fold_label} ===")
        train_subjects = [s for s in subjects if s != held_out]
        train_clips = [c for s in train_subjects for c in cohort[s]]
        held_clips = cohort[held_out]
        log(
            f"  train clips: {len(train_clips)} "
            f"held clips: {len(held_clips)}"
        )

        try:
            model, _hist = train_one_fold_mm_b(
                train_clips,
                epochs=epochs, clips_per_step=clips_per_step,
                lam=lam, seed=fold_i, fold_label=fold_label,
            )
        except Exception as e:
            log(f"  TRAINING FAILED: {e!r}")
            traceback.print_exc(file=sys.stdout)
            continue

        agg, per_clip = evaluate_fold(model, held_clips)
        log(
            f"  [{fold_label}] held-out pooled |r| = "
            f"{agg['pooled']['mean_abs_r']:.4f} "
            f"(n={agg['pooled']['n']})"
        )
        for m, v in agg["per_metric"].items():
            log(
                f"    {m:20s}  |r|={v['mean_abs_r']:.3f}  "
                f"n_clips={v['n_clips']}"
            )

        fold_results.append({
            "held_out": held_out,
            "source": "opencap" if held_out.startswith("opencap_") else "aspset",
            "fold_idx": fold_i,
            "n_train_clips": len(train_clips),
            "n_held_clips": len(held_clips),
            "aggregate": agg,
            "per_clip": per_clip,
        })

        last_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        pooled_r = agg["pooled"]["mean_abs_r"]
        if np.isfinite(pooled_r) and pooled_r > best_pooled:
            best_pooled = pooled_r
            best_subject = held_out
            best_state = last_state
            best_ys_mean = np.asarray(model.ys_mean)  # type: ignore[attr-defined]
            best_ys_std = np.asarray(model.ys_std)    # type: ignore[attr-defined]
            best_ya_mean = np.asarray(model.ya_mean)  # type: ignore[attr-defined]
            best_ya_std = np.asarray(model.ya_std)    # type: ignore[attr-defined]

        _save_intermediate(
            fold_results, OUT_DIR_B, "mm_b_persource_romaware_v1"
        )

    overall = _aggregate_overall(fold_results)
    log(
        f"=== MM-B OVERALL pooled |r| = "
        f"{overall['pooled_subject_mean_abs_r']['mean_abs_r']:.4f} "
        f"(folds={overall['pooled_subject_mean_abs_r']['n_folds']}) ==="
    )

    if best_state is not None:
        MODEL_OUT_B.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": best_state,
            "ys_mean": best_ys_mean,
            "ys_std": best_ys_std,
            "ya_mean": best_ya_mean,
            "ya_std": best_ya_std,
            "n_keys": N_KEYS,
            "temporal_t": TEMPORAL_T,
            "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
            "metrics": list(DEPLOY_METRICS),
            "per_source_idx": list(PER_SOURCE_IDX),
            "model_class": "TemporalKeypointCNNConfPerSource",
            "training": "mm_b_persource_romaware_v1",
            "lam": lam,
            "best_loso_subject": best_subject,
            "best_loso_pooled_abs_r": best_pooled,
        }, MODEL_OUT_B)
        log(
            f"saved best-fold MM-B checkpoint -> {MODEL_OUT_B} "
            f"(held={best_subject} |r|={best_pooled:.3f})"
        )

    return {
        "version": "mm_b_persource_romaware_v1",
        "epochs": epochs,
        "clips_per_step": clips_per_step,
        "lam": lam,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "metrics": list(DEPLOY_METRICS),
        "fold_results": fold_results,
        "overall": overall,
    }


# -------------------------------------------------------------------------
# Aggregator + intermediate save helpers.
# -------------------------------------------------------------------------


def _aggregate_overall(fold_results: list[dict]) -> dict:
    overall_metric_vals: dict[str, list[float]] = {
        m: [] for m in DEPLOY_METRICS
    }
    overall_all = []
    by_source = {"opencap": [], "aspset": []}
    by_source_per_metric: dict[str, dict[str, list[float]]] = {
        src: {m: [] for m in DEPLOY_METRICS} for src in by_source
    }
    for fr in fold_results:
        for m, v in fr["aggregate"]["per_metric"].items():
            if np.isfinite(v["mean_abs_r"]):
                overall_metric_vals[m].append(v["mean_abs_r"])
        if np.isfinite(fr["aggregate"]["pooled"]["mean_abs_r"]):
            overall_all.append(fr["aggregate"]["pooled"]["mean_abs_r"])
        src = fr["source"]
        p = fr["aggregate"]["pooled"]["mean_abs_r"]
        if np.isfinite(p) and src in by_source:
            by_source[src].append(p)
        for m, v in fr["aggregate"]["per_metric"].items():
            if np.isfinite(v["mean_abs_r"]) and src in by_source_per_metric:
                by_source_per_metric[src][m].append(v["mean_abs_r"])

    return {
        "per_metric_subject_mean_abs_r": {
            m: {
                "n_folds": len(vals),
                "mean_abs_r": float(np.mean(vals)) if vals else float("nan"),
                "std_abs_r": float(np.std(vals)) if vals else float("nan"),
            } for m, vals in overall_metric_vals.items()
        },
        "pooled_subject_mean_abs_r": {
            "n_folds": len(overall_all),
            "mean_abs_r": (
                float(np.mean(overall_all)) if overall_all else float("nan")
            ),
        },
        "by_source": {
            src: {
                "n_folds": len(vals),
                "pooled_mean_abs_r": (
                    float(np.mean(vals)) if vals else float("nan")
                ),
            } for src, vals in by_source.items()
        },
        "by_source_per_metric": {
            src: {
                m: {
                    "n_folds": len(vals),
                    "mean_abs_r": (
                        float(np.mean(vals)) if vals else float("nan")
                    ),
                } for m, vals in d.items()
            } for src, d in by_source_per_metric.items()
        },
    }


def _save_intermediate(
    fold_results: list[dict], out_dir: Path, version: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_slot_results.json"
    out_path.write_text(json.dumps({
        "version": version,
        "intermediate": True,
        "n_folds_completed": len(fold_results),
        "fold_results": fold_results,
    }, indent=2, default=str))


# -------------------------------------------------------------------------
# All-data training (no LOSO at L2) — used by Phase B Layer 3.
# -------------------------------------------------------------------------


def train_all_data_mm_a(
    *,
    epochs: int = 25,
    batch_size: int = 256,
    train_stride: int = 1,
    seed: int = 42,
) -> tuple[TemporalKeypointCNNConfPerSource, dict]:
    cohort = build_combined_cohort()
    if not cohort:
        raise RuntimeError("combined cohort is empty; cannot train")
    all_clips = [c for clips in cohort.values() for c in clips]
    log(
        f"MM-A all-data: {len(cohort)} subjects, {len(all_clips)} clips, "
        f"stride={train_stride}"
    )
    X_train, Ys_train, Ya_train = stack_clips_persource(
        all_clips, stride=train_stride
    )
    log(
        f"  stacked all-data windows: X={X_train.shape} "
        f"Ys={Ys_train.shape} Ya={Ya_train.shape}"
    )
    model, hist = train_one_fold_mm_a(
        X_train, Ys_train, Ya_train,
        epochs=epochs, batch_size=batch_size, seed=seed,
        fold_label="ALL-DATA-MM-A",
    )
    return model, {
        "hist": hist,
        "n_subjects": len(cohort),
        "n_clips": len(all_clips),
        "subjects": sorted(cohort.keys()),
    }


def train_all_data_mm_b(
    *,
    epochs: int = 25,
    clips_per_step: int = 4,
    lam: float = 1.0,
    seed: int = 42,
) -> tuple[TemporalKeypointCNNConfPerSource, dict]:
    cohort = build_combined_cohort()
    if not cohort:
        raise RuntimeError("combined cohort is empty; cannot train")
    all_clips = [c for clips in cohort.values() for c in clips]
    log(
        f"MM-B all-data: {len(cohort)} subjects, {len(all_clips)} clips"
    )
    model, hist = train_one_fold_mm_b(
        all_clips,
        epochs=epochs, clips_per_step=clips_per_step,
        lam=lam, seed=seed, fold_label="ALL-DATA-MM-B",
    )
    return model, {
        "hist": hist,
        "n_subjects": len(cohort),
        "n_clips": len(all_clips),
        "subjects": sorted(cohort.keys()),
    }


# -------------------------------------------------------------------------
# Shim model for FF / Layer 3 wrapper compatibility.
# -------------------------------------------------------------------------


class _SharedHeadShim(nn.Module):
    """Wrap a TemporalKeypointCNNConfPerSource so callers expecting the
    HH2 single-head API (`model(x) -> (B, n_metrics)`) see only the
    shared head. FF's Layer 3 retrain code calls `model(x_t).numpy()`
    expecting a tensor, not a tuple."""

    def __init__(self, base: TemporalKeypointCNNConfPerSource) -> None:
        super().__init__()
        self.base = base
        # Forward the y_mean / y_std attributes the FF runtime expects.
        self.y_mean = np.asarray(getattr(base, "ys_mean"))
        self.y_std = np.asarray(getattr(base, "ys_std"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.forward_shared(x)


def wrap_for_ff(
    base: TemporalKeypointCNNConfPerSource,
) -> _SharedHeadShim:
    """Wrap an MM model so its shared head looks like a HH2 single-head
    model to FF/KK/LL Layer 3 ridge retrain code."""
    return _SharedHeadShim(base)


# -------------------------------------------------------------------------
# Main (LOSO eval driver).
# -------------------------------------------------------------------------


def main_mm_a(
    epochs: int = 25,
    batch_size: int = 256,
    train_stride: int = 1,
    max_subjects: int | None = None,
) -> dict:
    OUT_DIR_A.mkdir(parents=True, exist_ok=True)
    log("=== learned_layer2_combined_persource MM-A START ===")
    log(
        f"epochs={epochs} batch_size={batch_size} "
        f"T={TEMPORAL_T} N_KEYS={N_KEYS} "
        f"train_stride={train_stride}"
    )
    cohort = build_combined_cohort()
    if not cohort:
        log("EMPTY COHORT; aborting")
        return {}
    if max_subjects is not None:
        keep = sorted(cohort.keys())[:max_subjects]
        cohort = {k: cohort[k] for k in keep}
        log(f"DEBUG: limited to {len(cohort)} subjects: {keep}")
    results = run_loso_mm_a(
        cohort, epochs=epochs, batch_size=batch_size,
        train_stride=train_stride,
    )
    out_path = OUT_DIR_A / "per_slot_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote final MM-A results to {out_path}")
    log("=== learned_layer2_combined_persource MM-A DONE ===")
    return results


def main_mm_b(
    epochs: int = 25,
    clips_per_step: int = 4,
    lam: float = 1.0,
    max_subjects: int | None = None,
) -> dict:
    OUT_DIR_B.mkdir(parents=True, exist_ok=True)
    log("=== learned_layer2_combined_persource MM-B START ===")
    log(
        f"epochs={epochs} clips_per_step={clips_per_step} lam={lam} "
        f"T={TEMPORAL_T} N_KEYS={N_KEYS}"
    )
    cohort = build_combined_cohort()
    if not cohort:
        log("EMPTY COHORT; aborting")
        return {}
    if max_subjects is not None:
        keep = sorted(cohort.keys())[:max_subjects]
        cohort = {k: cohort[k] for k in keep}
        log(f"DEBUG: limited to {len(cohort)} subjects: {keep}")
    results = run_loso_mm_b(
        cohort, epochs=epochs, clips_per_step=clips_per_step, lam=lam,
    )
    out_path = OUT_DIR_B / "per_slot_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"wrote final MM-B results to {out_path}")
    log("=== learned_layer2_combined_persource MM-B DONE ===")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=["mm_a", "mm_b"], default="mm_a",
        help="MM-A = per-source heads + per-frame SmoothL1; "
             "MM-B = per-source heads + ROM-aware",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="MM-A frame batch size")
    parser.add_argument("--train-stride", type=int, default=4,
                        help="MM-A train stride")
    parser.add_argument("--clips-per-step", type=int, default=4,
                        help="MM-B clips per step")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="MM-B extrema-loss weight")
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()
    if args.variant == "mm_a":
        main_mm_a(
            epochs=args.epochs, batch_size=args.batch_size,
            train_stride=args.train_stride,
            max_subjects=args.max_subjects,
        )
    else:
        main_mm_b(
            epochs=args.epochs,
            clips_per_step=args.clips_per_step,
            lam=args.lam,
            max_subjects=args.max_subjects,
        )
