"""Agent OO — Extrema-aware learned Layer 3 (per-slot MLP) with auxiliary
max/min supervision.

Cameron's hypothesis: v32 has 11 Good slots. The 3 knee slots + 1 hip_flexion
slot all have strong CCC (0.84-0.93) but fail the LoA +/-10 deg gate by
1-3 deg. LoA is dominated by extrema accuracy. Agent NN's learned L3
(``learned_layer3.py``) only supervises ROM scalar (max - min). Directly
supervising max and min should tighten LoA.

This is GG2's ROM-aware loss logic, but applied at L3 (not L2). Two output
heads (pred_max, pred_min), ROM = pred_max - pred_min.

Loss = alpha * SmoothL1(pred_rom, gt_rom)
     + beta * SmoothL1(pred_max, gt_max)
     + beta * SmoothL1(pred_min, gt_min)

Defaults: alpha=1.0, beta=0.5.

# CLI

  * ``--l2 v23`` -> tag ``v33_extrema_l3``   (HH2 combined alldata L2)
  * ``--l2 v29`` -> tag ``v34_mirrorflip_extrema_l3``  (NN mirror-flip L2)

# Outputs

  * ``data/learned_layer3_extrema/per_slot_validity_{tag}.json``
  * ``results/deploy_ready_models_{tag}.json``
  * ``data/learned_layer3_extrema/REPORT.md``
"""
from __future__ import annotations

import argparse
import builtins
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.train_regression_poc import fit_ridge  # noqa: E402
from harness.learned_layer3 import (  # noqa: E402
    L2_V23_CKPT, L2_V29_CKPT,
    _setup_l2_v23, _setup_l2_v29, _patch_motion_data,
    _ccc, _pearson, _classify, _stats_per_subject, _loso_ridge,
    BASELINE_PATH, V17_PATH,
)
from harness.per_slot_prediction_dump import DumpAccumulator  # noqa: E402


# Agent SS per-clip dump infrastructure (extrema-aware variant). Mirrors
# the v30/v31 dump logic; records the chosen (ridge or learned_extrema)
# predictions per held-out subject.
DUMP_ACCUMULATOR: DumpAccumulator | None = None


warnings_filter = logging.getLogger("py.warnings")
warnings_filter.setLevel(logging.ERROR)


# -------------------------------------------------------------------------
# Paths.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
OUT_DIR: Final[Path] = DATA_ROOT / "learned_layer3_extrema"

LAB: Final[Path] = DATA_ROOT / "LabValidation_withVideos"
ASPSET: Final[Path] = DATA_ROOT / "aspset510" / "ASPset-510" / "trainval"


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[OO-L3 t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Per-clip ground-truth extrema computation.
# -------------------------------------------------------------------------


def _gt_extrema_from_mot(
    subject: str, task: str, col: str,
) -> tuple[float, float]:
    """OpenCap mocap IK output: max/min of the target angle column.

    Mirrors ``train_v9_phased.gt_rom_col`` -> returns (max, min) so that
    `gt_rom_col == max - min`.
    """
    from harness.parsers import parse_mot
    p = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not p.exists():
        return float("nan"), float("nan")
    arr = parse_mot(p).column(col)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan"), float("nan")
    return float(np.nanmax(arr)), float(np.nanmin(arr))


def _gt_extrema_from_aspset(
    c3d_path: Path, target_gt: str,
) -> tuple[float, float]:
    """ASPset c3d-derived joint angles: max/min for the target angle.

    Mirrors the inline computation in
    ``train_v12_combined.extract_aspset_features`` (line 145-149).
    """
    from harness.aspset_loader import load_clip, joint_angles_from_aspset
    try:
        clip = load_clip(c3d_path)
        angles = joint_angles_from_aspset(clip)
    except Exception:
        return float("nan"), float("nan")
    if target_gt not in angles:
        return float("nan"), float("nan")
    arr = angles[target_gt]
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        return float("nan"), float("nan")
    if not np.any(np.isfinite(arr)):
        return float("nan"), float("nan")
    return float(np.nanmax(arr)), float(np.nanmin(arr))


# -------------------------------------------------------------------------
# Forked builders: emit per-row (gt_max, gt_min) alongside X, y, subjects.
#
# We FORK rather than monkey-patch so the order of rows is guaranteed to
# match the X order. Each fork is a near-verbatim copy of the original
# builder with an added gt_max/gt_min capture.
# -------------------------------------------------------------------------


def _build_dataset_v9_with_extrema(target_gt: str, cam: str):
    """Fork of ``train_v9_phased.build_dataset_v9`` with extrema capture.

    Returns (X, y, gt_max, gt_min, subjects, tasks) tuple, or None if
    no rows survive feature/GT filters.
    """
    from harness import train_v9_phased as v9
    from harness.train_v9_phased import (
        extract_v9_features, gt_rom_col, MIRROR,
    )

    X_rows, y, subjects, tasks = [], [], [], []
    gt_max_list, gt_min_list = [], []

    for kp_path in sorted(v9.KP_DIR.glob(f"*_{cam}.json")):
        base = kp_path.stem
        parts = base.split("_")
        subject = parts[0]
        task = "_".join(parts[1:-1])
        if not task.startswith("DJ"):
            continue
        # Original sample only (flip_aug=False is the default used by
        # build_combined_dataset).
        gt = gt_rom_col(subject, task, target_gt)
        if not np.isfinite(gt):
            continue
        try:
            feats = extract_v9_features(kp_path, subject, cam, target_gt, flip=False)
        except Exception:
            continue
        vals = list(feats.values())
        if not all(np.isfinite(vals)):
            continue
        gt_mx, gt_mn = _gt_extrema_from_mot(subject, task, target_gt)
        if not np.isfinite(gt_mx) or not np.isfinite(gt_mn):
            continue
        X_rows.append(vals)
        y.append(float(gt))
        gt_max_list.append(gt_mx)
        gt_min_list.append(gt_mn)
        subjects.append(subject)
        tasks.append(task)

    if not X_rows:
        return None
    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y, dtype=np.float64),
        np.array(gt_max_list, dtype=np.float64),
        np.array(gt_min_list, dtype=np.float64),
        subjects, tasks,
    )


def _build_combined_dataset_with_extrema(target_gt: str, view: str):
    """Fork of ``train_v12_combined.build_combined_dataset`` with extrema
    capture.

    Returns (X, y, gt_max, gt_min, subjects, sources) or None.
    """
    from harness import train_v9_phased as v9
    from harness import train_v12_combined as v12
    from harness.train_v12_combined import (
        ASPSET_VALID_TARGETS, ASPSET_COMPATIBLE_TARGETS,
        compute_aspset_camera_yaws, classify_view_aspset,
        extract_aspset_features,
    )
    from harness.train_multi_metric import VIEW_BUCKET_NAMES

    if target_gt not in ASPSET_COMPATIBLE_TARGETS:
        return None

    X_rows, y, subjects, sources = [], [], [], []
    gt_max_list, gt_min_list = [], []

    # OpenCap path.
    cam_id = {v: k for k, v in VIEW_BUCKET_NAMES.items()}.get(view)
    if cam_id is not None:
        original = v9.TARGETS
        v9.TARGETS = {
            **original,
            target_gt: ASPSET_COMPATIBLE_TARGETS[target_gt],
        }
        try:
            oc = _build_dataset_v9_with_extrema(target_gt, cam_id)
        finally:
            v9.TARGETS = original
        if oc is not None:
            X_oc, y_oc, mx_oc, mn_oc, subj_oc, _ = oc
            for i in range(len(y_oc)):
                X_rows.append(X_oc[i].tolist())
                y.append(float(y_oc[i]))
                gt_max_list.append(float(mx_oc[i]))
                gt_min_list.append(float(mn_oc[i]))
                subjects.append(f"opencap_{subj_oc[i]}")
                sources.append("opencap")

    # ASPset path.
    if target_gt in ASPSET_VALID_TARGETS:
        for subj_dir in sorted((ASPSET / "joints_3d").iterdir()):
            subject = subj_dir.name
            for clip_path in sorted(subj_dir.glob("*.c3d")):
                clip_id = clip_path.stem.split("-", 1)[1]
                for asp_view in ["left", "mid", "right"]:
                    yaw = compute_aspset_camera_yaws(subject, asp_view)
                    if yaw is None:
                        continue
                    bucket = classify_view_aspset(yaw)
                    if bucket != view:
                        continue
                    result = extract_aspset_features(
                        clip_id, subject, asp_view, target_gt,
                    )
                    if result is None:
                        continue
                    feats, gt = result
                    vals = list(feats.values())
                    if not all(np.isfinite(vals)):
                        continue
                    # ASPset gt filter mirrors extract_aspset_features.
                    if (
                        target_gt == "hip_adduction_r" and gt > 90
                    ):
                        continue
                    c3d_path = (
                        ASPSET / "joints_3d" / subject
                        / f"{subject}-{clip_id}.c3d"
                    )
                    gt_mx, gt_mn = _gt_extrema_from_aspset(
                        c3d_path, target_gt,
                    )
                    if not np.isfinite(gt_mx) or not np.isfinite(gt_mn):
                        continue
                    X_rows.append(vals)
                    y.append(float(gt))
                    gt_max_list.append(gt_mx)
                    gt_min_list.append(gt_mn)
                    subjects.append(f"aspset_{subject}")
                    sources.append("aspset")

    if not X_rows:
        return None
    return (
        np.array(X_rows, dtype=np.float64),
        np.array(y, dtype=np.float64),
        np.array(gt_max_list, dtype=np.float64),
        np.array(gt_min_list, dtype=np.float64),
        subjects, sources,
    )


def _build_opencap_only_with_extrema(target_gt: str, view: str):
    """v14 OpenCap-only path (used for ankle_angle_r). Returns
    (X, y, gt_max, gt_min, subjects, sources) or None."""
    from harness.train_multi_metric import VIEW_BUCKET_NAMES
    cam_id = {v: k for k, v in VIEW_BUCKET_NAMES.items()}.get(view)
    if cam_id is None:
        return None
    oc = _build_dataset_v9_with_extrema(target_gt, cam_id)
    if oc is None:
        return None
    X_oc, y_oc, mx_oc, mn_oc, subj_oc, _ = oc
    subjects = [f"opencap_{s}" for s in subj_oc]
    sources = ["opencap"] * len(y_oc)
    return X_oc, y_oc, mx_oc, mn_oc, subjects, sources


# -------------------------------------------------------------------------
# Slot dispatcher.
# -------------------------------------------------------------------------


def _collect_features_with_extrema_for_slot(
    target: str, view: str, approach: str,
):
    """Return (X, y, gt_max, gt_min, subjects, sources) for one slot, using
    the same builder dispatch as ``learned_layer3._collect_features_for_slot``.

    Returns None if the slot's builder isn't supported by the extrema path
    (e.g. event_anchored / event_anchored_bilateral / v9_phased -- those
    contribute to a small handful of slots that we will fall back to ridge
    for).
    """
    from harness import train_v9_phased as v9
    from harness import train_v12_combined as v12

    def _build_v12(target, view):
        v9.KP_DIR = DATA_ROOT / "couro_keypoints"
        v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_couro_keypoints"
        return _build_combined_dataset_with_extrema(target, view)

    def _build_v13(target, view):
        v9.KP_DIR = DATA_ROOT / "couro_keypoints"
        v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
        return _build_combined_dataset_with_extrema(target, view)

    def _build_v14(target, view):
        v9.KP_DIR = DATA_ROOT / "opencap_dwpose_keypoints"
        v12.ASPSET_KEYPOINTS = DATA_ROOT / "aspset510_dwpose_keypoints"
        if target == "ankle_angle_r":
            return _build_opencap_only_with_extrema(target, view)
        return _build_combined_dataset_with_extrema(target, view)

    def _build_v9p(target, view):
        v9.KP_DIR = DATA_ROOT / "couro_keypoints"
        from harness.train_multi_metric import VIEW_BUCKET_NAMES
        cam = {v: k for k, v in VIEW_BUCKET_NAMES.items()}.get(view)
        if cam is None:
            return None
        oc = _build_dataset_v9_with_extrema(target, cam)
        if oc is None:
            return None
        X_oc, y_oc, mx_oc, mn_oc, subj_oc, _ = oc
        subjects = list(subj_oc)
        sources = ["opencap"] * len(y_oc)
        return X_oc, y_oc, mx_oc, mn_oc, subjects, sources

    BUILDERS = {
        "v12_combined": _build_v12,
        "v13_dwpose_hybrid": _build_v13,
        "v14_full_dwpose": _build_v14,
        "v9_phased": _build_v9p,
        # event_anchored / event_anchored_bilateral / unknown -> not supported;
        # caller falls back to v32 entry for these slots.
    }
    builder = BUILDERS.get(approach)
    if builder is None:
        return None
    try:
        result = builder(target, view)
    except Exception as e:
        log(f"  extrema builder for {target}/{view}/{approach} failed: {e!r}")
        return None
    return result


# -------------------------------------------------------------------------
# Extrema-aware TinyMLP with two heads.
# -------------------------------------------------------------------------


class TinyMLPL3Extrema(nn.Module):
    """Trunk MLP with two output heads predicting (max, min) per clip.

    ROM is derived as max - min downstream. Trunk: 2 hidden layers of
    ``hidden`` units, ReLU + dropout 0.2. Same capacity as NN's L3.
    """

    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_max = nn.Linear(hidden, 1)
        self.head_min = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        pred_max = self.head_max(h).squeeze(-1)
        pred_min = self.head_min(h).squeeze(-1)
        return pred_max, pred_min


@dataclass(frozen=True)
class LearnedL3ExtremaModel:
    """Per-slot extrema-aware L3 wrapper. Stores torch state dict, feature
    z-score, and de-normalized linear params for both heads.
    """

    state_dict: dict
    feature_mean: np.ndarray
    feature_std: np.ndarray
    hidden: int
    n_features: int

    def to_module(self) -> TinyMLPL3Extrema:
        mod = TinyMLPL3Extrema(self.n_features, self.hidden)
        mod.load_state_dict(self.state_dict)
        mod.eval()
        return mod

    def predict_max_min(
        self, X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        Xz = (X - self.feature_mean) / np.maximum(self.feature_std, 1e-9)
        mod = self.to_module()
        with torch.no_grad():
            pmx, pmn = mod(torch.from_numpy(Xz.astype(np.float32)))
        return (
            np.atleast_1d(pmx.numpy()),
            np.atleast_1d(pmn.numpy()),
        )

    def predict_rom(self, X: np.ndarray) -> np.ndarray:
        pmx, pmn = self.predict_max_min(X)
        return pmx - pmn


def _train_learned_l3_extrema(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gt_max_train: np.ndarray,
    gt_min_train: np.ndarray,
    *,
    hidden: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    alpha: float = 1.0,
    beta: float = 0.5,
    seed: int = 0,
) -> LearnedL3ExtremaModel:
    """Train one extrema-aware TinyMLP. Z-scores features inside. Inner 85/15
    split for early stopping. Returns a model with linear-output params
    de-normalized to original angle units.
    """
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    if n < 4:
        raise RuntimeError(f"learned L3 extrema needs n>=4, got n={n}")

    # Z-score features.
    fm = X_train.mean(axis=0)
    fs = X_train.std(axis=0)
    fs = np.where(fs < 1e-9, 1.0, fs)
    Xz = (X_train - fm) / fs

    # Z-score the max/min targets independently (helps optimizer; we
    # de-normalize at the end).
    max_mean = float(gt_max_train.mean())
    max_std = (
        float(gt_max_train.std()) if gt_max_train.std() > 1e-9 else 1.0
    )
    min_mean = float(gt_min_train.mean())
    min_std = (
        float(gt_min_train.std()) if gt_min_train.std() > 1e-9 else 1.0
    )
    # Shared scale for ROM term (so all three losses live in similar units).
    # We use the mean of max_std and min_std; both are typically ~similar.
    rom_scale = 0.5 * (max_std + min_std)
    if rom_scale < 1e-9:
        rom_scale = 1.0

    mx_n = (gt_max_train - max_mean) / max_std
    mn_n = (gt_min_train - min_mean) / min_std

    # Inner train/val split for early stopping.
    n_val = max(1, int(0.15 * n))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr = torch.from_numpy(Xz[tr_idx].astype(np.float32))
    X_va = torch.from_numpy(Xz[val_idx].astype(np.float32))
    mx_tr = torch.from_numpy(mx_n[tr_idx].astype(np.float32))
    mx_va = torch.from_numpy(mx_n[val_idx].astype(np.float32))
    mn_tr = torch.from_numpy(mn_n[tr_idx].astype(np.float32))
    mn_va = torch.from_numpy(mn_n[val_idx].astype(np.float32))
    # gt_rom in normalized space requires de-normalizing predictions
    # back into raw degrees inside the loss. For numerical stability we
    # compute ROM in raw degrees and re-scale by rom_scale before applying
    # SmoothL1.
    gt_rom_tr = torch.from_numpy(
        (y_train[tr_idx] / rom_scale).astype(np.float32)
    )
    gt_rom_va = torch.from_numpy(
        (y_train[val_idx] / rom_scale).astype(np.float32)
    )

    torch.manual_seed(seed)
    model = TinyMLPL3Extrema(n_features=X_train.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    def _step(X, gt_mx, gt_mn, gt_rom_norm):
        pmx_n, pmn_n = model(X)
        # ROM in (de-normalized -> raw degrees) / rom_scale.
        pmx_raw = pmx_n * max_std + max_mean
        pmn_raw = pmn_n * min_std + min_mean
        pred_rom = (pmx_raw - pmn_raw) / rom_scale
        loss_rom = nn.functional.smooth_l1_loss(pred_rom, gt_rom_norm)
        loss_max = nn.functional.smooth_l1_loss(pmx_n, gt_mx)
        loss_min = nn.functional.smooth_l1_loss(pmn_n, gt_mn)
        return alpha * loss_rom + beta * loss_max + beta * loss_min

    best_val = float("inf")
    best_state: dict | None = None
    patience = 25
    patience_left = patience

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = _step(X_tr, mx_tr, mn_tr, gt_rom_tr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = float(_step(X_va, mx_va, mn_va, gt_rom_va).item())
        if v_loss + 1e-6 < best_val:
            best_val = v_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    # Re-bake (mean, std) into the head layers so predict_max_min returns
    # raw degrees. head_max.weight = old * max_std; head_max.bias =
    # old * max_std + max_mean. Same for head_min.
    best_state["head_max.weight"] = (
        best_state["head_max.weight"] * max_std
    )
    best_state["head_max.bias"] = (
        best_state["head_max.bias"] * max_std + max_mean
    )
    best_state["head_min.weight"] = (
        best_state["head_min.weight"] * min_std
    )
    best_state["head_min.bias"] = (
        best_state["head_min.bias"] * min_std + min_mean
    )

    return LearnedL3ExtremaModel(
        state_dict=best_state,
        feature_mean=fm,
        feature_std=fs,
        hidden=hidden,
        n_features=X_train.shape[1],
    )


def _loso_learned_extrema(
    X: np.ndarray, y: np.ndarray,
    gt_max: np.ndarray, gt_min: np.ndarray,
    subjects: list[str],
    *,
    hidden: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    alpha: float = 1.0,
    beta: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LOSO at L3 with extrema-aware loss. Returns
    (pred_rom, pred_max, pred_min, obs_rom, subjects_arr).
    """
    subj_arr = np.array(subjects)
    unique = sorted(set(subjects))
    pred_rom = np.full(len(y), np.nan)
    pred_mx = np.full(len(y), np.nan)
    pred_mn = np.full(len(y), np.nan)
    for fi, s in enumerate(unique):
        train_mask = subj_arr != s
        test_mask = subj_arr == s
        if train_mask.sum() < 4 or test_mask.sum() < 1:
            continue
        try:
            m = _train_learned_l3_extrema(
                X[train_mask], y[train_mask],
                gt_max[train_mask], gt_min[train_mask],
                hidden=hidden, epochs=epochs, lr=lr,
                weight_decay=weight_decay,
                alpha=alpha, beta=beta, seed=seed + fi,
            )
            pmx, pmn = m.predict_max_min(X[test_mask])
            pred_mx[test_mask] = pmx
            pred_mn[test_mask] = pmn
            pred_rom[test_mask] = pmx - pmn
        except Exception:
            # Fold-level ridge fallback for stability.
            try:
                m_r = fit_ridge(
                    X[train_mask], y[train_mask], alpha=10.0,
                )
                pred_rom[test_mask] = m_r.predict(X[test_mask])
            except Exception:
                pred_rom[test_mask] = float(np.mean(y[train_mask]))
    return pred_rom, pred_mx, pred_mn, y, subj_arr


# -------------------------------------------------------------------------
# Tier rank for promotion checks.
# -------------------------------------------------------------------------


TIER_RANK: Final[dict[str, int]] = {
    "Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1,
}


# -------------------------------------------------------------------------
# Orchestrator.
# -------------------------------------------------------------------------


def _orchestrate(tag: str, l2_kind: str) -> dict:
    log(f"=== orchestrate tag={tag} l2={l2_kind} ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if l2_kind == "v23":
        model, halpe_idx_arr = _setup_l2_v23()
    elif l2_kind == "v29":
        model, halpe_idx_arr = _setup_l2_v29()
    else:
        raise RuntimeError(f"unknown l2_kind {l2_kind}")

    _patch_motion_data(model, halpe_idx_arr)

    v17 = json.loads(V17_PATH.read_text())
    baseline = json.loads(BASELINE_PATH.read_text())
    baseline_by_slot = {
        (s["target"], s["view"]): s for s in baseline["slots"]
    }

    slot_keys: list[tuple[str, str, dict]] = []
    for target, slots in v17["models"].items():
        for view, entry in slots.items():
            slot_keys.append((target, view, entry))
    log(f"=== {len(slot_keys)} slots to process ===")

    per_slot_validity: list[dict] = []
    deploy_models: dict[str, dict[str, dict]] = {}
    tier_counter = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    learned_wins: list[tuple[str, str, float, float]] = []
    ridge_wins: list[tuple[str, str, float, float]] = []
    ties: list[tuple[str, str, float]] = []
    extrema_diag: list[dict] = []

    for i, (target, view, entry) in enumerate(slot_keys):
        approach = entry.get("approach", "unknown")
        t0 = time.time()
        log(
            f"--- slot {i+1}/{len(slot_keys)} {target}/{view} "
            f"approach={approach} ---"
        )
        result = _collect_features_with_extrema_for_slot(
            target, view, approach,
        )
        if result is None:
            base = baseline_by_slot.get((target, view), {})
            per_slot_validity.append({
                **base, f"{tag}_skipped": True,
                "skip_reason": (
                    f"extrema builder not supported for {approach}"
                ),
            })
            deploy_models.setdefault(target, {})[view] = entry
            continue

        X, y, gt_max, gt_min, subjects, sources = result

        # Sanity check: gt_max - gt_min must equal y per row (within 1e-3
        # tol). If not, fall back.
        diff = gt_max - gt_min - y
        max_abs_diff = float(np.max(np.abs(diff))) if len(diff) else 0.0
        if max_abs_diff > 1e-3:
            log(
                f"  EXTREMA ALIGN MISMATCH: max |gt_max-gt_min - y| "
                f"= {max_abs_diff:.4f}; skipping extrema for this slot"
            )
            base = baseline_by_slot.get((target, view), {})
            per_slot_validity.append({
                **base, f"{tag}_skipped": True,
                "skip_reason": f"extrema alignment mismatch ({max_abs_diff:.4f})",
            })
            deploy_models.setdefault(target, {})[view] = entry
            continue

        if len(y) < 20:
            base = baseline_by_slot.get((target, view), {})
            per_slot_validity.append({
                **base, f"{tag}_skipped": True,
                "skip_reason": f"only n={len(y)} samples",
            })
            deploy_models.setdefault(target, {})[view] = entry
            continue

        # Ridge LOSO baseline (same as Agent NN).
        try:
            pred_r, obs_r, subj_r = _loso_ridge(
                X, y, list(subjects), alpha=10.0,
            )
            stats_ridge = _stats_per_subject(pred_r, obs_r, subj_r)
        except Exception as e:
            log(f"  ridge LOSO failed: {e!r}")
            stats_ridge = {
                "ccc_lin": float("nan"), "pearson_r": float("nan"),
                "loa_half_width_deg": float("nan"),
                "classification": "Poor",
            }

        # Extrema-aware learned LOSO.
        try:
            pred_l, pmx_l, pmn_l, obs_l, subj_l = _loso_learned_extrema(
                X, y, gt_max, gt_min, list(subjects),
                hidden=32, epochs=200, lr=1e-2,
                weight_decay=1e-3, alpha=1.0, beta=0.5, seed=0,
            )
            stats_learned = _stats_per_subject(pred_l, obs_l, subj_l)
            # Diagnostic per-slot mean absolute error on max/min.
            mask_mx = np.isfinite(pmx_l) & np.isfinite(gt_max)
            mask_mn = np.isfinite(pmn_l) & np.isfinite(gt_min)
            mae_max = (
                float(np.mean(np.abs(pmx_l[mask_mx] - gt_max[mask_mx])))
                if mask_mx.any() else float("nan")
            )
            mae_min = (
                float(np.mean(np.abs(pmn_l[mask_mn] - gt_min[mask_mn])))
                if mask_mn.any() else float("nan")
            )
            extrema_diag.append({
                "target": target, "view": view,
                "mae_max_deg": mae_max,
                "mae_min_deg": mae_min,
                "n": int(mask_mx.sum()),
            })
        except Exception as e:
            log(f"  extrema LOSO failed: {e!r}; falling back to ridge")
            stats_learned = stats_ridge

        ccc_r = stats_ridge.get("ccc_lin")
        ccc_l = stats_learned.get("ccc_lin")

        def _val(c):
            if c is None or (isinstance(c, float) and math.isnan(c)):
                return -float("inf")
            return float(c)

        cr = _val(ccc_r)
        cl = _val(ccc_l)
        use_learned = cl >= cr - 0.05

        if use_learned and cl > cr + 1e-6:
            learned_wins.append((target, view, cr, cl))
        elif use_learned:
            ties.append((target, view, cr))
        else:
            ridge_wins.append((target, view, cr, cl))

        if use_learned:
            chosen = "learned_extrema"
            chosen_stats = stats_learned
            try:
                final_model = _train_learned_l3_extrema(
                    X.astype(np.float64), y.astype(np.float64),
                    gt_max.astype(np.float64), gt_min.astype(np.float64),
                    hidden=32, epochs=200, lr=1e-2,
                    weight_decay=1e-3, alpha=1.0, beta=0.5, seed=0,
                )
                final_block = {
                    "l3_type": "learned_extrema",
                    "hidden": final_model.hidden,
                    "n_features": final_model.n_features,
                    "feature_mean": final_model.feature_mean.tolist(),
                    "feature_std": final_model.feature_std.tolist(),
                    "state_dict": {
                        k: v.numpy().tolist()
                        for k, v in final_model.state_dict.items()
                    },
                }
            except Exception as e:
                log(f"  final extrema fit failed: {e!r}; -> ridge")
                chosen = "ridge"
                chosen_stats = stats_ridge
                final_block = None
        else:
            chosen = "ridge"
            chosen_stats = stats_ridge
            final_block = None

        if chosen == "ridge":
            try:
                final_ridge = fit_ridge(
                    X.astype(np.float64), y.astype(np.float64), alpha=10.0,
                )
                final_block = {
                    "l3_type": "ridge",
                    "weights": final_ridge.weights.tolist(),
                    "bias": float(final_ridge.bias),
                    "feature_mean": final_ridge.feature_mean.tolist(),
                    "feature_std": final_ridge.feature_std.tolist(),
                }
            except Exception as e:
                log(f"  final ridge fit failed: {e!r}")
                final_block = None

        new_tier = chosen_stats.get("classification", "Poor") or "Poor"
        tier_counter[new_tier] = tier_counter.get(new_tier, 0) + 1

        if DUMP_ACCUMULATOR is not None:
            if chosen == "learned_extrema":
                _pred_dump, _obs_dump, _subj_dump = pred_l, obs_l, subj_l
            else:
                _pred_dump, _obs_dump, _subj_dump = pred_r, obs_r, subj_r
            DUMP_ACCUMULATOR.add_slot(
                target=target, view=view,
                approach=f"{approach}|{chosen}",
                pred=_pred_dump, obs=_obs_dump, subjects=_subj_dump,
            )

        slot_record = {
            "target": target,
            "view": view,
            "approach": approach,
            "l2_source": l2_kind,
            "n_subjects": chosen_stats.get("n_subjects"),
            "n_trials": chosen_stats.get("n_trials"),
            "pearson_r": chosen_stats.get("pearson_r"),
            "ccc_lin": chosen_stats.get("ccc_lin"),
            "loa_half_width_deg": chosen_stats.get("loa_half_width_deg"),
            "mae_deg": chosen_stats.get("mae_deg"),
            "rmse_deg": chosen_stats.get("rmse_deg"),
            "classification": new_tier,
            "l3_kind": chosen,
            "ridge_ccc": ccc_r,
            "learned_ccc": ccc_l,
            "ridge_loa_half": stats_ridge.get("loa_half_width_deg"),
            "learned_loa_half": stats_learned.get("loa_half_width_deg"),
        }
        per_slot_validity.append(slot_record)

        deploy_entry = dict(entry)
        deploy_entry[f"{tag}_block"] = final_block
        deploy_entry[f"{tag}_loso_stats"] = {
            k: chosen_stats.get(k) for k in [
                "n_subjects", "n_trials", "mean_bias_deg",
                "sd_bias_deg", "loa_upper_deg", "loa_lower_deg",
                "loa_half_width_deg", "pearson_r", "ccc_lin",
                "mae_deg", "rmse_deg", "classification",
            ]
        }
        deploy_entry[f"{tag}_l3_kind"] = chosen
        deploy_entry[f"{tag}_l2_source"] = l2_kind
        deploy_models.setdefault(target, {})[view] = deploy_entry

        log(
            f"  ridge CCC={cr:+.3f} loa="
            f"{stats_ridge.get('loa_half_width_deg', float('nan')):.2f} | "
            f"extrema CCC={cl:+.3f} loa="
            f"{stats_learned.get('loa_half_width_deg', float('nan')):.2f} "
            f"-> picked={chosen} tier={new_tier} "
            f"({time.time()-t0:.1f}s)"
        )

    log(f"=== {tag} complete: tier counts {tier_counter} ===")
    log(
        f"   extrema wins: {len(learned_wins)} | "
        f"ties / within-0.05: {len(ties)} | "
        f"ridge wins: {len(ridge_wins)}"
    )

    deploy_path = RESULTS_DIR / f"deploy_ready_models_{tag}.json"
    deploy_obj = {
        "version": tag,
        "produced_by": "harness.learned_layer3_extrema (Agent OO)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "description": (
            f"L2 source = {l2_kind} (v23 = HH2 combined alldata L2, "
            "v29 = NN mirror-flip per-source alldata L2). L3 = per-slot "
            "extrema-aware TinyMLP (hidden=32, two heads pred_max/pred_min, "
            "loss = SmoothL1(rom) + 0.5*SmoothL1(max) + 0.5*SmoothL1(min)). "
            "Per-slot fallback to ridge if extrema-aware CCC underperforms "
            "ridge CCC by > 0.05. LOSO at L3; L2 trained on all 24 cohort "
            "subjects."
        ),
        "loso_discipline": (
            "Layer-3-LOSO-only (L2 trained on all cohort subjects)"
        ),
        "v17_base_path": str(V17_PATH),
        "approaches": v17.get("approaches"),
        "training_dataset": v17.get("training_dataset"),
        "models": deploy_models,
        "calibration_fix": v17.get("calibration_fix"),
        "selective_adoption": v17.get("selective_adoption"),
    }
    deploy_path.write_text(json.dumps(deploy_obj, indent=2, default=str))
    log(f"wrote {deploy_path}")

    per_slot_path = OUT_DIR / f"per_slot_validity_{tag}.json"
    per_slot_obj = {
        "version": tag,
        "produced_by": "harness.learned_layer3_extrema (Agent OO)",
        "l2_source": l2_kind,
        "loss_alpha": 1.0,
        "loss_beta": 0.5,
        "tier_counts": tier_counter,
        "extrema_diagnostics": extrema_diag,
        "learned_wins": [
            {
                "target": t, "view": v,
                "ridge_ccc": rc, "extrema_ccc": lc,
            }
            for (t, v, rc, lc) in learned_wins
        ],
        "ridge_wins": [
            {
                "target": t, "view": v,
                "ridge_ccc": rc, "extrema_ccc": lc,
            }
            for (t, v, rc, lc) in ridge_wins
        ],
        "ties": [
            {"target": t, "view": v, "ridge_ccc": rc}
            for (t, v, rc) in ties
        ],
        "slots": per_slot_validity,
    }
    per_slot_path.write_text(
        json.dumps(per_slot_obj, indent=2, default=str)
    )
    log(f"wrote {per_slot_path}")

    return per_slot_obj


def main() -> None:
    global DUMP_ACCUMULATOR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l2", choices=["v23", "v29"], default="v23",
        help="Which L2 model to use as feature source.",
    )
    parser.add_argument(
        "--dump-predictions", default=None,
        help="If set, also write per-subject (pred, gt) dumps for the "
             "chosen (ridge|learned_extrema) L3 head.",
    )
    args = parser.parse_args()
    tag = {
        "v23": "v33_extrema_l3",
        "v29": "v34_mirrorflip_extrema_l3",
    }[args.l2]
    log(f"=== Agent OO ({tag}) START ===")
    if args.dump_predictions:
        DUMP_ACCUMULATOR = DumpAccumulator(
            reader=args.dump_predictions,
            loso_discipline=(
                f"Layer-3-LOSO-only (OO: L2={args.l2}, extrema-aware L3)"
            ),
        )
        log(f"[SS dump] DUMP_ACCUMULATOR set for '{args.dump_predictions}'")
    _orchestrate(tag=tag, l2_kind=args.l2)
    if DUMP_ACCUMULATOR is not None:
        out_path = DUMP_ACCUMULATOR.write()
        log(f"[SS dump] wrote per-slot predictions to {out_path}")
    log(f"=== Agent OO ({tag}) DONE ===")


if __name__ == "__main__":
    main()
