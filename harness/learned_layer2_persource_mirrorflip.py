"""Learned Layer 2 (v29) — MM-A backbone + mirror-flip augmentation.

Agent NN build (Phase 1). Extends MM-A
(``learned_layer2_combined_persource.py``) — the per-source heads + per-frame
SmoothL1 model that won the most v28 slots — by adding left-right flip
augmentation at clip-load time.

# Rationale

v28 has a clear mirror-twin asymmetry pattern: certain (metric, view) slots
score very well on one side but poorly on the symmetric sister:

  * hip_adduction_r / side_right: best CCC 0.27 (sister side_left: 0.94)
  * hip_flexion_r   / side_right: best CCC 0.58 (sister side_left: 0.86)
  * ankle_angle_r   / side_left:  best CCC 0.38 (sister side_right: 0.64)
  * ankle_angle_r   / front_oblique_left: best CCC 0.56 (sister F_O_right: 0.73)

This smells like training-data left/right asymmetry rather than geometry.
Adding mirror flips during training (horizontal flip + L/R label swap)
doubles effective coverage on both sides.

# Implementation

1. Cohort loader (``build_combined_cohort_lr``) loads BOTH ``_r`` AND ``_l``
   joint angles for OpenCap, and both for ASPset (which now exposes
   hip_flexion_l, knee_angle_l, hip_adduction_l alongside the _r versions).
2. ``flip_clip()`` returns a *new* clip dict with:
     - x-coordinate of every keypoint negated (already centered, so x-flip
       is x = -x).
     - L/R keypoint pairs swapped in HALPE_KEYS_WITH_SMPL ordering.
     - The "angles" 5-vector is populated from the _l columns instead of _r
       (matches the deploy interface: the model always predicts the
       *visible right side*, so the flipped sample's "right side" really is
       the original-left side).
3. Training cohort = original clips + flipped clips (subject id unchanged).
   24-fold subject LOSO is preserved: held-out subject's flipped copies
   stay in held-out, not training.
4. Loss/architecture identical to MM-A:
     - TemporalKeypointCNNConfPerSource (shared 5d + ASPset 2d heads)
     - per-frame masked SmoothL1
     - AdamW lr=1e-3 wd=1e-4 batch_size=256 cosine LR 25 epochs.

# Caveats

* For ASPset, ``hip_adduction_l`` and ``lumbar_extension`` go through the
  same convention-mismatch routing as the _r versions (lumbar is bilateral
  so flipping it is identity-of-target; hip_adduction_l flips sign of the
  ASPset asin convention exactly the same as hip_adduction_r). Routing is
  handled by MM-A's ``route_targets()`` unchanged because we present the
  flipped clip with values already in the L slot mapped onto the R slot in
  the DEPLOY_METRICS vector.
* ankle_angle_l for OpenCap only (ASPset has no foot keypoints) — same as
  ankle_angle_r.
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

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.learned_layer2_combined import (  # noqa: E402
    DEPLOY_METRICS,
    HALPE_KEYS_WITH_SMPL,
    LAB,
    N_KEYS,
    OPENCAP_KP,
    TEMPORAL_T,
    _load_opencap_keypoints,
    _normalize_kp,
    _resample_gt,
    build_combined_cohort,
)
from harness.learned_layer2_combined_persource import (  # noqa: E402
    PER_SOURCE_IDX,
    TemporalKeypointCNNConfPerSource,
    _aggregate_overall,
    _save_intermediate,
    evaluate_fold,
    stack_clips_persource,
    train_one_fold_mm_a,
)
from harness.smpl_layer2_poc import HALPE26_INDEX  # noqa: E402
from harness.parsers import parse_mot  # noqa: E402


# -------------------------------------------------------------------------
# Paths.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
OUT_DIR: Final[Path] = DATA_ROOT / "learned_layer2_persource_mirrorflip"
MODEL_OUT: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_persource_mirrorflip_v1.pt"
)
MODEL_OUT_ALLDATA: Final[Path] = (
    REPO_ROOT / "models"
    / "learned_layer2_persource_mirrorflip_alldata_v1.pt"
)


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[NN-L2 t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# L/R metric mapping used by the flipper.
# -------------------------------------------------------------------------

# DEPLOY_METRICS  = ("hip_flexion_r", "hip_adduction_r", "knee_angle_r",
#                    "ankle_angle_r", "lumbar_extension")
# For mirror flip, when we flip the body, the new "right side" is the old
# left side. So we need _l targets where _r used to be. Lumbar is bilateral
# so it stays the same.
DEPLOY_METRICS_L: Final[tuple[str, ...]] = (
    "hip_flexion_l",
    "hip_adduction_l",
    "knee_angle_l",
    "ankle_angle_l",
    "lumbar_extension",  # bilateral, unchanged
)

# Halpe26 L/R name pairs (we always swap these on flip).
LR_HALPE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("L_eye", "R_eye"), ("L_ear", "R_ear"),
    ("L_sho", "R_sho"), ("L_elb", "R_elb"), ("L_wri", "R_wri"),
    ("L_hip", "R_hip"), ("L_kne", "R_kne"), ("L_ank", "R_ank"),
    ("L_big_toe", "R_big_toe"), ("L_small_toe", "R_small_toe"),
    ("L_heel", "R_heel"),
)


def _build_lr_swap_indices() -> np.ndarray:
    """Build the permutation that maps HALPE_KEYS_WITH_SMPL ordering to
    its L/R-swapped equivalent."""
    name_to_idx = {n: i for i, n in enumerate(HALPE_KEYS_WITH_SMPL)}
    perm = np.arange(len(HALPE_KEYS_WITH_SMPL), dtype=np.int64)
    for a, b in LR_HALPE_PAIRS:
        if a in name_to_idx and b in name_to_idx:
            perm[name_to_idx[a]] = name_to_idx[b]
            perm[name_to_idx[b]] = name_to_idx[a]
    return perm


LR_SWAP_IDX: Final[np.ndarray] = _build_lr_swap_indices()


# -------------------------------------------------------------------------
# Cohort loader that ALSO extracts _l angles per clip.
# -------------------------------------------------------------------------


def _load_opencap_clip_lr(
    kp_path: Path, halpe_idx_arr: np.ndarray
) -> dict | None:
    """Same as build_opencap_clip but also extracts _l angles into a
    secondary array `angles_l` so we can later build a flipped twin."""
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
    angles_r = np.full(
        (n_frames, len(DEPLOY_METRICS)), np.nan, dtype=np.float32
    )
    angles_l = np.full(
        (n_frames, len(DEPLOY_METRICS)), np.nan, dtype=np.float32
    )
    for k, metric_r in enumerate(DEPLOY_METRICS):
        if gt.has(metric_r):
            angles_r[:, k] = _resample_gt(
                gt.time, gt.column(metric_r), t_sec
            ).astype(np.float32)
        metric_l = DEPLOY_METRICS_L[k]
        if gt.has(metric_l):
            angles_l[:, k] = _resample_gt(
                gt.time, gt.column(metric_l), t_sec
            ).astype(np.float32)

    if not np.any(np.isfinite(angles_r)) and not np.any(
        np.isfinite(angles_l)
    ):
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
        "angles": angles_r,
        "angles_l": angles_l,
        "flipped": False,
    }


def _load_aspset_clip_lr(
    kp_path: Path, halpe_idx_arr: np.ndarray
) -> dict | None:
    """ASPset clip with both _r and _l angle vectors. Reuses the same
    convention alignment as ``_align_aspset_angles`` for both sides."""
    from harness.learned_layer2_combined import (
        _load_aspset_dwpose,
        ASPSET_C3D,
    )
    from harness.aspset_loader import load_clip as aspset_load_clip
    from harness.aspset_loader import joint_angles_from_aspset

    base = kp_path.stem
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

    def _align_one_side(side: str) -> np.ndarray:
        """side in {'r','l'}. Returns (n_frames_dw, 5) array in OpenCap
        convention for that side."""
        out = np.full(
            (n_frames_dw, len(DEPLOY_METRICS)), np.nan, dtype=np.float32
        )

        def _fit(arr: np.ndarray | None) -> np.ndarray:
            if arr is None:
                return np.full(n_frames_dw, np.nan, dtype=np.float32)
            if arr.shape[0] == n_frames_dw:
                return arr.astype(np.float32)
            src = np.linspace(0, 1, arr.shape[0])
            tgt = np.linspace(0, 1, n_frames_dw)
            return np.interp(tgt, src, arr).astype(np.float32)

        hf_key = f"hip_flexion_{side}"
        ha_key = f"hip_adduction_{side}"
        kn_key = f"knee_angle_{side}"
        out[:, 0] = _fit(angles_raw.get(hf_key))
        hip_add = angles_raw.get(ha_key)
        if hip_add is not None:
            hip_add = hip_add.copy()
            hip_add[np.abs(hip_add) > 85] = np.nan
            out[:, 1] = _fit(hip_add)
        knee = angles_raw.get(kn_key)
        if knee is not None:
            out[:, 2] = _fit(180.0 - knee.astype(np.float32))
        # ankle: ASPset has no foot keypoints — NaN
        lumbar = angles_raw.get("lumbar_extension")
        if lumbar is not None:
            out[:, 4] = _fit(lumbar.astype(np.float32) - 180.0)
        return out

    angles_r = _align_one_side("r")
    angles_l = _align_one_side("l")

    if not np.any(np.isfinite(angles_r)) and not np.any(
        np.isfinite(angles_l)
    ):
        return None

    kp_sub = clip["kp"][:, halpe_idx_arr, :]
    conf_sub = clip["conf"][:, halpe_idx_arr]
    kp_norm = np.zeros_like(kp_sub, dtype=np.float32)
    for i in range(n_frames_dw):
        kp_norm[i] = _normalize_kp(kp_sub[i]).astype(np.float32)

    return {
        "source": "aspset",
        "subject": f"aspset_{subject}",
        "cam": view,
        "trial": clip_id,
        "clip": base,
        "kp": kp_norm,
        "conf": conf_sub.astype(np.float32),
        "t_sec": (clip["t_ms"] / 1000.0).astype(np.float32),
        "angles": angles_r,
        "angles_l": angles_l,
        "flipped": False,
    }


def build_combined_cohort_lr() -> dict[str, list[dict]]:
    """Like ``build_combined_cohort`` but each clip carries both _r and _l
    angles for flip augmentation."""
    halpe_idx_arr = np.array(
        [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL],
        dtype=np.int64,
    )
    cohort: dict[str, list[dict]] = {}

    # OpenCap
    oc_files = sorted(OPENCAP_KP.glob("*.json"))
    log(f"  scanning {len(oc_files)} OpenCap DWPose JSONs ...")
    for i, kp_path in enumerate(oc_files):
        if (i + 1) % 50 == 0:
            log(f"    OpenCap scanned {i+1}/{len(oc_files)}")
        rec = _load_opencap_clip_lr(kp_path, halpe_idx_arr)
        if rec is not None:
            cohort.setdefault(rec["subject"], []).append(rec)

    # ASPset
    from harness.learned_layer2_combined import ASPSET_KP
    asp_files = sorted(ASPSET_KP.glob("*.json"))
    log(f"  scanning {len(asp_files)} ASPset DWPose JSONs ...")
    for i, kp_path in enumerate(asp_files):
        if (i + 1) % 200 == 0:
            log(f"    ASPset scanned {i+1}/{len(asp_files)}")
        rec = _load_aspset_clip_lr(kp_path, halpe_idx_arr)
        if rec is not None:
            cohort.setdefault(rec["subject"], []).append(rec)

    log(
        f"  cohort_lr: {len(cohort)} subjects, "
        f"{sum(len(v) for v in cohort.values())} clips"
    )
    return cohort


# -------------------------------------------------------------------------
# Mirror-flip clip generator.
# -------------------------------------------------------------------------


def flip_clip(rec: dict) -> dict:
    """Return a *new* clip dict with keypoints horizontally flipped, L/R
    HALPE pairs swapped, and the right-side target replaced by the
    original left-side target. Subject id unchanged so LOSO is preserved.

    `_normalize_kp` already centered each frame at the mean and unit-scaled,
    so flipping x is just multiplication by -1.
    """
    kp = rec["kp"].copy()
    conf = rec["conf"].copy()
    # x-flip
    kp[..., 0] = -kp[..., 0]
    # L/R swap
    kp = kp[:, LR_SWAP_IDX, :]
    conf = conf[:, LR_SWAP_IDX]
    # The flipped right-side targets come from the original left-side
    # angles.
    return {
        **rec,
        "kp": kp,
        "conf": conf,
        "angles": rec["angles_l"],
        "angles_l": rec["angles"],
        "clip": rec["clip"] + "_FLIP",
        "flipped": True,
    }


def augment_cohort_with_flips(
    cohort: dict[str, list[dict]],
    *,
    flip_prob: float = 1.0,
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Generate flipped duplicates and append them per subject.

    flip_prob=1.0 (default) => every clip gets a flipped twin (doubles
    training data). subject id preserved so LOSO discipline is intact.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list[dict]] = {}
    for subj, clips in cohort.items():
        out[subj] = list(clips)
        for rec in clips:
            if flip_prob >= 1.0 or rng.random() < flip_prob:
                out[subj].append(flip_clip(rec))
    log(
        f"  augmented cohort: {sum(len(v) for v in out.values())} "
        f"clips (vs original "
        f"{sum(len(v) for v in cohort.values())})"
    )
    return out


# -------------------------------------------------------------------------
# LOSO driver — identical to MM-A but operates on flip-augmented cohort.
# -------------------------------------------------------------------------


def run_loso_mirrorflip(
    cohort: dict[str, list[dict]],
    *,
    epochs: int = 25,
    batch_size: int = 256,
    train_stride: int = 1,
) -> dict:
    subjects = sorted(cohort.keys())
    log(
        f"NN LOSO over {len(subjects)} subjects "
        f"(train_stride={train_stride}, mirror flips enabled)"
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
        # Only evaluate on UNFLIPPED held-out clips so the metric matches
        # the deploy interface (predict the visible right side).
        held_clips = [
            c for c in cohort[held_out] if not c.get("flipped", False)
        ]
        log(
            f"  train clips (incl flips): {len(train_clips)} "
            f"held clips (orig only): {len(held_clips)}"
        )

        X_train, Ys_train, Ya_train = stack_clips_persource(
            train_clips, stride=train_stride
        )
        log(
            f"  stacked windows: X={X_train.shape} "
            f"Ys={Ys_train.shape} Ya={Ya_train.shape}"
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
            "source": (
                "opencap" if held_out.startswith("opencap_") else "aspset"
            ),
            "fold_idx": fold_i,
            "n_train_windows": int(X_train.shape[0]),
            "n_held_clips": len(held_clips),
            "aggregate": agg,
            "per_clip": per_clip,
        })

        last_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        pooled_r = agg["pooled"]["mean_abs_r"]
        if np.isfinite(pooled_r) and pooled_r > best_pooled:
            best_pooled = pooled_r
            best_subject = held_out
            best_state = last_state
            best_ys_mean = np.asarray(model.ys_mean)  # type: ignore
            best_ys_std = np.asarray(model.ys_std)    # type: ignore
            best_ya_mean = np.asarray(model.ya_mean)  # type: ignore
            best_ya_std = np.asarray(model.ya_std)    # type: ignore

        _save_intermediate(
            fold_results, OUT_DIR, "nn_persource_mirrorflip_v1"
        )

    overall = _aggregate_overall(fold_results)
    log(
        f"=== NN OVERALL pooled |r| = "
        f"{overall['pooled_subject_mean_abs_r']['mean_abs_r']:.4f} "
        f"(folds={overall['pooled_subject_mean_abs_r']['n_folds']}) ==="
    )

    if best_state is not None:
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
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
            "training": "nn_persource_mirrorflip_v1",
            "best_loso_subject": best_subject,
            "best_loso_pooled_abs_r": best_pooled,
            "augmentation": "mirror_flip_prob_1.0",
        }, MODEL_OUT)
        log(
            f"saved best-fold NN checkpoint -> {MODEL_OUT} "
            f"(held={best_subject} |r|={best_pooled:.3f})"
        )

    return {
        "version": "nn_persource_mirrorflip_v1",
        "epochs": epochs,
        "batch_size": batch_size,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "metrics": list(DEPLOY_METRICS),
        "augmentation": "mirror_flip_prob_1.0",
        "fold_results": fold_results,
        "overall": overall,
    }


# -------------------------------------------------------------------------
# All-data training for Phase B Layer 3 (no LOSO at L2).
# -------------------------------------------------------------------------


def train_all_data_mirrorflip(
    *,
    epochs: int = 25,
    batch_size: int = 256,
    train_stride: int = 1,
    seed: int = 42,
) -> tuple[TemporalKeypointCNNConfPerSource, dict]:
    cohort = build_combined_cohort_lr()
    if not cohort:
        raise RuntimeError("combined cohort_lr is empty; cannot train")
    cohort = augment_cohort_with_flips(cohort, flip_prob=1.0, seed=seed)
    all_clips = [c for clips in cohort.values() for c in clips]
    log(
        f"NN all-data (mirror-flip): {len(cohort)} subjects, "
        f"{len(all_clips)} clips (incl flips), "
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
        fold_label="ALL-DATA-NN",
    )
    return model, {
        "hist": hist,
        "n_subjects": len(cohort),
        "n_clips": len(all_clips),
        "subjects": sorted(cohort.keys()),
    }


# -------------------------------------------------------------------------
# Main.
# -------------------------------------------------------------------------


def main(
    epochs: int = 25,
    batch_size: int = 256,
    train_stride: int = 1,
    max_subjects: int | None = None,
    do_alldata: bool = True,
    do_loso: bool = True,
    flip_seed: int = 0,
) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== learned_layer2_persource_mirrorflip START ===")
    log(
        f"epochs={epochs} batch_size={batch_size} "
        f"train_stride={train_stride}"
    )

    results: dict = {
        "version": "nn_persource_mirrorflip_v1",
        "epochs": epochs,
        "batch_size": batch_size,
        "train_stride": train_stride,
    }

    if do_loso:
        # LOSO eval
        cohort = build_combined_cohort_lr()
        if not cohort:
            log("EMPTY COHORT; aborting")
            return {}
        cohort = augment_cohort_with_flips(
            cohort, flip_prob=1.0, seed=flip_seed
        )
        if max_subjects is not None:
            keep = sorted(cohort.keys())[:max_subjects]
            cohort = {k: cohort[k] for k in keep}
            log(f"DEBUG: limited to {len(cohort)} subjects: {keep}")
        loso_results = run_loso_mirrorflip(
            cohort, epochs=epochs, batch_size=batch_size,
            train_stride=train_stride,
        )
        out_path = OUT_DIR / "per_slot_results.json"
        out_path.write_text(json.dumps(loso_results, indent=2, default=str))
        log(f"wrote final NN LOSO results to {out_path}")
        results["loso"] = loso_results

    if do_alldata:
        # All-data (for Phase B L3)
        if MODEL_OUT_ALLDATA.exists():
            log(f"  ALL-DATA checkpoint already exists at {MODEL_OUT_ALLDATA}")
        else:
            model, meta = train_all_data_mirrorflip(
                epochs=epochs, batch_size=batch_size,
                train_stride=train_stride, seed=42,
            )
            MODEL_OUT_ALLDATA.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "ys_mean": np.asarray(model.ys_mean),  # type: ignore
                "ys_std": np.asarray(model.ys_std),    # type: ignore
                "ya_mean": np.asarray(model.ya_mean),  # type: ignore
                "ya_std": np.asarray(model.ya_std),    # type: ignore
                "n_keys": N_KEYS,
                "temporal_t": TEMPORAL_T,
                "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
                "metrics": list(DEPLOY_METRICS),
                "per_source_idx": list(PER_SOURCE_IDX),
                "model_class": "TemporalKeypointCNNConfPerSource",
                "training": "nn_persource_mirrorflip_alldata_v1",
                "epochs": epochs,
                "batch_size": batch_size,
                "train_stride": train_stride,
                "augmentation": "mirror_flip_prob_1.0",
                "loso_discipline": "ALL_DATA_NO_LOSO_AT_L2",
                "cohort": {
                    "n_subjects": meta["n_subjects"],
                    "n_clips": meta["n_clips"],
                    "subjects": meta["subjects"],
                },
            }, MODEL_OUT_ALLDATA)
            log(
                f"  saved all-data NN mirror-flip L2 -> "
                f"{MODEL_OUT_ALLDATA}"
            )
            results["alldata_model_path"] = str(MODEL_OUT_ALLDATA)

    log("=== learned_layer2_persource_mirrorflip DONE ===")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--no-loso", action="store_true",
        help="Skip per-subject LOSO eval (still train all-data)",
    )
    parser.add_argument(
        "--no-alldata", action="store_true",
        help="Skip all-data training",
    )
    parser.add_argument("--flip-seed", type=int, default=0)
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_stride=args.train_stride,
        max_subjects=args.max_subjects,
        do_loso=not args.no_loso,
        do_alldata=not args.no_alldata,
        flip_seed=args.flip_seed,
    )
