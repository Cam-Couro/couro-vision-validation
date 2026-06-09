"""Layer 2 reader using VideoPose3D 2D->3D pose lifting (Agent TT).

Lifts Halpe-26 2D keypoints (already cached from DWPose-L) into Human3.6M
canonical 3D joint positions via the pretrained VideoPose3D temporal model
(Pavllo et al. 2019, FAIR, Apache 2.0). Joint angles are then computed
directly from 3D vectors in the canonical body-centered frame, which is
view-invariant by construction -- the geometric step-change vs Couro's
deployed 2D-keypoint Layer 2.

Output API matches Agent FF's ``predict_l2_for_series`` contract: a
``(T, 5)`` float32 array of angle traces in OpenSim sign conventions, ready
to be drop-in monkey-patched into ``keypoints_to_motion_data`` via FF's
factory. This is what makes the v44 retrain a single-line swap.

3D angle conventions
--------------------
All angles are computed from purely segment-vector geometry in
VideoPose3D's canonical frame. The H36M skeleton output by VideoPose3D
has root at origin; no need to identify a global "up". Each angle is a
function only of relative joint positions, so it is camera-agnostic.

* ``knee_angle_r``:  ``180 - angle((R_hip - R_kne), (R_ank - R_kne))``
  (180 = straight leg, 0 = full flexion -- OpenSim convention).
* ``hip_flexion_r``: ``180 - angle((thorax - R_hip), (R_kne - R_hip))``
  computed in the **sagittal plane** of the body. The sagittal plane is
  defined by the cross product of the pelvis (L_hip - R_hip) axis and
  the spine (thorax - hip) axis; vectors are projected into that plane
  before the angle is taken. This isolates the flex/extend component
  from the abduction/adduction component (the geometric advantage of 3D).
* ``hip_adduction_r``: angle of (R_kne - R_hip) projected onto the
  **frontal plane** (plane perpendicular to the body's anterior axis),
  signed positive for adduction (knee toward midline). This is the slot
  that should benefit MOST from 3D vs 2D: in 2D, oblique views see this
  motion as an in-plane vector and confound it with hip flexion. In 3D,
  it's recovered cleanly regardless of camera angle.
* ``ankle_angle_r``: ``angle((R_kne - R_ank), foot_vector) - 90``. H36M
  does not have toes; we use a synthetic foot vector pointing in the
  anterior direction (cross product of pelvis x body-up axis) scaled to
  a unit length. This is a rough approximation -- H36M's ankle landmark
  alone is not enough to recover ankle dorsi/plantarflexion precisely.
* ``lumbar_extension``: signed angle between (thorax - hip) and the
  body's vertical axis (defined as the pelvis-frame-vertical: the mean
  spine direction across the clip). Positive = forward lean (consistent
  with Couro's existing convention).

Right-side only by default; left-side is obtained via the same FF
mirror-flip trick (flip x-axis and swap L/R indices on the 2D keypoints).

LOSO discipline
---------------
VideoPose3D was pretrained on Human3.6M, which has 11 subjects, none of
which overlap with OpenCap or ASPset. The L2 lifter is therefore
disjoint from the L3 LOSO pool, so this is a clean drop-in: no double-
LOSO violation.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.videopose3d import (
    HALPE26,
    H36M17_NAMES,
    TemporalModel,
    load_pretrained,
    halpe26_to_h36m17,
    normalize_screen_coords,
)


DATA_ROOT: Final[Path] = REPO_ROOT / "data"
OPENCAP_KP: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"
OUT_DIR: Final[Path] = DATA_ROOT / "videopose3d_layer2"
CHECKPOINT_PATH: Final[Path] = REPO_ROOT / "models" / "videopose3d_h36m.bin"
CACHE_PATH: Final[Path] = OUT_DIR / "per_clip_angles.json"

# Match Agent FF's L2 metric order so the monkey-patch plugs in cleanly.
DEPLOY_METRICS: Final[tuple[str, ...]] = (
    "hip_flexion_r",
    "hip_adduction_r",
    "knee_angle_r",
    "ankle_angle_r",
    "lumbar_extension",
)

# H36M-17 joint index lookup (must match harness.videopose3d.H36M17_NAMES).
H36M = {name: i for i, name in enumerate(H36M17_NAMES)}


# -------------------------------------------------------------------------
# 3D -> 5 angles.
# -------------------------------------------------------------------------


def _safe_unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize last-axis vectors; zero-norm vectors become zero."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm > eps, v / np.where(norm > eps, norm, 1.0), v * 0.0)


def _angle_between_3d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Per-frame interior angle in degrees (0..180)."""
    u1 = _safe_unit(v1)
    u2 = _safe_unit(v2)
    cos = np.clip((u1 * u2).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _project_onto_plane(v: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Project v onto the plane perpendicular to plane_normal.

    Vectorized over leading axes (e.g. time). plane_normal is broadcast.
    """
    n = _safe_unit(plane_normal)
    # Component of v along n:
    along = (v * n).sum(axis=-1, keepdims=True) * n
    return v - along


def angles_from_h36m_3d(pos_3d: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Compute the 5 deploy angles from a (T, 17, 3) H36M trajectory.

    Args:
        pos_3d: (T, 17, 3) joint positions in VideoPose3D's canonical frame.
        mirror: if True, swap L/R indices before computing the right-side
            angles -- this is the 3D analogue of FF's keypoint mirror flip.
            Used to extract LEFT-side angles by inferring the "right" model
            on a mirrored skeleton.

    Returns:
        (T, 5) float32 array, columns in DEPLOY_METRICS order.
    """
    p = np.asarray(pos_3d, dtype=np.float64)
    if mirror:
        # Reflect across x=0 (the sagittal plane) and swap L/R index sets.
        p = p.copy()
        p[..., 0] = -p[..., 0]
        lr_swaps = [
            ("R_hip", "L_hip"), ("R_kne", "L_kne"), ("R_ank", "L_ank"),
            ("R_sho", "L_sho"), ("R_elb", "L_elb"), ("R_wri", "L_wri"),
        ]
        for a, b in lr_swaps:
            ia, ib = H36M[a], H36M[b]
            p[:, [ia, ib]] = p[:, [ib, ia]]

    T = p.shape[0]
    hip = p[:, H36M["hip"]]
    spine = p[:, H36M["spine"]]
    thorax = p[:, H36M["thorax"]]
    neck = p[:, H36M["neck"]]
    head = p[:, H36M["head"]]
    R_hip, R_kne, R_ank = p[:, H36M["R_hip"]], p[:, H36M["R_kne"]], p[:, H36M["R_ank"]]
    L_hip = p[:, H36M["L_hip"]]

    # --- Body local frame, computed PER FRAME for view invariance. ---
    # Lateral axis: from R_hip to L_hip (sign chosen so +x_body = "left").
    lateral = L_hip - R_hip                              # (T, 3)
    # Body-up axis: from hip up to neck.
    up_axis = neck - hip                                 # (T, 3)
    # Forward axis: orthogonal to lateral and up (right-handed cross).
    fwd_axis = np.cross(lateral, up_axis)                # (T, 3)
    lateral_u = _safe_unit(lateral)
    up_u = _safe_unit(up_axis)
    fwd_u = _safe_unit(fwd_axis)

    # --- knee_angle_r: 180 - interior angle (full straight = 180 deg). ---
    knee_int = _angle_between_3d(R_hip - R_kne, R_ank - R_kne)
    knee_angle_r = 180.0 - knee_int

    # --- hip_flexion_r: 180 - interior angle in sagittal plane (lateral normal). ---
    # Project both segments into the sagittal plane (plane normal = lateral).
    torso_vec_proj = _project_onto_plane(thorax - R_hip, lateral_u)
    thigh_vec_proj = _project_onto_plane(R_kne - R_hip, lateral_u)
    hip_int = _angle_between_3d(torso_vec_proj, thigh_vec_proj)
    hip_flexion_r = 180.0 - hip_int

    # --- hip_adduction_r: angle of thigh in the frontal plane (forward normal). ---
    # Frontal plane is perpendicular to fwd_axis. Project thigh vec into it
    # and measure its signed angle from the body-up axis (negative-y in image
    # convention since up_axis is from hip TO neck, i.e. positive direction).
    # Sign: positive when knee moves TOWARD midline (adduction).
    thigh_vec = R_kne - R_hip
    thigh_frontal = _project_onto_plane(thigh_vec, fwd_u)
    # Decompose thigh_frontal into (along lateral_u, along up_u).
    comp_lat = (thigh_frontal * lateral_u).sum(axis=-1)
    comp_up = (thigh_frontal * up_u).sum(axis=-1)
    # In our convention lateral_u points TOWARD the LEFT hip (positive = away
    # from midline for right leg, since the right thigh sits on the -lateral
    # side). So when the right knee crosses the midline (adduction), comp_lat
    # becomes POSITIVE. Angle from -up_axis direction, atan2.
    # Use atan2(comp_lat, -comp_up) so neutral standing (knee directly below
    # hip, -up_axis direction) -> 0; adducted (knee toward midline) -> positive.
    hip_adduction_r = np.degrees(np.arctan2(comp_lat, -comp_up))

    # --- ankle_angle_r: dorsiflexion proxy (H36M has no toes). ---
    # Without a foot landmark, we synthesise a foot direction as the
    # projection of the body-forward axis onto the plane perpendicular
    # to the shank, then take the angle between shank and that foot
    # direction. Returns 0 in neutral stance; positive = dorsiflexion.
    shank_vec = R_ank - R_kne
    # Anatomical foot direction approximation: world-forward.
    foot_dir = fwd_u
    # Project shank into the sagittal plane (lateral normal) for cleaner sign.
    shank_proj = _project_onto_plane(shank_vec, lateral_u)
    foot_proj = _project_onto_plane(foot_dir, lateral_u)
    # Interior angle.
    ankle_int = _angle_between_3d(shank_proj, foot_proj)
    # Neutral stance: shank vertical, foot horizontal -> interior 90.
    # Couro's convention: ankle_angle = 90 - interior. Dorsiflexion = positive.
    ankle_angle_r = 90.0 - ankle_int

    # --- lumbar_extension: signed angle of spine from body-up. ---
    # We compute "body-up" via a clip-stable estimate: the mean of (neck - hip)
    # over the clip, then measure each-frame trunk lean in the sagittal plane.
    spine_vec = thorax - hip   # (T, 3)
    body_up_canon = _safe_unit((neck - hip).mean(axis=0, keepdims=True))   # (1, 3)
    body_lat_canon = _safe_unit((L_hip - R_hip).mean(axis=0, keepdims=True))  # (1, 3)
    spine_proj = _project_onto_plane(spine_vec, np.broadcast_to(body_lat_canon, spine_vec.shape))
    up_proj = _project_onto_plane(body_up_canon, body_lat_canon).reshape(3)
    # Signed angle: positive = forward lean (trunk fwd of vertical).
    # Find the perpendicular-in-sagittal vector to up_proj:
    fwd_canon = _safe_unit(np.broadcast_to(
        _safe_unit(np.cross(body_lat_canon, body_up_canon)), spine_vec.shape
    ))
    spine_along_up = (spine_proj * _safe_unit(up_proj)).sum(axis=-1)
    spine_along_fwd = (spine_proj * fwd_canon).sum(axis=-1)
    lumbar_extension = np.degrees(np.arctan2(spine_along_fwd, spine_along_up))

    out = np.stack([
        hip_flexion_r.astype(np.float32),
        hip_adduction_r.astype(np.float32),
        knee_angle_r.astype(np.float32),
        ankle_angle_r.astype(np.float32),
        lumbar_extension.astype(np.float32),
    ], axis=-1)
    # Guard against any residual NaNs.
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out


# -------------------------------------------------------------------------
# Inference pipeline.
# -------------------------------------------------------------------------


@dataclass
class ClipKeypoints:
    """Halpe-26 keypoint container for one OpenCap clip."""
    subject: str
    cam: str
    trial: str
    clip_id: str
    kp_2d: np.ndarray         # (T, 26, 2)
    width: int
    height: int
    fps: float


def load_opencap_clip_kp(kp_path: Path) -> ClipKeypoints | None:
    """Load a single OpenCap DWPose JSON into a ClipKeypoints record."""
    with open(kp_path) as f:
        d = json.load(f)
    seq = d.get("keypoints_sequence")
    if not seq:
        return None
    T = len(seq)
    if T < 3:
        return None
    kp = np.zeros((T, 26, 2), dtype=np.float32)
    for i, fr in enumerate(seq):
        arr = np.asarray(fr["keypoints"], dtype=np.float32)
        if arr.shape[0] >= 26:
            kp[i] = arr[:26]
    # Fill NaNs with linear interpolation per keypoint/dim.
    for k in range(26):
        for d_ax in range(2):
            col = kp[:, k, d_ax]
            mask = np.isfinite(col)
            if mask.sum() == 0:
                kp[:, k, d_ax] = 0.0
                continue
            if mask.sum() < T:
                idx = np.arange(T)
                kp[:, k, d_ax] = np.interp(idx, idx[mask], col[mask])

    meta = d["video_metadata"]
    base = kp_path.stem
    parts = base.split("_")
    subject = parts[0]
    cam = parts[-1]
    trial = "_".join(parts[1:-1])
    return ClipKeypoints(
        subject=subject, cam=cam, trial=trial, clip_id=base,
        kp_2d=kp, width=int(meta["width"]), height=int(meta["height"]),
        fps=float(meta["fps"]),
    )


def lift_clip_to_3d(
    model: TemporalModel, clip: ClipKeypoints, *, device: str = "cpu",
) -> np.ndarray:
    """Lift a clip's 2D keypoints to 3D positions. Wraps videopose3d.lift_sequence."""
    from harness.videopose3d import lift_sequence
    return lift_sequence(
        model, clip.kp_2d, width=clip.width, height=clip.height, device=device,
    )


def angles_for_clip(
    model: TemporalModel, clip: ClipKeypoints, *, device: str = "cpu",
) -> dict:
    """Run VideoPose3D + angle reader on a single clip. Returns angle dict."""
    pos_3d = lift_clip_to_3d(model, clip, device=device)   # (T, 17, 3)
    pred_r = angles_from_h36m_3d(pos_3d, mirror=False)
    pred_l = angles_from_h36m_3d(pos_3d, mirror=True)
    return {
        "clip_id": clip.clip_id,
        "subject": clip.subject,
        "cam": clip.cam,
        "trial": clip.trial,
        "n_frames": int(pos_3d.shape[0]),
        "fps": float(clip.fps),
        "pred_r": pred_r,
        "pred_l": pred_l,
    }


# -------------------------------------------------------------------------
# Bulk run over OpenCap cohort + cache.
# -------------------------------------------------------------------------


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[TT t={elapsed:6.1f}s] {msg}", flush=True)


def run_inference_all_clips(
    *,
    device: str = "cpu",
    force: bool = False,
) -> dict[str, dict]:
    """Run VideoPose3D inference on every OpenCap clip, cache angles to JSON.

    Returns: dict[clip_id -> {"pred_r": (T,5), "pred_l": (T,5), ...}].

    The cache JSON stores per-clip {clip_id, subject, cam, trial, n_frames,
    fps, pred_r (list-of-list), pred_l (list-of-list)} so downstream
    deserialisation is trivial.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists() and not force:
        log(f"loading cached per-clip angles from {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            raw = json.load(f)
        out: dict[str, dict] = {}
        for clip_id, rec in raw["clips"].items():
            out[clip_id] = {
                "clip_id": clip_id,
                "subject": rec["subject"],
                "cam": rec["cam"],
                "trial": rec["trial"],
                "n_frames": rec["n_frames"],
                "fps": rec["fps"],
                "pred_r": np.asarray(rec["pred_r"], dtype=np.float32),
                "pred_l": np.asarray(rec["pred_l"], dtype=np.float32),
            }
        log(f"loaded {len(out)} cached clips")
        return out

    log(f"loading VideoPose3D pretrained checkpoint from {CHECKPOINT_PATH}")
    model = load_pretrained(CHECKPOINT_PATH, device=device)

    kp_files = sorted(OPENCAP_KP.glob("*.json"))
    log(f"found {len(kp_files)} OpenCap DWPose JSONs")

    cache: dict[str, dict] = {}
    timings: list[float] = []
    last_log = time.time()
    fail = 0
    for i, kp_path in enumerate(kp_files):
        try:
            clip = load_opencap_clip_kp(kp_path)
            if clip is None:
                fail += 1
                continue
            t0 = time.time()
            rec = angles_for_clip(model, clip, device=device)
            dt = time.time() - t0
            timings.append(dt)
            cache[clip.clip_id] = rec
        except Exception as e:
            fail += 1
            log(f"  WARN: clip {kp_path.name} failed: {e!r}")
            continue

        if time.time() - last_log > 10.0:
            mean_t = float(np.mean(timings[-32:])) if timings else 0.0
            log(
                f"  [{i+1}/{len(kp_files)}] cached={len(cache)} "
                f"fail={fail} last32_mean={mean_t*1000:.1f}ms/clip"
            )
            last_log = time.time()

    log(f"inference complete: {len(cache)} ok, {fail} failed")

    # Serialize: angles as lists for JSON portability.
    serial = {
        "version": "videopose3d_layer2_v1",
        "produced_by": "harness.learned_layer2_videopose3d (Agent TT)",
        "produced_date": time.strftime("%Y-%m-%d"),
        "model_checkpoint": str(CHECKPOINT_PATH.name),
        "model_license": "Apache-2.0",
        "model_citation": (
            "Pavllo et al., '3D human pose estimation in video with "
            "temporal convolutions and semi-supervised training', CVPR 2019."
        ),
        "deploy_metrics": list(DEPLOY_METRICS),
        "n_clips": len(cache),
        "inference_timing_ms_per_clip": {
            "mean": float(np.mean(timings)) * 1000 if timings else 0.0,
            "p50": float(np.median(timings)) * 1000 if timings else 0.0,
            "p95": float(np.percentile(timings, 95)) * 1000 if timings else 0.0,
            "min": float(np.min(timings)) * 1000 if timings else 0.0,
            "max": float(np.max(timings)) * 1000 if timings else 0.0,
        },
        "device": device,
        "clips": {
            cid: {
                "subject": rec["subject"], "cam": rec["cam"],
                "trial": rec["trial"], "n_frames": rec["n_frames"],
                "fps": rec["fps"],
                "pred_r": rec["pred_r"].tolist(),
                "pred_l": rec["pred_l"].tolist(),
            }
            for cid, rec in cache.items()
        },
    }
    CACHE_PATH.write_text(json.dumps(serial))
    log(f"wrote cache -> {CACHE_PATH}  ({len(cache)} clips)")
    return cache


# -------------------------------------------------------------------------
# Drop-in adapter for FF's keypoints_to_motion_data monkey-patch.
# -------------------------------------------------------------------------


def get_predict_fn(cache: dict[str, dict] | None = None):
    """Build a callable that mimics FF's ``predict_l2_for_series`` signature.

    The closure looks up the angles cache by clip identifier (which we
    derive from the clip's kp_path on disk -- the FF pipeline always passes
    a CouroKeypointSeries that was loaded from a known file). When the
    callable can't resolve the clip (e.g. ASPset clips, which aren't in
    our cache), it returns ``None`` so FF's machinery falls back to the
    original hand-engineered series.
    """
    if cache is None:
        cache = run_inference_all_clips()

    def predict(clip_id: str, mirror: bool = False) -> np.ndarray | None:
        rec = cache.get(clip_id)
        if rec is None:
            return None
        return rec["pred_l"] if mirror else rec["pred_r"]

    return predict


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-run inference even if cache exists.")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    log("=== Agent TT VideoPose3D L2 inference START ===")
    run_inference_all_clips(device=args.device, force=args.force)
    log("=== Agent TT done ===")


if __name__ == "__main__":
    main()
