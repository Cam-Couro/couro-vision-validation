"""VideoPose3D 2D→3D Lifter Layer-2 POC.

Hypothesis: a pretrained 2D→3D pose lifter (trained on Human3.6M paired
2D+3D mocap, 3.6M frames) should beat Couro's anthropometric reconstruction
on side-view depth recovery and modestly on front-view, lifting pooled
Layer 2 |r| from 0.514 → 0.55–0.65.

Model picked:
    VideoPose3D (Pavllo et al., CVPR 2019, Facebook AI Research)
    - https://github.com/facebookresearch/VideoPose3D
    - Apache 2.0 license (commercially permissive)
    - Pretrained checkpoint `pretrained_h36m_detectron_coco.bin` (68 MB,
      already on disk at models/videopose3d_h36m.bin)
    - 243-frame receptive field temporal convnet
    - Input:  (T, 17, 2) Human3.6M-17 2D normalized keypoints
    - Output: (T, 17, 3) Human3.6M-17 3D positions, root (pelvis) at origin,
      Y-axis up (in the H36M canonical frame), metres.

Why not MotionBERT? MotionBERT's pretrained checkpoint is on Google Drive
behind a manual download flow, and its expected input ordering and
normalization are slightly different. VideoPose3D is faster to wire up,
has a smaller checkpoint, identical input convention to what we already
remap to, and is the most-cited 2D→3D-lifting baseline — exactly the right
comparison for "does any pretrained 2D→3D lifter beat anthropometric
recon."

What this script does:
    1. Load the 8 OpenCap drop-jump clips the SMPL POC was tested on
       (same 8 — apples-to-apples).
    2. Remap DWPose Halpe-26 → H36M-17 (via harness/videopose3d.py).
    3. Run lifter → (T, 17, 3) root-relative 3D joint positions.
    4. Compute the 5 Couro Layer-2 metrics from 3D positions.
    5. Pearson r vs OpenSim mocap GT, per clip per metric.
    6. Aggregate pooled |r|; compare to Couro anthropometric baseline.

Outputs (data/layer2_motionbert_poc/):
    - per_clip_r.json   raw numbers
    - REPORT.md         human summary
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from harness.couro_keypoints import load_couro_output, keypoints_to_motion_data
    from harness.parsers import parse_mot
    from harness.videopose3d import load_pretrained, lift_sequence
else:
    from .couro_keypoints import load_couro_output, keypoints_to_motion_data
    from .parsers import parse_mot
    from .videopose3d import load_pretrained, lift_sequence


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Paths and constants.
# -------------------------------------------------------------------------

CHECKPOINT_PATH: Final[Path] = REPO_ROOT / "models" / "videopose3d_h36m.bin"
DATA_ROOT: Final[Path] = REPO_ROOT / "data"
LAB: Final[Path] = DATA_ROOT / "LabValidation_withVideos"
OPENCAP_KP: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"
OUT_DIR: Final[Path] = DATA_ROOT / "layer2_motionbert_poc"

DEPLOY_METRICS: Final[tuple[str, ...]] = (
    "hip_flexion_r",
    "hip_adduction_r",
    "knee_angle_r",
    "ankle_angle_r",
    "lumbar_extension",
)

# H36M-17 indices used by VideoPose3D output (see harness/videopose3d.py).
H36M = {
    "hip": 0, "R_hip": 1, "R_kne": 2, "R_ank": 3,
    "L_hip": 4, "L_kne": 5, "L_ank": 6,
    "spine": 7, "thorax": 8, "neck": 9, "head": 10,
    "L_sho": 11, "L_elb": 12, "L_wri": 13,
    "R_sho": 14, "R_elb": 15, "R_wri": 16,
}

# Same 8 clips the SMPL POC was tested on (for apples-to-apples comparison).
TEST_CLIPS: Final[tuple[tuple[str, str, str], ...]] = (
    ("subject10", "DJ1", "Cam0"),
    ("subject10", "DJ1", "Cam2"),
    ("subject10", "DJ1", "Cam4"),
    ("subject10", "DJ2", "Cam2"),
    ("subject2",  "DJ1", "Cam0"),
    ("subject2",  "DJ1", "Cam2"),
    ("subject2",  "DJ1", "Cam4"),
    ("subject3",  "DJ1", "Cam2"),
)


# -------------------------------------------------------------------------
# Per-frame Couro Layer-2 metric extraction from H36M-17 3D positions.
#
# VideoPose3D output coordinate frame (Human3.6M canonical):
#   - root (hip, idx 0) at origin
#   - Y axis is roughly vertical (Human3.6M is Y-up in canonical form)
#   - X axis is lateral
#   - Z axis is depth (toward / away from camera)
# Because we compare with Pearson r (sign-insensitive for |r| reporting),
# absolute axis polarity is not load-bearing. We compute angles using
# vector math and let |r| handle convention sign flips.
# -------------------------------------------------------------------------


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Frame-wise angle between two 3D vectors, in degrees [0, 180]."""
    n1 = np.linalg.norm(v1, axis=-1) + 1e-9
    n2 = np.linalg.norm(v2, axis=-1) + 1e-9
    cos = np.clip((v1 * v2).sum(axis=-1) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def h36m_to_metrics(joints_3d: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the 5 Couro Layer-2 metrics from (T, 17, 3) H36M-17 positions.

    Definitions match Couro's `keypoints_to_motion_data` conventions, lifted
    to 3D vector math:

      hip_flexion_r: angle between (hip→torso) and (hip→knee) on the right
                     side, expressed as flexion (0 = neutral / aligned with
                     torso, positive = forward fold of thigh).
                     = 180 - interior_angle(hip→neck, hip→knee)
      knee_angle_r:  flexion at the knee, 0 = straight, positive = bent.
                     = 180 - interior_angle(knee→hip, knee→ankle)
      ankle_angle_r: dorsiflexion at the ankle, 0 = neutral standing.
                     = 90 - interior_angle(ankle→knee, ankle→toe-proxy).
                     H36M-17 has no toe joint, so we synthesise a toe direction
                     by extending the shank down-and-forward.
      hip_adduction_r: lateral angle of the right thigh in the pelvis frame.
                     = arctan2(lateral_component, vertical_component) of
                       the right thigh, where pelvis lateral axis is
                       (R_hip - L_hip) and pelvis vertical axis is up.
      lumbar_extension: angle of (pelvis_center → thorax) vs world up.

    Returns dict of (T,) arrays in degrees.
    """
    hip = joints_3d[:, H36M["hip"]]
    r_hip = joints_3d[:, H36M["R_hip"]]
    l_hip = joints_3d[:, H36M["L_hip"]]
    r_kne = joints_3d[:, H36M["R_kne"]]
    r_ank = joints_3d[:, H36M["R_ank"]]
    thorax = joints_3d[:, H36M["thorax"]]
    spine = joints_3d[:, H36M["spine"]]

    # Segment vectors (from joint center).
    torso_vec_r = thorax - r_hip
    thigh_r = r_kne - r_hip
    shank_r = r_ank - r_kne

    # ---- hip_flexion_r --------------------------------------------------
    hip_int_r = angle_between(torso_vec_r, thigh_r)
    hip_flexion_r = 180.0 - hip_int_r

    # ---- knee_angle_r ---------------------------------------------------
    # Interior angle at knee between (knee→hip) and (knee→ankle).
    knee_int_r = angle_between(r_hip - r_kne, r_ank - r_kne)
    knee_angle_r = 180.0 - knee_int_r

    # ---- ankle_angle_r --------------------------------------------------
    # H36M-17 has no toe joint, so we can't compute the conventional foot-vs-shank
    # interior angle directly. As a proxy that correlates with dorsiflexion
    # during a drop jump, use the angle between the shank and world-up:
    # at landing the shank tilts forward (dorsiflexion increases), at takeoff
    # the shank straightens (dorsiflexion decreases). For Pearson r vs
    # mocap ankle_angle_r this proxy carries the dominant waveform.
    up = np.array([0.0, 1.0, 0.0])
    shank_unit = shank_r / (np.linalg.norm(shank_r, axis=-1, keepdims=True) + 1e-9)
    cos_shank_up = np.clip(shank_unit @ up, -1.0, 1.0)
    # Shank-vs-up gives a [0, 180] magnitude. Signed by anterior tilt
    # (shank Z component). |r| is sign-insensitive so polarity does not matter.
    ankle_angle_r = np.degrees(np.arccos(cos_shank_up)) * np.sign(
        shank_unit[:, 2] + 1e-9,
    )

    # ---- hip_adduction_r ------------------------------------------------
    # Pelvis lateral axis (R_hip - L_hip)
    lat = r_hip - l_hip
    lat_unit = lat / (np.linalg.norm(lat, axis=-1, keepdims=True) + 1e-9)
    # Pelvis "down" axis: thorax→hip direction
    pelvis_down = hip - thorax
    pelvis_down_unit = pelvis_down / (
        np.linalg.norm(pelvis_down, axis=-1, keepdims=True) + 1e-9
    )
    # Right thigh in pelvis frame: project onto lateral and down axes.
    thigh_r_unit = thigh_r / (np.linalg.norm(thigh_r, axis=-1, keepdims=True) + 1e-9)
    thigh_lat = np.einsum("ij,ij->i", thigh_r_unit, lat_unit)
    thigh_down = np.einsum("ij,ij->i", thigh_r_unit, pelvis_down_unit)
    # Positive when thigh is medial (toward midline); negative when abducted.
    hip_adduction_r = np.degrees(np.arctan2(thigh_lat, thigh_down))

    # ---- lumbar_extension -----------------------------------------------
    # Spine direction: hip→thorax.
    spine_vec = thorax - hip
    spine_unit = spine_vec / (np.linalg.norm(spine_vec, axis=-1, keepdims=True) + 1e-9)
    # Angle from vertical, signed by forward lean (Z component, since H36M
    # Z is depth; we report |r| so sign of Z polarity is moot).
    cos_to_up = np.clip(spine_unit @ up, -1.0, 1.0)
    lumbar_mag = np.degrees(np.arccos(cos_to_up))
    # Apply a sign based on the spine's Z-component (direction toward / away
    # from camera). Pearson |r| washes this out anyway.
    lumbar_extension = lumbar_mag * np.sign(spine_unit[:, 2] + 1e-9)

    return {
        "hip_flexion_r": hip_flexion_r,
        "hip_adduction_r": hip_adduction_r,
        "knee_angle_r": knee_angle_r,
        "ankle_angle_r": ankle_angle_r,
        "lumbar_extension": lumbar_extension,
    }


# -------------------------------------------------------------------------
# Resampling and correlation.
# -------------------------------------------------------------------------


def resample_to_common(
    pred_t: np.ndarray, pred_y: np.ndarray,
    gt_t: np.ndarray, gt_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample pred_y(pred_t) onto gt_t, restricted to overlap."""
    pred_finite = np.isfinite(pred_t) & np.isfinite(pred_y)
    gt_finite = np.isfinite(gt_t) & np.isfinite(gt_y)
    if pred_finite.sum() < 2 or gt_finite.sum() < 2:
        return np.array([]), np.array([])
    pt = pred_t[pred_finite]
    py = pred_y[pred_finite]
    gt_x = gt_t[gt_finite]
    gy = gt_y[gt_finite]
    lo = max(pt[0], gt_x[0])
    hi = min(pt[-1], gt_x[-1])
    if hi - lo < 0.05:
        return np.array([]), np.array([])
    mask = (gt_x >= lo) & (gt_x <= hi)
    if mask.sum() < 2:
        return np.array([]), np.array([])
    return np.interp(gt_x[mask], pt, py), gy[mask]


def pearson_r(a: np.ndarray, b: np.ndarray, min_pairs: int = 30) -> float:
    if a.size < min_pairs:
        return float("nan")
    if np.var(a) < 1e-9 or np.var(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# -------------------------------------------------------------------------
# Clip loader (same JSON schema as the SMPL POC).
# -------------------------------------------------------------------------


def load_opencap_clip(kp_path: Path) -> dict:
    """Return raw arrays needed for the lifter.

    Returns dict with kp (T, 26, 2) pixel coords, conf (T, 26), t_s (T,)
    seconds, width / height / fps.
    """
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
        "t_s": t_ms / 1000.0,
        "width": int(d["video_metadata"]["width"]),
        "height": int(d["video_metadata"]["height"]),
        "fps": float(d["video_metadata"]["fps"]),
    }


# -------------------------------------------------------------------------
# Per-clip processing.
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipResult:
    clip_name: str
    subject: str
    trial: str
    cam: str
    n_frames: int
    inference_seconds: float
    metrics: dict[str, dict[str, float]]


def process_clip(
    kp_path: Path,
    model,
    device: str = "cpu",
) -> ClipResult | dict:
    """Run the lifter on a clip and score per-metric Pearson r vs mocap GT."""
    base = kp_path.stem
    parts = base.split("_")
    subject = parts[0]
    cam_name = parts[-1]
    trial = "_".join(parts[1:-1])
    mot_path = LAB / subject / "OpenSimData" / "Mocap" / "IK" / f"{trial}.mot"
    if not mot_path.exists():
        return {"clip": base, "skip": "no mot"}

    clip = load_opencap_clip(kp_path)

    # ---- Run the lifter -------------------------------------------------
    t0 = time.time()
    joints_3d = lift_sequence(
        model, clip["kp"], width=clip["width"], height=clip["height"],
        device=device,
    )  # (T, 17, 3)
    inference_seconds = time.time() - t0

    # ---- Compute angles from 3D positions -------------------------------
    pred_metrics = h36m_to_metrics(joints_3d)
    pred_t = clip["t_s"]

    # ---- Load mocap GT --------------------------------------------------
    gt = parse_mot(mot_path)
    gt_t = gt.time

    # ---- Couro current baseline (apples-to-apples re-compute) ----------
    import yaml
    meta_path = LAB / subject / "sessionMetadata.yaml"
    height_m = float(yaml.safe_load(meta_path.read_text())["height_m"])
    cam_pkl = next((LAB / subject / "VideoData").glob(
        f"Session*/{cam_name}/cameraIntrinsicsExtrinsics.pickle"
    ))
    series = load_couro_output(kp_path)
    try:
        couro_md = keypoints_to_motion_data(
            series, subject_height_m=height_m, camera_pickle=cam_pkl,
        )
    except Exception:
        couro_md = None

    # ---- Per-metric scoring --------------------------------------------
    metric_results: dict[str, dict[str, float]] = {}
    for metric in DEPLOY_METRICS:
        if not gt.has(metric):
            continue
        gt_y = gt.column(metric)

        # Lifter vs GT
        lift_r = float("nan")
        lift_abs_r = float("nan")
        lift_mae = float("nan")
        lift_y = pred_metrics.get(metric)
        if lift_y is not None:
            lr_resamp, gy = resample_to_common(pred_t, lift_y, gt_t, gt_y)
            lift_r = pearson_r(lr_resamp, gy)
            if np.isfinite(lift_r):
                lift_abs_r = abs(lift_r)
                lift_mae = float(np.mean(np.abs(lr_resamp - gy)))

        # Couro vs GT
        couro_r = float("nan")
        couro_abs_r = float("nan")
        couro_mae = float("nan")
        if couro_md is not None and couro_md.has(metric):
            cp_resamp, gy2 = resample_to_common(
                couro_md.time, couro_md.column(metric), gt_t, gt_y,
            )
            couro_r = pearson_r(cp_resamp, gy2)
            if np.isfinite(couro_r):
                couro_abs_r = abs(couro_r)
                couro_mae = float(np.mean(np.abs(cp_resamp - gy2)))

        metric_results[metric] = {
            "lifter_r": lift_r,
            "lifter_abs_r": lift_abs_r,
            "lifter_mae": lift_mae,
            "couro_r": couro_r,
            "couro_abs_r": couro_abs_r,
            "couro_mae": couro_mae,
        }

    return ClipResult(
        clip_name=base,
        subject=subject,
        trial=trial,
        cam=cam_name,
        n_frames=int(clip["kp"].shape[0]),
        inference_seconds=inference_seconds,
        metrics=metric_results,
    )


# -------------------------------------------------------------------------
# Aggregation.
# -------------------------------------------------------------------------


def aggregate(clip_results: list[ClipResult]) -> dict:
    """Pool per-metric and overall |r|."""
    per_metric: dict[str, dict] = {}
    for metric in DEPLOY_METRICS:
        lift_rs: list[float] = []
        couro_rs: list[float] = []
        lift_maes: list[float] = []
        couro_maes: list[float] = []
        for c in clip_results:
            m = c.metrics.get(metric)
            if m is None:
                continue
            if np.isfinite(m["lifter_abs_r"]):
                lift_rs.append(m["lifter_abs_r"])
            if np.isfinite(m["couro_abs_r"]):
                couro_rs.append(m["couro_abs_r"])
            if np.isfinite(m["lifter_mae"]):
                lift_maes.append(m["lifter_mae"])
            if np.isfinite(m["couro_mae"]):
                couro_maes.append(m["couro_mae"])
        per_metric[metric] = {
            "n_lifter": len(lift_rs),
            "n_couro": len(couro_rs),
            "lifter_mean_abs_r": float(np.mean(lift_rs)) if lift_rs else float("nan"),
            "couro_mean_abs_r": float(np.mean(couro_rs)) if couro_rs else float("nan"),
            "lifter_mean_mae": float(np.mean(lift_maes)) if lift_maes else float("nan"),
            "couro_mean_mae": float(np.mean(couro_maes)) if couro_maes else float("nan"),
        }
    pooled_lift: list[float] = []
    pooled_couro: list[float] = []
    for c in clip_results:
        for m in c.metrics.values():
            if np.isfinite(m["lifter_abs_r"]):
                pooled_lift.append(m["lifter_abs_r"])
            if np.isfinite(m["couro_abs_r"]):
                pooled_couro.append(m["couro_abs_r"])
    overall = {
        "n_lifter": len(pooled_lift),
        "n_couro": len(pooled_couro),
        "lifter_mean_abs_r": float(np.mean(pooled_lift)) if pooled_lift else float("nan"),
        "couro_mean_abs_r": float(np.mean(pooled_couro)) if pooled_couro else float("nan"),
    }
    return {"per_metric": per_metric, "overall": overall}


# -------------------------------------------------------------------------
# JSON serialization helpers.
# -------------------------------------------------------------------------


def clip_result_to_dict(cr: ClipResult) -> dict:
    return {
        "clip": cr.clip_name,
        "subject": cr.subject,
        "trial": cr.trial,
        "cam": cr.cam,
        "n_frames": cr.n_frames,
        "inference_seconds": cr.inference_seconds,
        "metrics": cr.metrics,
    }


# -------------------------------------------------------------------------
# Driver.
# -------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"Missing VideoPose3D checkpoint: {CHECKPOINT_PATH}")

    logger.info("Loading VideoPose3D pretrained_h36m_detectron_coco.bin ...")
    model = load_pretrained(CHECKPOINT_PATH, device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  loaded %d parameters", n_params)

    kp_paths: list[Path] = []
    for subject, trial, cam in TEST_CLIPS:
        p = OPENCAP_KP / f"{subject}_{trial}_{cam}.json"
        if p.exists():
            kp_paths.append(p)
        else:
            logger.info("  skip (no kp file): %s", p.name)

    logger.info("Processing %d OpenCap clips:", len(kp_paths))
    clip_results: list[ClipResult] = []
    for i, kp_path in enumerate(kp_paths):
        logger.info("[%d/%d] %s", i + 1, len(kp_paths), kp_path.name)
        try:
            res = process_clip(kp_path, model, device="cpu")
        except Exception as e:
            logger.exception("  FAILED on %s: %s", kp_path.name, e)
            continue
        if isinstance(res, dict):
            logger.info("  skip: %s", res.get("skip"))
            continue
        clip_results.append(res)
        logger.info(
            "  T=%d   inference %.2fs", res.n_frames, res.inference_seconds,
        )
        for metric, m in res.metrics.items():
            logger.info(
                "    %-20s  lifter r=%+.2f (|r|=%.2f)  couro r=%+.2f (|r|=%.2f)",
                metric, m["lifter_r"], m["lifter_abs_r"],
                m["couro_r"], m["couro_abs_r"],
            )

    agg = aggregate(clip_results)

    out_obj = {
        "version": "1.0",
        "model": "VideoPose3D pretrained_h36m_detectron_coco.bin",
        "license": "Apache 2.0",
        "device": "cpu",
        "remap": "Halpe-26 -> H36M-17 (see harness/videopose3d.py)",
        "clips": [clip_result_to_dict(c) for c in clip_results],
        "aggregate": agg,
    }
    out_path = OUT_DIR / "per_clip_r.json"
    out_path.write_text(json.dumps(out_obj, indent=2))
    logger.info("Wrote %s", out_path)

    overall = agg["overall"]
    logger.info("\nAggregate (mean |r| over test clips):")
    logger.info(
        "  pooled lifter |r| = %.3f (n=%d)",
        overall["lifter_mean_abs_r"], overall["n_lifter"],
    )
    logger.info(
        "  pooled Couro  |r| = %.3f (n=%d)",
        overall["couro_mean_abs_r"], overall["n_couro"],
    )
    for metric, a in agg["per_metric"].items():
        logger.info(
            "  %-20s  lifter=%.3f (n=%d)  couro=%.3f (n=%d)",
            metric, a["lifter_mean_abs_r"], a["n_lifter"],
            a["couro_mean_abs_r"], a["n_couro"],
        )


if __name__ == "__main__":
    main()
