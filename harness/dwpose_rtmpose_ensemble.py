"""Agent VV2 - Layer 1 two-ARCHITECTURE detector ensemble (DWPose + RTMPose).

This composes two existing, proven single-camera pipelines:
  * DWPose half: from harness/dwpose_flip_tta.py (DWPose -> COCO-WholeBody-133
    -> remapped to Halpe-26 via dwpose_to_halpe26). Run WITHOUT the flip
    (original frame only).
  * RTMPose half: from harness/aspset_infer_local.py (RTMPose-Halpe26 ONNX,
    INPUT_W,INPUT_H=288,384; SIMCC_SPLIT_RATIO=2.0; 25% padding). RTMPose-Halpe26
    outputs Halpe-26 DIRECTLY (no remap). We VERIFY index correspondence against
    the DWPose remap layout in the smoke check before trusting any full run.

Ensemble
--------
For each frame, run BOTH detectors on the SAME person bbox (the per-frame ROI
derived from the cached opencap_dwpose_keypoints visible-joint bounding box --
exactly the bbox logic from dwpose_flip_tta.py). Then confidence-weighted
average per keypoint:

    kp = (c_dw * kp_dw + c_rt * kp_rt) / (c_dw + c_rt)
    ensembled conf = mean(c_dw, c_rt)

``--mean`` switches to plain per-keypoint average (ablation).

SINGLE CAMERA ONLY. This runs TWO DETECTORS (two architectures) on the SAME
single-camera frame. It is NOT multi-camera fusion. The inference contract is
unchanged: 1 video stream -> keypoints -> 5 angles.

Output
------
``data/dwpose_rtmpose_ensemble_keypoints/{subject}_{task}_{cam}.json`` --
IDENTICAL schema to ``data/opencap_dwpose_keypoints`` (couro-dwpose-halpe26-v1),
so the entire downstream L2/L3 pipeline reads it as a drop-in keypoint source.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
LAB = DATA_ROOT / "LabValidation_withVideos"
ORIG_KP_DIR = DATA_ROOT / "opencap_dwpose_keypoints"
OUT_DIR = DATA_ROOT / "dwpose_rtmpose_ensemble_keypoints"

DWPOSE_MODEL = MODELS_DIR / "dw-ll_ucoco_384.onnx"
RTMPOSE_MODEL = MODELS_DIR / "rtmpose-halpe26.onnx"

# Shared preprocessing constants (both detectors use the same crop geometry).
INPUT_W, INPUT_H = 288, 384
SIMCC_SPLIT_RATIO = 2.0
MEAN_BGR = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD_BGR = np.array([58.395, 57.12, 57.375], dtype=np.float32)
N_KEYPOINTS = 26

# Halpe-26 indices, matching couro_keypoints.KP and the layout written by
# opencap_infer_dwpose_gpu.dwpose_to_halpe26 (and emitted by RTMPose-Halpe26).
#  0 nose | 1 L_eye 2 R_eye | 3 L_ear 4 R_ear | 5 L_sho 6 R_sho
#  7 L_elb 8 R_elb | 9 L_wri 10 R_wri | 11 L_hip 12 R_hip
# 13 L_kne 14 R_kne | 15 L_ank 16 R_ank | 17 head_top 18 neck 19 hip_center
# 20 L_big_toe 21 R_big_toe | 22 L_small_toe 23 R_small_toe | 24 L_heel 25 R_heel
HALPE_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sho", "R_sho", "L_elb", "R_elb", "L_wri", "R_wri",
    "L_hip", "R_hip", "L_kne", "R_kne", "L_ank", "R_ank",
    "head_top", "neck", "hip_center",
    "L_big_toe", "R_big_toe", "L_small_toe", "R_small_toe", "L_heel", "R_heel",
]
VISIBLE_JOINTS = [i for i in range(26) if i not in (17, 18, 19)]


# ---------------------------------------------------------------------------
# DWPose half (verbatim from dwpose_flip_tta.py)
# ---------------------------------------------------------------------------
def dwpose_to_halpe26(xy_dw, conf_dw):
    """COCO-WholeBody (DWPose output) -> Halpe-26, byte-compatible with cache."""
    CW = {
        "nose": 0, "L_eye": 1, "R_eye": 2, "L_ear": 3, "R_ear": 4,
        "L_sho": 5, "R_sho": 6, "L_elb": 7, "R_elb": 8,
        "L_wri": 9, "R_wri": 10,
        "L_hip": 11, "R_hip": 12, "L_kne": 13, "R_kne": 14,
        "L_ank": 15, "R_ank": 16,
        "L_big_toe": 17, "L_small_toe": 18, "L_heel": 19,
        "R_big_toe": 20, "R_small_toe": 21, "R_heel": 22,
    }
    xy = np.zeros((26, 2), dtype=np.float64)
    conf = np.zeros((26,), dtype=np.float64)
    direct = {
        0: "nose", 1: "L_eye", 2: "R_eye", 3: "L_ear", 4: "R_ear",
        5: "L_sho", 6: "R_sho", 7: "L_elb", 8: "R_elb",
        9: "L_wri", 10: "R_wri",
        11: "L_hip", 12: "R_hip", 13: "L_kne", 14: "R_kne",
        15: "L_ank", 16: "R_ank",
        20: "L_big_toe", 21: "R_big_toe",
        22: "L_small_toe", 23: "R_small_toe",
        24: "L_heel", 25: "R_heel",
    }
    for halpe_idx, name in direct.items():
        cw_idx = CW[name]
        xy[halpe_idx] = xy_dw[cw_idx]
        conf[halpe_idx] = conf_dw[cw_idx]
    eye_mid = (xy_dw[CW["L_eye"]] + xy_dw[CW["R_eye"]]) / 2
    nose = xy_dw[CW["nose"]]
    xy[17] = eye_mid + (eye_mid - nose) * 1.5
    conf[17] = float(np.mean([conf_dw[CW["L_eye"]], conf_dw[CW["R_eye"]], conf_dw[CW["nose"]]]))
    xy[18] = (xy_dw[CW["L_sho"]] + xy_dw[CW["R_sho"]]) / 2
    conf[18] = float(min(conf_dw[CW["L_sho"]], conf_dw[CW["R_sho"]]))
    xy[19] = (xy_dw[CW["L_hip"]] + xy_dw[CW["R_hip"]]) / 2
    conf[19] = float(min(conf_dw[CW["L_hip"]], conf_dw[CW["R_hip"]]))
    return xy, conf


def load_session_cpu(onnx_path: Path) -> ort.InferenceSession:
    """DWPose runs on CPU (matches dwpose_flip_tta.py / the original cache)."""
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def load_session_coreml(onnx_path: Path) -> ort.InferenceSession:
    """RTMPose runs on CoreML when available (matches aspset_infer_local.py)."""
    return ort.InferenceSession(
        str(onnx_path),
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )


def preprocess_crop(frame_bgr, bbox):
    """Affine-crop the bbox to the model input (288x384), 25% pad. Returns
    (chw_float, M_inv). Identical geometry for both detectors."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = x2 - x1, y2 - y1
    aspect_input = INPUT_W / INPUT_H
    if w / max(h, 1e-9) > aspect_input:
        h = w / aspect_input
    else:
        w = h * aspect_input
    w *= 1.25
    h *= 1.25
    src = np.array([
        [cx - 0.5 * w, cy - 0.5 * h],
        [cx + 0.5 * w, cy - 0.5 * h],
        [cx - 0.5 * w, cy + 0.5 * h],
    ], dtype=np.float32)
    dst = np.array([[0, 0], [INPUT_W, 0], [0, INPUT_H]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    M_inv = cv2.invertAffineTransform(M)
    crop = cv2.warpAffine(frame_bgr, M, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
    crop = crop.astype(np.float32)
    crop = (crop - MEAN_BGR) / STD_BGR
    chw = crop.transpose(2, 0, 1)
    return chw, M_inv


def decode_simcc(simcc_x, simcc_y):
    x_idx = np.argmax(simcc_x, axis=-1)
    y_idx = np.argmax(simcc_y, axis=-1)
    sx = np.max(simcc_x, axis=-1)
    sy = np.max(simcc_y, axis=-1)
    scores = np.clip(np.minimum(sx, sy), 0.0, 1.0)
    xy = np.stack([x_idx / SIMCC_SPLIT_RATIO, y_idx / SIMCC_SPLIT_RATIO], axis=-1)
    return xy, scores


def transform_back(xy_crop, M_inv):
    ones = np.ones((xy_crop.shape[0], 1), dtype=np.float32)
    xy_h = np.hstack([xy_crop.astype(np.float32), ones])
    return xy_h @ M_inv.T


def roi_from_keypoints(kp_xy, conf, width, height, pad_frac=0.25, conf_min=0.2):
    """Per-frame ROI from cached visible keypoints' bounding box, padded.
    Identical to dwpose_flip_tta.roi_from_keypoints."""
    pts = kp_xy[VISIBLE_JOINTS]
    cf = conf[VISIBLE_JOINTS]
    mask = (cf >= conf_min) & np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    if mask.sum() < 4:
        return np.array([0.0, 0.0, float(width), float(height)], dtype=np.float32)
    p = pts[mask]
    x1, y1 = p[:, 0].min(), p[:, 1].min()
    x2, y2 = p[:, 0].max(), p[:, 1].max()
    w, h = x2 - x1, y2 - y1
    x1 -= pad_frac * w
    x2 += pad_frac * w
    y1 -= pad_frac * h
    y2 += pad_frac * h
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(width), x2)
    y2 = min(float(height), y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return np.array([0.0, 0.0, float(width), float(height)], dtype=np.float32)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def run_dwpose(frame, bbox, sess, input_name):
    """DWPose pass -> Halpe-26 xy, conf in ORIGINAL image coords."""
    chw, M_inv = preprocess_crop(frame, bbox)
    out = sess.run(None, {input_name: chw[None, ...]})
    sx, sy = out[0][0], out[1][0]
    xy_crop, conf_dw = decode_simcc(sx, sy)
    xy_orig = transform_back(xy_crop, M_inv)
    xy_h, conf_h = dwpose_to_halpe26(xy_orig, conf_dw)
    return xy_h, conf_h


def run_rtmpose(frame, bbox, sess, input_name):
    """RTMPose-Halpe26 pass -> Halpe-26 xy, conf in ORIGINAL image coords.
    RTMPose emits Halpe-26 directly (no remap)."""
    chw, M_inv = preprocess_crop(frame, bbox)
    out = sess.run(None, {input_name: chw[None, ...]})
    sx, sy = out[0][0], out[1][0]
    xy_crop, conf_rt = decode_simcc(sx, sy)
    xy_orig = transform_back(xy_crop, M_inv)
    return (np.asarray(xy_orig, dtype=np.float64),
            np.asarray(conf_rt, dtype=np.float64))


def ensemble_average(xy_dw, c_dw, xy_rt, c_rt, conf_weighted=True):
    """Confidence-weighted average per keypoint (or plain mean if --mean)."""
    xy_dw = np.asarray(xy_dw); xy_rt = np.asarray(xy_rt)
    c_dw = np.asarray(c_dw); c_rt = np.asarray(c_rt)
    if conf_weighted:
        w_dw = c_dw[:, None]
        w_rt = c_rt[:, None]
        denom = w_dw + w_rt
        denom = np.where(denom > 1e-6, denom, 1.0)
        xy = (xy_dw * w_dw + xy_rt * w_rt) / denom
        zero = (c_dw + c_rt) <= 1e-6
        xy[zero] = 0.5 * (xy_dw[zero] + xy_rt[zero])
    else:
        xy = 0.5 * (xy_dw + xy_rt)
    conf = 0.5 * (c_dw + c_rt)
    return xy, conf


def video_path_for(subject: str, task: str, cam: str) -> Path | None:
    base = LAB / subject / "VideoData"
    for sess in ("Session0", "Session1"):
        vp = base / sess / cam / task / f"{task}_syncdWithMocap.avi"
        if vp.exists():
            return vp
    return None


def process_clip(orig_kp_path: Path, dw_sess, dw_in, rt_sess, rt_in,
                 out_path: Path, conf_weighted: bool = True):
    if out_path.exists() and out_path.stat().st_size > 1000:
        return "cached", 0.0, 0
    stem = orig_kp_path.stem  # subject_task_cam
    parts = stem.split("_")
    subject = parts[0]
    cam = parts[-1]
    task = "_".join(parts[1:-1])
    vp = video_path_for(subject, task, cam)
    if vp is None:
        return "no_video", 0.0, 0

    orig = json.loads(orig_kp_path.read_text())
    seq = orig["keypoints_sequence"]
    meta = orig["video_metadata"]
    width = int(meta["width"])
    height = int(meta["height"])
    fps = float(meta["fps"])

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return "bad_video", 0.0, 0

    out_seq = []
    t0 = time.time()
    frame_idx = 0
    n_frames_proc = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= len(seq):
            break
        fr = seq[frame_idx]
        kp_xy = np.asarray(fr["keypoints"], dtype=np.float64)
        kp_conf = np.asarray(fr["keypoint_scores"], dtype=np.float64)
        bbox = roi_from_keypoints(kp_xy, kp_conf, width, height)

        xy_dw, c_dw = run_dwpose(frame, bbox, dw_sess, dw_in)
        xy_rt, c_rt = run_rtmpose(frame, bbox, rt_sess, rt_in)
        xy_e, c_e = ensemble_average(xy_dw, c_dw, xy_rt, c_rt, conf_weighted)

        out_seq.append({
            "frame_idx": frame_idx,
            "timestamp_ms": fr.get("timestamp_ms", frame_idx / fps * 1000.0),
            "keypoints": xy_e.tolist(),
            "keypoint_scores": c_e.tolist(),
        })
        n_frames_proc += 1
        frame_idx += 1
    cap.release()

    payload = {
        "schema_version": "couro-dwpose-halpe26-v1",
        "ensemble": {
            "method": "layer1_dwpose_rtmpose_detector_ensemble",
            "detectors": ["dwpose_dw-ll_ucoco_384", "rtmpose_halpe26"],
            "average": "confidence_weighted" if conf_weighted else "mean",
            "single_camera": True,
            "agent": "VV2",
        },
        "video_metadata": {
            "width": width, "height": height, "fps": fps,
            "total_frames": len(out_seq),
        },
        "keypoints_sequence": out_seq,
    }
    out_path.write_text(json.dumps(payload))
    return "ok", time.time() - t0, n_frames_proc


def smoke_check(dw_sess, dw_in, rt_sess, rt_in):
    """1-frame sanity check: both detectors produce 26 keypoints, print a few
    corresponding keypoints (L/R hip, L/R knee) from each detector, and compare
    the DWPose pass against the cached opencap_dwpose_keypoints for the same
    clip. STOP if indices don't correspond."""
    print("=== SMOKE CHECK (1 frame) ===", flush=True)
    clips = sorted(ORIG_KP_DIR.glob("*.json"))
    # prefer a side-cam clip (Cam0) so the smoke is representative of the run
    side = [c for c in clips if c.stem.split("_")[-1] in ("Cam0", "Cam4")]
    cand = side or clips
    chosen = None
    for cp in cand:
        parts = cp.stem.split("_")
        subject, cam = parts[0], parts[-1]
        task = "_".join(parts[1:-1])
        if video_path_for(subject, task, cam) is not None:
            chosen = cp
            break
    if chosen is None:
        print("SMOKE FAIL: no clip with an available video found", flush=True)
        return False
    parts = chosen.stem.split("_")
    subject, cam = parts[0], parts[-1]
    task = "_".join(parts[1:-1])
    vp = video_path_for(subject, task, cam)
    print(f"clip: {chosen.stem}  video: {vp}", flush=True)

    orig = json.loads(chosen.read_text())
    seq = orig["keypoints_sequence"]
    meta = orig["video_metadata"]
    width, height = int(meta["width"]), int(meta["height"])
    fr0 = seq[0]
    cached_xy = np.asarray(fr0["keypoints"], dtype=np.float64)
    cached_conf = np.asarray(fr0["keypoint_scores"], dtype=np.float64)
    bbox = roi_from_keypoints(cached_xy, cached_conf, width, height)

    cap = cv2.VideoCapture(str(vp))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("SMOKE FAIL: could not read first video frame", flush=True)
        return False

    try:
        xy_dw, c_dw = run_dwpose(frame, bbox, dw_sess, dw_in)
    except Exception as e:
        print(f"SMOKE FAIL: DWPose I/O error: {e!r}", flush=True)
        return False
    try:
        xy_rt, c_rt = run_rtmpose(frame, bbox, rt_sess, rt_in)
    except Exception as e:
        print(f"SMOKE FAIL: RTMPose I/O error: {e!r}", flush=True)
        return False

    if xy_dw.shape[0] != N_KEYPOINTS:
        print(f"SMOKE FAIL: DWPose produced {xy_dw.shape[0]} kp, expected 26", flush=True)
        return False
    if xy_rt.shape[0] != N_KEYPOINTS:
        print(f"SMOKE FAIL: RTMPose produced {xy_rt.shape[0]} kp, expected 26", flush=True)
        return False
    print(f"OK: both detectors produced {N_KEYPOINTS} keypoints", flush=True)

    # Print corresponding keypoints to confirm index correspondence
    probe = {"L_hip": 11, "R_hip": 12, "L_kne": 13, "R_kne": 14,
             "L_sho": 5, "R_sho": 6, "nose": 0}
    print("\n--- corresponding keypoints (DWPose vs RTMPose) ---", flush=True)
    for name, idx in probe.items():
        print(f"  [{idx:2d}] {name:7s}  DW=({xy_dw[idx][0]:7.1f},{xy_dw[idx][1]:7.1f}) c={c_dw[idx]:.2f}"
              f"   RT=({xy_rt[idx][0]:7.1f},{xy_rt[idx][1]:7.1f}) c={c_rt[idx]:.2f}", flush=True)

    # Correspondence heuristic: on the major torso/limb joints both detectors
    # should land within a sane pixel radius of each other AND of the cache.
    # If RTMPose's L_hip is closer to DWPose's R_hip than to DWPose's L_hip, the
    # index order is swapped -> STOP.
    corr_idxs = [5, 6, 11, 12, 13, 14]
    same_dist = float(np.mean([np.linalg.norm(xy_dw[i] - xy_rt[i]) for i in corr_idxs]))
    # cross-check L/R swap for the hip pair specifically
    lhip_same = np.linalg.norm(xy_dw[11] - xy_rt[11])
    lhip_cross = np.linalg.norm(xy_dw[11] - xy_rt[12])
    print(f"\nmean |DW-RT| over {corr_idxs} = {same_dist:.1f} px", flush=True)
    print(f"L_hip: |DW11-RT11|={lhip_same:.1f}px  |DW11-RT12|(crossed)={lhip_cross:.1f}px", flush=True)

    # DWPose-vs-cache agreement (re-inference should reproduce the cache to a
    # few px on confident joints, modulo decode noise + provider differences).
    cache_diff = float(np.mean([np.linalg.norm(xy_dw[i] - cached_xy[i])
                                for i in corr_idxs if cached_conf[i] > 0.2]))
    print(f"mean |DWPose_reinfer - cache| over confident corr joints = {cache_diff:.1f} px", flush=True)

    ok_corr = True
    if lhip_cross < lhip_same:
        print("SMOKE FAIL: RTMPose L_hip matches DWPose R_hip better than L_hip "
              "-> index order or L/R labelling is SWAPPED. STOP.", flush=True)
        ok_corr = False
    if same_dist > 80.0:
        print(f"SMOKE WARN: DW/RT disagree by {same_dist:.0f}px on torso joints "
              "-- large but not necessarily an index error (architecture diff). "
              "Inspect printed coords above.", flush=True)
    if cache_diff > 40.0:
        print(f"SMOKE WARN: DWPose re-inference differs from cache by {cache_diff:.0f}px "
              "(expected a few px). Provider/bbox difference -- inspect.", flush=True)
    if ok_corr:
        print("\nSMOKE PASS: 26 kp each, indices correspond (no L/R swap "
              "detected on hip pair). Safe to run full job.", flush=True)
    return ok_corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dwpose-model", default=str(DWPOSE_MODEL))
    ap.add_argument("--rtmpose-model", default=str(RTMPOSE_MODEL))
    ap.add_argument("--subjects", nargs="*", help="restrict to these subjects")
    ap.add_argument("--cams", nargs="*", help="restrict to these cams e.g. Cam0")
    ap.add_argument("--limit", type=int, default=0, help="max clips")
    ap.add_argument("--mean", action="store_true",
                    help="plain mean instead of confidence-weighted average")
    ap.add_argument("--smoke", action="store_true",
                    help="run 1-frame sanity check only, then exit")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dw_sess = load_session_cpu(Path(args.dwpose_model))
    dw_in = dw_sess.get_inputs()[0].name
    rt_sess = load_session_coreml(Path(args.rtmpose_model))
    rt_in = rt_sess.get_inputs()[0].name
    print(f"DWPose loaded provider={dw_sess.get_providers()[0]} input={dw_in!r}", flush=True)
    print(f"RTMPose loaded provider={rt_sess.get_providers()[0]} input={rt_in!r}", flush=True)

    if args.smoke:
        ok = smoke_check(dw_sess, dw_in, rt_sess, rt_in)
        raise SystemExit(0 if ok else 2)

    clips = sorted(ORIG_KP_DIR.glob("*.json"))
    if args.subjects:
        sset = set(args.subjects)
        clips = [c for c in clips if c.stem.split("_")[0] in sset]
    if args.cams:
        cset = set(args.cams)
        clips = [c for c in clips if c.stem.split("_")[-1] in cset]
    if args.limit:
        clips = clips[:args.limit]

    print(f"Processing {len(clips)} clips", flush=True)
    stats = {"ok": 0, "cached": 0, "no_video": 0, "bad_video": 0}
    per_clip_ms = []
    per_frame_ms = []
    t_start = time.time()
    for i, cp in enumerate(clips):
        out_path = OUT_DIR / cp.name
        try:
            status, dt, nfr = process_clip(
                cp, dw_sess, dw_in, rt_sess, rt_in, out_path,
                conf_weighted=not args.mean,
            )
        except Exception as e:
            print(f"  ERR {cp.stem}: {e!r}", flush=True)
            status, dt, nfr = "bad_video", 0.0, 0
        stats[status] = stats.get(status, 0) + 1
        if status == "ok" and nfr > 0:
            per_clip_ms.append(dt * 1000.0)
            per_frame_ms.append(dt * 1000.0 / nfr)
        if (i + 1) % 5 == 0:
            el = time.time() - t_start
            eta = el * (len(clips) - i - 1) / (i + 1)
            print(f"  [{i+1}/{len(clips)}] {stats} elapsed={el:.0f}s eta={eta:.0f}s",
                  flush=True)

    timing = {}
    if per_frame_ms:
        a = np.array(per_frame_ms)
        ac = np.array(per_clip_ms)
        timing = {
            "per_frame_ms": {
                "mean": float(a.mean()), "p50": float(np.median(a)),
                "p95": float(np.percentile(a, 95)),
                "min": float(a.min()), "max": float(a.max()),
            },
            "per_clip_ms": {"mean": float(ac.mean()), "p50": float(np.median(ac))},
            "note": "two detector passes/frame (DWPose CPU + RTMPose CoreML)",
            "dwpose_device": dw_sess.get_providers()[0],
            "rtmpose_device": rt_sess.get_providers()[0],
            "n_clips_timed": len(per_clip_ms),
        }
        (OUT_DIR / "_timing.json").write_text(json.dumps(timing, indent=2))
    print(f"Done. {stats} in {time.time()-t_start:.0f}s", flush=True)
    if timing:
        print(f"Latency: {timing['per_frame_ms']['mean']:.1f} ms/frame "
              f"(2 detectors), {timing['per_clip_ms']['mean']:.0f} ms/clip mean",
              flush=True)


if __name__ == "__main__":
    main()
