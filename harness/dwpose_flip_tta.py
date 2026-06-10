"""Agent UU - Layer 1 test-time horizontal-flip augmentation for DWPose.

This is the FIRST build in the project to touch Layer 1 (the keypoint
detector). Every prior agent (KK..TT) operated at Layer 2 or Layer 3.

Motivation (Agent TT, v45 clean negative)
------------------------------------------
``hip_adduction_r / side_right`` is stuck at CCC 0.27 -- the mirror twin of
``hip_adduction_r / side_left`` (CCC 0.94). Mirror-flip L2 *training* (Agent NN)
did not fix it; VideoPose3D 2D->3D lifting (Agent TT) made it worse. Both
clean negatives rule out downstream geometry / training data. The remaining
suspect is the DWPose detector itself: a systematic right-side keypoint
position/depth bias when the subject is viewed from oblique/side angles.

Hypothesis
----------
TRUE test-time flip augmentation cancels left/right detector asymmetry. Per
frame:
  1. Run DWPose on the original crop  -> keypoints_orig
  2. Run DWPose on the HORIZONTALLY-MIRRORED crop -> keypoints_flipped
  3. Un-flip keypoints_flipped: mirror x back to image space AND swap L/R
     keypoint labels (left_hip<->right_hip, ...).
  4. Average orig + unflipped (confidence-weighted) -> keypoints_tta.

This is still ONE camera. We run the detector twice on the same single frame
(original + its mirror), not two cameras. The inference contract is unchanged:
1 video stream -> keypoints -> 5 angles. It is NOT multi-camera fusion.

Why this is real flip-TTA (not a cached-keypoint identity)
----------------------------------------------------------
The flipped pass runs the ONNX detector on physically mirrored pixels. The
"right side" in the flipped image is read by the same network weights that read
the (better-performing) left side in the original. If DWPose has a left/right
asymmetry, the two passes disagree and the average pulls the right side toward
the better-conditioned estimate. Flipping cached keypoints alone would be a
geometric identity that adds zero information -- we do NOT do that.

Bounding boxes
--------------
The original OpenCap inference (opencap_infer_dwpose_gpu.py) used person-detector
bboxes that are not shipped in the repo. We reconstruct a per-frame ROI from the
cached original keypoints' visible-joint bounding box (padded). The SAME ROI
geometry is used for the original re-inference pass (which reproduces the cache
within decode noise -- a built-in sanity check) and, mirrored, for the flipped
pass. The ROI is just where to look; the detector still re-reads raw pixels.

Outputs
-------
``data/dwpose_flip_tta_keypoints/{subject}_{task}_{cam}.json`` -- identical
schema to ``data/opencap_dwpose_keypoints`` (couro-dwpose-halpe26-v1), so the
entire downstream L2/L3 pipeline reads it as a drop-in keypoint source.
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
OUT_DIR = DATA_ROOT / "dwpose_flip_tta_keypoints"

INPUT_W, INPUT_H = 288, 384
SIMCC_SPLIT_RATIO = 2.0
MEAN_BGR = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD_BGR = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# Halpe-26 indices, matching couro_keypoints.KP and the layout written by
# opencap_infer_dwpose_gpu.dwpose_to_halpe26.
#  0 nose | 1 L_eye 2 R_eye | 3 L_ear 4 R_ear | 5 L_sho 6 R_sho
#  7 L_elb 8 R_elb | 9 L_wri 10 R_wri | 11 L_hip 12 R_hip
# 13 L_kne 14 R_kne | 15 L_ank 16 R_ank | 17 head_top 18 neck 19 hip_center
# 20 L_big_toe 21 R_big_toe | 22 L_small_toe 23 R_small_toe | 24 L_heel 25 R_heel
HALPE_LR_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
    (20, 21), (22, 23), (24, 25),
]
# Midline (no swap): 0 nose, 17 head_top, 18 neck, 19 hip_center.
# Visible detector joints (exclude derived 17/18/19) for ROI estimation.
VISIBLE_JOINTS = [i for i in range(26) if i not in (17, 18, 19)]


# COCO-WholeBody (DWPose output) -> Halpe-26, copied verbatim from
# opencap_infer_dwpose_gpu so the cache layout is byte-compatible.
def dwpose_to_halpe26(xy_dw, conf_dw):
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


def load_session(onnx_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )


def preprocess_crop(frame_bgr, bbox, flip: bool):
    """Affine-crop the bbox to the model input. If flip=True, the crop pixels
    are horizontally mirrored before normalization (TRUE pixel flip)."""
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
    if flip:
        crop = cv2.flip(crop, 1)  # mirror raw pixels in crop space
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


def unflip_crop_coords(xy_crop):
    """Undo the in-crop horizontal mirror: x -> INPUT_W - 1 - x (crop space)."""
    out = xy_crop.copy()
    out[:, 0] = (INPUT_W - 1) - out[:, 0]
    return out


def swap_lr_halpe(xy, conf):
    """Swap left/right Halpe-26 keypoint labels in-place-safe (returns copies)."""
    xy2 = xy.copy()
    conf2 = conf.copy()
    for a, b in HALPE_LR_PAIRS:
        xy2[[a, b]] = xy[[b, a]]
        conf2[[a, b]] = conf[[b, a]]
    return xy2, conf2


def roi_from_keypoints(kp_xy, conf, width, height, pad_frac=0.25, conf_min=0.2):
    """Per-frame ROI from visible keypoints' bounding box, padded.

    Falls back to full frame if too few confident joints.
    """
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


def run_pass(frame, bbox, sess, input_name, flip: bool):
    """Single detector pass (original or flipped). Returns Halpe-26 xy, conf in
    ORIGINAL image coordinates (flip already undone)."""
    chw, M_inv = preprocess_crop(frame, bbox, flip=flip)
    out = sess.run(None, {input_name: chw[None, ...]})
    sx, sy = out[0][0], out[1][0]
    xy_crop, conf_dw = decode_simcc(sx, sy)
    if flip:
        xy_crop = unflip_crop_coords(xy_crop)
    xy_orig = transform_back(xy_crop, M_inv)
    xy_h, conf_h = dwpose_to_halpe26(xy_orig, conf_dw)
    if flip:
        # The crop was mirrored, so the network's "left" outputs are the
        # subject's right and vice-versa. Swap L/R labels to restore identity.
        xy_h, conf_h = swap_lr_halpe(xy_h, conf_h)
    return xy_h, conf_h


def tta_average(xy_o, c_o, xy_f, c_f, conf_weighted=True):
    """Average original + unflipped passes. Confidence-weighted per keypoint."""
    xy_o = np.asarray(xy_o)
    xy_f = np.asarray(xy_f)
    c_o = np.asarray(c_o)
    c_f = np.asarray(c_f)
    if conf_weighted:
        w_o = c_o[:, None]
        w_f = c_f[:, None]
        denom = (w_o + w_f)
        denom = np.where(denom > 1e-6, denom, 1.0)
        xy = (xy_o * w_o + xy_f * w_f) / denom
        # where both confidences ~0, fall back to plain mean
        zero = (c_o + c_f) <= 1e-6
        xy[zero] = 0.5 * (xy_o[zero] + xy_f[zero])
    else:
        xy = 0.5 * (xy_o + xy_f)
    conf = 0.5 * (c_o + c_f)
    return xy, conf


def video_path_for(subject: str, task: str, cam: str) -> Path | None:
    base = LAB / subject / "VideoData"
    for sess in ("Session0", "Session1"):
        vp = base / sess / cam / task / f"{task}_syncdWithMocap.avi"
        if vp.exists():
            return vp
    return None


def process_clip(orig_kp_path: Path, sess, input_name, out_path: Path,
                 conf_weighted: bool = True):
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

        xy_o, c_o = run_pass(frame, bbox, sess, input_name, flip=False)
        xy_f, c_f = run_pass(frame, bbox, sess, input_name, flip=True)
        xy_tta, c_tta = tta_average(xy_o, c_o, xy_f, c_f, conf_weighted)

        out_seq.append({
            "frame_idx": frame_idx,
            "timestamp_ms": fr.get("timestamp_ms", frame_idx / fps * 1000.0),
            "keypoints": xy_tta.tolist(),
            "keypoint_scores": c_tta.tolist(),
        })
        n_frames_proc += 1
        frame_idx += 1
    cap.release()

    payload = {
        "schema_version": "couro-dwpose-halpe26-v1",
        "tta": {
            "method": "layer1_horizontal_flip_tta",
            "passes": ["original", "flipped_unflipped"],
            "average": "confidence_weighted" if conf_weighted else "mean",
            "agent": "UU",
        },
        "video_metadata": {
            "width": width, "height": height, "fps": fps,
            "total_frames": len(out_seq),
        },
        "keypoints_sequence": out_seq,
    }
    out_path.write_text(json.dumps(payload))
    return "ok", time.time() - t0, n_frames_proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS_DIR / "dw-ll_ucoco_384.onnx"))
    ap.add_argument("--subjects", nargs="*", help="restrict to these subjects")
    ap.add_argument("--cams", nargs="*", help="restrict to these cams e.g. Cam0")
    ap.add_argument("--limit", type=int, default=0, help="max clips (smoke test)")
    ap.add_argument("--mean", action="store_true",
                    help="use plain mean instead of confidence-weighted")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sess = load_session(Path(args.model))
    input_name = sess.get_inputs()[0].name
    print(f"DWPose loaded. provider={sess.get_providers()[0]}", flush=True)

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
            status, dt, nfr = process_clip(cp, sess, input_name, out_path,
                                           conf_weighted=not args.mean)
        except Exception as e:
            print(f"  ERR {cp.stem}: {e!r}", flush=True)
            status, dt, nfr = "bad_video", 0.0, 0
        stats[status] = stats.get(status, 0) + 1
        if status == "ok" and nfr > 0:
            per_clip_ms.append(dt * 1000.0)
            per_frame_ms.append(dt * 1000.0 / nfr)
        if (i + 1) % 10 == 0:
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
            "per_clip_ms": {
                "mean": float(ac.mean()), "p50": float(np.median(ac)),
            },
            "note": "two detector passes/frame (original + flipped)",
            "device": sess.get_providers()[0],
            "n_clips_timed": len(per_clip_ms),
        }
        (OUT_DIR / "_timing.json").write_text(json.dumps(timing, indent=2))
    print(f"Done. {stats} in {time.time()-t_start:.0f}s", flush=True)
    if timing:
        print(f"Latency: {timing['per_frame_ms']['mean']:.1f} ms/frame "
              f"(2 passes), {timing['per_clip_ms']['mean']:.0f} ms/clip mean",
              flush=True)


if __name__ == "__main__":
    main()
