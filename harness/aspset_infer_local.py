"""Local RTMPose-X Halpe-26 inference driver for ASPset-510 videos.

Runs on Apple Silicon via CoreMLExecutionProvider — ~60-100fps on M-chip
Neural Engine.

For each video, uses ASPset's pre-computed bounding boxes (top-down RTMPose
needs a bbox per frame to crop the person from the scene).

Output schema matches what the rest of our pipeline expects (same format as
the OpenCap Couro keypoint JSONs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# RTMPose-X Halpe-26 model settings (matches the production checkpoint)
INPUT_W, INPUT_H = 288, 384       # (W, H) for preprocessing crop
SIMCC_SPLIT_RATIO = 2.0           # SimCC bins = input_dim * split_ratio
N_KEYPOINTS = 26
# RTMPose normalization
MEAN_BGR = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # imagenet RGB → BGR
STD_BGR = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def load_session(onnx_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        onnx_path,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )


def read_bboxes(box_csv: Path) -> np.ndarray:
    """Per-frame bboxes from ASPset CSV. Returns (N, 4) array of x1,y1,x2,y2."""
    boxes = []
    with open(box_csv) as f:
        header = f.readline()  # x1,y1,x2,y2
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                boxes.append([float(p) for p in parts[:4]])
    return np.asarray(boxes, dtype=np.float32)


def preprocess_crop(frame_bgr: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop person, resize to (288, 384), normalize. Returns (chw_float, affine_inv).

    affine_inv: 2x3 matrix to map predicted (x, y) in 288x384 cropped frame
    back to original video coordinates.
    """
    x1, y1, x2, y2 = bbox
    bx_cx, bx_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bx_w, bx_h = x2 - x1, y2 - y1

    # Match RTMPose's "aspect-ratio-corrected, padded" pre-processing
    aspect_input = INPUT_W / INPUT_H            # 288 / 384 = 0.75
    aspect_box = bx_w / max(bx_h, 1e-9)
    if aspect_box > aspect_input:
        # Box too wide — pad height
        bx_h = bx_w / aspect_input
    else:
        # Box too tall — pad width
        bx_w = bx_h * aspect_input
    # 25% padding (RTMPose convention)
    bx_w *= 1.25; bx_h *= 1.25

    # Build affine: source rect → (288, 384) dest
    src = np.array([
        [bx_cx - 0.5 * bx_w, bx_cy - 0.5 * bx_h],
        [bx_cx + 0.5 * bx_w, bx_cy - 0.5 * bx_h],
        [bx_cx - 0.5 * bx_w, bx_cy + 0.5 * bx_h],
    ], dtype=np.float32)
    dst = np.array([[0, 0], [INPUT_W, 0], [0, INPUT_H]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    M_inv = cv2.invertAffineTransform(M)

    crop = cv2.warpAffine(frame_bgr, M, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
    crop = crop.astype(np.float32)
    crop = (crop - MEAN_BGR) / STD_BGR
    # HWC → CHW
    chw = crop.transpose(2, 0, 1)
    return chw, M_inv


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode SimCC outputs to keypoints. Returns (xy, scores).

    simcc_x: (n_kp, n_bins_x) — n_bins_x = INPUT_W * SIMCC_SPLIT_RATIO
    simcc_y: (n_kp, n_bins_y) — n_bins_y = INPUT_H * SIMCC_SPLIT_RATIO
    """
    x_idx = np.argmax(simcc_x, axis=-1)
    y_idx = np.argmax(simcc_y, axis=-1)
    score_x = np.max(simcc_x, axis=-1)
    score_y = np.max(simcc_y, axis=-1)
    scores = np.minimum(score_x, score_y)
    xy = np.stack([
        x_idx / SIMCC_SPLIT_RATIO,
        y_idx / SIMCC_SPLIT_RATIO,
    ], axis=-1)
    # Mask: keypoints with very low conf get coords zeroed
    bad = scores < 0
    xy[bad] = 0
    scores = np.clip(scores, 0.0, 1.0)
    return xy, scores


def transform_back(xy_crop: np.ndarray, M_inv: np.ndarray) -> np.ndarray:
    """Map (n_kp, 2) keypoints from 288x384 crop coords back to original frame."""
    ones = np.ones((xy_crop.shape[0], 1), dtype=np.float32)
    xy_h = np.hstack([xy_crop, ones])
    return xy_h @ M_inv.T


def process_video(
    video_path: Path,
    box_csv: Path,
    sess: ort.InferenceSession,
    out_path: Path,
    batch_size: int = 8,
) -> bool:
    """Run RTMPose on every frame of one ASPset video."""
    if out_path.exists() and out_path.stat().st_size > 1000:
        return True  # idempotent

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bboxes = read_bboxes(box_csv)

    all_xy = []
    all_conf = []
    all_time = []

    frame_idx = 0
    batch_imgs: list[np.ndarray] = []
    batch_minv: list[np.ndarray] = []
    batch_frame_idx: list[int] = []

    def flush_batch():
        if not batch_imgs:
            return
        x = np.stack(batch_imgs, axis=0)
        out = sess.run(None, {"input": x})
        simcc_x_batch, simcc_y_batch = out[0], out[1]
        for i, (simcc_x, simcc_y, M_inv, fi) in enumerate(
            zip(simcc_x_batch, simcc_y_batch, batch_minv, batch_frame_idx)
        ):
            xy_crop, conf = decode_simcc(simcc_x, simcc_y)
            xy_orig = transform_back(xy_crop, M_inv)
            all_xy.append(xy_orig.tolist())
            all_conf.append(conf.tolist())
            all_time.append(fi / fps)
        batch_imgs.clear(); batch_minv.clear(); batch_frame_idx.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= len(bboxes):
            # bboxes truncated — use last available
            bbox = bboxes[-1] if len(bboxes) > 0 else np.array([0, 0, width, height], dtype=np.float32)
        else:
            bbox = bboxes[frame_idx]
        chw, M_inv = preprocess_crop(frame, bbox)
        batch_imgs.append(chw)
        batch_minv.append(M_inv)
        batch_frame_idx.append(frame_idx)
        if len(batch_imgs) >= batch_size:
            flush_batch()
        frame_idx += 1
    flush_batch()
    cap.release()

    # Couro production schema (matches load_couro_output expectations)
    keypoints_sequence = []
    for i in range(frame_idx):
        keypoints_sequence.append({
            "frame_idx": i,
            "timestamp_ms": all_time[i] * 1000.0,
            "keypoints": all_xy[i] if i < len(all_xy) else [[0.0, 0.0]] * 26,
            "keypoint_scores": all_conf[i] if i < len(all_conf) else [0.0] * 26,
        })
    payload = {
        "schema_version": "couro-rtmpose-halpe26-v1",
        "video_metadata": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": frame_idx,
        },
        "keypoints_sequence": keypoints_sequence,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos_root", help="path to dir containing per-subject video folders")
    p.add_argument("boxes_root", help="path to dir containing per-subject box CSVs")
    p.add_argument("out_dir", help="where to save keypoint JSONs")
    p.add_argument("--model", default="/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/models/rtmpose-halpe26.onnx")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--subjects", nargs="*", help="if set, only process these subject ids")
    args = p.parse_args()

    videos_root = Path(args.videos_root)
    boxes_root = Path(args.boxes_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = load_session(args.model)
    print(f"Model loaded. Provider: {sess.get_providers()[0]}", flush=True)

    # Find all video files matching subjects filter
    if args.subjects:
        video_files = []
        for s in args.subjects:
            video_files.extend(sorted((videos_root / s).glob("*.mkv")))
            video_files.extend(sorted((videos_root / s).glob("*.mp4")))
    else:
        video_files = sorted(videos_root.rglob("*.mkv")) + sorted(videos_root.rglob("*.mp4"))

    print(f"Processing {len(video_files)} videos with batch={args.batch}", flush=True)
    t_start = time.time()
    n_ok = 0
    for i, vp in enumerate(video_files):
        # Find the matching box CSV
        # video filename: {subject}-{clip}-{view}.mkv → box CSV: {subject}-{clip}-{view}.csv
        rel = vp.relative_to(videos_root)
        box_csv = boxes_root / rel.parent.name / (vp.stem + ".csv")
        if not box_csv.exists():
            print(f"  SKIP no bbox: {vp.name}", flush=True)
            continue
        out_path = out_dir / (vp.stem + ".json")
        try:
            ok = process_video(vp, box_csv, sess, out_path, batch_size=args.batch)
            if ok:
                n_ok += 1
        except Exception as e:
            print(f"  ERROR {vp.name}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            eta_s = elapsed * (len(video_files) - i - 1) / (i + 1)
            print(f"  [{i+1}/{len(video_files)}] elapsed={elapsed:.0f}s eta={eta_s:.0f}s", flush=True)
    print(f"Done. {n_ok}/{len(video_files)} videos processed in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
