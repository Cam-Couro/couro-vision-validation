"""DWPose-L inference driver for OpenCap LabValidation videos.

Differs from aspset_infer_dwpose_gpu.py only in input layout:
  - Reads from a manifest.json listing (subject, task, cam, video_path, bbox_csv)
  - Output naming: {subject}_{task}_{cam}.json (matches existing couro_keypoints/ layout)

Reuses the COCO-WholeBody → Halpe-26 remap so downstream training is identical.
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

INPUT_W, INPUT_H = 288, 384
SIMCC_SPLIT_RATIO = 2.0
MEAN_BGR = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD_BGR = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def load_session(onnx_path):
    return ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def read_bboxes(box_csv):
    boxes = []
    with open(box_csv) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                boxes.append([float(p) for p in parts[:4]])
    return np.asarray(boxes, dtype=np.float32)


def preprocess_crop(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = x2 - x1, y2 - y1
    aspect_input = INPUT_W / INPUT_H
    if w / max(h, 1e-9) > aspect_input:
        h = w / aspect_input
    else:
        w = h * aspect_input
    w *= 1.25; h *= 1.25
    src = np.array([
        [cx - 0.5*w, cy - 0.5*h],
        [cx + 0.5*w, cy - 0.5*h],
        [cx - 0.5*w, cy + 0.5*h],
    ], dtype=np.float32)
    dst = np.array([[0, 0], [INPUT_W, 0], [0, INPUT_H]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    M_inv = cv2.invertAffineTransform(M)
    crop = cv2.warpAffine(frame_bgr, M, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR).astype(np.float32)
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
    xy_h = np.hstack([xy_crop, ones])
    return xy_h @ M_inv.T


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


def process_video(video_path, box_csv, sess, out_path, batch_size=8):
    if out_path.exists() and out_path.stat().st_size > 1000:
        return True
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bboxes = read_bboxes(box_csv)
    input_name = sess.get_inputs()[0].name

    all_xy, all_conf, all_time = [], [], []
    frame_idx = 0
    batch_imgs, batch_minv, batch_fi = [], [], []

    def flush():
        if not batch_imgs:
            return
        x = np.stack(batch_imgs, axis=0)
        out = sess.run(None, {input_name: x})
        sx_batch, sy_batch = out[0], out[1]
        for sx, sy, M_inv, fi in zip(sx_batch, sy_batch, batch_minv, batch_fi):
            xy_crop, conf_dw = decode_simcc(sx, sy)
            xy_orig = transform_back(xy_crop, M_inv)
            xy_halpe, conf_halpe = dwpose_to_halpe26(xy_orig, conf_dw)
            all_xy.append(xy_halpe.tolist())
            all_conf.append(conf_halpe.tolist())
            all_time.append(fi / fps)
        batch_imgs.clear(); batch_minv.clear(); batch_fi.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bbox = bboxes[min(frame_idx, len(bboxes) - 1)] if len(bboxes) > 0 else np.array([0, 0, width, height], dtype=np.float32)
        chw, M_inv = preprocess_crop(frame, bbox)
        batch_imgs.append(chw); batch_minv.append(M_inv); batch_fi.append(frame_idx)
        if len(batch_imgs) >= batch_size:
            flush()
        frame_idx += 1
    flush()
    cap.release()

    seq = [
        {
            "frame_idx": i,
            "timestamp_ms": all_time[i] * 1000.0,
            "keypoints": all_xy[i],
            "keypoint_scores": all_conf[i],
        }
        for i in range(frame_idx)
    ]
    payload = {
        "schema_version": "couro-dwpose-halpe26-v1",
        "video_metadata": {
            "width": width, "height": height, "fps": fps, "total_frames": frame_idx,
        },
        "keypoints_sequence": seq,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest_path")
    p.add_argument("videos_dir", help="local dir where videos are downloaded")
    p.add_argument("bbox_dir", help="local dir with bbox CSVs")
    p.add_argument("out_dir")
    p.add_argument("--model", default="dw-ll_ucoco_384.onnx")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--subjects", nargs="*")
    args = p.parse_args()

    with open(args.manifest_path) as f:
        manifest = json.load(f)
    if args.subjects:
        wanted = set(args.subjects)
        manifest = [m for m in manifest if m["subject"] in wanted]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = Path(args.videos_dir)
    bbox_dir = Path(args.bbox_dir)

    sess = load_session(args.model)
    print(f"Model loaded. Provider: {sess.get_providers()[0]}", flush=True)
    print(f"Processing {len(manifest)} clips", flush=True)

    t0 = time.time()
    n_ok = 0
    for i, item in enumerate(manifest):
        name = item["name"]
        # Video local path follows S3 layout under videos_dir
        # s3://.../inputs/subject10/VideoData/Session0/Cam0/DJ1/DJ1_syncdWithMocap.avi
        vp = videos_dir / item["subject"] / "VideoData" / "Session0" / item["cam"] / item["task"] / f"{item['task']}_syncdWithMocap.avi"
        if not vp.exists():
            print(f"  SKIP no video: {vp}", flush=True)
            continue
        bp = bbox_dir / item["bbox_csv"]
        out_path = out_dir / f"{name}.json"
        try:
            if process_video(vp, bp, sess, out_path, batch_size=args.batch):
                n_ok += 1
        except Exception as e:
            print(f"  ERR {name}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (len(manifest) - i - 1) / (i + 1)
            print(f"  [{i+1}/{len(manifest)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    print(f"Done. {n_ok}/{len(manifest)} processed in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
