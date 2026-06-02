"""Run DWPose CoreML/ONNX on a single MPI-INF-3DHP camera video, using a CSV
of pre-generated bounding boxes (one per frame) for the cropping step.

Output JSON matches the Couro production schema (Halpe-26, 26 keypoints).
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

# Reuse helpers from the ASPset DWPose driver to avoid duplication.
HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))

from aspset_infer_dwpose import (  # noqa: E402
    INPUT_H,
    INPUT_W,
    decode_simcc,
    dwpose_to_halpe26,
    load_session,
    preprocess_crop,
    read_bboxes,
    transform_back,
)


def run_video(
    video_path: Path,
    box_csv: Path,
    sess: ort.InferenceSession,
    out_path: Path,
    *,
    batch_size: int = 8,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bboxes = read_bboxes(box_csv)
    input_name = sess.get_inputs()[0].name

    all_xy: list[list] = []
    all_conf: list[list] = []
    all_time: list[float] = []
    all_frame_idx: list[int] = []

    batch_imgs: list[np.ndarray] = []
    batch_minv: list[np.ndarray] = []
    batch_fi: list[int] = []

    def flush() -> None:
        if not batch_imgs:
            return
        x = np.stack(batch_imgs, axis=0)
        out = sess.run(None, {input_name: x})
        sx_batch, sy_batch = out[0], out[1]
        for sx, sy, M_inv, fi in zip(sx_batch, sy_batch, batch_minv, batch_fi):
            xy_crop, conf_dw = decode_simcc(sx, sy)
            xy_orig = transform_back(xy_crop, M_inv)
            xy_h, conf_h = dwpose_to_halpe26(xy_orig, conf_dw)
            all_xy.append(xy_h.tolist())
            all_conf.append(conf_h.tolist())
            all_time.append(fi / fps if fps > 0 else fi / 50.0)
            all_frame_idx.append(fi)
        batch_imgs.clear(); batch_minv.clear(); batch_fi.clear()

    frame_idx = 0
    processed = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            if frame_idx >= len(bboxes):
                break
            bbox = bboxes[frame_idx]
            chw, M_inv = preprocess_crop(frame, bbox)
            batch_imgs.append(chw); batch_minv.append(M_inv); batch_fi.append(frame_idx)
            if len(batch_imgs) >= batch_size:
                flush()
                processed += batch_size
                if processed % 200 == 0:
                    elapsed = time.time() - t0
                    rate = processed / max(elapsed, 1e-6)
                    print(f"  [{video_path.name}] processed {processed} frames "
                          f"({rate:.1f} fps)", flush=True)
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break
    flush()
    cap.release()

    seq = [
        {
            "frame_idx": all_frame_idx[i],
            "timestamp_ms": all_time[i] * 1000.0,
            "keypoints": all_xy[i],
            "keypoint_scores": all_conf[i],
        }
        for i in range(len(all_frame_idx))
    ]
    payload = {
        "schema_version": "couro-dwpose-halpe26-v1",
        "video_metadata": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total,
            "processed_frames": len(seq),
            "frame_stride": frame_stride,
            "source_video": video_path.name,
            "source_dataset": "mpi_inf_3dhp",
        },
        "keypoints_sequence": seq,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    elapsed = time.time() - t0
    print(f"  -> wrote {out_path} ({len(seq)} frames, {elapsed:.0f}s)", flush=True)
    return payload["video_metadata"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("boxes_csv")
    p.add_argument("out_json")
    p.add_argument(
        "--model",
        default=str(HARNESS.parent / "models" / "dw-ll_ucoco_384.onnx"),
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame (default 1)")
    args = p.parse_args()

    sess = load_session(args.model)
    print(f"Model loaded. Provider: {sess.get_providers()[0]}", flush=True)
    run_video(
        Path(args.video),
        Path(args.boxes_csv),
        sess,
        Path(args.out_json),
        batch_size=args.batch,
        max_frames=args.max_frames,
        frame_stride=args.stride,
    )


if __name__ == "__main__":
    main()
