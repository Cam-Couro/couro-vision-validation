"""ASPset-510 RTMPose inference driver — runs INSIDE Couro's prod container
on a g5.xlarge instance.

Usage (inside container):
    python aspset_infer.py /work/ASPset-510/trainval/videos /work/keypoints_out

For each .mkv in the videos directory:
    1. Downsample to 1280px max dim (matches Couro prod's MAX_DIMENSION=1280)
    2. Run RTMPose-X Halpe-26 inference → 26 keypoints per frame
    3. Save keypoint JSON in our standard schema (same as OpenCap outputs)

Output naming: {subject}-{clip_id}-{view}.json  (matches the .mkv filename)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Couro's path inside the container (Dockerfile flattens /app/services/ecs/rtm_pose_inference → /app/rtm_pose_inference)
sys.path.insert(0, "/app")

from rtm_pose_inference.core.inference import InferenceEngine
from rtm_pose_inference.config.config import Config


# Same MAX_DIMENSION Couro uses in production
MAX_DIMENSION = Config.VIDEO_PROCESSING["MAX_DIMENSION"]


def process_one_video(video_path: Path, out_dir: Path, engine: InferenceEngine) -> bool:
    """Process one video → save Couro keypoint JSON."""
    out_path = out_dir / f"{video_path.stem}.json"
    if out_path.exists():
        return True  # idempotent — skip if already processed

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: could not open {video_path}", flush=True)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute downsample target — matches Couro's VideoProcessor behavior
    max_dim = max(width_orig, height_orig)
    scale = MAX_DIMENSION / max_dim if max_dim > MAX_DIMENSION else 1.0
    width = int(width_orig * scale)
    height = int(height_orig * scale)

    keypoints_sequence = []
    confidences = []
    times = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Single-person inference (ASPset is mostly single-athlete clips)
        result = engine.infer_single_frame(frame)
        # Result schema: per-frame [n_people, 26, 3] (x, y, conf)
        if result and len(result) > 0:
            person = max(result, key=lambda p: p[:, 2].mean())  # pick highest-conf person
            kp_xy = person[:, :2].tolist()
            kp_conf = person[:, 2].tolist()
        else:
            kp_xy = [[0.0, 0.0]] * 26
            kp_conf = [0.0] * 26
        keypoints_sequence.append(kp_xy)
        confidences.append(kp_conf)
        times.append(frame_idx / fps)
        frame_idx += 1

    cap.release()

    payload = {
        "schema_version": "couro-rtmpose-halpe26-v1",
        "video": str(video_path.name),
        "width": width,
        "height": height,
        "fps": fps,
        "n_frames": frame_idx,
        "time": times,
        "keypoints_sequence": keypoints_sequence,
        "confidence": confidences,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return True


def main():
    if len(sys.argv) != 3:
        print("usage: aspset_infer.py <videos_root> <keypoints_out>")
        sys.exit(1)
    videos_root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = InferenceEngine(Config.INFERENCE_CONFIG["track_mode"])
    print(f"Engine loaded. MAX_DIMENSION={MAX_DIMENSION}", flush=True)

    videos = sorted(videos_root.rglob("*.mkv"))
    print(f"Processing {len(videos)} videos...", flush=True)
    n_ok = 0
    for i, v in enumerate(videos):
        if i % 50 == 0:
            print(f"[{i}/{len(videos)}] {v.name}", flush=True)
        if process_one_video(v, out_dir, engine):
            n_ok += 1
    print(f"Done: {n_ok}/{len(videos)} videos processed", flush=True)


if __name__ == "__main__":
    main()
