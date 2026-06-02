"""Run the full rear-camera validation pipeline for a single MPI-INF-3DHP
subject:

    1. Identify rear camera (largest |rel_az| from cam0; should be cam 7 for
       all subjects since they share the studio rig, but we verify per-subject
       and fall back to whatever camera is closest to -159 deg).
    2. Generate bboxes from 2D GT joint annotations for the rear cam.
    3. Run DWPose-L inference on the rear cam video.
    4. Compute GT joint angles from annot3 mocap (camera-invariant).
    5. Compute predicted joint angles from DWPose Halpe-26 keypoints.
    6. Pearson r per metric, written as r_S{N}_cam{rear_id}.json.

This is the multi-subject counterpart to the original S1 pipeline. We only
run the rear camera per subject to stay within the time budget.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
sys.path.insert(0, str(HARNESS))

from mpi3dhp_bboxes import make_bboxes, write_bbox_csv  # noqa: E402
from mpi3dhp_camera_analysis import classify_cameras, parse_calibration  # noqa: E402
from mpi3dhp_correlate import correlate  # noqa: E402
from mpi3dhp_gt_angles import compute_gt_angles  # noqa: E402
from mpi3dhp_infer_dwpose import run_video  # noqa: E402
from mpi3dhp_pred_angles import compute_pred_angles  # noqa: E402
from aspset_infer_dwpose import load_session  # noqa: E402

DATA_ROOT = ROOT / "data" / "mpi_inf_3dhp"
DWPOSE_OUT_ROOT = ROOT / "data" / "mpi_inf_3dhp_dwpose"
GT_OUT_ROOT = ROOT / "data" / "mpi_inf_3dhp_gt"
PRED_OUT_ROOT = ROOT / "data" / "mpi_inf_3dhp_rear_validation"
BOX_OUT_ROOT = PRED_OUT_ROOT / "boxes"
MODEL_PATH = ROOT / "models" / "dw-ll_ucoco_384.onnx"


def select_rear_camera(layout: dict) -> dict:
    """Return the camera dict with rel_az closest to -159 deg (cam 7 in S1).

    Falls back to whichever camera is most rearward (largest |rel_az|).
    """
    cams = layout["cameras"]
    target = -159.0
    best = min(cams, key=lambda c: abs(c["rel_az_from_cam0"] - target))
    # If the best match is not actually rearward, just take max abs az
    if abs(best["rel_az_from_cam0"]) < 100:
        best = max(cams, key=lambda c: abs(c["rel_az_from_cam0"]))
    return best


def run_subject(
    subject_id: int,
    *,
    stride: int = 2,
    batch_size: int = 8,
    rear_cam_override: int | None = None,
    rerun_dwpose: bool = False,
) -> dict:
    sx = f"S{subject_id}"
    seq_dir = DATA_ROOT / sx / "Seq2"
    annot_mat = seq_dir / "annot.mat"
    calib = seq_dir / "camera.calibration"
    img_dir = seq_dir / "imageSequence"

    report: dict = {"subject": sx, "steps": []}

    # 1. Camera layout
    if not calib.exists():
        report["status"] = "missing_calibration"
        return report
    cams = parse_calibration(calib)
    layout = classify_cameras(cams)
    PRED_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (PRED_OUT_ROOT / f"camera_layout_{sx}.json").write_text(json.dumps(layout, indent=2))

    if rear_cam_override is not None:
        rear_id = rear_cam_override
        rear_meta = next((c for c in layout["cameras"] if c["cam_id"] == rear_id), {})
    else:
        rear_meta = select_rear_camera(layout)
        rear_id = rear_meta["cam_id"]
    report["rear_cam"] = {
        "cam_id": rear_id,
        "rel_az_from_cam0": rear_meta.get("rel_az_from_cam0"),
        "height_mm": rear_meta.get("height_mm"),
        "radius_mm": rear_meta.get("radius_mm"),
    }
    report["steps"].append(f"camera_analysis: rear=cam{rear_id}")

    video_path = img_dir / f"video_{rear_id}.avi"
    if not video_path.exists():
        report["status"] = "missing_rear_video"
        return report

    # 2. Bboxes
    box_csv = BOX_OUT_ROOT / f"{sx}_Seq2_cam{rear_id}.csv"
    BOX_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not box_csv.exists():
        boxes = make_bboxes(annot_mat, rear_id, pad_frac=0.15)
        write_bbox_csv(boxes, box_csv)
        report["steps"].append(f"bboxes: {len(boxes)} frames")
    else:
        report["steps"].append("bboxes: reused")

    # 3. DWPose inference
    dw_json = DWPOSE_OUT_ROOT / f"{sx}_Seq2_cam{rear_id}.json"
    DWPOSE_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if dw_json.exists() and not rerun_dwpose:
        report["steps"].append(f"dwpose: reused {dw_json.name}")
    else:
        t0 = time.time()
        sess = load_session(str(MODEL_PATH))
        meta = run_video(
            video_path,
            box_csv,
            sess,
            dw_json,
            batch_size=batch_size,
            max_frames=None,
            frame_stride=stride,
        )
        report["steps"].append(
            f"dwpose: {meta['processed_frames']} frames "
            f"in {time.time() - t0:.0f}s"
        )

    # 4. GT angles
    gt_json = GT_OUT_ROOT / f"{sx}_Seq2_angles.json"
    GT_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not gt_json.exists():
        angles = compute_gt_angles(annot_mat, cam_id=0)
        out = {k: v.tolist() for k, v in angles.items()}
        out["_source"] = {"annot_mat": str(annot_mat), "cam_id": 0}
        gt_json.write_text(json.dumps(out))
        report["steps"].append(f"gt_angles: {len(angles)} metrics")
    else:
        report["steps"].append("gt_angles: reused")

    # 5. Predicted angles
    pred_json = PRED_OUT_ROOT / f"pred_{sx}_cam{rear_id}.json"
    pred = compute_pred_angles(dw_json, conf_threshold=0.3)
    out = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in pred.items()}
    out["_source"] = {"dwpose_json": str(dw_json)}
    pred_json.write_text(json.dumps(out))
    report["steps"].append(f"pred_angles -> {pred_json.name}")

    # 6. Correlation
    r_json = PRED_OUT_ROOT / f"r_{sx}_cam{rear_id}.json"
    results = correlate(pred_json, gt_json)
    r_payload = {
        "subject": sx,
        "cam_id": rear_id,
        "label": f"{sx} rear (cam{rear_id})",
        "rear_meta": rear_meta,
        "results": results,
    }
    r_json.write_text(json.dumps(r_payload, indent=2))
    report["steps"].append(f"correlation -> {r_json.name}")
    report["r_summary"] = {
        m: round(results[m]["pearson_r"], 3) for m in results
    }
    report["status"] = "ok"
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("subject_id", type=int)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--rear-cam", type=int, default=None,
                   help="Override rear camera ID (default: auto-detect)")
    p.add_argument("--rerun-dwpose", action="store_true")
    args = p.parse_args()
    report = run_subject(
        args.subject_id,
        stride=args.stride,
        batch_size=args.batch,
        rear_cam_override=args.rear_cam,
        rerun_dwpose=args.rerun_dwpose,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
