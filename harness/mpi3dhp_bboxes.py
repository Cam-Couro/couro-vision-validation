"""Generate per-frame bounding boxes for a given MPI-INF-3DHP camera by
taking the 2D joint annotations (annot2) for that camera and computing a
padded tight box around all 28 joints.

Output format matches the CSV format expected by aspset_infer_dwpose.py:
    x1,y1,x2,y2 per frame, one line per frame, with a header.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


def make_bboxes(
    annot_mat_path: Path,
    cam_id: int,
    pad_frac: float = 0.15,
) -> np.ndarray:
    """Return (n_frames, 4) array of [x1, y1, x2, y2] tight padded boxes."""
    m = sio.loadmat(str(annot_mat_path))
    a2 = m["annot2"][cam_id, 0]  # (n_frames, 56)
    n_frames = a2.shape[0]
    kps = a2.reshape(n_frames, 28, 2)
    x_min = kps[:, :, 0].min(axis=1)
    x_max = kps[:, :, 0].max(axis=1)
    y_min = kps[:, :, 1].min(axis=1)
    y_max = kps[:, :, 1].max(axis=1)
    w = x_max - x_min
    h = y_max - y_min
    px = w * pad_frac
    py = h * pad_frac
    boxes = np.stack([x_min - px, y_min - py, x_max + px, y_max + py], axis=1)
    # Clamp to image bounds — MPI-INF-3DHP frames are 2048x2048
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 2048)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 2048)
    return boxes


def write_bbox_csv(boxes: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("x1,y1,x2,y2\n")
        for b in boxes:
            f.write(f"{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("annot_mat")
    p.add_argument("cam_id", type=int)
    p.add_argument("out_csv")
    p.add_argument("--pad", type=float, default=0.15)
    args = p.parse_args()
    boxes = make_bboxes(Path(args.annot_mat), args.cam_id, args.pad)
    write_bbox_csv(boxes, Path(args.out_csv))
    print(f"Wrote {len(boxes)} bboxes to {args.out_csv}")


if __name__ == "__main__":
    main()
