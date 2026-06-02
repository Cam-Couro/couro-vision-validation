"""Download MPI-INF-3DHP Seq2 data for additional subjects (S2-S8).

For the cross-subject rear-view validation we only need:
    - annot.mat (mocap GT)
    - camera.calibration (per-subject extrinsics)
    - video_7.avi from vnect_cameras.zip (rear camera, ~-159 deg from cam 0)

The dataset has TWO zips per subject:
    - vnect_cameras.zip: cams 0,1,2,4,5,6,7,8 (the standard VNect set, includes
      the rearward cam 7)
    - other_angled_cameras.zip: cams 3, 9, 10 (extra side / rear-oblique wall cams)

We download vnect_cameras.zip and extract only video_7.avi to save disk
space. Designed to be run in parallel: each subject is independent.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://gvv.mpi-inf.mpg.de/3dhp-dataset"
DEST_ROOT = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/"
    "research-agent/multiview-validation/data/mpi_inf_3dhp"
)
SEQ = "Seq2"
TARGET_VIDEO = "video_7.avi"  # Rear camera (cam 7, ~-159 deg)


def fetch(url: str, dst: Path, min_bytes: int = 1000) -> None:
    if dst.exists() and dst.stat().st_size > min_bytes:
        print(
            f"  [{dst.parent.parent.name}] exists: {dst.name} "
            f"({dst.stat().st_size:,} bytes)",
            flush=True,
        )
        return
    print(f"  [{dst.parent.parent.name}] GET  {url}", flush=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "couro-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out)
    tmp.rename(dst)
    elapsed = time.time() - t0
    print(
        f"  [{dst.parent.parent.name}] saved {dst.name} "
        f"({dst.stat().st_size:,} bytes, {elapsed:.1f}s)",
        flush=True,
    )


def extract_one(zip_path: Path, target_name: str, dst_dir: Path) -> bool:
    """Extract a single video file from the wall-cameras zip."""
    if (dst_dir / target_name).exists():
        print(
            f"  [{zip_path.parent.parent.parent.name}] {target_name} "
            f"already extracted",
            flush=True,
        )
        return True
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if Path(name).name == target_name:
                with zf.open(name) as src, open(dst_dir / target_name, "wb") as out:
                    shutil.copyfileobj(src, out)
                print(
                    f"  [{zip_path.parent.parent.parent.name}] extracted "
                    f"{target_name}",
                    flush=True,
                )
                return True
    print(
        f"  [{zip_path.parent.parent.parent.name}] WARN: {target_name} "
        f"not in zip",
        flush=True,
    )
    return False


def download_subject(subject_id: int, keep_zip: bool = False) -> dict:
    """Download annot, calibration, and video_7 for subject SX/Seq2."""
    sx = f"S{subject_id}"
    seq_dir = DEST_ROOT / sx / SEQ
    img_dir = seq_dir / "imageSequence"
    img_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"subject": sx, "status": "ok", "errors": []}
    try:
        # Annotations and calibration
        fetch(f"{BASE}/{sx}/{SEQ}/annot.mat", seq_dir / "annot.mat")
        fetch(
            f"{BASE}/{sx}/{SEQ}/camera.calibration",
            seq_dir / "camera.calibration",
            min_bytes=100,
        )

        # vnect_cameras zip (contains cams 0,1,2,4,5,6,7,8 incl. rear cam 7)
        vnect_zip = img_dir / "vnect_cameras.zip"
        fetch(
            f"{BASE}/{sx}/{SEQ}/imageSequence/vnect_cameras.zip",
            vnect_zip,
            min_bytes=10_000_000,
        )

        # Extract just video_7.avi (rear camera at ~-159 deg)
        ok = extract_one(vnect_zip, TARGET_VIDEO, img_dir)
        if not ok:
            result["status"] = "missing_video_7"
            result["errors"].append("video_7 not found in vnect zip")

        # Optionally remove the zip to save disk per subject
        if not keep_zip and vnect_zip.exists():
            vnect_zip.unlink()
            print(f"  [{sx}] removed vnect zip", flush=True)

    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["errors"].append(str(e))
        print(f"  [{sx}] ERROR: {e}", flush=True)

    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--subjects",
        default="2,3,4,5,6,7,8",
        help="Comma-separated subject IDs (default 2-8, skip S1)",
    )
    p.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the other_angled_cameras.zip files (default deletes them)",
    )
    args = p.parse_args()

    subjects = [int(s) for s in args.subjects.split(",") if s.strip()]
    print(f"Downloading subjects: {subjects}", flush=True)
    results: list[dict] = []
    for sid in subjects:
        results.append(download_subject(sid, keep_zip=args.keep_zip))
    print("\nSummary:")
    for r in results:
        print(f"  {r['subject']}: {r['status']}  {r.get('errors', [])}")


if __name__ == "__main__":
    sys.exit(main())
