"""Download MPI-INF-3DHP S2-S8 Seq1 (Walking/Standing/Sitting) for cross-subject
walking-gait rear-view validation. Variant of mpi3dhp_download_multi.py
targeting Seq1 instead of Seq2.

Downloads only the rear camera video (video_7.avi) from vnect_cameras.zip
plus annot.mat and camera.calibration per subject.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://gvv.mpi-inf.mpg.de/3dhp-dataset"
DEST_ROOT = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/"
    "research-agent/multiview-validation/data/mpi_inf_3dhp"
)
SEQ = "Seq1"
TARGET_VIDEO = "video_7.avi"


def fetch(url: str, dst: Path, min_bytes: int = 1000) -> None:
    if dst.exists() and dst.stat().st_size > min_bytes:
        print(f"  [{dst.parent.parent.name}] exists: {dst.name}", flush=True)
        return
    print(f"  [{dst.parent.parent.name}] GET  {url}", flush=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "couro-research/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out)
    tmp.rename(dst)
    print(f"  [{dst.parent.parent.name}] saved {dst.name} ({time.time()-t0:.1f}s)", flush=True)


def extract_one(zip_path: Path, target_name: str, dst_dir: Path) -> bool:
    if (dst_dir / target_name).exists():
        return True
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if Path(name).name == target_name:
                with zf.open(name) as src, open(dst_dir / target_name, "wb") as out:
                    shutil.copyfileobj(src, out)
                return True
    return False


def download_subject(subject_id: int, keep_zip: bool = False) -> dict:
    sx = f"S{subject_id}"
    seq_dir = DEST_ROOT / sx / SEQ
    img_dir = seq_dir / "imageSequence"
    img_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"subject": sx, "status": "ok", "errors": []}
    try:
        fetch(f"{BASE}/{sx}/{SEQ}/annot.mat", seq_dir / "annot.mat")
        fetch(f"{BASE}/{sx}/{SEQ}/camera.calibration", seq_dir / "camera.calibration", min_bytes=100)

        vnect_zip = img_dir / "vnect_cameras.zip"
        fetch(f"{BASE}/{sx}/{SEQ}/imageSequence/vnect_cameras.zip", vnect_zip, min_bytes=10_000_000)

        ok = extract_one(vnect_zip, TARGET_VIDEO, img_dir)
        if not ok:
            result["status"] = "missing_video_7"
            result["errors"].append("video_7 not found in vnect zip")

        if not keep_zip and vnect_zip.exists():
            vnect_zip.unlink()
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["errors"].append(str(e))
        print(f"  [{sx}] ERROR: {e}", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_subject, s): s for s in args.subjects}
        for fut in as_completed(futures):
            r = fut.result()
            print(f"=== {r['subject']}: {r['status']} ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
