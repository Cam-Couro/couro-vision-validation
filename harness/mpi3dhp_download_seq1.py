"""Download MPI-INF-3DHP S1/Seq1 (Walking) data including wall cameras.

Variant of mpi3dhp_download.py targeting Seq1 (walking gait) for rear-view
walking validation, as a proxy for running gait in the existing harness.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://gvv.mpi-inf.mpg.de/3dhp-dataset"
SUBJECT = "S1"
SEQ = "Seq1"
DEST_ROOT = Path(
    "/Users/cameronvan/Documents/Claude/Projects/Couro/"
    "research-agent/multiview-validation/data/mpi_inf_3dhp"
)


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1000:
        print(f"  exists: {dst.name} ({dst.stat().st_size:,} bytes)", flush=True)
        return
    print(f"  GET   {url}", flush=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out)
    tmp.rename(dst)
    print(f"  saved {dst.name} ({dst.stat().st_size:,} bytes)", flush=True)


def unzip(zip_path: Path, dst_dir: Path) -> None:
    if not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if not base:
                continue
            target = dst_dir / base
            if target.exists():
                continue
            with zf.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            print(f"  unzip {base}", flush=True)


def main() -> None:
    seq_dir = DEST_ROOT / SUBJECT / SEQ
    img_dir = seq_dir / "imageSequence"
    img_dir.mkdir(parents=True, exist_ok=True)

    fetch(f"{BASE}/{SUBJECT}/{SEQ}/annot.mat", seq_dir / "annot.mat")
    fetch(f"{BASE}/{SUBJECT}/{SEQ}/camera.calibration", seq_dir / "camera.calibration")

    vnect_zip = img_dir / "vnect_cameras.zip"
    wall_zip = img_dir / "other_angled_cameras.zip"
    fetch(f"{BASE}/{SUBJECT}/{SEQ}/imageSequence/vnect_cameras.zip", vnect_zip)
    fetch(f"{BASE}/{SUBJECT}/{SEQ}/imageSequence/other_angled_cameras.zip", wall_zip)

    print("Unzipping...", flush=True)
    unzip(vnect_zip, img_dir)
    unzip(wall_zip, img_dir)

    print("Done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
