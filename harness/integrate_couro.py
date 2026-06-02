"""Post-process Couro inference outputs:
1. Read keypoint JSONs from local sync dir
2. Convert each to OpenSim .mot via couro_keypoints, using anthropometric
   foreshortening (segment-length 3D recovery) when subject height + camera
   intrinsics are available.
3. Write into the existing OpenSimData/Video/Couro_TrackMode/N-cameras/IK/ layout
   so the existing sweep harness picks them up as another 'source' automatically.

After this runs, `python -m harness.sweep` produces the Couro vs Vicon vs HRNet vs OpenPose
comparison table in one pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.couro_keypoints import (
        load_couro_output, write_as_mot, keypoints_to_motion_data,
    )
else:
    from .couro_keypoints import (
        load_couro_output, write_as_mot, keypoints_to_motion_data,
    )


def _load_subject_metadata(data_root: Path) -> dict[str, dict]:
    """Return {subject: {mass_kg, height_m}} from each subject's sessionMetadata.yaml."""
    meta: dict[str, dict] = {}
    for yml in data_root.glob("subject*/sessionMetadata.yaml"):
        with open(yml) as f:
            d = yaml.safe_load(f)
        meta[yml.parent.name] = {
            "mass_kg": float(d["mass_kg"]),
            "height_m": float(d["height_m"]),
        }
    return meta


def _camera_pickle_path(data_root: Path, subject: str, cam: str) -> Path | None:
    for sess in sorted((data_root / subject / "VideoData").glob("Session*")):
        pkl = sess / cam / "cameraIntrinsicsExtrinsics.pickle"
        if pkl.exists():
            return pkl
    return None


DEFAULT_DATA = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/LabValidation_withVideos").expanduser()
DEFAULT_KP_DIR = Path("~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/couro_keypoints").expanduser()


def integrate_keypoints(kp_dir: Path, data_root: Path, use_anthro: bool = True) -> dict:
    """Read every couro keypoint JSON, convert to .mot with optional anthropometric
    foreshortening (out-of-plane angle recovery), and write to
    OpenSimData/Video/Couro_TrackMode_{cam}/1-cameras/IK/{task}.mot.

    If `use_anthro=True` (default), uses subject height + camera intrinsics to
    recover out-of-plane joint angles. Writes to suffix `Couro_Anthro3D_{cam}`
    so the new source coexists with the original naive-2D source in the sweep.
    """
    subject_meta = _load_subject_metadata(data_root)
    written: list[Path] = []
    failures: list[tuple[str, str]] = []

    for jf in sorted(kp_dir.glob("*.json")):
        if jf.name.startswith("_"):
            continue
        try:
            # Filename: "{subject}_{task}_{cam}.json"
            base = jf.stem
            parts = base.split("_")
            if len(parts) < 3:
                continue
            subject = parts[0]
            cam = parts[-1]
            task = "_".join(parts[1:-1])

            series = load_couro_output(jf)
            if use_anthro and subject in subject_meta:
                cam_pkl = _camera_pickle_path(data_root, subject, cam)
                motion = keypoints_to_motion_data(
                    series,
                    subject_height_m=subject_meta[subject]["height_m"],
                    camera_pickle=cam_pkl,
                )
                source_dir = f"Couro_Anthro3D_{cam}"
            else:
                motion = keypoints_to_motion_data(series)
                source_dir = f"Couro_TrackMode_{cam}"

            out_path = (
                data_root / subject / "OpenSimData" / "Video" /
                source_dir / "1-cameras" / "IK" / f"{task}.mot"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Write .mot directly from MotionData (skip write_as_mot which re-parses)
            with open(out_path, "w") as f:
                f.write("Coordinates\nversion=1\n")
                f.write(f"nRows={motion.time.size}\n")
                f.write(f"nColumns={len(motion.columns)}\n")
                f.write("inDegrees=yes\n\n")
                f.write("endheader\n")
                f.write("\t".join(motion.columns) + "\n")
                import numpy as np
                for i in range(motion.time.size):
                    row = [f"{motion.time[i]:.4f}"]
                    row.extend(
                        f"{v:.4f}" if np.isfinite(v) else "nan"
                        for v in motion.values[i]
                    )
                    f.write("\t".join(row) + "\n")
            written.append(out_path)
        except Exception as e:  # pragma: no cover
            failures.append((jf.name, f"{type(e).__name__}: {e}"))

    return {
        "n_written": len(written),
        "n_failed": len(failures),
        "failures": failures[:5],
        "sample": [str(p) for p in written[:5]],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--keypoints-dir", type=Path, default=DEFAULT_KP_DIR)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    args = p.parse_args(argv)
    result = integrate_keypoints(args.keypoints_dir, args.data_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
