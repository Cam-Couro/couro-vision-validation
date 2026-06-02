"""End-to-end smoke-test runner: subject × task × (Vicon GT vs N-camera academic IK).

Couro CV outputs slot in as additional comparison rows once available — see
`load_couro_outputs()` for the expected input format.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Allow running as either `python -m harness.run` or `python harness/run.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.metrics import METRICS, extract_all
    from harness.parsers import parse_camera_pickle, parse_mot
    from harness.stats import PairedStats, compute_paired_stats, quality_tier
    from harness.views import bucket_summary, classify_camera
else:
    from .metrics import METRICS, extract_all
    from .parsers import parse_camera_pickle, parse_mot
    from .stats import PairedStats, compute_paired_stats, quality_tier
    from .views import bucket_summary, classify_camera


DEFAULT_DATA = Path(
    "~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data"
    "/LabValidation_withVideos"
).expanduser()
DEFAULT_RESULTS = Path(
    "~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results"
).expanduser()


@dataclass(frozen=True)
class TrialPaths:
    subject: str
    task: str
    mocap_ik: Path
    video_iks: dict[str, Path]              # source_label -> .mot path
    camera_pickles: dict[str, Path]         # cam_id -> pickle


def discover_trial(data_root: Path, subject: str, task: str) -> TrialPaths:
    sub_root = data_root / subject
    mocap_ik = sub_root / "OpenSimData" / "Mocap" / "IK" / f"{task}.mot"
    if not mocap_ik.exists():
        raise FileNotFoundError(f"missing Vicon GT IK: {mocap_ik}")

    video_iks: dict[str, Path] = {}
    video_root = sub_root / "OpenSimData" / "Video"
    for model_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        for ncam_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            ik = ncam_dir / "IK" / f"{task}.mot"
            if ik.exists():
                video_iks[f"{model_dir.name}__{ncam_dir.name}"] = ik

    camera_pickles: dict[str, Path] = {}
    for session_dir in sorted((sub_root / "VideoData").glob("Session*")):
        for cam_dir in sorted(session_dir.glob("Cam*")):
            pkl = cam_dir / "cameraIntrinsicsExtrinsics.pickle"
            if pkl.exists():
                camera_pickles[cam_dir.name] = pkl

    return TrialPaths(
        subject=subject, task=task, mocap_ik=mocap_ik,
        video_iks=video_iks, camera_pickles=camera_pickles,
    )


def _stats_dict(s: PairedStats) -> dict[str, float | int]:
    d = asdict(s)
    d["quality"] = quality_tier(s.rmse)
    return d


def _compare_waveforms(gt_path: Path, test_path: Path) -> dict[str, dict]:
    """Time-series comparison per metric: aligns the two .mot files and runs
    paired stats on each angle column extracted by the metric extractor.

    For peak/mean/ROM scalars, the time-series comparison returns the agreement
    of the *underlying joint angle waveform* — more informative than a single
    paired point. The scalar-level comparison happens separately in run().
    """
    gt = parse_mot(gt_path)
    test = parse_mot(test_path)

    # Resample test to GT time grid using linear interpolation so column-aligned
    # paired comparisons make sense.
    out: dict[str, dict] = {}
    columns_to_compare = sorted(set(gt.columns) & set(test.columns) - {"time"})
    for col in columns_to_compare:
        gt_v = gt.column(col)
        test_v_raw = test.column(col)
        if test.time.size < 2 or gt.time.size < 2:
            continue
        test_v = np.interp(gt.time, test.time, test_v_raw,
                           left=np.nan, right=np.nan)
        s = compute_paired_stats(test_v, gt_v)
        out[col] = _stats_dict(s)
    return out


def _compare_metric_scalars(gt_path: Path, test_paths: dict[str, Path]) -> dict[str, dict]:
    """Per Couro-metric scalar: GT value vs each test source's value.

    Single trial gives one paired point per metric per source — not enough for
    stats, but it sets up the table. In the multi-trial scaled run, each row
    becomes a paired sample for proper RMSE/MAE/ICC.
    """
    gt_motion = parse_mot(gt_path)
    gt_scalars = extract_all(gt_motion)

    rows: dict[str, dict] = {}
    for spec in METRICS:
        gt_val = gt_scalars[spec.name]
        rows[spec.name] = {
            "gt": gt_val,
            "by_source": {},
            "spec": {
                "body_region": spec.body_region,
                "plane": spec.plane,
                "units": spec.units,
                "description": spec.description,
            },
        }
        for source, path in test_paths.items():
            test_motion = parse_mot(path)
            test_val = extract_all(test_motion)[spec.name]
            rows[spec.name]["by_source"][source] = {
                "test": test_val,
                "abs_error": (
                    abs(test_val - gt_val) if np.isfinite(gt_val) and np.isfinite(test_val)
                    else float("nan")
                ),
            }
    return rows


def run_trial(
    data_root: Path,
    subject: str,
    task: str,
    couro_ik_paths: dict[str, Path] | None = None,
) -> dict:
    """Run smoke-test for one (subject, task). Returns a JSON-serializable result."""
    paths = discover_trial(data_root, subject, task)

    classifications = []
    for cam_id, pkl in sorted(paths.camera_pickles.items()):
        cam = parse_camera_pickle(pkl, cam_id)
        classifications.append(classify_camera(cam))
    view_buckets = bucket_summary(classifications)

    test_paths = dict(paths.video_iks)
    if couro_ik_paths:
        for label, p in couro_ik_paths.items():
            test_paths[f"couro__{label}"] = p

    waveform_stats: dict[str, dict] = {}
    for source, p in test_paths.items():
        waveform_stats[source] = _compare_waveforms(paths.mocap_ik, p)

    scalar_stats = _compare_metric_scalars(paths.mocap_ik, test_paths)

    return {
        "subject": subject,
        "task": task,
        "ground_truth": str(paths.mocap_ik.relative_to(data_root)),
        "sources_compared": sorted(test_paths.keys()),
        "view_buckets": view_buckets,
        "camera_classifications": [
            {"cam_id": c.cam_id, "yaw_deg": c.yaw_deg, "bucket": c.bucket}
            for c in classifications
        ],
        "waveform_stats": waveform_stats,
        "scalar_stats": scalar_stats,
    }


def load_couro_outputs(spec_path: Path | None) -> dict[str, Path]:
    """Couro CV stub: when CV pipeline outputs land, point at a spec JSON of:
        {"side": "/path/to/side_cam_only.mot", "front": "...", ...}
    All other steps stay identical.
    """
    if spec_path is None or not spec_path.exists():
        return {}
    with open(spec_path) as f:
        return {k: Path(v).expanduser() for k, v in json.load(f).items()}


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Couro multi-view validation smoke test")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--subject", default="subject2")
    p.add_argument("--task", default="DJ1")
    p.add_argument("--couro-spec", type=Path, default=None,
                   help="Optional JSON spec mapping view-name -> Couro IK .mot path")
    args = p.parse_args(argv)

    couro_paths = load_couro_outputs(args.couro_spec)
    result = run_trial(args.data_root, args.subject, args.task, couro_paths)
    args.results_root.mkdir(parents=True, exist_ok=True)
    out_path = args.results_root / f"{args.subject}_{args.task}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: (
            None if isinstance(x, float) and not np.isfinite(x) else x
        ))
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
