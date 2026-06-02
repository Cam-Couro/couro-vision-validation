"""Multi-subject, multi-task sweep with proper paired-sample aggregate stats.

Each trial contributes one paired observation per (metric × source). Across the
full sweep, this gives N=subjects×trials paired points per (metric × source) —
enough for credible RMSE/MAE/ICC/Bland–Altman per cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.metrics import METRICS, extract_all
    from harness.parsers import parse_camera_pickle, parse_mot
    from harness.stats import compute_paired_stats, quality_tier
    from harness.views import bucket_summary, classify_camera
    from harness.run import DEFAULT_DATA, DEFAULT_RESULTS, discover_trial
else:
    from .metrics import METRICS, extract_all
    from .parsers import parse_camera_pickle, parse_mot
    from .stats import compute_paired_stats, quality_tier
    from .views import bucket_summary, classify_camera
    from .run import DEFAULT_DATA, DEFAULT_RESULTS, discover_trial


def discover_subjects(data_root: Path) -> list[str]:
    return sorted(p.name for p in data_root.glob("subject*") if p.is_dir())


def discover_tasks_for_subject(data_root: Path, subject: str) -> list[str]:
    ik_dir = data_root / subject / "OpenSimData" / "Mocap" / "IK"
    if not ik_dir.exists():
        return []
    return sorted(p.stem for p in ik_dir.glob("*.mot"))


def task_family(task: str) -> str:
    """Group similar tasks for stratified analysis."""
    base = task.lower()
    if base.startswith("djasym"):
        return "drop_jump_asymmetric"
    if base.startswith("dj"):
        return "drop_jump"
    if base.startswith("squatsasym"):
        return "squat_asymmetric"
    if base.startswith("squats"):
        return "squat"
    if base.startswith("stsweak"):
        return "sit_to_stand_weak_legs"
    if base.startswith("sts"):
        return "sit_to_stand"
    if base.startswith("walkingts"):
        return "walking_trunk_sway"
    if base.startswith("walking"):
        return "walking"
    if base.startswith("static"):
        return "static"
    return base


def run_sweep(
    data_root: Path,
    subjects: list[str] | None = None,
    task_filter: Iterable[str] | None = None,
) -> dict:
    """Walk all (subject, task) pairs; collect per-trial scalars per metric per source.

    Aggregates into per-(metric × source) paired stats and per-(metric × source × task_family).
    Camera classifications are recorded per subject.
    """
    if subjects is None:
        subjects = discover_subjects(data_root)
    task_filter_set: set[str] | None = set(task_filter) if task_filter else None

    # Long-form record per trial-source-metric
    records: list[dict] = []
    cam_classifications_by_subject: dict[str, list[dict]] = {}

    n_trials_attempted = 0
    n_trials_succeeded = 0
    skipped: list[tuple[str, str, str]] = []

    for subject in subjects:
        sub_root = data_root / subject
        if not sub_root.exists():
            continue
        # camera classifications
        cam_rows: list[dict] = []
        for sess in sorted(sub_root.glob("VideoData/Session*")):
            for cam_dir in sorted(sess.glob("Cam*")):
                pkl = cam_dir / "cameraIntrinsicsExtrinsics.pickle"
                if pkl.exists():
                    cam = parse_camera_pickle(pkl, cam_dir.name)
                    cls = classify_camera(cam)
                    cam_rows.append({
                        "cam_id": cls.cam_id, "yaw_deg": cls.yaw_deg, "bucket": cls.bucket,
                    })
        cam_classifications_by_subject[subject] = cam_rows

        for task in discover_tasks_for_subject(data_root, subject):
            if task_filter_set and task_family(task) not in task_filter_set and task not in task_filter_set:
                continue
            n_trials_attempted += 1
            try:
                paths = discover_trial(data_root, subject, task)
            except FileNotFoundError as e:
                skipped.append((subject, task, str(e)))
                continue
            try:
                gt = parse_mot(paths.mocap_ik)
                gt_scalars = extract_all(gt)
                for source, p in paths.video_iks.items():
                    test = parse_mot(p)
                    test_scalars = extract_all(test)
                    for spec in METRICS:
                        records.append({
                            "subject": subject,
                            "task": task,
                            "task_family": task_family(task),
                            "metric": spec.name,
                            "body_region": spec.body_region,
                            "plane": spec.plane,
                            "source": source,
                            "gt": float(gt_scalars[spec.name]),
                            "test": float(test_scalars[spec.name]),
                        })
                n_trials_succeeded += 1
            except Exception as e:  # pragma: no cover - data quality issues should not kill sweep
                skipped.append((subject, task, f"parse error: {e}"))

    # Aggregate per (metric × source)
    by_ms: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"gt": [], "test": []}
    )
    for r in records:
        if not (np.isfinite(r["gt"]) and np.isfinite(r["test"])):
            continue
        by_ms[(r["metric"], r["source"])]["gt"].append(r["gt"])
        by_ms[(r["metric"], r["source"])]["test"].append(r["test"])

    metric_source_stats: list[dict] = []
    for (metric, source), arrs in sorted(by_ms.items()):
        s = compute_paired_stats(np.array(arrs["test"]), np.array(arrs["gt"]))
        row = {"metric": metric, "source": source}
        row.update(asdict(s))
        row["quality"] = quality_tier(s.rmse)
        metric_source_stats.append(row)

    # Aggregate per (metric × source × task_family) for stratification
    by_mst: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"gt": [], "test": []}
    )
    for r in records:
        if not (np.isfinite(r["gt"]) and np.isfinite(r["test"])):
            continue
        key = (r["metric"], r["source"], r["task_family"])
        by_mst[key]["gt"].append(r["gt"])
        by_mst[key]["test"].append(r["test"])
    metric_source_task_stats: list[dict] = []
    for (metric, source, fam), arrs in sorted(by_mst.items()):
        s = compute_paired_stats(np.array(arrs["test"]), np.array(arrs["gt"]))
        row = {"metric": metric, "source": source, "task_family": fam}
        row.update(asdict(s))
        row["quality"] = quality_tier(s.rmse)
        metric_source_task_stats.append(row)

    # Camera bucket summary (aggregate across subjects)
    flat_buckets: dict[str, int] = defaultdict(int)
    for rows in cam_classifications_by_subject.values():
        for r in rows:
            flat_buckets[r["bucket"]] += 1

    return {
        "subjects": subjects,
        "n_trials_attempted": n_trials_attempted,
        "n_trials_succeeded": n_trials_succeeded,
        "n_skipped": len(skipped),
        "skipped_examples": skipped[:10],
        "camera_buckets_aggregate": dict(flat_buckets),
        "cameras_by_subject": cam_classifications_by_subject,
        "records": records,
        "metric_source_stats": metric_source_stats,
        "metric_source_task_stats": metric_source_task_stats,
    }


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Multi-subject sweep")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Restrict to task families: drop_jump, drop_jump_asymmetric, "
                        "squat, sit_to_stand, walking, walking_trunk_sway, static")
    p.add_argument("--out", default="sweep_results.json")
    args = p.parse_args(argv)

    result = run_sweep(args.data_root, args.subjects, args.tasks)
    args.results_root.mkdir(parents=True, exist_ok=True)
    out_path = args.results_root / args.out
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: (
            None if isinstance(x, float) and not np.isfinite(x) else x
        ))
    print(f"trials attempted: {result['n_trials_attempted']} "
          f"succeeded: {result['n_trials_succeeded']} "
          f"skipped: {result['n_skipped']}")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
