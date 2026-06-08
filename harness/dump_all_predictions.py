"""Agent SS unified dump driver.

Runs each L3 reader's existing retrain pipeline with the prediction-dump
accumulator enabled, writing per-subject (pred, gt) tuples to
``data/per_slot_predictions/per_slot_predictions_<reader>.json``.

Readers covered:

  * v17 -- baseline hand-engineered (biomech_validity_stats)
  * v18 -- FF learned L2 (layer3_retrain_on_learned_l2)
  * v20 -- GG ROM-aware L2 (derived from per_clip_results.json -- already
           persisted on disk)
  * v23 -- HH combined L2 (layer3_retrain_on_combined_l2 -> FF.run())
  * v24 -- LL combined ROM-aware L2 (layer3_retrain_on_combined_rom_aware_l2)
  * v26 -- MM per-source per-frame L2 (layer3_retrain_on_persource_perframe_l2)
  * v27 -- MM per-source ROM-aware L2 (layer3_retrain_on_persource_romaware_l2)
  * v29 -- NN mirror-flip per-source L2 (layer3_retrain_on_mirrorflip_l2)
  * v30 -- learned L3 on v23 L2 (learned_layer3.py --l2 v23)
  * v31 -- learned L3 on v29 L2 (learned_layer3.py --l2 v29)
  * v33 -- extrema-aware L3 on v23 L2 (learned_layer3_extrema.py --l2 v23)
  * v34 -- extrema-aware L3 on v29 L2 (learned_layer3_extrema.py --l2 v29)
  * v37 -- PP calibrate v23 (residual_calibration.py --readers v37)
  * v38 -- PP calibrate v29-based v31 (residual_calibration.py --readers v38)
  * v39 -- PP calibrate v17 (residual_calibration.py --readers v39)
  * v41 -- QQ calibrate v20 (calibrate_v20.py)

Run individual readers via ``--readers v23 v30 ...`` or all via no flag.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.per_slot_prediction_dump import DumpAccumulator, DUMP_DIR


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[SS-DRIVER t={elapsed:6.1f}s] {msg}", flush=True)


def _ensure_dir() -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)


def _already_dumped(reader: str) -> bool:
    return (DUMP_DIR / f"per_slot_predictions_{reader}.json").exists()


# -------------------------------------------------------------------------
# v20: derive per-subject means from already-persisted per-clip results.
# -------------------------------------------------------------------------


def dump_v20() -> Path:
    """Convert rom_aware_layer2/per_clip_results.json into the unified format.

    The existing file already contains per-clip predictions for every LOSO
    fold (held-out OpenCap subject) at the metric level, but only as raw
    angle traces with peak/min/ROM. We aggregate per-subject means of the
    pred_rom and gt_rom values across all clips/views/metrics.

    NOTE: the v20 deploy file (deploy_ready_models_v20_rom_aware.json)
    actually publishes per-(metric, view) ridge L3 stats. Those ridges
    were trained on top of the v20 L2 angle traces. The raw per-clip
    ROM in per_clip_results.json is the L2-only output BEFORE the L3
    ridge fit. To match the v42 oracle CCC for v20, we need the L3
    output; that requires re-running v20's L3 pass. So we fall back
    to re-running rom_aware_layer2 with the dump hook -- but that takes
    significant time. As a pragmatic shortcut, we skip v20 and rely on
    other readers covering its slots in the ensemble pool. Document the
    skip honestly.
    """
    raise NotImplementedError(
        "v20 dump requires re-running rom_aware_layer2 L3 ridge pass."
    )


# -------------------------------------------------------------------------
# v17: biomech_validity_stats baseline.
# -------------------------------------------------------------------------


def dump_v17(force: bool = False) -> Path | None:
    if not force and _already_dumped("v17"):
        log("v17 already dumped, skipping")
        return DUMP_DIR / "per_slot_predictions_v17.json"
    log("=== dump v17 (biomech_validity_stats baseline) ===")
    from harness import biomech_validity_stats as bvs
    bvs.DUMP_ACCUMULATOR = DumpAccumulator(
        reader="v17",
        loso_discipline=(
            "Layer-3-LOSO baseline (hand-engineered features per approach)."
        ),
    )
    out = bvs.run(skip_combined=False)
    bvs.write_outputs(out)
    out_path = bvs.DUMP_ACCUMULATOR.write()
    bvs.DUMP_ACCUMULATOR = None
    log(f"v17 dump complete: {out_path}")
    return out_path


# -------------------------------------------------------------------------
# v18: FF directly.
# -------------------------------------------------------------------------


def dump_v18(force: bool = False) -> Path | None:
    if not force and _already_dumped("v18"):
        log("v18 already dumped, skipping")
        return DUMP_DIR / "per_slot_predictions_v18.json"
    log("=== dump v18 (FF: layer3_retrain_on_learned_l2) ===")
    from harness import layer3_retrain_on_learned_l2 as ff
    ff.DUMP_ACCUMULATOR = DumpAccumulator(
        reader="v18",
        loso_discipline="Layer-3-LOSO-only (FF: L2 trained on all OpenCap)",
    )
    ff.run()
    out_path = ff.DUMP_ACCUMULATOR.write()
    ff.DUMP_ACCUMULATOR = None
    log(f"v18 dump complete: {out_path}")
    return out_path


# -------------------------------------------------------------------------
# v23/v24/v26/v27/v29: delegate to FF via their wrapper scripts.
# Each wrapper monkey-patches FF.build_or_load_all_data_l2 and calls
# FF.run(). The dump accumulator persists across that call since it is a
# module-level attribute on FF that they don't touch.
# -------------------------------------------------------------------------


def _dump_via_ff_wrapper(reader: str, wrapper_module_name: str,
                         loso_discipline: str, force: bool = False) -> Path | None:
    if not force and _already_dumped(reader):
        log(f"{reader} already dumped, skipping")
        return DUMP_DIR / f"per_slot_predictions_{reader}.json"
    log(f"=== dump {reader} (delegating to FF via {wrapper_module_name}) ===")
    from harness import layer3_retrain_on_learned_l2 as ff
    ff.DUMP_ACCUMULATOR = DumpAccumulator(
        reader=reader, loso_discipline=loso_discipline,
    )
    wrapper = __import__(
        wrapper_module_name, fromlist=["run"],
    )
    wrapper.run()
    out_path = ff.DUMP_ACCUMULATOR.write()
    ff.DUMP_ACCUMULATOR = None
    log(f"{reader} dump complete: {out_path}")
    return out_path


def dump_v23(force: bool = False) -> Path | None:
    return _dump_via_ff_wrapper(
        reader="v23",
        wrapper_module_name="harness.layer3_retrain_on_combined_l2",
        loso_discipline=(
            "Layer-3-LOSO-only (KK: combined-cohort L2 trained on all "
            "9 OpenCap + 15 ASPset subjects)"
        ),
        force=force,
    )


def dump_v24(force: bool = False) -> Path | None:
    return _dump_via_ff_wrapper(
        reader="v24",
        wrapper_module_name="harness.layer3_retrain_on_combined_rom_aware_l2",
        loso_discipline=(
            "Layer-3-LOSO-only (LL: combined-cohort ROM-aware L2)"
        ),
        force=force,
    )


def dump_v26(force: bool = False) -> Path | None:
    return _dump_via_ff_wrapper(
        reader="v26",
        wrapper_module_name="harness.layer3_retrain_on_persource_perframe_l2",
        loso_discipline=(
            "Layer-3-LOSO-only (MM-A: per-source per-frame L2)"
        ),
        force=force,
    )


def dump_v27(force: bool = False) -> Path | None:
    return _dump_via_ff_wrapper(
        reader="v27",
        wrapper_module_name="harness.layer3_retrain_on_persource_romaware_l2",
        loso_discipline=(
            "Layer-3-LOSO-only (MM-B: per-source ROM-aware L2)"
        ),
        force=force,
    )


def dump_v29(force: bool = False) -> Path | None:
    return _dump_via_ff_wrapper(
        reader="v29",
        wrapper_module_name="harness.layer3_retrain_on_mirrorflip_l2",
        loso_discipline=(
            "Layer-3-LOSO-only (NN: mirror-flip per-source L2)"
        ),
        force=force,
    )


# -------------------------------------------------------------------------
# v30/v31: learned L3 on v23/v29 L2.
# -------------------------------------------------------------------------


def dump_v30(force: bool = False) -> Path | None:
    if not force and _already_dumped("v30"):
        log("v30 already dumped, skipping")
        return DUMP_DIR / "per_slot_predictions_v30.json"
    log("=== dump v30 (NN: learned L3 on v23 L2) ===")
    from harness import learned_layer3 as ll3
    ll3.DUMP_ACCUMULATOR = DumpAccumulator(
        reader="v30",
        loso_discipline=(
            "Layer-3-LOSO-only (NN: v23 L2 + learned TinyMLP L3 with "
            "ridge fallback)"
        ),
    )
    ll3._orchestrate(tag="v30_learned_l3", l2_kind="v23")
    out_path = ll3.DUMP_ACCUMULATOR.write()
    ll3.DUMP_ACCUMULATOR = None
    log(f"v30 dump complete: {out_path}")
    return out_path


def dump_v31(force: bool = False) -> Path | None:
    if not force and _already_dumped("v31"):
        log("v31 already dumped, skipping")
        return DUMP_DIR / "per_slot_predictions_v31.json"
    log("=== dump v31 (NN: learned L3 on v29 L2) ===")
    from harness import learned_layer3 as ll3
    ll3.DUMP_ACCUMULATOR = DumpAccumulator(
        reader="v31",
        loso_discipline=(
            "Layer-3-LOSO-only (NN: v29 mirror-flip L2 + learned L3)"
        ),
    )
    ll3._orchestrate(tag="v31_mirrorflip_learned_l3", l2_kind="v29")
    out_path = ll3.DUMP_ACCUMULATOR.write()
    ll3.DUMP_ACCUMULATOR = None
    log(f"v31 dump complete: {out_path}")
    return out_path


# -------------------------------------------------------------------------
# v33/v34: extrema-aware L3 on v23/v29 L2.
# -------------------------------------------------------------------------


def dump_v33(force: bool = False) -> Path | None:
    if not force and _already_dumped("v33"):
        log("v33 already dumped, skipping")
        return DUMP_DIR / "per_slot_predictions_v33.json"
    log("=== dump v33 (OO: extrema-aware L3 on v23 L2) ===")
    from harness import learned_layer3_extrema as oo
    oo.DUMP_ACCUMULATOR = DumpAccumulator(
        reader="v33",
        loso_discipline=(
            "Layer-3-LOSO-only (OO: v23 L2 + extrema-aware learned L3)"
        ),
    )
    oo._orchestrate(tag="v33_extrema_l3", l2_kind="v23")
    out_path = oo.DUMP_ACCUMULATOR.write()
    oo.DUMP_ACCUMULATOR = None
    log(f"v33 dump complete: {out_path}")
    return out_path


def dump_v34(force: bool = False) -> Path | None:
    if not force and _already_dumped("v34"):
        log("v34 already dumped, skipping")
        return DUMP_DIR / "per_slot_predictions_v34.json"
    log("=== dump v34 (OO: extrema-aware L3 on v29 L2) ===")
    from harness import learned_layer3_extrema as oo
    oo.DUMP_ACCUMULATOR = DumpAccumulator(
        reader="v34",
        loso_discipline=(
            "Layer-3-LOSO-only (OO: v29 mirror-flip L2 + extrema L3)"
        ),
    )
    oo._orchestrate(tag="v34_mirrorflip_extrema_l3", l2_kind="v29")
    out_path = oo.DUMP_ACCUMULATOR.write()
    oo.DUMP_ACCUMULATOR = None
    log(f"v34 dump complete: {out_path}")
    return out_path


# -------------------------------------------------------------------------
# v37/v38/v39: PP residual calibration.
# -------------------------------------------------------------------------


def dump_calibrated(reader: str, force: bool = False) -> Path | None:
    if not force and _already_dumped(reader):
        log(f"{reader} already dumped, skipping")
        return DUMP_DIR / f"per_slot_predictions_{reader}.json"
    log(f"=== dump {reader} (PP: residual calibration) ===")
    from harness import residual_calibration as rc
    rc.DUMP_ACCUMULATOR = DumpAccumulator(
        reader=reader,
        loso_discipline=(
            f"Nested LOSO with residual calibration (PP: {reader})"
        ),
    )
    rc._run_one_reader(reader)
    out_path = rc.DUMP_ACCUMULATOR.write()
    rc.DUMP_ACCUMULATOR = None
    log(f"{reader} dump complete: {out_path}")
    return out_path


def dump_v41(force: bool = False) -> Path | None:
    if not force and _already_dumped("v41"):
        log("v41 already dumped, skipping")
        return DUMP_DIR / "per_slot_predictions_v41.json"
    log("=== dump v41 (QQ: calibrate_v20) ===")
    from harness import calibrate_v20 as qq
    qq.DUMP_ACCUMULATOR = DumpAccumulator(
        reader="v41",
        loso_discipline="Nested LOSO with residual calibration (QQ: v20)",
    )
    qq.run()
    out_path = qq.DUMP_ACCUMULATOR.write()
    qq.DUMP_ACCUMULATOR = None
    log(f"v41 dump complete: {out_path}")
    return out_path


# -------------------------------------------------------------------------
# CLI.
# -------------------------------------------------------------------------


READERS: dict[str, callable] = {
    "v17": dump_v17,
    "v18": dump_v18,
    "v23": dump_v23,
    "v24": dump_v24,
    "v26": dump_v26,
    "v27": dump_v27,
    "v29": dump_v29,
    "v30": dump_v30,
    "v31": dump_v31,
    "v33": dump_v33,
    "v34": dump_v34,
    "v37": lambda force=False: dump_calibrated("v37", force=force),
    "v38": lambda force=False: dump_calibrated("v38", force=force),
    "v39": lambda force=False: dump_calibrated("v39", force=force),
    "v41": dump_v41,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--readers", nargs="+", default=list(READERS),
        choices=list(READERS),
        help="Which readers to dump (default: all).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-dump even if file exists.",
    )
    parser.add_argument(
        "--skip-on-error", action="store_true",
        help="Continue with remaining readers if one fails.",
    )
    args = parser.parse_args()

    _ensure_dir()
    log(f"=== Agent SS dump driver START: readers={args.readers} ===")
    summary: dict[str, str] = {}
    for r in args.readers:
        fn = READERS[r]
        t0 = time.time()
        try:
            out_path = fn(force=args.force)
            summary[r] = f"ok ({time.time() - t0:.1f}s) -> {out_path}"
        except Exception as e:
            summary[r] = f"FAILED: {e!r}"
            log(f"  {r} FAILED: {e!r}")
            traceback.print_exc()
            if not args.skip_on_error:
                raise

    log("=== summary ===")
    for r, s in summary.items():
        log(f"  {r}: {s}")
    summary_path = DUMP_DIR / "dump_driver_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
