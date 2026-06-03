"""Layer 3 retrain on MM-A per-source-heads + per-frame SmoothL1 Layer 2.

Sister of:
  - ``harness.layer3_retrain_on_learned_l2``           (FF → v18)
  - ``harness.layer3_retrain_on_rom_aware_l2``         (GG2 L3 wrapper → v20)
  - ``harness.layer3_retrain_on_combined_l2``          (KK → v23)
  - ``harness.layer3_retrain_on_combined_rom_aware_l2``(LL → v24)

Same Phase B pipeline: train ONE Layer 2 model on the FULL combined cohort
(24 subjects, no LOSO at L2), then re-fit per-(metric × view) Layer 3 ridge
with subject-level LOSO at L3 only. Tag outputs as **v26**.

# Pipeline
1. Train MM-A per-source-heads L2 on all 24 cohort subjects (per-frame
   SmoothL1 loss). Cache to
   ``models/learned_layer2_persource_perframe_alldata_v1.pt``.
2. Wrap with `_SharedHeadShim` so FF's runtime sees an HH2-compatible
   single-head model returning the SHARED (OpenCap-convention) outputs.
3. Monkey-patch FF's MODEL_OUT, output paths, and `build_or_load_all_data_l2`
   to load the wrapped MM-A model.
4. Reuse FF's `run()` (slot iteration, ridge fit, validity stats).
5. Tag outputs as v26.

# LOSO discipline (same caveat as v18/v20/v23/v24)
L2 trained on ALL 24 subjects; L3 LOSO at subject level only. Not
double-LOSO. Tier promotions are upper bounds.
"""
from __future__ import annotations

import builtins
import json
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np

for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.learned_layer2_combined import (
    DEPLOY_METRICS as L2_METRICS,
    HALPE_KEYS_WITH_SMPL,
    N_KEYS,
    TEMPORAL_T,
)
from harness.learned_layer2_combined_persource import (
    PER_SOURCE_IDX,
    TemporalKeypointCNNConfPerSource,
    train_all_data_mm_a,
    wrap_for_ff,
)


# -------------------------------------------------------------------------
# Paths.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"

OUT_DIR: Final[Path] = DATA_ROOT / "layer3_retrain_persource_perframe"
ALLDATA_MODEL_OUT: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_persource_perframe_alldata_v1.pt"
)
V17_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v17_selective.json"
V26_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v26_persource_perframe.json"
)
PER_SLOT_V26_OUT: Final[Path] = OUT_DIR / "per_slot_validity_v26.json"


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[MM-A-L3 t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Step 1: Train (or load) all-data MM-A L2.
# -------------------------------------------------------------------------


def build_or_load_all_data_l2_mm_a(
    *,
    epochs: int = 25,
    batch_size: int = 256,
    train_stride: int = 1,
    seed: int = 42,
    force_retrain: bool = False,
) -> tuple[TemporalKeypointCNNConfPerSource, dict]:
    """Train ONE MM-A L2 on ALL 24 cohort subjects. No LOSO at L2."""
    if ALLDATA_MODEL_OUT.exists() and not force_retrain:
        log(f"loading cached MM-A all-data L2 from {ALLDATA_MODEL_OUT}")
        ckpt = torch.load(ALLDATA_MODEL_OUT, weights_only=False)
        model = TemporalKeypointCNNConfPerSource()
        model.load_state_dict(ckpt["model_state"])
        model.ys_mean = ckpt["ys_mean"]  # type: ignore[attr-defined]
        model.ys_std = ckpt["ys_std"]    # type: ignore[attr-defined]
        model.ya_mean = ckpt["ya_mean"]  # type: ignore[attr-defined]
        model.ya_std = ckpt["ya_std"]    # type: ignore[attr-defined]
        return model, ckpt

    log("training all-data MM-A L2 (no LOSO at L2) ...")
    log(
        f"  epochs={epochs} batch_size={batch_size} "
        f"train_stride={train_stride}"
    )
    model, meta = train_all_data_mm_a(
        epochs=epochs, batch_size=batch_size,
        train_stride=train_stride, seed=seed,
    )

    ALLDATA_MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        },
        "ys_mean": np.asarray(model.ys_mean),  # type: ignore[attr-defined]
        "ys_std": np.asarray(model.ys_std),    # type: ignore[attr-defined]
        "ya_mean": np.asarray(model.ya_mean),  # type: ignore[attr-defined]
        "ya_std": np.asarray(model.ya_std),    # type: ignore[attr-defined]
        "n_keys": N_KEYS,
        "temporal_t": TEMPORAL_T,
        "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
        "metrics": list(L2_METRICS),
        "per_source_idx": list(PER_SOURCE_IDX),
        "model_class": "TemporalKeypointCNNConfPerSource",
        "training": "mm_a_persource_perframe_alldata_v1",
        "epochs": epochs,
        "batch_size": batch_size,
        "train_stride": train_stride,
        "loso_discipline": "ALL_DATA_NO_LOSO_AT_L2",
        "cohort": {
            "n_subjects": meta["n_subjects"],
            "n_clips": meta["n_clips"],
            "subjects": meta["subjects"],
        },
    }, ALLDATA_MODEL_OUT)
    log(f"  saved all-data MM-A L2 -> {ALLDATA_MODEL_OUT}")

    return model, {
        "ys_mean": np.asarray(model.ys_mean),  # type: ignore[attr-defined]
        "ys_std": np.asarray(model.ys_std),    # type: ignore[attr-defined]
    }


# -------------------------------------------------------------------------
# Step 2: Drive FF's run() with our shared-head shim.
# -------------------------------------------------------------------------


def run(
    *,
    epochs: int = 25,
    batch_size: int = 256,
    train_stride: int = 1,
    force_retrain_l2: bool = False,
) -> dict:
    base, _ = build_or_load_all_data_l2_mm_a(
        epochs=epochs, batch_size=batch_size,
        train_stride=train_stride,
        force_retrain=force_retrain_l2,
    )
    shim = wrap_for_ff(base)
    log("MM-A L2 ready (wrapped for FF compatibility)")

    from harness import layer3_retrain_on_learned_l2 as ff

    ff.MODEL_OUT = ALLDATA_MODEL_OUT
    ff.V18_PATH = V26_PATH
    ff.OUT_DIR = OUT_DIR

    def _bypass_build_or_load(
        *, epochs: int = 25, batch_size: int = 256,
        force_retrain: bool = False,
    ):
        log(
            "  [MM-A-L3 patch] bypassing FF.build_or_load_all_data_l2; "
            "returning pre-trained MM-A L2 (shared head)"
        )
        return shim, {
            "model_state": shim.state_dict(),
            "y_mean": shim.y_mean,
            "y_std": shim.y_std,
        }

    ff.build_or_load_all_data_l2 = _bypass_build_or_load

    log("delegating to FF.run() with MM-A L2 shim patched in ...")
    per_slot_out = ff.run()

    # Rename FF's default per-slot output file (v18 -> v26).
    ff_default = OUT_DIR / "per_slot_validity_v18.json"
    if ff_default.exists() and ff_default != PER_SLOT_V26_OUT:
        try:
            content = json.loads(ff_default.read_text())
            content["version"] = "v26_persource_perframe"
            content["produced_by"] = (
                "harness.layer3_retrain_on_persource_perframe_l2 (Agent MM)"
            )
            content["caveat"] = (
                "L2 model trained on ALL 24 cohort subjects (9 OpenCap + 15 "
                "ASPset) with per-source target heads (HH2's own recommended "
                "fix). Shared head (deployed) sees OpenCap convention for "
                "hip_adduction_r + lumbar_extension; ASPset head absorbs "
                "ASPset convention for those two metrics. Other metrics "
                "(hip_flexion_r, knee_angle_r, ankle_angle_r) use a shared "
                "head trained on both cohorts. Per-frame masked SmoothL1 "
                "loss (HH2 regime). L3 ridge LOSO at subject level only. "
                "Tier promotions involving any cohort subject are upper "
                "bounds; per HH2's per-fold variance the true double-LOSO "
                "number could be ~0.05-0.10 |r| lower."
            )
            PER_SLOT_V26_OUT.write_text(json.dumps(content, indent=2))
            log(f"copied + retagged {ff_default} -> {PER_SLOT_V26_OUT}")
        except Exception as e:
            log(f"  WARN: could not retag v18 -> v26: {e!r}")

    return per_slot_out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain-l2", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=1)
    args = parser.parse_args()
    log("=== Agent MM Phase B (v26 per-source per-frame) START ===")
    if args.force_retrain_l2 and ALLDATA_MODEL_OUT.exists():
        ALLDATA_MODEL_OUT.unlink()
        log(f"removed cached L2 checkpoint at {ALLDATA_MODEL_OUT}")
    run(
        epochs=args.epochs, batch_size=args.batch_size,
        train_stride=args.train_stride,
        force_retrain_l2=args.force_retrain_l2,
    )
    log("=== Agent MM Phase B (v26) DONE ===")


if __name__ == "__main__":
    main()
