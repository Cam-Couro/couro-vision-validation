"""Run FF's layer3_retrain pipeline using the ROM-aware GG checkpoint.

We monkey-patch FF's MODEL_OUT to point at the GG all-data checkpoint, then
invoke FF's `run()`. Output paths are redirected to the GG output dir.

Pre-req: rom_aware_layer2_alldata_v1.pt must exist. We build it here if
missing by running the GG training on ALL 9 subjects (no LOSO).
"""
from __future__ import annotations

import builtins
import json
import math
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

from harness.rom_aware_layer2 import (
    DEPLOY_METRICS,
    TemporalKeypointCNNConf,
    build_cohort,
    train_one_fold,
)
from harness.learned_layer2_real_gt import N_KEYS, TEMPORAL_T, HALPE_KEYS_WITH_SMPL


DATA_ROOT: Final[Path] = REPO_ROOT / "data"
OUT_DIR: Final[Path] = DATA_ROOT / "rom_aware_layer2"
ALLDATA_MODEL_OUT: Final[Path] = REPO_ROOT / "models" / "rom_aware_layer2_alldata_v1.pt"


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[GG-L3 t={elapsed:6.1f}s] {msg}", flush=True)


def build_or_load_alldata_gg(
    *, epochs: int = 25, clip_batch: int = 16,
    loss_kind: str = "extrema", lambda_extrema: float = 0.5,
    force_retrain: bool = False,
) -> TemporalKeypointCNNConf:
    """Train ONE GG model on all 9 subjects (no LOSO at L2)."""
    if ALLDATA_MODEL_OUT.exists() and not force_retrain:
        log(f"loading existing all-data GG checkpoint from {ALLDATA_MODEL_OUT}")
        ckpt = torch.load(ALLDATA_MODEL_OUT, weights_only=False)
        model = TemporalKeypointCNNConf()
        model.load_state_dict(ckpt["model_state"])
        model.y_mean = ckpt["y_mean"]  # type: ignore[attr-defined]
        model.y_std = ckpt["y_std"]    # type: ignore[attr-defined]
        return model

    log("training all-data GG L2 ...")
    cohort = build_cohort()
    all_clips = [c for clips in cohort.values() for c in clips]
    log(f"all-data cohort: {len(all_clips)} clips across {len(cohort)} subjects")
    model, _ = train_one_fold(
        all_clips,
        epochs=epochs, clip_batch=clip_batch,
        seed=42, fold_label="ALL-DATA-GG",
        loss_kind=loss_kind, lambda_extrema=lambda_extrema,
    )
    ALLDATA_MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
        "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
        "n_keys": N_KEYS,
        "temporal_t": TEMPORAL_T,
        "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
        "metrics": list(DEPLOY_METRICS),
        "model_class": "TemporalKeypointCNNConf",
        "loss_kind": loss_kind,
        "lambda_extrema": lambda_extrema,
        "loso_discipline": "ALL_DATA_NO_LOSO_AT_L2",
    }, ALLDATA_MODEL_OUT)
    log(f"saved all-data GG checkpoint to {ALLDATA_MODEL_OUT}")
    return model


def run_l3_retrain_with_gg() -> dict:
    """Hijack FF's run() by swapping its MODEL_OUT to the GG checkpoint, then
    re-running the L3 ridge fit per slot."""
    # Pre-build GG all-data checkpoint.
    build_or_load_alldata_gg()

    # Now redirect FF's MODEL_OUT to point to ours.
    import harness.layer3_retrain_on_learned_l2 as ff
    ff.MODEL_OUT = ALLDATA_MODEL_OUT
    ff.OUT_DIR = OUT_DIR
    ff.TRACE_CACHE = OUT_DIR / "l2_traces_cache_gg.npz"
    log(f"redirected FF MODEL_OUT -> {ff.MODEL_OUT}")
    log(f"redirected FF OUT_DIR -> {ff.OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build_or_load_all_data_l2 in FF reads MODEL_OUT; we've already saved
    # there, so it will load our GG model.
    log("running FF's L3 retrain pipeline with GG model ...")
    return ff.run()


def main() -> None:
    log("=== GG-L3 retrain START ===")
    try:
        results = run_l3_retrain_with_gg()
    except Exception as e:
        log(f"FATAL: {e!r}")
        import traceback
        traceback.print_exc()
        raise

    # Save the slot validity in v19 format.
    if isinstance(results, dict):
        v19_path = OUT_DIR / "per_slot_validity_v19.json"
        # results may already have slots; write whatever FF returns.
        v19_path.write_text(json.dumps(results, indent=2, default=str))
        log(f"wrote v19 slot validity to {v19_path}")
    log("=== GG-L3 retrain DONE ===")


if __name__ == "__main__":
    main()
