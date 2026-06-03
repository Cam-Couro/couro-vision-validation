"""Layer 3 retrain on the mirror-flip per-source per-frame L2 (Agent NN -> v29).

Sister of `layer3_retrain_on_persource_perframe_l2.py` (v26). Replaces the
MM-A all-data L2 with the NN mirror-flip all-data L2 and routes outputs to
v29 paths.

Pipeline reuses FF's slot iteration / ridge fit / validity stats — just
with the v29 L2 shim monkey-patched in.
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

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from harness.learned_layer2_combined import (  # noqa: E402
    DEPLOY_METRICS as L2_METRICS,
    HALPE_KEYS_WITH_SMPL,
    N_KEYS,
    TEMPORAL_T,
)
from harness.learned_layer2_combined_persource import (  # noqa: E402
    PER_SOURCE_IDX,
    TemporalKeypointCNNConfPerSource,
    wrap_for_ff,
)


DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
OUT_DIR: Final[Path] = DATA_ROOT / "layer3_retrain_mirrorflip"
ALLDATA_MODEL_OUT: Final[Path] = (
    REPO_ROOT / "models"
    / "learned_layer2_persource_mirrorflip_alldata_v1.pt"
)
V29_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v29_mirrorflip.json"
)
PER_SLOT_V29_OUT: Final[Path] = OUT_DIR / "per_slot_validity_v29.json"


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[NN-L3-MF t={elapsed:6.1f}s] {msg}", flush=True)


def _load_alldata_l2():
    if not ALLDATA_MODEL_OUT.exists():
        raise RuntimeError(
            f"mirror-flip all-data L2 not found at {ALLDATA_MODEL_OUT}; "
            "run harness/learned_layer2_persource_mirrorflip.py first."
        )
    ckpt = torch.load(ALLDATA_MODEL_OUT, weights_only=False)
    base = TemporalKeypointCNNConfPerSource()
    base.load_state_dict(ckpt["model_state"])
    base.ys_mean = ckpt["ys_mean"]  # type: ignore[attr-defined]
    base.ys_std = ckpt["ys_std"]    # type: ignore[attr-defined]
    base.ya_mean = ckpt["ya_mean"]  # type: ignore[attr-defined]
    base.ya_std = ckpt["ya_std"]    # type: ignore[attr-defined]
    return wrap_for_ff(base)


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shim = _load_alldata_l2()
    log(f"loaded mirror-flip all-data L2 from {ALLDATA_MODEL_OUT}")

    from harness import layer3_retrain_on_learned_l2 as ff
    ff.MODEL_OUT = ALLDATA_MODEL_OUT
    ff.V18_PATH = V29_PATH
    ff.OUT_DIR = OUT_DIR

    def _bypass_build_or_load(
        *, epochs: int = 25, batch_size: int = 256,
        force_retrain: bool = False,
    ):
        log("  [NN-L3-MF] bypassing FF.build_or_load_all_data_l2")
        return shim, {
            "model_state": shim.state_dict(),
            "y_mean": shim.y_mean,
            "y_std": shim.y_std,
        }

    ff.build_or_load_all_data_l2 = _bypass_build_or_load

    log("delegating to FF.run() ...")
    per_slot_out = ff.run()

    # Retag default v18 -> v29.
    ff_default = OUT_DIR / "per_slot_validity_v18.json"
    if ff_default.exists() and ff_default != PER_SLOT_V29_OUT:
        try:
            content = json.loads(ff_default.read_text())
            content["version"] = "v29_mirrorflip"
            content["produced_by"] = (
                "harness.layer3_retrain_on_mirrorflip_l2 (Agent NN)"
            )
            content["caveat"] = (
                "L2 model trained on ALL 24 cohort subjects WITH "
                "mirror-flip augmentation (L/R label-swapped duplicates). "
                "Per-source heads inherited from MM-A. L3 ridge re-fit "
                "with LOSO at subject level only. Tier promotions "
                "involving any cohort subject are upper bounds."
            )
            PER_SLOT_V29_OUT.write_text(json.dumps(content, indent=2))
            log(f"retagged {ff_default} -> {PER_SLOT_V29_OUT}")
        except Exception as e:
            log(f"  WARN: could not retag v18 -> v29: {e!r}")

    # Move the v18 deploy alias to v29 explicitly (FF writes its file to
    # V18_PATH, which we already redirected to V29_PATH, but tag the
    # version).
    if V29_PATH.exists():
        try:
            content = json.loads(V29_PATH.read_text())
            content["version"] = "v29_mirrorflip"
            content["produced_by"] = (
                "harness.layer3_retrain_on_mirrorflip_l2 (Agent NN)"
            )
            V29_PATH.write_text(json.dumps(content, indent=2))
            log(f"retagged v29 deploy -> {V29_PATH}")
        except Exception as e:
            log(f"  WARN: could not retag {V29_PATH}: {e!r}")

    return per_slot_out


if __name__ == "__main__":
    log("=== Agent NN Phase 1B (v29 ridge L3) START ===")
    run()
    log("=== Agent NN Phase 1B (v29 ridge L3) DONE ===")
