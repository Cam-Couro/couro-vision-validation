"""Layer 3 retrain on ROM-aware learned Layer 2 (Agent GG2).

Sister script to harness/layer3_retrain_on_learned_l2.py (Agent FF). Same
pipeline, same LOSO discipline (Layer-3-LOSO-only), but uses Agent GG2's
extrema-aware learned Layer 2 instead of EE2's per-frame learned Layer 2.

# Pipeline
1. Train ONE GG2-style ROM-aware Layer 2 on all 9 OpenCap subjects (no LOSO
   at L2). Cache to models/rom_aware_layer2_alldata_v1.pt.
2. Monkey-patch FF's MODEL_OUT and build_or_load_all_data_l2 to load that
   model.
3. Reuse FF's run() (slot iteration, ridge fit, validity stats, REPORT.md).
4. Write the validity stats and v20 deploy plan to data/rom_aware_layer2/.

# LOSO discipline
Same Layer-3-LOSO-only caveat as FF. Tier promotions involving OpenCap
subjects are upper bounds.

# Print discipline
Inherits FF's per-slot logging. GG2's training is also logged.

# Time
~50s extra for the all-data L2 training. Otherwise same cost as FF (~5-10
minutes per slot orchestration).
"""
from __future__ import annotations

import builtins
import json
import math
import sys
import time
import traceback
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

from harness.learned_layer2_real_gt import (
    TemporalKeypointCNNConf,
    HALPE_KEYS_WITH_SMPL,
    N_KEYS,
    TEMPORAL_T,
    DEPLOY_METRICS as L2_METRICS,
    build_clip_dataset,
)
from harness.smpl_layer2_poc import HALPE26_INDEX
from harness.rom_aware_layer2 import train_one_fold_romaware


# Paths.
DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
OPENCAP_KP: Final[Path] = DATA_ROOT / "opencap_dwpose_keypoints"

OUT_DIR: Final[Path] = DATA_ROOT / "rom_aware_layer2"
ALLDATA_MODEL_OUT: Final[Path] = (
    REPO_ROOT / "models" / "rom_aware_layer2_alldata_v1.pt"
)
V17_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v17_selective.json"
V20_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v20_rom_aware.json"
PER_SLOT_OUT: Final[Path] = OUT_DIR / "per_slot_validity_v20.json"
BASELINE_PATH: Final[Path] = (
    DATA_ROOT / "biomech_validity_stats" / "per_slot_validity.json"
)


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    print(f"[GG2-L3 t={elapsed:6.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Step 1: Train all-data GG2 ROM-aware Layer 2.
# ----------------------------------------------------------------------------


def build_or_load_all_data_l2_romaware(
    *,
    epochs: int = 12,
    clips_per_step: int = 4,
    lam: float = 1.0,
    force_retrain: bool = False,
) -> tuple[TemporalKeypointCNNConf, dict]:
    """Train ONE GG2-style ROM-aware Layer 2 on ALL 9 OpenCap subjects.

    No LOSO at L2. Used as the canonical L2 model for Layer 3 retraining.
    """
    if ALLDATA_MODEL_OUT.exists() and not force_retrain:
        log(f"loading cached all-data ROM-aware L2 from {ALLDATA_MODEL_OUT}")
        ckpt = torch.load(ALLDATA_MODEL_OUT, weights_only=False)
        model = TemporalKeypointCNNConf()
        model.load_state_dict(ckpt["model_state"])
        model.y_mean = ckpt["y_mean"]  # type: ignore[attr-defined]
        model.y_std = ckpt["y_std"]    # type: ignore[attr-defined]
        return model, ckpt

    log("training all-data GG2 L2 (no LOSO at L2) ...")
    halpe_idx_arr = np.array(
        [HALPE26_INDEX[name] for name in HALPE_KEYS_WITH_SMPL], dtype=np.int64
    )
    cohort: dict[str, list[dict]] = {}
    kp_files = sorted(OPENCAP_KP.glob("*.json"))
    log(f"  scanning {len(kp_files)} OpenCap DWPose JSONs ...")
    last_log = time.time()
    for i, kp_path in enumerate(kp_files):
        if (time.time() - last_log) > 15.0:
            log(f"  scanned {i+1}/{len(kp_files)} (kept_subjects={len(cohort)})")
            last_log = time.time()
        rec = build_clip_dataset(kp_path, halpe_idx_arr)
        if rec is None:
            continue
        cohort.setdefault(rec["subject"], []).append(rec)
    log(
        f"  cohort: {len(cohort)} subjects, "
        f"{sum(len(v) for v in cohort.values())} clips"
    )

    all_clips = [c for clips in cohort.values() for c in clips]
    model, _ = train_one_fold_romaware(
        all_clips,
        epochs=epochs,
        clips_per_step=clips_per_step,
        lam=lam,
        seed=42,
        fold_label="ALL-DATA-ROM-AWARE",
    )

    ALLDATA_MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
        "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
        "n_keys": N_KEYS,
        "temporal_t": TEMPORAL_T,
        "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
        "metrics": list(L2_METRICS),
        "model_class": "TemporalKeypointCNNConf",
        "training": "rom_aware_layer2_alldata_v1",
        "lam": lam,
        "loso_discipline": "ALL_DATA_NO_LOSO_AT_L2",
    }, ALLDATA_MODEL_OUT)
    log(f"  saved all-data ROM-aware L2 to {ALLDATA_MODEL_OUT}")
    return model, {
        "model_state": model.state_dict(),
        "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
        "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
    }


# ----------------------------------------------------------------------------
# Step 2: Monkey-patch FF's module to use GG2's L2 + new output paths.
# ----------------------------------------------------------------------------


def run() -> dict:
    """Drive FF's Layer 3 retrain with GG2's L2 model and new output paths."""
    # 1. Train/load GG2 all-data L2.
    model, _ = build_or_load_all_data_l2_romaware()
    log("GG2 L2 ready")

    # 2. Import FF and monkey-patch its key entry point + output paths.
    from harness import layer3_retrain_on_learned_l2 as ff

    ff.MODEL_OUT = ALLDATA_MODEL_OUT
    ff.V18_PATH = V20_PATH
    ff.OUT_DIR = OUT_DIR

    def _bypass_build_or_load(*, epochs=25, batch_size=256, force_retrain=False):
        log(
            "  [GG2 patch] bypassing FF.build_or_load_all_data_l2; "
            "returning pre-trained GG2 model"
        )
        return model, {
            "model_state": model.state_dict(),
            "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
            "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
        }

    ff.build_or_load_all_data_l2 = _bypass_build_or_load

    # Run FF's pipeline. It will:
    # - monkey-patch keypoints_to_motion_data
    # - iterate the 23-24 slots and re-fit ridge per slot
    # - compute validity stats and write outputs
    # FF's run() returns the per_slot_out dict.
    log("delegating to FF.run() with GG2 L2 model patched in ...")
    per_slot_out = ff.run()

    # 3. Rename FF's default output file (per_slot_validity_v18.json -> _v20).
    ff_default = OUT_DIR / "per_slot_validity_v18.json"
    if ff_default.exists() and ff_default != PER_SLOT_OUT:
        # Copy contents into our target path; keep the FF original around too
        # in case FF's REPORT.md references it.
        try:
            content = ff_default.read_text()
            PER_SLOT_OUT.write_text(content)
            log(f"copied {ff_default} -> {PER_SLOT_OUT}")
        except Exception as e:
            log(f"  WARN: could not copy v18 -> v20: {e!r}")

    return per_slot_out


def main() -> None:
    log("=== GG2 Layer 3 retrain START ===")
    run()
    log("=== GG2 Layer 3 retrain DONE ===")


if __name__ == "__main__":
    main()
