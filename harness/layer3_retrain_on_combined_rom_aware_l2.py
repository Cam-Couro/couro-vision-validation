"""Layer 3 retrain on LL combined-cohort + ROM-aware Layer 2 (Agent LL → v24).

Sister script to:
  - ``harness.layer3_retrain_on_learned_l2``         (FF → v18, EE2 L2)
  - ``harness.layer3_retrain_on_rom_aware_l2``       (GG2 L3 wrapper → v20)
  - ``harness.layer3_retrain_on_combined_l2``        (KK → v23, HH2 L2)

Same Phase B pipeline as KK: train ONE Layer 2 model on the FULL combined
cohort (24 subjects, no LOSO at L2), then re-fit per-(metric × view) Layer 3
ridge with subject-level LOSO at L3 only.

# Pipeline
1. Train ONE LL combined+ROM-aware L2 on all 24 cohort subjects.
2. Cache to ``models/learned_layer2_combined_rom_aware_alldata_v1.pt``.
3. Monkey-patch FF's MODEL_OUT, output paths, and
   ``build_or_load_all_data_l2`` to load LL's model.
4. Reuse FF's ``run()`` (slot iteration, ridge fit, validity stats).
5. Tag outputs as v24.

# LOSO discipline (same caveat as v18/v20/v23)
L2 trained on ALL 24 subjects; L3 LOSO at subject level only. Not
double-LOSO. Tier promotions involving any cohort subject are upper bounds;
true double-LOSO would likely be ~0.05-0.10 |r| lower per HH2's per-fold
variance.

# What this produces
- ``models/learned_layer2_combined_rom_aware_alldata_v1.pt``
- ``data/layer3_retrain_combined_rom_aware/per_slot_validity_v24.json``
- ``results/deploy_ready_models_v24_combined_rom_aware.json``
- ``data/layer3_retrain_combined_rom_aware/REPORT.md``
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
    TemporalKeypointCNNConf,
)
from harness.learned_layer2_combined_rom_aware import train_all_data_ll


# -------------------------------------------------------------------------
# Paths.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"

OUT_DIR: Final[Path] = DATA_ROOT / "layer3_retrain_combined_rom_aware"
ALLDATA_MODEL_OUT: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_combined_rom_aware_alldata_v1.pt"
)
V17_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v17_selective.json"
)
V24_PATH: Final[Path] = (
    RESULTS_DIR / "deploy_ready_models_v24_combined_rom_aware.json"
)
PER_SLOT_V24_OUT: Final[Path] = OUT_DIR / "per_slot_validity_v24.json"


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    # print intentional: matches FF / KK progress contract.
    print(f"[LL-L3 t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Step 1: Train (or load) all-data combined + ROM-aware L2.
# -------------------------------------------------------------------------


def build_or_load_all_data_l2_combined_rom_aware(
    *,
    epochs: int = 25,
    clips_per_step: int = 4,
    lam: float = 1.0,
    drop_aspset_hipadd: bool = True,
    seed: int = 42,
    force_retrain: bool = False,
) -> tuple[TemporalKeypointCNNConf, dict]:
    """Train ONE LL-style L2 on ALL 24 cohort subjects. No LOSO at L2."""
    if ALLDATA_MODEL_OUT.exists() and not force_retrain:
        log(f"loading cached all-data LL L2 from {ALLDATA_MODEL_OUT}")
        ckpt = torch.load(ALLDATA_MODEL_OUT, weights_only=False)
        model = TemporalKeypointCNNConf()
        model.load_state_dict(ckpt["model_state"])
        model.y_mean = ckpt["y_mean"]  # type: ignore[attr-defined]
        model.y_std = ckpt["y_std"]    # type: ignore[attr-defined]
        return model, ckpt

    log("training all-data LL L2 (combined + ROM-aware, no LOSO at L2) ...")
    log(
        f"  epochs={epochs} clips_per_step={clips_per_step} lam={lam} "
        f"drop_aspset_hipadd={drop_aspset_hipadd}"
    )

    model, meta = train_all_data_ll(
        epochs=epochs,
        clips_per_step=clips_per_step,
        lam=lam,
        drop_aspset_hipadd=drop_aspset_hipadd,
        seed=seed,
    )

    ALLDATA_MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        },
        "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
        "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
        "n_keys": N_KEYS,
        "temporal_t": TEMPORAL_T,
        "halpe_keys": list(HALPE_KEYS_WITH_SMPL),
        "metrics": list(L2_METRICS),
        "model_class": "TemporalKeypointCNNConf",
        "training": "learned_layer2_combined_rom_aware_alldata_v1",
        "epochs": epochs,
        "clips_per_step": clips_per_step,
        "lam": lam,
        "drop_aspset_hipadd": drop_aspset_hipadd,
        "loso_discipline": "ALL_DATA_NO_LOSO_AT_L2",
        "cohort": {
            "n_subjects": meta["n_subjects"],
            "n_clips": meta["n_clips"],
            "subjects": meta["subjects"],
        },
    }, ALLDATA_MODEL_OUT)
    log(f"  saved all-data LL L2 -> {ALLDATA_MODEL_OUT}")

    return model, {
        "model_state": model.state_dict(),
        "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
        "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
    }


# -------------------------------------------------------------------------
# Step 2: Drive FF's run() with our model + output paths.
# -------------------------------------------------------------------------


def run(
    *,
    epochs: int = 25,
    clips_per_step: int = 4,
    lam: float = 1.0,
    drop_aspset_hipadd: bool = True,
    force_retrain_l2: bool = False,
) -> dict:
    """Drive FF's Layer 3 retrain pipeline with LL's combined+ROM-aware L2."""
    model, _ = build_or_load_all_data_l2_combined_rom_aware(
        epochs=epochs,
        clips_per_step=clips_per_step,
        lam=lam,
        drop_aspset_hipadd=drop_aspset_hipadd,
        force_retrain=force_retrain_l2,
    )
    log("LL L2 ready")

    from harness import layer3_retrain_on_learned_l2 as ff

    ff.MODEL_OUT = ALLDATA_MODEL_OUT
    ff.V18_PATH = V24_PATH
    ff.OUT_DIR = OUT_DIR

    def _bypass_build_or_load(
        *, epochs: int = 25, batch_size: int = 256, force_retrain: bool = False,
    ):
        log(
            "  [LL-L3 patch] bypassing FF.build_or_load_all_data_l2; "
            "returning pre-trained LL L2"
        )
        return model, {
            "model_state": model.state_dict(),
            "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
            "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
        }

    ff.build_or_load_all_data_l2 = _bypass_build_or_load

    log("delegating to FF.run() with LL L2 model patched in ...")
    per_slot_out = ff.run()

    # Rename FF's default per-slot output file (v18 -> v24).
    ff_default = OUT_DIR / "per_slot_validity_v18.json"
    if ff_default.exists() and ff_default != PER_SLOT_V24_OUT:
        try:
            content = json.loads(ff_default.read_text())
            content["version"] = "v24_combined_rom_aware"
            content["produced_by"] = (
                "harness.layer3_retrain_on_combined_rom_aware_l2 (Agent LL)"
            )
            content["caveat"] = (
                "L2 model trained on ALL 24 cohort subjects (9 OpenCap + 15 "
                "ASPset) with combined-cohort masked SmoothL1 loss + ROM-aware "
                "extrema-aware extra terms (lam=1.0). ASPset hip_adduction_r "
                "supervision was dropped per HH2's recommendation to avoid the "
                "convention-mismatch regression. L3 ridge LOSO at subject "
                "level only. Tier promotions involving any cohort subject are "
                "upper bounds; per HH2's per-fold variance the true "
                "double-LOSO number could be ~0.05-0.10 |r| lower."
            )
            PER_SLOT_V24_OUT.write_text(json.dumps(content, indent=2))
            log(f"copied + retagged {ff_default} -> {PER_SLOT_V24_OUT}")
        except Exception as e:
            log(f"  WARN: could not retag v18 -> v24: {e!r}")

    return per_slot_out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain-l2", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--clips-per-step", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--keep-aspset-hipadd", action="store_true")
    args = parser.parse_args()
    log("=== Agent LL Phase B START (v24 combined + ROM-aware L2) ===")
    if args.force_retrain_l2 and ALLDATA_MODEL_OUT.exists():
        ALLDATA_MODEL_OUT.unlink()
        log(f"removed cached L2 checkpoint at {ALLDATA_MODEL_OUT}")
    run(
        epochs=args.epochs,
        clips_per_step=args.clips_per_step,
        lam=args.lam,
        drop_aspset_hipadd=not args.keep_aspset_hipadd,
        force_retrain_l2=args.force_retrain_l2,
    )
    log("=== Agent LL Phase B DONE ===")


if __name__ == "__main__":
    main()
