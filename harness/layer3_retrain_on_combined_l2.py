"""Layer 3 retrain on combined-cohort learned Layer 2 (Agent HH/HH2 → v23).

Sister script to ``harness.layer3_retrain_on_learned_l2`` (Agent FF → v18) and
``harness.layer3_retrain_on_rom_aware_l2`` (Agent GG2 → v20). Same Phase B
pipeline: train ONE Layer 2 model on the FULL combined cohort
(9 OpenCap + 15 ASPset subjects = 24 subjects, no LOSO at L2), then re-fit
the per-(metric x view) Layer 3 ridge with subject-level LOSO at L3 only.

# Why a separate script
HH2 (``learned_layer2_combined.py``) trained 24 LOSO folds; the best-fold
checkpoint shipped at ``models/learned_layer2_combined_v1.pt`` has
opencap_subject8 held out and so is **not** L2-trained on 8 of the 9 OpenCap
subjects in the L3 LOSO pool. For the v18/v20 analog ("L2 on all, L3 LOSO")
we need an L2 trained on **every** cohort subject. This script does that
once and caches it.

# LOSO discipline (inherited from FF/v18, GG2/v20)
*L2 is trained on all 24 subjects.* L3 ridge is re-fit per slot with
subject-level LOSO at L3 only. This leaks L2 information from the held-out
L3 subject into the L2 training data. Same caveat as v18 and v20: tier
promotions involving OpenCap or ASPset subjects in the L3 fold are upper
bounds; the true double-LOSO number could be ~0.05-0.10 |r| lower
(HH2 LOSO eval: pooled OpenCap-held |r| 0.670, pooled ASPset-held |r| was
the lower side; per HH2 REPORT, OpenCap-held best fold 0.744).

# What this produces
- ``models/learned_layer2_combined_alldata_v1.pt`` — all-data combined L2 cache
- ``data/layer3_retrain_combined_l2/per_slot_validity_v23.json``
- ``results/deploy_ready_models_v23_combined_l2.json``
- ``data/layer3_retrain_combined_l2/REPORT.md``
"""
from __future__ import annotations

import builtins
import json
import logging
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np

# numpy 2.x compat for any chumpy-backed loads.
for _name in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, _name):
        setattr(np, _name, getattr(builtins, _name, None))

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

# Pull HH2's combined-cohort training infrastructure.
from harness.learned_layer2_combined import (
    DEPLOY_METRICS as L2_METRICS,
    HALPE_KEYS_WITH_SMPL,
    N_KEYS,
    TEMPORAL_T,
    TemporalKeypointCNNConf,
    build_combined_cohort,
    stack_clips,
    train_one_fold,
)


# -------------------------------------------------------------------------
# Paths and constants.
# -------------------------------------------------------------------------

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"

OUT_DIR: Final[Path] = DATA_ROOT / "layer3_retrain_combined_l2"
ALLDATA_MODEL_OUT: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_combined_alldata_v1.pt"
)
V17_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v17_selective.json"
V23_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v23_combined_l2.json"
PER_SLOT_V23_OUT: Final[Path] = OUT_DIR / "per_slot_validity_v23.json"


_T0 = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - _T0
    # NOTE: print here is intentional — matches FF / GG2 progress-logging
    # contract for the slot orchestration harness.
    print(f"[KK t={elapsed:6.1f}s] {msg}", flush=True)


# -------------------------------------------------------------------------
# Step 1: Train (or load) all-data combined L2 on 24 subjects.
# -------------------------------------------------------------------------


def build_or_load_all_data_l2_combined(
    *,
    epochs: int = 15,
    batch_size: int = 256,
    train_stride: int = 4,
    seed: int = 42,
    force_retrain: bool = False,
) -> tuple[TemporalKeypointCNNConf, dict]:
    """Train ONE HH-style Layer 2 model on ALL 24 cohort subjects.

    Mirrors HH2's ``train_one_fold`` exactly (same architecture, same masked
    SmoothL1 loss, same target normalization), but with no held-out subject.
    Saves the checkpoint in the same schema as HH2's per-fold checkpoint plus
    the ``loso_discipline`` flag used by FF/GG2.

    Defaults (epochs=15, train_stride=4) match HH2's main() CLI defaults so
    the resulting model is comparable to HH2's per-fold models.
    """
    if ALLDATA_MODEL_OUT.exists() and not force_retrain:
        log(f"loading cached all-data combined L2 from {ALLDATA_MODEL_OUT}")
        ckpt = torch.load(ALLDATA_MODEL_OUT, weights_only=False)
        model = TemporalKeypointCNNConf()
        model.load_state_dict(ckpt["model_state"])
        model.y_mean = ckpt["y_mean"]  # type: ignore[attr-defined]
        model.y_std = ckpt["y_std"]    # type: ignore[attr-defined]
        return model, ckpt

    log("training all-data combined L2 (no LOSO at L2) ...")
    log(f"  epochs={epochs} batch_size={batch_size} train_stride={train_stride}")

    cohort = build_combined_cohort()
    if not cohort:
        raise RuntimeError("combined cohort is empty; cannot train")

    all_clips = [c for clips in cohort.values() for c in clips]
    log(f"  total clips across {len(cohort)} subjects: {len(all_clips)}")

    X_train, Y_train = stack_clips(all_clips, stride=train_stride)
    log(f"  stacked training windows: X={X_train.shape} Y={Y_train.shape}")

    model, _hist = train_one_fold(
        X_train, Y_train,
        epochs=epochs, batch_size=batch_size, seed=seed,
        fold_label="ALL-DATA-COMBINED",
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
        "training": "learned_layer2_combined_alldata_v1",
        "epochs": epochs,
        "train_stride": train_stride,
        "loso_discipline": "ALL_DATA_NO_LOSO_AT_L2",
        "cohort": {
            "n_subjects": len(cohort),
            "subjects": sorted(cohort.keys()),
        },
    }, ALLDATA_MODEL_OUT)
    log(f"  saved all-data combined L2 -> {ALLDATA_MODEL_OUT}")

    return model, {
        "model_state": model.state_dict(),
        "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
        "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
    }


# -------------------------------------------------------------------------
# Step 2: Drive FF's run() with our model + output paths.
# -------------------------------------------------------------------------


def run() -> dict:
    """Drive FF's Layer 3 retrain pipeline with HH's combined-cohort L2."""
    model, _ = build_or_load_all_data_l2_combined()
    log("combined L2 ready")

    # Import FF and monkey-patch its key entry point + output paths.
    from harness import layer3_retrain_on_learned_l2 as ff

    ff.MODEL_OUT = ALLDATA_MODEL_OUT
    ff.V18_PATH = V23_PATH
    ff.OUT_DIR = OUT_DIR

    def _bypass_build_or_load(*, epochs: int = 25, batch_size: int = 256,
                              force_retrain: bool = False):
        log("  [KK patch] bypassing FF.build_or_load_all_data_l2; "
            "returning pre-trained combined L2")
        return model, {
            "model_state": model.state_dict(),
            "y_mean": np.asarray(model.y_mean),  # type: ignore[attr-defined]
            "y_std": np.asarray(model.y_std),    # type: ignore[attr-defined]
        }

    ff.build_or_load_all_data_l2 = _bypass_build_or_load

    log("delegating to FF.run() with combined L2 model patched in ...")
    per_slot_out = ff.run()

    # Rename FF's default per-slot output file (v18 -> v23) and tag the data.
    ff_default = OUT_DIR / "per_slot_validity_v18.json"
    if ff_default.exists() and ff_default != PER_SLOT_V23_OUT:
        try:
            content = json.loads(ff_default.read_text())
            content["version"] = "v23_combined_l2"
            content["produced_by"] = (
                "harness.layer3_retrain_on_combined_l2 (Agent KK)"
            )
            content["caveat"] = (
                "L2 model trained on ALL 24 cohort subjects (9 OpenCap + 15 "
                "ASPset). L3 LOSO at subject level only. Tier promotions "
                "involving any cohort subject are upper bounds; true "
                "double-LOSO would likely show ~0.05-0.10 |r| lower per HH2's "
                "per-fold variance. ASPset hip_adduction_r convention mismatch "
                "(see HH2 REPORT) causes regression on hip_adduction slots."
            )
            PER_SLOT_V23_OUT.write_text(json.dumps(content, indent=2))
            log(f"copied + retagged {ff_default} -> {PER_SLOT_V23_OUT}")
        except Exception as e:
            log(f"  WARN: could not retag v18 -> v23: {e!r}")

    return per_slot_out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain-l2", action="store_true")
    args = parser.parse_args()
    log("=== Agent KK Phase B START (v23 combined L2) ===")
    if args.force_retrain_l2 and ALLDATA_MODEL_OUT.exists():
        ALLDATA_MODEL_OUT.unlink()
        log(f"removed cached L2 checkpoint at {ALLDATA_MODEL_OUT}")
    run()
    log("=== Agent KK Phase B DONE ===")


if __name__ == "__main__":
    main()
