"""Per-slot prediction dump helper for Agent SS ensemble experiment.

This module provides a single utility that any L3 reader script can call to
persist held-out (subject, pred, gt) tuples in a unified format on disk.

The dump JSON format is:

    {
      "version": "per_slot_predictions",
      "reader": "<reader_tag>",
      "loso_discipline": "<discipline description>",
      "aggregation": "per_subject",  # one row per (slot, fold/subject)
      "slots": [
        {
          "target": "hip_flexion_r",
          "view": "side_left",
          "approach": "<builder approach>",
          "fold_records": [
            {"subject": "opencap_subject2", "pred": 78.31, "gt": 80.05,
             "n_clips": 4},
            ...
          ]
        },
        ...
      ]
    }

Each fold_record corresponds to one held-out subject in the LOSO loop.
``pred`` and ``gt`` are subject-level MEANS over that subject's held-out
clips. This matches the per-subject aggregation used by the deploy reports
(see ``_compute_stats_per_subject`` in ``layer3_retrain_on_learned_l2``)
and is the right aggregation for honest ensembling: CCC and Bland-Altman
LoA in the deploy reports are computed at the subject level, so ensemble
weights must be too.

Per-clip (rather than per-subject) ensembling would require clip-level
alignment across readers, but readers do not share a common clip key when
the underlying builder differs (v17 hand-features vs v18 learned L2 vs
v20 ROM-aware L2 use different feature pipelines). Subject-level
ensembling is robust because every reader uses the same subject pool.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DUMP_DIR = REPO_ROOT / "data" / "per_slot_predictions"


def _per_subject_means(
    pred: np.ndarray, obs: np.ndarray, subjects: np.ndarray
) -> list[dict]:
    """Aggregate clip-level (pred, obs) into per-subject means.

    Identical aggregation to the deploy reports' ``_compute_stats_per_subject``
    helper. Returns a list of fold records (one per held-out subject).
    """
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    subjects = np.asarray(subjects)
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[mask]
    obs = obs[mask]
    subjects = subjects[mask]
    unique = sorted(set(subjects.tolist()))
    records = []
    for s in unique:
        sel = subjects == s
        n = int(sel.sum())
        if n == 0:
            continue
        records.append({
            "subject": str(s),
            "pred": float(np.mean(pred[sel])),
            "gt": float(np.mean(obs[sel])),
            "n_clips": n,
        })
    return records


class DumpAccumulator:
    """Accumulate per-slot fold records during a single retrain pass."""

    def __init__(
        self, *, reader: str, loso_discipline: str = "Layer-3-LOSO",
    ) -> None:
        self.reader = reader
        self.loso_discipline = loso_discipline
        self.slots: list[dict] = []

    def add_slot(
        self,
        *,
        target: str,
        view: str,
        approach: str,
        pred: np.ndarray,
        obs: np.ndarray,
        subjects: Iterable,
    ) -> None:
        subj_arr = np.asarray(list(subjects))
        fold_records = _per_subject_means(pred, obs, subj_arr)
        self.slots.append({
            "target": target,
            "view": view,
            "approach": approach,
            "fold_records": fold_records,
        })

    def add_slot_skip(self, *, target: str, view: str, reason: str) -> None:
        self.slots.append({
            "target": target,
            "view": view,
            "skipped": True,
            "skip_reason": reason,
            "fold_records": [],
        })

    def write(self, path: Path | None = None) -> Path:
        if path is None:
            DUMP_DIR.mkdir(parents=True, exist_ok=True)
            path = DUMP_DIR / f"per_slot_predictions_{self.reader}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "version": "per_slot_predictions_v1",
            "reader": self.reader,
            "loso_discipline": self.loso_discipline,
            "aggregation": "per_subject",
            "produced_by": "Agent SS (per_slot_prediction_dump)",
            "slots": self.slots,
        }
        path.write_text(json.dumps(out, indent=2))
        return path
