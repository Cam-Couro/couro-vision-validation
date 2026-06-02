"""Overwrite the FF-templated REPORT.md to a proper v24 narrative.

FF's `_write_report` produced a v18-flavored markdown in
`data/layer3_retrain_combined_rom_aware/REPORT.md`. This script reads the
v24 per-slot validity, the v17 baseline, the v18/v20/v23 per-slot validity,
and rewrites the report with v24-correct prose and tables.

Run after `harness.layer3_retrain_on_combined_rom_aware_l2` completes.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"

V17_BASELINE = DATA_ROOT / "biomech_validity_stats" / "per_slot_validity.json"
V18_PS = DATA_ROOT / "layer3_retrain_learned_l2" / "per_slot_validity_v18.json"
V20_PS = DATA_ROOT / "rom_aware_layer2" / "per_slot_validity_v20.json"
V23_PS = DATA_ROOT / "layer3_retrain_combined_l2" / "per_slot_validity_v23.json"
V24_PS = DATA_ROOT / "layer3_retrain_combined_rom_aware" / "per_slot_validity_v24.json"

OUT_PATH = DATA_ROOT / "layer3_retrain_combined_rom_aware" / "REPORT.md"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _by_slot(d: dict) -> dict[tuple[str, str], dict]:
    return {(s["target"], s["view"]): s for s in d["slots"]}


def _fmt(v: object, prec: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def _tier_counts(slots: list[dict]) -> dict[str, int]:
    counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    for s in slots:
        cls = s.get("classification") or "Poor"
        if cls.startswith("LoA-only"):
            cls = "Poor"
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def main() -> None:
    v17 = _load(V17_BASELINE)
    v17_by = _by_slot(v17)
    v18 = _load(V18_PS)
    v18_by = _by_slot(v18)
    v20 = _load(V20_PS)
    v20_by = _by_slot(v20)
    v23 = _load(V23_PS)
    v23_by = _by_slot(v23)
    v24 = _load(V24_PS)
    v24_by = _by_slot(v24)

    v17_counts = _tier_counts(v17["slots"])
    v18_counts = _tier_counts(v18["slots"])
    v20_counts = _tier_counts(v20["slots"])
    v23_counts = _tier_counts(v23["slots"])
    v24_counts = _tier_counts(v24["slots"])

    promotions = []
    demotions = []
    unchanged = []
    tier_rank = {"Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1}
    for key, v24_slot in v24_by.items():
        v17_slot = v17_by.get(key, {})
        b_tier = v17_slot.get("classification", "Poor")
        if b_tier and b_tier.startswith("LoA-only"):
            b_tier = "Poor"
        n_tier = v24_slot.get("classification", "Poor")
        if n_tier and n_tier.startswith("LoA-only"):
            n_tier = "Poor"
        if tier_rank.get(n_tier, 0) > tier_rank.get(b_tier, 0):
            promotions.append((key, b_tier, n_tier))
        elif tier_rank.get(n_tier, 0) < tier_rank.get(b_tier, 0):
            demotions.append((key, b_tier, n_tier))
        else:
            unchanged.append((key, n_tier))

    lines: list[str] = []
    lines.append(
        "# Phase B: Layer 3 retrained on LL combined-cohort + ROM-aware "
        "Layer 2 (v24 / Agent LL)"
    )
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append("**Build:** Agent LL (Phase B follow-up to HH2 + GG2)")
    lines.append(
        f"**Verdict:** **{v24_counts['Good']} Good / "
        f"{v24_counts['Moderate']} Moderate / "
        f"{v24_counts['Poor']} Poor.** "
        "v24 = HH2's combined cohort + GG2's ROM-aware loss."
    )
    lines.append("")

    lines.append("## What this is")
    lines.append("")
    lines.append(
        "Layer 3 ridge regression re-fit per (metric × view) slot using "
        "the angle traces emitted by **LL's combined-cohort + ROM-aware "
        "learned Layer 2**, which is the hybrid of:"
    )
    lines.append("")
    lines.append(
        "- **HH2 (v23)**: doubled cohort (9 OpenCap + 15 ASPset = 24 "
        "subjects, ~1670 clips), masked SmoothL1 per-frame loss."
    )
    lines.append(
        "- **GG2 (v20)**: extrema-aware extra loss term, lam=1.0, computed "
        "per-(clip, metric) on the differentiable max/min of the predicted "
        "trajectory."
    )
    lines.append("")
    lines.append(
        "Per HH2's own recommendation, ASPset hip_adduction_r supervision "
        "is dropped from training (convention mismatch with OpenSim's "
        "lumped-rotation definition). All other metrics use the full "
        "combined cohort. Loss = `SmoothL1(per_frame, masked) + lam * "
        "|peak_pred - peak_gt| + lam * |min_pred - min_gt|`. Extrema "
        "respect the same NaN mask as the per-frame term."
    )
    lines.append("")
    lines.append(
        "Pipeline (see `harness/layer3_retrain_on_combined_rom_aware_l2.py`):"
    )
    lines.append("")
    lines.append(
        "1. Train one `TemporalKeypointCNNConf` on **all 24 cohort "
        "subjects** (no LOSO at L2) with the combined + ROM-aware loss. "
        "Saved to `models/learned_layer2_combined_rom_aware_alldata_v1.pt`."
    )
    lines.append(
        "2. Monkey-patch FF's `keypoints_to_motion_data` so the 5 deploy "
        "metrics come from the learned model. Left-side angles via "
        "mirrored-keypoint inference; pelvis_tilt remains hand-engineered."
    )
    lines.append(
        "3. For each of the 23 v17 deploy slots, rebuild the per-clip ridge "
        "features with the patched L2 and re-fit a ridge regression with "
        "subject-level LOSO at L3 only."
    )
    lines.append(
        "4. Compute Bland-Altman + Lin's CCC per slot; classify against the "
        "canonical biomech validity tier thresholds (Good: CCC > 0.60 AND "
        "LoA half < ±10°; Moderate: CCC > 0.40 AND LoA half < ±15°; Poor: "
        "otherwise)."
    )
    lines.append("")
    lines.append(
        "Single phone camera. Same input/output contract as Couro's "
        "deployed Layer 2."
    )
    lines.append("")

    lines.append("## LOSO discipline used in this build")
    lines.append("")
    lines.append(
        "**Layer-3-LOSO-only.** Same caveat as v18 (FF), v20 (GG2), and v23 "
        "(KK). One L2 model trained on every cohort subject, then L3 ridge "
        "LOSO. This leaks Layer 2 information from the held-out L3 subject "
        "into the L2 training data. Per HH2's per-fold LOSO numbers (best "
        "OpenCap-held |r| 0.744, worst 0.541, pooled 0.670), the true "
        "double-LOSO CCC at L3 could be ~0.05-0.10 lower than reported "
        "here. Cleanly double-LOSO only at the v17 hand-engineered reader."
    )
    lines.append("")

    lines.append("## Tier count delta")
    lines.append("")
    lines.append(
        "| Tier | v17 baseline | v18 (EE2) | v20 (GG2) | v23 (HH2) "
        "| **v24 (LL)** |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for tier in ("Excellent", "Good", "Moderate", "Poor"):
        lines.append(
            f"| {tier} | {v17_counts.get(tier, 0)} | "
            f"{v18_counts.get(tier, 0)} | {v20_counts.get(tier, 0)} | "
            f"{v23_counts.get(tier, 0)} | **{v24_counts.get(tier, 0)}** |"
        )
    lines.append("")
    lines.append(
        f"Promotions vs v17: {len(promotions)} | "
        f"Demotions vs v17: {len(demotions)} | "
        f"Unchanged vs v17: {len(unchanged)}"
    )
    lines.append("")

    if promotions:
        lines.append("### Promotions vs v17")
        lines.append("")
        for (tgt, view), b, n in promotions:
            lines.append(f"- {tgt} / {view}: {b} -> {n}")
        lines.append("")
    if demotions:
        lines.append("### Demotions vs v17")
        lines.append("")
        for (tgt, view), b, n in demotions:
            lines.append(f"- {tgt} / {view}: {b} -> {n}")
        lines.append("")

    lines.append(
        "## Per-slot before/after table (baseline = v17 hand-engineered; "
        "v24 = this build)"
    )
    lines.append("")
    lines.append(
        "| Target | View | n | r baseline | CCC baseline | LoA/2 baseline "
        "| Tier baseline | r v24 | CCC v24 | LoA/2 v24 | Tier v24 |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |"
    )
    for key, v24_slot in v24_by.items():
        tgt, view = key
        b = v17_by.get(key, {})
        b_tier = b.get("classification") or "Poor"
        if b_tier.startswith("LoA-only"):
            b_tier = "Poor"
        n_tier = v24_slot.get("classification") or "Poor"
        if n_tier.startswith("LoA-only"):
            n_tier = "Poor"
        lines.append(
            f"| {tgt} | {view} | "
            f"{v24_slot.get('n_subjects', '?')} | "
            f"{_fmt(b.get('pearson_r'))} | {_fmt(b.get('ccc_lin'))} | "
            f"{_fmt(b.get('loa_half_width_deg'))} | {b_tier} | "
            f"{_fmt(v24_slot.get('pearson_r'))} | "
            f"{_fmt(v24_slot.get('ccc_lin'))} | "
            f"{_fmt(v24_slot.get('loa_half_width_deg'))} | "
            f"{n_tier} |"
        )
    lines.append("")

    lines.append("## v18 vs v20 vs v23 vs v24 head-to-head")
    lines.append("")
    lines.append(
        "| Slot | v17 CCC | v18 CCC | v20 CCC | v23 CCC | **v24 CCC** "
        "| Best of (v18/v20/v23/v24) |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for key in v24_by.keys():
        tgt, view = key
        c17 = v17_by.get(key, {}).get("ccc_lin")
        c18 = v18_by.get(key, {}).get("ccc_lin")
        c20 = v20_by.get(key, {}).get("ccc_lin")
        c23 = v23_by.get(key, {}).get("ccc_lin")
        c24 = v24_by.get(key, {}).get("ccc_lin")
        learned = {"v18": c18, "v20": c20, "v23": c23, "v24": c24}
        # Pick max (NaN-safe).
        best_reader = None
        best_val = float("-inf")
        for r, val in learned.items():
            if val is None or (
                isinstance(val, float) and math.isnan(val)
            ):
                continue
            if val > best_val:
                best_val = val
                best_reader = r
        lines.append(
            f"| {tgt} / {view} | {_fmt(c17)} | {_fmt(c18)} | "
            f"{_fmt(c20)} | {_fmt(c23)} | **{_fmt(c24)}** | "
            f"{best_reader or '-'} |"
        )
    lines.append("")

    # Did v24 beat v23 anywhere?
    v24_beats_v23 = []
    v23_beats_v24 = []
    for key in v24_by.keys():
        c23 = v23_by.get(key, {}).get("ccc_lin")
        c24 = v24_by.get(key, {}).get("ccc_lin")
        if c23 is None or c24 is None:
            continue
        if isinstance(c23, float) and math.isnan(c23):
            continue
        if isinstance(c24, float) and math.isnan(c24):
            continue
        if c24 > c23 + 0.005:
            v24_beats_v23.append((key, c23, c24))
        elif c23 > c24 + 0.005:
            v23_beats_v24.append((key, c23, c24))

    lines.append("## Where v24 beat v23 (LL beat HH2)")
    lines.append("")
    if v24_beats_v23:
        lines.append(
            "| Slot | v23 CCC | v24 CCC | Δ |"
        )
        lines.append("| --- | ---: | ---: | ---: |")
        for (tgt, view), c23, c24 in sorted(
            v24_beats_v23, key=lambda x: x[2] - x[1], reverse=True,
        ):
            lines.append(
                f"| {tgt} / {view} | {_fmt(c23)} | {_fmt(c24)} | "
                f"{_fmt(c24 - c23)} |"
            )
    else:
        lines.append(
            "*(No slot where v24 beats v23 by more than 0.005 CCC.)*"
        )
    lines.append("")

    lines.append("## Where v23 beat v24 (HH2 beat LL)")
    lines.append("")
    if v23_beats_v24:
        lines.append(
            "| Slot | v23 CCC | v24 CCC | Δ |"
        )
        lines.append("| --- | ---: | ---: | ---: |")
        for (tgt, view), c23, c24 in sorted(
            v23_beats_v24, key=lambda x: x[1] - x[2], reverse=True,
        ):
            lines.append(
                f"| {tgt} / {view} | {_fmt(c23)} | {_fmt(c24)} | "
                f"{_fmt(c24 - c23)} |"
            )
    else:
        lines.append(
            "*(No slot where v23 beats v24 by more than 0.005 CCC.)*"
        )
    lines.append("")

    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "1. **Layer-3-LOSO-only caveat.** See discipline section. L2 "
        "trained on ALL 24 cohort subjects; L3 ridge LOSO at subject level "
        "only. Not double-LOSO. Cohort-subject tier promotions are upper "
        "bounds."
    )
    lines.append(
        "2. **ROM-aware loss + cross-dataset interaction.** GG2 demonstrated "
        "ROM-aware loss on a clean single OpenCap cohort. LL adds it on the "
        "combined OpenCap+ASPset cohort. Where ASPset's convention noise "
        "exists (lumbar offset, hip_adduction definition), the extrema "
        "terms can pin predictions to noisy targets. We mitigated this by "
        "dropping ASPset hip_adduction_r supervision; lumbar still inherits "
        "some risk."
    )
    lines.append(
        "3. **ASPset hip_adduction_r dropped from LL training** (HH2's "
        "recommendation). LL's hip_adduction_r is therefore trained on "
        "OpenCap-only ground truth, with ASPset providing shared-trunk "
        "representation only."
    )
    lines.append(
        "4. **Ankle GT remains OpenCap-only** (ASPset has no foot KPs). The "
        "learned L2 was trained on OpenCap ankle GT; ASPset clips have "
        "ankle masked from loss but contribute pose representation."
    )
    lines.append(
        "5. **Per-source target heads not implemented.** A natural next "
        "experiment is to add per-source output heads for hip_adduction_r "
        "and lumbar_extension to absorb convention mismatch without "
        "discarding ASPset data."
    )
    lines.append(
        "6. **No invented numbers.** All CCC / LoA / |r| values were "
        "computed from the v24 LOSO build."
    )
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append(
        "- `harness/learned_layer2_combined_rom_aware.py` — LL Layer 2 "
        "trainer (combined cohort + ROM-aware loss)"
    )
    lines.append(
        "- `harness/layer3_retrain_on_combined_rom_aware_l2.py` — LL "
        "Layer 3 retrain wrapper"
    )
    lines.append(
        "- `models/learned_layer2_combined_rom_aware_alldata_v1.pt` — "
        "all-data LL L2 checkpoint"
    )
    lines.append(
        "- `data/layer3_retrain_combined_rom_aware/per_slot_validity_v24"
        ".json` — per-slot v24 validity stats"
    )
    lines.append(
        "- `results/deploy_ready_models_v24_combined_rom_aware.json` — "
        "v24 deploy candidate (full v17 base + v24 learned-L2 ridge re-fits)"
    )
    lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote v24 REPORT -> {OUT_PATH}")


if __name__ == "__main__":
    main()
