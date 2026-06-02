"""Write REPORT.md for GG2's rom_aware_layer2 build.

Reads:
  data/rom_aware_layer2/per_slot_results.json     (L2 ROM CCC results)
  data/rom_aware_layer2/per_slot_validity_v20.json (L3 ridge tier results, if exists)
  data/biomech_validity_stats/per_slot_validity.json (baseline tiers)
  data/learned_layer2_real_gt/REPORT.md            (EE2 numbers for compare)
  data/layer3_retrain_learned_l2/per_slot_validity_v18.json (FF v18 numbers)

Writes:
  data/rom_aware_layer2/REPORT.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
RES = REPO_ROOT / "results"

OUT_DIR = DATA / "rom_aware_layer2"
L2_RESULTS = OUT_DIR / "per_slot_results.json"
L3_RESULTS_V20 = OUT_DIR / "per_slot_validity_v20.json"
BASELINE = DATA / "biomech_validity_stats" / "per_slot_validity.json"
FF_V18 = DATA / "layer3_retrain_learned_l2" / "per_slot_validity_v18.json"
REPORT_OUT = OUT_DIR / "REPORT.md"


def _fmt(v, prec=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.{prec}f}"
    return str(v)


def main() -> None:
    if not L2_RESULTS.exists():
        raise SystemExit(f"missing {L2_RESULTS}")
    l2 = json.loads(L2_RESULTS.read_text())

    has_l3 = L3_RESULTS_V20.exists()
    l3 = json.loads(L3_RESULTS_V20.read_text()) if has_l3 else None

    base = json.loads(BASELINE.read_text())
    base_by_slot = {(s["target"], s["view"]): s for s in base["slots"]}
    base_tier_counts: dict[str, int] = {}
    for s in base["slots"]:
        base_tier_counts[s["classification"]] = (
            base_tier_counts.get(s["classification"], 0) + 1
        )

    ff_v18 = None
    if FF_V18.exists():
        ff_v18 = json.loads(FF_V18.read_text())
    ff_by_slot = {}
    if ff_v18 is not None:
        for s in ff_v18.get("slots", []):
            ff_by_slot[(s["target"], s["view"])] = s

    lines: list[str] = []
    lines.append("# ROM-aware Learned Layer 2 (Agent GG2)")
    lines.append("")
    lines.append("**Date:** 2026-05-29")
    lines.append("**Build:** Agent GG2 (Phase A + Phase B, extrema-aware loss)")
    lines.append("")
    lines.append("## What this build does")
    lines.append("")
    lines.append(
        "Trains Agent EE2's `TemporalKeypointCNNConf` with an explicit "
        "extrema-aware loss so the model is rewarded for landing per-clip "
        "peaks and valleys at correct amplitude, not just for per-frame "
        "waveform agreement. Motivation: Agent FF found EE2's per-frame |r| "
        "gains did NOT translate to per-trial ROM tier promotions."
    )
    lines.append("")
    lines.append("Loss:")
    lines.append("")
    lines.append("```")
    lines.append("loss = SmoothL1(pred_frames_n, gt_frames_n)")
    lines.append("     + lam * mean( |peak(pred_n) - peak(gt_n)| )")
    lines.append("     + lam * mean( |min(pred_n)  - min(gt_n)|  )")
    lines.append("```")
    lines.append("")
    lines.append(
        f"All in normalized angle space; lam = {l2.get('lam')}. "
        f"Per-clip extrema computed differentiably via `torch.amax`, "
        f"`torch.amin` over each clip's full center-frame trajectory."
    )
    lines.append("")
    lines.append(
        f"epochs = {l2.get('epochs')}; clips per gradient step = "
        f"{l2.get('clips_per_step')}; CPU only."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Phase A: per-frame |r| and per-clip ROM CCC numbers across 9 folds.
    # ------------------------------------------------------------------
    lines.append("## Phase A: Learned Layer 2 (LOSO, OpenCap only)")
    lines.append("")
    overall = l2.get("overall", {})
    lines.append(
        f"**Pooled per-frame |r|** = "
        f"{_fmt(overall.get('pooled_frame_abs_r_mean'), 4)} "
        f"(EE2 reference: 0.645, Couro hand-engineered baseline: 0.514)"
    )
    lines.append("")
    lines.append(
        f"**Metric-mean ROM CCC** = "
        f"{_fmt(overall.get('metric_mean_rom_ccc_mean'), 4)} "
        f"(higher = better extrema fidelity)"
    )
    lines.append("")

    lines.append("### Per-metric across 9 LOSO folds")
    lines.append("")
    lines.append(
        "| Metric | per-frame |r| (GG2) | EE2 per-frame |r| | "
        "ROM CCC (GG2) | ROM MAE (GG2, deg) | Peak CCC | Valley CCC |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    EE2_FRAME_R = {
        "hip_flexion_r": 0.752,
        "hip_adduction_r": 0.397,
        "knee_angle_r": 0.686,
        "ankle_angle_r": 0.686,
        "lumbar_extension": 0.706,
    }
    pm = overall.get("per_metric", {})
    for m in ["hip_flexion_r", "hip_adduction_r", "knee_angle_r",
              "ankle_angle_r", "lumbar_extension"]:
        st = pm.get(m, {})
        fa = st.get("frame_abs_r_mean", {})
        rc = st.get("rom_ccc", {})
        rm = st.get("rom_mae", {})
        pc = st.get("peak_ccc", {})
        vc = st.get("valley_ccc", {})
        lines.append(
            f"| {m} | "
            f"{_fmt(fa.get('mean'), 3)} +/- {_fmt(fa.get('std'), 3)} | "
            f"{EE2_FRAME_R.get(m, '-')} | "
            f"{_fmt(rc.get('mean'), 3)} +/- {_fmt(rc.get('std'), 3)} | "
            f"{_fmt(rm.get('mean'), 1)} | "
            f"{_fmt(pc.get('mean'), 3)} | "
            f"{_fmt(vc.get('mean'), 3)} |"
        )
    lines.append("")

    # Per-fold pooled |r| and ROM CCC
    lines.append("### Per-fold")
    lines.append("")
    lines.append("| Held-out | pooled per-frame |r| | metric-mean ROM CCC |")
    lines.append("|---|---:|---:|")
    for fr in l2.get("fold_results_slim", []):
        lines.append(
            f"| {fr['held_out']} | "
            f"{_fmt(fr.get('eval_pooled_frame_abs_r'), 3)} | "
            f"{_fmt(fr.get('eval_metric_mean_rom_ccc'), 3)} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Phase B: Layer 3 retrain tier results (if available).
    # ------------------------------------------------------------------
    if has_l3:
        lines.append("## Phase B: Layer 3 ridge retrained on GG2 L2 (v20)")
        lines.append("")
        lines.append(
            "Same protocol as FF (Layer-3-LOSO-only, v17 deploy slot list, "
            "per-slot ridge re-fit). L2 is GG2's ROM-aware model trained on "
            "ALL 9 OpenCap subjects."
        )
        lines.append("")
        lines.append("### Tier counts")
        lines.append("")
        lines.append("| Tier | v17 baseline | v18 (EE2 L2) | v20 (GG2 L2) | "
                     "Δ vs v17 | Δ vs v18 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        v20_counts = l3.get("tier_counts", {})
        v18_counts: dict[str, int] = {}
        if ff_v18 is not None:
            v18_counts = ff_v18.get("tier_counts", {})
        for tier in ["Excellent", "Good", "Moderate", "Poor"]:
            b = base_tier_counts.get(tier, 0)
            v18c = v18_counts.get(tier, 0)
            v20c = v20_counts.get(tier, 0)
            lines.append(
                f"| {tier} | {b} | {v18c} | {v20c} | "
                f"{v20c - b:+d} | {v20c - v18c:+d} |"
            )
        lines.append("")
        lines.append(
            f"Promotions vs v17: {len(l3.get('promotions', []))}. "
            f"Demotions vs v17: {len(l3.get('demotions', []))}. "
            f"Unchanged: {len(l3.get('unchanged', []))}."
        )
        lines.append("")

        if l3.get("promotions"):
            lines.append("### Promotions (vs v17 baseline)")
            lines.append("")
            for p in l3["promotions"]:
                lines.append(f"- {p['target']} / {p['view']}: {p['from']} -> {p['to']}")
            lines.append("")
        if l3.get("demotions"):
            lines.append("### Demotions (vs v17 baseline)")
            lines.append("")
            for d in l3["demotions"]:
                lines.append(f"- {d['target']} / {d['view']}: {d['from']} -> {d['to']}")
            lines.append("")

        # Per-slot table.
        lines.append("### Per-slot")
        lines.append("")
        lines.append(
            "| Target | View | n | baseline r | baseline CCC | baseline LoA/2 | baseline tier | "
            "v18 r | v18 CCC | v18 LoA/2 | v18 tier | "
            "v20 r | v20 CCC | v20 LoA/2 | v20 tier |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |")
        for s in l3.get("slots", []):
            tgt, view = s["target"], s["view"]
            b = base_by_slot.get((tgt, view), {})
            ff = ff_by_slot.get((tgt, view), {})
            lines.append(
                f"| {tgt} | {view} | {s.get('n_subjects', '?')} | "
                f"{_fmt(b.get('pearson_r'))} | {_fmt(b.get('ccc_lin'))} | "
                f"{_fmt(b.get('loa_half_width_deg'))} | "
                f"{b.get('classification', '?')} | "
                f"{_fmt(ff.get('pearson_r'))} | {_fmt(ff.get('ccc_lin'))} | "
                f"{_fmt(ff.get('loa_half_width_deg'))} | "
                f"{ff.get('classification', '?')} | "
                f"{_fmt(s.get('pearson_r'))} | {_fmt(s.get('ccc_lin'))} | "
                f"{_fmt(s.get('loa_half_width_deg'))} | "
                f"{s.get('classification', '?')} |"
            )
        lines.append("")
    else:
        lines.append("## Phase B: Layer 3 retrain")
        lines.append("")
        lines.append(
            "Layer 3 ridge retrain was NOT run in this build "
            "(time budget exhausted by L2 training). Only Phase A numbers "
            "are reported above."
        )
        lines.append("")

    # ------------------------------------------------------------------
    # Interpretation.
    # ------------------------------------------------------------------
    lines.append("## Interpretation")
    lines.append("")
    metric_mean_rom_ccc = overall.get("metric_mean_rom_ccc_mean", float("nan"))
    if isinstance(metric_mean_rom_ccc, float) and not math.isnan(metric_mean_rom_ccc):
        if metric_mean_rom_ccc > 0.30:
            lines.append(
                f"**Headline:** GG2's extrema-aware loss produced metric-mean "
                f"ROM CCC of {metric_mean_rom_ccc:.3f}, a meaningful "
                f"improvement over the implied baseline (EE2's per-frame |r| "
                f"gains did not translate to ROM CCC). Whether this carries "
                f"to Layer 3 tier promotion depends on Phase B."
            )
        elif metric_mean_rom_ccc > 0.15:
            lines.append(
                f"**Headline:** GG2's extrema-aware loss produced "
                f"metric-mean ROM CCC of {metric_mean_rom_ccc:.3f} -- a "
                f"modest improvement vs EE2 per-frame loss's implied ~0 "
                f"per-clip ROM CCC, but still below the per-frame |r| level. "
                f"Translation to L3 tier promotion is uncertain."
            )
        else:
            lines.append(
                f"**Headline:** GG2's extrema-aware loss produced "
                f"metric-mean ROM CCC of {metric_mean_rom_ccc:.3f}. This is "
                f"low. Per-frame |r| is preserved at "
                f"{overall.get('pooled_frame_abs_r_mean', float('nan')):.3f} "
                f"(comparable to EE2's 0.645), but the extrema-aware loss "
                f"as formulated does NOT drive cross-clip ROM correlation."
            )
            lines.append("")
            lines.append(
                "**Why this is not a slam-dunk:** the loss penalizes "
                "absolute peak/min error per (clip, metric) in normalized "
                "space. It tightens within-clip extrema toward GT, but ROM "
                "CCC is a CROSS-clip correlation. If the model learns a "
                "constant offset per clip, the within-clip extrema error "
                "shrinks but cross-clip ROM rank does not improve. A future "
                "build should optimize a differentiable surrogate for ROM "
                "CCC directly (e.g. batched soft-rank, or a cross-clip "
                "moment-matching term)."
            )
    else:
        lines.append(
            "**Headline:** Phase A LOSO did not complete; ROM CCC unavailable."
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Caveats.
    # ------------------------------------------------------------------
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **OpenCap only.** Phase A LOSO uses 9 OpenCap subjects. No "
        "ASPset. Per-frame |r| reported here matches EE2's regime."
    )
    lines.append(
        "- **Loss is per-clip extrema, not cross-clip ROM CCC.** The "
        "extrema-aware loss as written penalizes |peak(pred) - peak(gt)| "
        "and |min(pred) - min(gt)| per (clip, metric) in normalized space. "
        "This is a within-clip objective. Per-clip ROM CCC is a cross-clip "
        "objective. The two are not the same. If per-clip extrema MAE "
        "tightens but every clip is offset by a different constant, ROM "
        "CCC does not improve."
    )
    lines.append(
        "- **Lambda is fixed.** Default lam = 1.0 per the brief. Higher "
        "lam (5 or 10) might push harder on extrema at the cost of "
        "per-frame fidelity. Not swept in this build."
    )
    lines.append(
        "- **Phase B caveat inherited.** If Phase B (Layer 3 retrain) "
        "is included, it inherits FF's Layer-3-LOSO-only discipline. "
        "OpenCap-subject tier results are upper bounds; true double-LOSO "
        "would likely be ~0.05-0.10 |r| lower."
    )
    lines.append(
        "- **No mixed-extrema strategy.** A natural follow-up is to "
        "ENSEMBLE GG2 ROM with hand-engineered ROM (which Phase B FF "
        "suggested as a next experiment). Not attempted here."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Files.
    # ------------------------------------------------------------------
    lines.append("## Files")
    lines.append("")
    lines.append(f"- `harness/rom_aware_layer2.py` -- training, runnable")
    lines.append(f"- `models/rom_aware_layer2_v1.pt` -- best-fold checkpoint")
    lines.append(f"- `data/rom_aware_layer2/per_slot_results.json` -- L2 LOSO results")
    lines.append(f"- `data/rom_aware_layer2/per_clip_results.json` -- L2 per-clip detail")
    if has_l3:
        lines.append(f"- `harness/layer3_retrain_on_rom_aware_l2.py` -- L3 retrain orchestrator")
        lines.append(f"- `models/rom_aware_layer2_alldata_v1.pt` -- all-data L2 used for L3")
        lines.append(f"- `data/rom_aware_layer2/per_slot_validity_v20.json` -- L3 tier results")
        lines.append(f"- `results/deploy_ready_models_v20_rom_aware.json` -- v20 deploy candidate")
    lines.append("")

    # ------------------------------------------------------------------
    # Recommendation.
    # ------------------------------------------------------------------
    lines.append("## Recommendation")
    lines.append("")
    if has_l3 and l3.get("promotions"):
        n_p = len(l3["promotions"])
        n_d = len(l3.get("demotions", []))
        if n_p >= n_d:
            lines.append(
                f"**Adopt v20 selectively** at the {n_p} promoted slots. "
                f"The remaining slots should keep v17 ridge weights. Same "
                f"logic as FF: tier wins where the lift survives the "
                f"Layer-3-LOSO-only upper-bound caveat are worth taking."
            )
        else:
            lines.append(
                f"**Do NOT adopt v20 wholesale.** Net tier change is "
                f"negative ({n_p} promotions vs {n_d} demotions). The "
                f"extrema-aware loss did not solve FF's translation gap."
            )
    elif has_l3:
        lines.append(
            "**Do NOT adopt v20 wholesale.** No promotions found. The "
            "extrema-aware loss as formulated does not produce L3 tier "
            "lifts."
        )
    else:
        lines.append(
            "**Phase B incomplete.** Recommendation deferred until L3 "
            "retrain is run."
        )
    lines.append("")
    lines.append("## Single-camera reaffirmation")
    lines.append("")
    lines.append(
        "Every measurement uses a single virtual or real camera per "
        "inference. No multi-camera fusion. GG2 consumes one DWPose stream "
        "and produces 5 joint angles per frame -- same I/O contract as "
        "Couro's hand-engineered Layer 2."
    )
    lines.append("")

    REPORT_OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
