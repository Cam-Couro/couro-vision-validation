"""Write data/learned_layer2_combined_rom_aware/REPORT.md (LL Phase A).

This is the Layer 2 training narrative. The full 24-fold LOSO Phase A
eval is the canonical artifact but blows the LL brief's 3 h budget when
combined with v24 (Phase B) + v25 (Phase C) + reports + git push. Per the
brief's honest-reporting contract, this script writes a Phase A REPORT
that:

1. Documents the hybrid build (combined cohort + ROM-aware loss).
2. Reports the all-data training loss curve (Phase B trains an all-data L2;
   loss history is captured in the model checkpoint metadata if available).
3. Reports LOSO results IF `data/learned_layer2_combined_rom_aware/
   per_slot_results.json` exists (from a prior LOSO run); otherwise marks
   LOSO Phase A as "not run this build cycle".

It cleanly defers to v24 (Phase B) for the tier-promotion claims, since
that's what Cameron's brief cares about.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Final

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

OUT_DIR: Final[Path] = REPO_ROOT / "data" / "learned_layer2_combined_rom_aware"
LOSO_RESULTS: Final[Path] = OUT_DIR / "per_slot_results.json"
ALLDATA_MODEL: Final[Path] = (
    REPO_ROOT / "models" / "learned_layer2_combined_rom_aware_alldata_v1.pt"
)
HH2_REPORT: Final[Path] = (
    REPO_ROOT / "data" / "learned_layer2_combined" / "REPORT.md"
)
GG2_REPORT: Final[Path] = (
    REPO_ROOT / "data" / "rom_aware_layer2" / "REPORT.md"
)
OUT_PATH: Final[Path] = OUT_DIR / "REPORT.md"


def _fmt(v: object, prec: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def _load_loso() -> dict | None:
    if not LOSO_RESULTS.exists():
        return None
    try:
        d = json.loads(LOSO_RESULTS.read_text())
        if d.get("intermediate"):
            return None  # only use final results
        if "overall" not in d:
            return None
        return d
    except Exception:
        return None


def _load_ckpt_meta() -> dict | None:
    if not ALLDATA_MODEL.exists():
        return None
    try:
        ckpt = torch.load(ALLDATA_MODEL, weights_only=False, map_location="cpu")
        return {
            "training": ckpt.get("training"),
            "epochs": ckpt.get("epochs"),
            "clips_per_step": ckpt.get("clips_per_step"),
            "lam": ckpt.get("lam"),
            "drop_aspset_hipadd": ckpt.get("drop_aspset_hipadd"),
            "loso_discipline": ckpt.get("loso_discipline"),
            "cohort": ckpt.get("cohort"),
        }
    except Exception:
        return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loso = _load_loso()
    ckpt = _load_ckpt_meta()

    lines: list[str] = []
    lines.append(
        "# Combined OpenCap + ASPset + ROM-Aware Learned Layer 2 "
        "(Agent LL, Phase A)"
    )
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    lines.append("**Build:** Agent LL (hybrid follow-up to HH2 + GG2)")
    lines.append("")
    lines.append("## What this is")
    lines.append("")
    lines.append(
        "Layer 2 training that combines the two prior Phase B wins:"
    )
    lines.append("")
    lines.append(
        "- **HH2 (combined cohort)**: doubled training cohort from 9 "
        "OpenCap subjects (EE2) to 24 subjects (9 OpenCap + 15 ASPset, "
        "~1670 clips). Lifted OpenCap-held pooled |r| from 0.645 → 0.670 "
        "(+0.025); ASPset-held pooled |r| 0.677. v23 Phase B Layer 3 "
        "retrain produced 4 Good slots in the v22 selective oracle."
    )
    lines.append(
        "- **GG2 (ROM-aware loss)**: added an extrema-aware loss term "
        "`lam * |peak_pred - peak_gt| + lam * |min_pred - min_gt|` "
        "(lam=1.0) to push the model toward landing per-clip extrema at "
        "the correct amplitude. v20 Phase B Layer 3 retrain produced 4 "
        "Good slots, +5 promotions vs v17."
    )
    lines.append("")
    lines.append(
        "LL combines both: HH2's data pipeline (24 subjects, masked NaN "
        "for ankle on ASPset) + GG2's extrema-aware loss. The hypothesis "
        "is that the two improvements compound on slots where the failure "
        "mode is extrema accuracy. Combined cohort = better generalization. "
        "ROM-aware loss = tighter LoA at the L3 ridge."
    )
    lines.append("")

    lines.append("## Training recipe")
    lines.append("")
    lines.append(
        "- **Architecture**: `TemporalKeypointCNNConf` (same as EE2/HH2/"
        "GG2), 66 input channels, 100,741 params, T=9 frames."
    )
    lines.append(
        "- **Loss**: `SmoothL1(per_frame, masked) + lam * |peak_pred - "
        "peak_gt| + lam * |min_pred - min_gt|`, lam=1.0. Extrema computed "
        "per-(clip, metric) via differentiable `torch.amax`/`torch.amin`. "
        "Extrema and per-frame both respect the per-(clip, metric, frame) "
        "finite-target mask."
    )
    lines.append(
        "- **Masking**: ASPset has no foot KPs → ankle GT = NaN → masked. "
        "ASPset hip_adduction_r supervision is **dropped** by default per "
        "HH2's recommendation (asin-based frontal-plane definition does "
        "not match OpenSim's lumped-rotation hip_adduction; HH2 saw "
        "-0.051 |r| regression on OpenCap-held folds). CLI flag "
        "`--keep-aspset-hipadd` disables the drop."
    )
    lines.append(
        "- **Optimizer**: AdamW lr=1e-3, weight_decay=1e-4, cosine LR."
    )
    lines.append(
        "- **Step**: each step processes `clips_per_step=4` random clips. "
        "Per-clip full forward pass; extrema computed across the clip's "
        "center-frame trajectory."
    )
    lines.append(
        "- **CPU only**, 24-fold subject-level LOSO discipline at L2 "
        "(applied via `harness.learned_layer2_combined_rom_aware.main()`)."
    )
    lines.append("")

    lines.append("## Cohort")
    lines.append("")
    lines.append(
        "- **OpenCap**: 9 subjects, ~270 clips, all 5 angles. CC-BY 4.0."
    )
    lines.append(
        "- **ASPset**: 15 of 17 subjects ingested (2 had c3d/parse "
        "failures, same as HH2), ~1350 clips, 3 angles after the "
        "hip_adduction_r drop (no ankle, no hip_adduction). CC0."
    )
    lines.append(
        "- **Total**: 24 subjects, ~1620 clips, single-camera DWPose."
    )
    lines.append("")
    lines.append(
        "Convention alignment ASPset → OpenCap (carried from HH2):"
    )
    lines.append("")
    lines.append("- hip_flexion_r: identity")
    lines.append(
        "- hip_adduction_r: identity for OpenCap; dropped from ASPset "
        "supervision"
    )
    lines.append("- knee_angle_r: `OC = 180 − ASP`")
    lines.append("- ankle_angle_r: NaN (no foot KPs; masked from loss)")
    lines.append("- lumbar_extension: `OC = ASP − 180`")
    lines.append("")

    # ----- All-data checkpoint metadata -----
    lines.append("## All-data L2 checkpoint (used by v24 Phase B)")
    lines.append("")
    if ckpt:
        lines.append(
            f"- **Training**: `{ckpt.get('training')}`"
        )
        lines.append(
            f"- **Epochs**: {ckpt.get('epochs')}, "
            f"**clips_per_step**: {ckpt.get('clips_per_step')}, "
            f"**lam**: {ckpt.get('lam')}, "
            f"**drop_aspset_hipadd**: {ckpt.get('drop_aspset_hipadd')}"
        )
        if ckpt.get("cohort"):
            c = ckpt["cohort"]
            lines.append(
                f"- **Cohort**: {c.get('n_subjects')} subjects, "
                f"{c.get('n_clips')} clips"
            )
        lines.append(
            f"- **LOSO discipline**: `{ckpt.get('loso_discipline')}` "
            "(no LOSO at L2, used as the cached L2 model for v24 "
            "Phase B L3 ridge re-fit)"
        )
    else:
        lines.append("*(checkpoint metadata unavailable.)*")
    lines.append("")

    # ----- LOSO Phase A results (if available) -----
    lines.append("## Phase A LOSO evaluation (24-fold)")
    lines.append("")
    if loso:
        overall = loso["overall"]
        lines.append(
            f"**Overall LOSO pooled per-frame |r| = "
            f"{_fmt(overall.get('pooled_frame_abs_r_mean'))}**; "
            f"metric-mean ROM CCC = "
            f"{_fmt(overall.get('metric_mean_rom_ccc_mean'))}."
        )
        lines.append("")
        by_src = overall.get("by_source", {})
        if by_src:
            lines.append("### Per-source pooled |r|")
            lines.append("")
            lines.append(
                "| Source | n folds | pooled per-frame |r| |"
            )
            lines.append("| --- | ---: | ---: |")
            for src in ("opencap", "aspset"):
                d = by_src.get(src, {})
                lines.append(
                    f"| {src} | {d.get('n_folds', '-')} | "
                    f"{_fmt(d.get('pooled_frame_abs_r_mean'))} |"
                )
            lines.append("")
        per_metric = overall.get("per_metric", {})
        if per_metric:
            lines.append("### Per-metric across all 24 LOSO folds")
            lines.append("")
            lines.append(
                "| Metric | frame |r| mean | ROM CCC mean | ROM MAE mean "
                "(deg) | n folds |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for m in (
                "hip_flexion_r", "hip_adduction_r", "knee_angle_r",
                "ankle_angle_r", "lumbar_extension",
            ):
                d = per_metric.get(m, {})
                fr = d.get("frame_abs_r_mean", {})
                rc = d.get("rom_ccc", {})
                rm = d.get("rom_mae", {})
                lines.append(
                    f"| {m} | {_fmt(fr.get('mean'))} | "
                    f"{_fmt(rc.get('mean'))} | "
                    f"{_fmt(rm.get('mean'), prec=2)} | "
                    f"{fr.get('n_folds', '-')} |"
                )
            lines.append("")
    else:
        lines.append(
            "**Not run this build cycle.** Per the LL brief's 3 h budget, "
            "this build prioritized:"
        )
        lines.append("")
        lines.append(
            "1. All-data L2 training (no LOSO at L2) → v24 Phase B Layer 3 "
            "ridge re-fit → tier-promotion measurement."
        )
        lines.append(
            "2. v25 selective oracle build (Phase C) — Cameron's core "
            "question is the floor-lift status of the bottom-5 sub-0.80 "
            "slots, answered by the v25 oracle."
        )
        lines.append("")
        lines.append(
            "Phase A 24-fold LOSO eval is the natural follow-up build. "
            "Estimated cost: ~10 min/fold × 24 folds ≈ 4 h CPU. The harness "
            "is in place — run `python3 -m harness."
            "learned_layer2_combined_rom_aware --epochs 15 --lam 1.0` to "
            "produce the full Phase A artifact."
        )
        lines.append("")

    lines.append("## Comparison summary (vs HH2 and GG2)")
    lines.append("")
    lines.append(
        "| Build | Cohort | Loss | Phase A pooled |r| | Phase B Good slots |"
    )
    lines.append("| --- | --- | --- | --- | ---: |")
    lines.append(
        "| EE2 (v18) | 9 OpenCap | per-frame SmoothL1 | 0.645 | 2 |"
    )
    lines.append(
        "| GG2 (v20) | 9 OpenCap | per-frame + extrema (lam=1.0) | "
        "0.6313 | 4 |"
    )
    lines.append(
        "| HH2 (v23) | 9 OC + 15 ASP (24) | per-frame SmoothL1 (masked) | "
        "0.670 (OC-held) / 0.677 (ASP-held) | 4 |"
    )
    if loso:
        ov = loso["overall"]
        bs = ov.get("by_source", {})
        oc = bs.get("opencap", {}).get("pooled_frame_abs_r_mean")
        ap = bs.get("aspset", {}).get("pooled_frame_abs_r_mean")
        ll_phaseA = (
            f"{_fmt(oc)} (OC-held) / {_fmt(ap)} (ASP-held)"
        )
    else:
        ll_phaseA = "*(not measured this build cycle — see Phase A LOSO section)*"
    lines.append(
        f"| **LL (v24)** | 9 OC + 15 ASP (24) | per-frame + extrema "
        f"(lam=1.0), ASPset hip_adduction dropped | {ll_phaseA} | "
        "*(see Phase B REPORT)* |"
    )
    lines.append("")

    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "1. **24-fold LOSO Phase A eval is not in this build cycle.** "
        "Prioritized v24 / v25 deploy artifacts. The harness is in place "
        "and runs via `python3 -m harness.learned_layer2_combined_rom_aware`."
    )
    lines.append(
        "2. **ROM-aware loss + cross-dataset convention may interact.** "
        "GG2 added the extrema-aware loss on a clean single OpenCap cohort. "
        "LL adds it on the combined OpenCap+ASPset cohort. Where ASPset's "
        "convention noise exists (lumbar offset, hip_adduction definition), "
        "the extrema terms can pin predictions to noisy targets. We "
        "mitigated by dropping ASPset hip_adduction supervision. Lumbar "
        "extrema still inherit some risk."
    )
    lines.append(
        "3. **Hip_adduction_r ASPset drop is a coarse fix.** A cleaner fix "
        "is per-source target heads (separate output projection per "
        "dataset). Out of scope for LL build cycle."
    )
    lines.append(
        "4. **No invented numbers.** All metrics that appear in the "
        "comparison table are from the cited prior builds (HH2 / GG2 / "
        "EE2 REPORTs)."
    )
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append(
        "- `harness/learned_layer2_combined_rom_aware.py` — LL trainer "
        "(this build)"
    )
    lines.append(
        "- `models/learned_layer2_combined_rom_aware_alldata_v1.pt` — "
        "all-data LL L2 checkpoint (used by v24 Phase B)"
    )
    if loso:
        lines.append(
            "- `data/learned_layer2_combined_rom_aware/per_slot_results"
            ".json` — Phase A LOSO per-fold results"
        )
    lines.append(
        "- `data/layer3_retrain_combined_rom_aware/REPORT.md` — v24 "
        "Phase B narrative (tier promotions and CCC table)"
    )
    lines.append(
        "- `data/v25_selective_oracle/REPORT.md` — v25 selective oracle "
        "narrative (the floor-lift verdict)"
    )
    lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote LL Phase A REPORT -> {OUT_PATH}")


if __name__ == "__main__":
    main()
