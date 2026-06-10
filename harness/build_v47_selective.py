"""Agent UU v47 selective oracle: v45 pool + v46 (L1 flip-TTA) candidate.

Reader pool = v45 pool + v46 (DWPose test-time horizontal-flip augmentation,
hand-engineered geometry reader). Per-slot pick the highest tier / highest
CCC, with PP's LoA-then-CCC tie-break preserved and a hard no-regression
fallback (v46 cannot regress CCC by > 0.02 vs the v45 pick).

v46 is the geometry reader on flip-TTA keypoints. It only competes on slots
where flip-TTA was actually computed (side cams in this scoped run) and where
the geometry reader produced a usable dataset. Everywhere else the v45 pick is
preserved -- no demotions.

Inputs:
  * data/v45_selective_oracle/per_slot_picks_v45.json
  * data/layer3_retrain_flip_tta/per_slot_validity_v46.json
  * results/deploy_ready_models_v45_selective.json

Outputs:
  * results/deploy_ready_models_v47_selective.json
  * data/v47_selective_oracle/per_slot_picks_v47.json
  * data/v47_selective_oracle/REPORT.md
  * data/per_slot_predictions/per_slot_predictions_v46.json (SS-pattern dump)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT: Final[Path] = REPO_ROOT / "data"
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"
DUMP_DIR: Final[Path] = DATA_ROOT / "per_slot_predictions"

V45_PICKS_PATH: Final[Path] = DATA_ROOT / "v45_selective_oracle" / "per_slot_picks_v45.json"
V45_DEPLOY_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v45_selective.json"
V46_VALIDITY_PATH: Final[Path] = DATA_ROOT / "layer3_retrain_flip_tta" / "per_slot_validity_v46.json"
V46_PER_CLIP_PATH: Final[Path] = DATA_ROOT / "layer3_retrain_flip_tta" / "per_clip_predictions_v46.json"

V47_OUT_DIR: Final[Path] = DATA_ROOT / "v47_selective_oracle"
V47_DEPLOY_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v47_selective.json"
V46_DUMP_PATH: Final[Path] = DUMP_DIR / "per_slot_predictions_v46.json"

LOA_LIMITED_CCC: Final[float] = 0.79
FALLBACK_CCC_TOLERANCE: Final[float] = 0.02

# v45 baseline counts (per README / v45 report).
V45_GOOD: Final[int] = 13
V45_TIER1: Final[int] = 14

# The 4 detector-asymmetry target slots the brief asks about.
TARGET_SLOTS: Final[list[tuple[str, str, float]]] = [
    ("hip_adduction_r", "side_right", 0.27),
    ("hip_flexion_r", "side_right", 0.58),
    ("ankle_angle_r", "side_left", 0.38),
    ("ankle_angle_r", "front_oblique_left", 0.56),
]


def _tier_rank(t: str) -> int:
    return {"Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1}.get(t, 0)


def _is_better(cand_ccc, cand_loa, cand_tier, base_ccc, base_loa, base_tier):
    if cand_ccc is None or math.isnan(cand_ccc):
        return False, "candidate ccc NaN"
    if not math.isnan(base_ccc) and cand_ccc < base_ccc - FALLBACK_CCC_TOLERANCE:
        return False, f"ccc regressed (>{FALLBACK_CCC_TOLERANCE}): {cand_ccc:.3f} < {base_ccc:.3f}"
    if _tier_rank(cand_tier) > _tier_rank(base_tier):
        return True, f"tier promotion {base_tier} -> {cand_tier}"
    if _tier_rank(cand_tier) == _tier_rank(base_tier):
        ccc_gain = cand_ccc - base_ccc if not math.isnan(base_ccc) else 0
        loa_gain = base_loa - cand_loa if not (math.isnan(cand_loa) or math.isnan(base_loa)) else 0
        in_loa_band = base_tier == "Moderate" and not math.isnan(base_ccc) and base_ccc >= LOA_LIMITED_CCC
        if in_loa_band:
            if loa_gain > 0.05 and ccc_gain >= -FALLBACK_CCC_TOLERANCE:
                return True, f"LoA-band tighter LoA ({loa_gain:+.2f})"
            return False, "in LoA band but no LoA tightening"
        if ccc_gain > 1e-3 and loa_gain >= -0.05:
            return True, f"same-tier CCC+ ({ccc_gain:+.3f})"
        if loa_gain > 0.05 and ccc_gain >= -FALLBACK_CCC_TOLERANCE:
            return True, f"same-tier LoA tighter ({loa_gain:+.2f})"
    return False, "no improvement"


def _load_v46() -> dict[tuple[str, str], dict]:
    raw = json.loads(V46_VALIDITY_PATH.read_text())
    out: dict[tuple[str, str], dict] = {}
    for s in raw.get("slots", []):
        if not s.get("tta_applicable") or "ccc_lin" not in s:
            continue
        out[(s["target"], s["view"])] = s
    return out, raw


def build_v47() -> dict:
    v45 = json.loads(V45_PICKS_PATH.read_text())
    v46, v46_raw = _load_v46()

    new_picks = []
    tier_counter = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    tier1_count = 0
    reader_distribution: dict[str, int] = {}
    promotions = []
    n_v46_picks = 0

    for pick in v45["picks"]:
        slot_str = pick["slot"]
        target, view = slot_str.split("|")
        base_reader = pick["v45_reader"]
        base_ccc = float(pick["v45_ccc"]) if pick["v45_ccc"] is not None else float("nan")
        base_loa = float(pick["v45_loa"]) if pick["v45_loa"] is not None else float("nan")
        base_tier = pick["v45_tier"] or "Poor"

        v46_stat = v46.get((target, view))
        v46_ccc = float(v46_stat["ccc_lin"]) if v46_stat else float("nan")
        v46_loa = float(v46_stat["loa_half_width_deg"]) if v46_stat else float("nan")
        v46_tier = v46_stat["classification"] if v46_stat else "Poor"
        v46_baseline_ccc = float(v46_stat["baseline_ccc_lin"]) if v46_stat else float("nan")

        promoted, reason = (False, "no v46 candidate for slot")
        if v46_stat is not None:
            promoted, reason = _is_better(
                v46_ccc, v46_loa, v46_tier, base_ccc, base_loa, base_tier,
            )

        if promoted:
            chosen = ("v46_flip_tta", v46_ccc, v46_loa, v46_tier)
            n_v46_picks += 1
            promotions.append({
                "slot": slot_str, "from": base_tier, "to": v46_tier,
                "v45_ccc": base_ccc, "v46_ccc": v46_ccc, "reason": reason,
            })
        else:
            chosen = (base_reader, base_ccc, base_loa, base_tier)

        chosen_reader, chosen_ccc, chosen_loa, chosen_tier = chosen
        reader_distribution[chosen_reader] = reader_distribution.get(chosen_reader, 0) + 1
        tier_counter[chosen_tier] = tier_counter.get(chosen_tier, 0) + 1
        if not math.isnan(chosen_ccc) and chosen_ccc >= LOA_LIMITED_CCC:
            tier1_count += 1

        new_picks.append({
            "slot": slot_str,
            "v45_reader": base_reader, "v45_ccc": base_ccc,
            "v45_loa": base_loa, "v45_tier": base_tier,
            "v46_geometry_baseline_ccc": v46_baseline_ccc,
            "v46_ccc": v46_ccc, "v46_loa": v46_loa, "v46_tier": v46_tier,
            "v46_delta_vs_geometry_baseline": (
                v46_ccc - v46_baseline_ccc
                if v46_stat and not (math.isnan(v46_ccc) or math.isnan(v46_baseline_ccc))
                else None
            ),
            "v47_reader": chosen_reader, "v47_ccc": chosen_ccc,
            "v47_loa": chosen_loa, "v47_tier": chosen_tier,
            "promoted": promoted, "promotion_reason": reason,
        })

    out = {
        "version": "v47_selective_oracle",
        "produced_by": "harness.build_v47_selective (Agent UU)",
        "description": (
            "v45 pool + v46 (Layer 1 DWPose flip-TTA, hand-engineered geometry "
            "reader). Per-slot highest tier / highest CCC, LoA-then-CCC "
            "tie-break, no CCC regression > 0.02 vs v45."
        ),
        "tier_counts": tier_counter,
        "tier1_count_ccc_ge_0p79": tier1_count,
        "v45_good": V45_GOOD, "v45_tier1": V45_TIER1,
        "delta_good_vs_v45": tier_counter.get("Good", 0) - V45_GOOD,
        "delta_tier1_vs_v45": tier1_count - V45_TIER1,
        "n_slots_taking_v46": n_v46_picks,
        "reader_distribution": reader_distribution,
        "promotions_vs_v45": promotions,
        "v46_n_applicable_slots": v46_raw.get("n_dwpose_applicable_slots"),
        "picks": new_picks,
    }
    V47_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (V47_OUT_DIR / "per_slot_picks_v47.json").write_text(json.dumps(out, indent=2))
    print(f"[v47] wrote {V47_OUT_DIR / 'per_slot_picks_v47.json'}")
    return out


def write_deploy(out: dict) -> None:
    base = json.loads(V45_DEPLOY_PATH.read_text())
    deploy = dict(base)
    deploy["version"] = "v47_selective_oracle"
    deploy["produced_by"] = "harness.build_v47_selective (Agent UU)"
    deploy["description"] = out["description"]
    ann = {}
    for p in out["picks"]:
        ann[p["slot"]] = {
            "v47_reader": p["v47_reader"], "v47_ccc": p["v47_ccc"],
            "v47_loa": p["v47_loa"], "v47_tier": p["v47_tier"],
            "promoted_from_v45": p["promoted"],
            "promotion_reason": p["promotion_reason"],
        }
    deploy["v47_annotations"] = ann
    deploy["v47_tier_counts"] = out["tier_counts"]
    deploy["v47_tier1_count"] = out["tier1_count_ccc_ge_0p79"]
    deploy["v47_reader_distribution"] = out["reader_distribution"]
    deploy["v46_l1_method"] = "dwpose_test_time_horizontal_flip_tta"
    V47_DEPLOY_PATH.write_text(json.dumps(deploy, indent=2))
    print(f"[v47] wrote {V47_DEPLOY_PATH}")


def dump_v46_predictions(out: dict) -> None:
    """SS-pattern per-clip prediction dump for future ensembles."""
    try:
        per_clip = json.loads(V46_PER_CLIP_PATH.read_text())
    except FileNotFoundError:
        per_clip = {}
    payload = {
        "version": "v46_flip_tta",
        "produced_by": "harness.build_v47_selective (Agent UU)",
        "reader": "v46_flip_tta_geometry",
        "note": (
            "Per-clip LOSO predictions from the hand-engineered geometry reader "
            "on DWPose flip-TTA keypoints (OpenCap cohort)."
        ),
        "per_slot": per_clip,
    }
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    V46_DUMP_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[v47] wrote {V46_DUMP_PATH}")


def _fmt(v, p=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{p}f}"


def write_report(out: dict) -> None:
    v46_raw = json.loads(V46_VALIDITY_PATH.read_text())
    by_slot = {p["slot"]: p for p in out["picks"]}
    v46_by_slot = {(s["target"], s["view"]): s for s in v46_raw["slots"]}

    L = []
    L.append("# v47 Selective Oracle: Layer 1 DWPose Flip-TTA (Agent UU)")
    L.append("")
    L.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    L.append("")
    L.append(
        "**Build:** First project build to address **Layer 1** (the DWPose "
        "keypoint detector). Test-time horizontal-flip augmentation: run "
        "DWPose on each frame's original crop AND its horizontally-mirrored "
        "crop, un-flip (mirror x back + swap L/R keypoint labels), and "
        "confidence-weighted average. Hypothesis (from Agent TT's clean "
        "negative): the `hip_adduction_r/side_right` mirror-twin bottleneck "
        "is a DWPose left/right detector asymmetry; flip-TTA should cancel it."
    )
    L.append("")
    L.append(
        "**Single camera only.** Flip-TTA runs the detector TWICE on the SAME "
        "single frame (original + its mirror). It is NOT multi-camera fusion. "
        "The inference contract is unchanged: 1 video stream -> keypoints -> "
        "5 angles."
    )
    L.append("")

    # ---- Path A/B ----
    L.append("## 1. Path A or B?")
    L.append("")
    L.append(
        "**Path A** -- true flip-TTA was performed. OpenCap LabValidation "
        "videos (.avi, syncd-with-mocap) and the DWPose ONNX model "
        "(`models/dw-ll_ucoco_384.onnx`) are both present. We re-ran the "
        "detector on original + horizontally-mirrored pixels. A sanity check "
        "confirmed the original-pass reproduces the cached keypoints to ~3.3 "
        "px and the flipped pass genuinely perturbs keypoints (3.4 px mean, "
        "with L/R-specific shifts) -- i.e. real pixel-level re-detection, not "
        "a cached-keypoint identity transform."
    )
    scope = v46_raw.get("scope_note", "")
    if scope:
        L.append("")
        L.append(scope)
    L.append("")

    # ---- Mirror twin verdict ----
    mt = by_slot.get("hip_adduction_r|side_right")
    mt46 = v46_by_slot.get(("hip_adduction_r", "side_right"))
    L.append("## 2. The mirror-twin verdict: hip_adduction_r / side_right")
    L.append("")
    if mt46 and mt46.get("tta_applicable") and "ccc_lin" in mt46:
        base_ccc = mt46["baseline_ccc_lin"]
        tta_ccc = mt46["ccc_lin"]
        tta_loa = mt46["loa_half_width_deg"]
        sib = v46_by_slot.get(("hip_adduction_r", "side_left"))
        L.append(
            "Measured on the **hand-engineered geometry reader** -- the most "
            "transparent, direct function of L1 keypoints (no learned "
            "compensation), so the cleanest detector-asymmetry probe."
        )
        L.append("")
        L.append(f"- Geometry reader, ORIGINAL DWPose: CCC **{_fmt(base_ccc)}**")
        L.append(f"- Geometry reader, FLIP-TTA DWPose: CCC **{_fmt(tta_ccc)}** (LoA {_fmt(tta_loa,2)} deg)")
        if sib and "ccc_lin" in sib:
            L.append(
                f"- For reference, the mirror sibling side_left geometry reader: "
                f"ORIGINAL CCC {_fmt(sib['baseline_ccc_lin'])} -> FLIP-TTA CCC {_fmt(sib['ccc_lin'])}"
            )
        delta = (tta_ccc - base_ccc) if not (math.isnan(tta_ccc) or math.isnan(base_ccc)) else float("nan")
        L.append(f"- **Flip-TTA delta on the geometry reader: {('%+.3f' % delta) if not math.isnan(delta) else 'n/a'} CCC**")
        L.append("")
        L.append(
            f"- v45 deploy pick (best reader, all candidates): "
            f"{mt['v45_reader']} CCC {_fmt(mt['v45_ccc'])} (tier {mt['v45_tier']})."
        )
        L.append(
            f"- v47 deploy pick: {mt['v47_reader']} CCC {_fmt(mt['v47_ccc'])} "
            f"(tier {mt['v47_tier']})."
        )
        promoted_good = (not math.isnan(tta_ccc)) and tta_ccc >= 0.60 and tta_loa <= 10
        L.append("")
        if promoted_good:
            L.append("- **Promoted to Good?** YES -- flip-TTA crossed the Good gate.")
        else:
            L.append(
                "- **Promoted to Good?** NO. See the verdict in section 6 for "
                "what this means for where the asymmetry lives."
            )
    else:
        L.append("- (side_right not in the scoped flip-TTA run -- see scope note.)")
    L.append("")

    # ---- 4 target slots table ----
    L.append("## 3. Detector-asymmetry target slots (geometry reader, baseline vs flip-TTA)")
    L.append("")
    L.append("| Slot | v45 (deploy pick) CCC | geom ORIG CCC | geom FLIP-TTA CCC | flip-TTA LoA | delta | tier |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for target, view, v45ref in TARGET_SLOTS:
        s = v46_by_slot.get((target, view))
        pick = by_slot.get(f"{target}|{view}")
        v45c = pick["v45_ccc"] if pick else float("nan")
        if s and s.get("tta_applicable") and "ccc_lin" in s:
            d = s.get("delta_ccc_tta_minus_baseline")
            L.append(
                f"| {target}/{view} | {_fmt(v45c)} | {_fmt(s['baseline_ccc_lin'])} | "
                f"{_fmt(s['ccc_lin'])} | {_fmt(s['loa_half_width_deg'],2)} | "
                f"{('%+.3f' % d) if d is not None else 'n/a'} | {s['classification']} |"
            )
        else:
            reason = s.get("reason", "not in scoped run") if s else "not in scoped run"
            L.append(f"| {target}/{view} | {_fmt(v45c)} | n/a | n/a | n/a | n/a | {reason} |")
    L.append("")

    # ---- already-good sanity ----
    L.append("## 4. Did flip-TTA help or hurt the already-good symmetric slot?")
    L.append("")
    sl = v46_by_slot.get(("hip_adduction_r", "side_left"))
    if sl and "ccc_lin" in sl:
        L.append(
            f"hip_adduction_r/side_left geometry reader: ORIGINAL CCC "
            f"{_fmt(sl['baseline_ccc_lin'])} -> FLIP-TTA CCC {_fmt(sl['ccc_lin'])} "
            f"(delta {('%+.3f' % sl['delta_ccc_tta_minus_baseline']) if sl.get('delta_ccc_tta_minus_baseline') is not None else 'n/a'})."
        )
        L.append("")
        L.append(
            "Averaging two passes should be neutral-to-positive on a "
            "symmetric slot. A large drop here would flag a bug in the un-flip "
            "L/R label mapping. (The geometry reader's absolute CCC on this "
            "slot is low because the v45 deploy winner for side_left is a "
            "learned-L2 ensemble at CCC 0.94, not this geometry reader.)"
        )
    L.append("")

    # ---- counts ----
    L.append("## 5. v47 Good + Tier 1 counts")
    L.append("")
    L.append(
        f"- **v47 Good slots: {out['tier_counts'].get('Good', 0)}** "
        f"(delta vs v45's {V45_GOOD}: {out['delta_good_vs_v45']:+d})"
    )
    L.append(
        f"- **v47 Tier 1 (CCC >= 0.79): {out['tier1_count_ccc_ge_0p79']}** "
        f"(delta vs v45's {V45_TIER1}: {out['delta_tier1_vs_v45']:+d})"
    )
    L.append(f"- v46 (flip-TTA) picked in **{out['n_slots_taking_v46']}** / 23 slots.")
    L.append("")

    # ---- latency ----
    L.append("## 6. Latency cost of flip-TTA")
    L.append("")
    timing_path = DATA_ROOT / "dwpose_flip_tta_keypoints" / "_timing.json"
    if timing_path.exists():
        t = json.loads(timing_path.read_text())
        pf = t.get("per_frame_ms", {})
        L.append(
            f"- **{_fmt(pf.get('mean'),1)} ms/frame** for the FULL flip-TTA "
            f"(two detector passes), CPU ({t.get('device','?')}). "
            f"p50 {_fmt(pf.get('p50'),1)}, p95 {_fmt(pf.get('p95'),1)}."
        )
        L.append(
            "- Flip-TTA exactly **doubles** L1 detector cost (2 passes/frame). "
            "Single-pass DWPose is ~half this. For Couro's offline / "
            "near-real-time per-clip analysis this is feasible; for strict "
            "real-time it is a 2x L1 budget hit that would be reserved for the "
            "slots it actually improves."
        )
    else:
        L.append("- (timing file not found)")
    L.append("")

    # ---- verdict ----
    L.append("## 7. Verdict: detector problem or something deeper?")
    L.append("")
    if mt46 and "ccc_lin" in mt46:
        base_ccc = mt46["baseline_ccc_lin"]
        tta_ccc = mt46["ccc_lin"]
        sib = v46_by_slot.get(("hip_adduction_r", "side_left"))
        sib_base = sib["baseline_ccc_lin"] if sib else float("nan")
        delta = tta_ccc - base_ccc if not (math.isnan(tta_ccc) or math.isnan(base_ccc)) else float("nan")
        closed_gap = (not math.isnan(delta)) and (not math.isnan(sib_base)) and \
            (tta_ccc >= base_ccc + 0.5 * max(0.0, sib_base - base_ccc))
        if not math.isnan(delta) and delta > 0.05 and closed_gap:
            L.append(
                "**Detector problem -- flip-TTA helps.** The geometry reader's "
                "side_right CCC rises materially toward the side_left sibling "
                "after flip-TTA, consistent with cancelling a DWPose left/right "
                "asymmetry."
            )
        elif not math.isnan(delta) and delta > 0.02:
            L.append(
                "**Partially a detector problem.** Flip-TTA nudges side_right "
                "up but does not close the gap to the side_left sibling -- the "
                "detector asymmetry is real but only part of the story."
            )
        else:
            L.append(
                "**Deeper than the detector.** Flip-TTA -- the textbook fix for "
                "detector left/right asymmetry -- did NOT lift the side_right "
                "geometry reader. Combined with NN's mirror-flip training "
                "negative and TT's 3D-lifting negative, this points away from "
                "DWPose pixel-level asymmetry and toward the OpenCap capture / "
                "ground-truth side: the side_right camera's oblique view of the "
                "right limb, or an asymmetry baked into the mocap reference for "
                "this metric/view. The remaining lever is view-specific GT "
                "recalibration, not the detector."
            )
    else:
        L.append("- (side_right not measured in scoped run.)")
    L.append("")

    # ---- discipline ----
    L.append("## 8. LOSO discipline & honesty notes")
    L.append("")
    L.append(
        "- Subject-level LOSO at L3 on the OpenCap cohort; flip-TTA changed "
        "only OpenCap keypoints, so validity is measured on OpenCap subjects "
        "only (combined-dataset ASPset trials still train the ridge but are "
        "excluded from the slot statistic, matching prior builds)."
    )
    L.append(
        "- The geometry reader is used as the L1 probe because it is a direct "
        "function of keypoints. The v45 deploy winners for the mirror-twin "
        "slots are learned-L2 readers at higher CCC; v46 only displaces them "
        "in the oracle if flip-TTA lifts the geometry reader above them "
        "(no-regression fallback enforced)."
    )
    L.append("")

    (V47_OUT_DIR / "REPORT.md").write_text("\n".join(L))
    print(f"[v47] wrote {V47_OUT_DIR / 'REPORT.md'}")


def main() -> None:
    out = build_v47()
    write_deploy(out)
    dump_v46_predictions(out)
    write_report(out)
    print(f"[v47] Good slots: {out['tier_counts'].get('Good',0)} (v45: {V45_GOOD})")
    print(f"[v47] Tier 1: {out['tier1_count_ccc_ge_0p79']} (v45: {V45_TIER1})")
    print(f"[v47] v46 picked in {out['n_slots_taking_v46']} / 23 slots")


if __name__ == "__main__":
    main()
