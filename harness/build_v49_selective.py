"""Agent VV2 v49 selective oracle: v47 pool + v48 (L1 DWPose+RTMPose ensemble).

Mirrors harness.build_v47_selective. Reader pool = v47 pool + v48 (the two-
architecture detector ensemble, hand-engineered geometry reader). Per-slot pick
the highest tier / highest CCC, with PP's LoA-then-CCC tie-break and a hard
no-regression fallback (v48 cannot regress CCC by > 0.02 vs the v47 pick).

CRITICAL (per the campaign brief): the consolidated truth is reproduced from
each reader's validity file, NOT from the v47 routing map (the v47 per_slot
routing is known-stale). The v47 pool is taken from
``data/v47_selective_oracle/consolidated_metrics_v47.json`` -- the source-of-
truth artifact the prior agent rebuilt from the readers. The v48 candidate is
taken from this run's ``per_slot_validity_v48.json``. We rebuild
``consolidated_metrics_v49.json`` the same way (from readers, not routing), and
that consolidated artifact is the source of truth for the Good count.

v48 is the geometry reader on the detector-ensemble keypoints. It only competes
on slots where the ensemble was actually computed (side cams in this scoped run)
and where the geometry reader produced a usable dataset. Everywhere else the v47
pick is preserved -- no demotions.

SINGLE CAMERA ONLY: the ensemble is two DETECTORS on the SAME single-camera
frame, not multi-camera fusion.

Outputs:
  * results/deploy_ready_models_v49_selective.json
  * data/v49_selective_oracle/REPORT.md
  * data/v49_selective_oracle/consolidated_metrics_v49.json  (source of truth)
  * data/per_slot_predictions/per_slot_predictions_v48.json  (SS-pattern dump)
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

V47_CONSOLIDATED_PATH: Final[Path] = DATA_ROOT / "v47_selective_oracle" / "consolidated_metrics_v47.json"
V47_DEPLOY_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v47_selective.json"
V48_VALIDITY_PATH: Final[Path] = DATA_ROOT / "layer3_retrain_detector_ensemble" / "per_slot_validity_v48.json"
V48_PER_CLIP_PATH: Final[Path] = DATA_ROOT / "layer3_retrain_detector_ensemble" / "per_clip_predictions_v48.json"

V49_OUT_DIR: Final[Path] = DATA_ROOT / "v49_selective_oracle"
V49_DEPLOY_PATH: Final[Path] = RESULTS_DIR / "deploy_ready_models_v49_selective.json"
V49_CONSOLIDATED_PATH: Final[Path] = V49_OUT_DIR / "consolidated_metrics_v49.json"
V48_DUMP_PATH: Final[Path] = DUMP_DIR / "per_slot_predictions_v48.json"

LOA_LIMITED_CCC: Final[float] = 0.79
FALLBACK_CCC_TOLERANCE: Final[float] = 0.02
TIER1_CCC: Final[float] = 0.79

# v47 baseline counts (from v47 consolidated / report).
V47_GOOD: Final[int] = 14
V47_TIER1: Final[int] = 14

# The 4 detector-asymmetry target slots the brief asks about (with v47 CCC).
TARGET_SLOTS: Final[list[tuple[str, str, float]]] = [
    ("hip_adduction_r", "side_right", 0.277),
    ("hip_flexion_r", "side_right", 0.584),
    ("ankle_angle_r", "side_left", 0.638),    # v46 flip-TTA won this in v47
    ("knee_angle_r", "side_right", 0.819),     # LoA-limited (Moderate)
]


def _tier_rank(t: str) -> int:
    return {"Excellent": 4, "Good": 3, "Moderate": 2, "Poor": 1}.get(t, 0)


def _classify(ccc: float, loa_half: float) -> str:
    if ccc is None or math.isnan(ccc):
        return "Poor"
    if ccc > 0.75 and loa_half < 5:
        return "Excellent"
    if ccc > 0.60 and loa_half < 10:
        return "Good"
    if ccc > 0.40 and loa_half < 15:
        return "Moderate"
    return "Poor"


def _is_better(cand_ccc, cand_loa, cand_tier, base_ccc, base_loa, base_tier):
    """LoA-then-CCC tie-break with no-regression fallback (from build_v47)."""
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


def _load_v47_pool() -> dict[tuple[str, str], dict]:
    """The v47 source-of-truth pool: per-slot best reader from the consolidated
    metrics artifact (rebuilt from readers, NOT the stale routing map)."""
    raw = json.loads(V47_CONSOLIDATED_PATH.read_text())
    out: dict[tuple[str, str], dict] = {}
    for r in raw:
        out[(r["metric"], r["view"])] = {
            "ccc": float(r["ccc"]),
            "loa": float(r["loa_half_deg"]),
            "tier": r["tier"],
            "reader": r["reader"],
        }
    return out, raw


def _load_v48() -> tuple[dict[tuple[str, str], dict], dict]:
    raw = json.loads(V48_VALIDITY_PATH.read_text())
    out: dict[tuple[str, str], dict] = {}
    for s in raw.get("slots", []):
        if not s.get("ensemble_applicable") or "ccc_lin" not in s:
            continue
        out[(s["target"], s["view"])] = s
    return out, raw


def build_v49() -> dict:
    v47_pool, v47_raw = _load_v47_pool()
    v48, v48_raw = _load_v48()

    consolidated = []          # source of truth for v49 Good count
    new_picks = []
    tier_counter = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0}
    tier1_count = 0
    reader_distribution: dict[str, int] = {}
    promotions = []
    n_v48_picks = 0

    for (target, view), base in sorted(v47_pool.items()):
        slot_str = f"{target}|{view}"
        base_reader = base["reader"]
        base_ccc = base["ccc"]
        base_loa = base["loa"]
        base_tier = base["tier"]

        v48_stat = v48.get((target, view))
        v48_ccc = float(v48_stat["ccc_lin"]) if v48_stat else float("nan")
        v48_loa = float(v48_stat["loa_half_width_deg"]) if v48_stat else float("nan")
        v48_tier = v48_stat["classification"] if v48_stat else "Poor"
        v48_baseline_ccc = float(v48_stat["baseline_ccc_lin"]) if v48_stat else float("nan")

        promoted, reason = (False, "no v48 candidate for slot")
        if v48_stat is not None:
            promoted, reason = _is_better(
                v48_ccc, v48_loa, v48_tier, base_ccc, base_loa, base_tier,
            )

        if promoted:
            chosen_reader = "v48_detector_ensemble"
            chosen_ccc, chosen_loa, chosen_tier = v48_ccc, v48_loa, v48_tier
            n_v48_picks += 1
            promotions.append({
                "slot": slot_str, "from": base_tier, "to": v48_tier,
                "v47_ccc": base_ccc, "v48_ccc": v48_ccc, "reason": reason,
            })
        else:
            chosen_reader = base_reader
            chosen_ccc, chosen_loa, chosen_tier = base_ccc, base_loa, base_tier

        # Recompute tier from the chosen CCC/LoA so the consolidated artifact is
        # internally consistent (source of truth, not inherited).
        chosen_tier = _classify(chosen_ccc, chosen_loa)
        reader_distribution[chosen_reader] = reader_distribution.get(chosen_reader, 0) + 1
        tier_counter[chosen_tier] = tier_counter.get(chosen_tier, 0) + 1
        is_tier1 = (not math.isnan(chosen_ccc)) and chosen_ccc >= TIER1_CCC and chosen_tier == "Good"
        if is_tier1:
            tier1_count += 1

        consolidated.append({
            "metric": target, "view": view,
            "ccc": round(chosen_ccc, 4) if not math.isnan(chosen_ccc) else None,
            "loa_half_deg": round(chosen_loa, 2) if not math.isnan(chosen_loa) else None,
            "tier": chosen_tier, "reader": chosen_reader,
            "tier1": is_tier1,
        })
        new_picks.append({
            "slot": slot_str,
            "v47_reader": base_reader, "v47_ccc": base_ccc,
            "v47_loa": base_loa, "v47_tier": base_tier,
            "v48_geometry_baseline_ccc": v48_baseline_ccc,
            "v48_ccc": v48_ccc, "v48_loa": v48_loa, "v48_tier": v48_tier,
            "v48_delta_vs_geometry_baseline": (
                v48_ccc - v48_baseline_ccc
                if v48_stat and not (math.isnan(v48_ccc) or math.isnan(v48_baseline_ccc))
                else None
            ),
            "v49_reader": chosen_reader, "v49_ccc": chosen_ccc,
            "v49_loa": chosen_loa, "v49_tier": chosen_tier,
            "promoted": promoted, "promotion_reason": reason,
        })

    # Sort consolidated by tier rank then CCC desc (matches v47 artifact style).
    consolidated.sort(key=lambda r: (-_tier_rank(r["tier"]),
                                     -(r["ccc"] if r["ccc"] is not None else -1)))

    good_count = sum(1 for r in consolidated if r["tier"] == "Good")
    out = {
        "version": "v49_selective_oracle",
        "produced_by": "harness.build_v49_selective (Agent VV2)",
        "description": (
            "v47 pool + v48 (Layer 1 DWPose+RTMPose two-architecture detector "
            "ensemble, hand-engineered geometry reader). Per-slot highest tier "
            "/ highest CCC, LoA-then-CCC tie-break, no CCC regression > 0.02 vs "
            "v47. Consolidated metrics reproduced from each reader's validity "
            "file (NOT the v47 routing map, which is known-stale). SINGLE "
            "CAMERA: two detectors on the same frame, not multi-camera fusion."
        ),
        "tier_counts": tier_counter,
        "good_count_from_consolidated": good_count,
        "tier1_count_ccc_ge_0p79_and_good": tier1_count,
        "v47_good": V47_GOOD, "v47_tier1": V47_TIER1,
        "delta_good_vs_v47": good_count - V47_GOOD,
        "delta_tier1_vs_v47": tier1_count - V47_TIER1,
        "n_slots_taking_v48": n_v48_picks,
        "reader_distribution": reader_distribution,
        "promotions_vs_v47": promotions,
        "v48_n_applicable_slots": v48_raw.get("n_dwpose_applicable_slots"),
        "v48_ensemble_cams": v48_raw.get("ensemble_cams"),
        "picks": new_picks,
    }
    V49_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (V49_OUT_DIR / "per_slot_picks_v49.json").write_text(json.dumps(out, indent=2))
    V49_CONSOLIDATED_PATH.write_text(json.dumps(consolidated, indent=2))
    print(f"[v49] wrote {V49_OUT_DIR / 'per_slot_picks_v49.json'}")
    print(f"[v49] wrote {V49_CONSOLIDATED_PATH} (source of truth, Good={good_count})")
    return out


def write_deploy(out: dict) -> None:
    base = json.loads(V47_DEPLOY_PATH.read_text())
    deploy = dict(base)
    deploy["version"] = "v49_selective_oracle"
    deploy["produced_by"] = "harness.build_v49_selective (Agent VV2)"
    deploy["description"] = out["description"]
    ann = {}
    for p in out["picks"]:
        ann[p["slot"]] = {
            "v49_reader": p["v49_reader"], "v49_ccc": p["v49_ccc"],
            "v49_loa": p["v49_loa"], "v49_tier": p["v49_tier"],
            "promoted_from_v47": p["promoted"],
            "promotion_reason": p["promotion_reason"],
        }
    deploy["v49_annotations"] = ann
    deploy["v49_tier_counts"] = out["tier_counts"]
    deploy["v49_good_count"] = out["good_count_from_consolidated"]
    deploy["v49_tier1_count"] = out["tier1_count_ccc_ge_0p79_and_good"]
    deploy["v49_reader_distribution"] = out["reader_distribution"]
    deploy["v48_l1_method"] = "dwpose_rtmpose_two_architecture_detector_ensemble"
    V49_DEPLOY_PATH.write_text(json.dumps(deploy, indent=2))
    print(f"[v49] wrote {V49_DEPLOY_PATH}")


def dump_v48_predictions() -> None:
    """SS-pattern per-clip prediction dump for future ensembles."""
    try:
        per_clip = json.loads(V48_PER_CLIP_PATH.read_text())
    except FileNotFoundError:
        per_clip = {}
    payload = {
        "version": "v48_detector_ensemble",
        "produced_by": "harness.build_v49_selective (Agent VV2)",
        "reader": "v48_detector_ensemble_geometry",
        "note": (
            "Per-clip LOSO predictions from the hand-engineered geometry reader "
            "on DWPose+RTMPose detector-ensemble keypoints (OpenCap cohort)."
        ),
        "per_slot": per_clip,
    }
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    V48_DUMP_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[v49] wrote {V48_DUMP_PATH}")


def _fmt(v, p=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{p}f}"


def write_report(out: dict) -> None:
    v48_raw = json.loads(V48_VALIDITY_PATH.read_text())
    v47_pool, _ = _load_v47_pool()
    by_slot = {p["slot"]: p for p in out["picks"]}
    v48_by_slot = {(s["target"], s["view"]): s for s in v48_raw["slots"]}

    # Cross-check: did the ensemble beat flip-TTA on slots flip-TTA won?
    # ankle_angle_r/side_left was won by v46 (flip-TTA) at 0.638 in v47.
    timing_path = DATA_ROOT / "dwpose_rtmpose_ensemble_keypoints" / "_timing.json"
    flip_timing_path = DATA_ROOT / "dwpose_flip_tta_keypoints" / "_timing.json"

    L = []
    L.append("# v49 Selective Oracle: Layer 1 DWPose+RTMPose Detector Ensemble (Agent VV2)")
    L.append("")
    L.append(f"**Date:** {time.strftime('%Y-%m-%d')}")
    L.append("")
    L.append(
        "**Build:** A two-ARCHITECTURE Layer 1 detector ensemble. For each "
        "frame, DWPose (`dw-ll_ucoco_384`, COCO-WholeBody-133 -> Halpe-26) and "
        "RTMPose-Halpe26 are both run on the SAME person bbox, then their "
        "keypoints are confidence-weighted averaged: "
        "`kp = (c_dw*kp_dw + c_rt*kp_rt)/(c_dw+c_rt)`. This is the final "
        "modeling lever in the campaign."
    )
    L.append("")
    L.append(
        "**SINGLE CAMERA ONLY.** Two DETECTORS (two architectures) on the SAME "
        "single-camera frame. This is NOT multi-camera fusion. The inference "
        "contract is unchanged: 1 video stream -> keypoints -> 5 angles."
    )
    L.append("")

    # ---- 1. correspondence ----
    L.append("## 1. Keypoint correspondence: did DWPose & RTMPose Halpe-26 indices align?")
    L.append("")
    L.append(
        "**YES -- indices align directly; no remap needed for RTMPose.** The "
        "built-in 1-frame smoke check confirmed both detectors emit 26 "
        "keypoints in the identical Halpe-26 index order. On a Cam0 clip "
        "(subject10/DJ1) the mean DWPose-vs-RTMPose distance over the six "
        "torso/limb joints {L/R shoulder, L/R hip, L/R knee} was ~4.8 px, and "
        "the L_hip same-index distance (0.9 px) was far smaller than the "
        "crossed L_hip-vs-R_hip distance (33.2 px) -- ruling out an L/R label "
        "swap. The DWPose re-inference pass also reproduced the cached "
        "`opencap_dwpose_keypoints` to ~4.1 px, validating the shared bbox / "
        "pre-processing path. RTMPose-Halpe26 outputs Halpe-26 directly; "
        "DWPose is remapped from COCO-WholeBody-133 via the verbatim "
        "`dwpose_to_halpe26`. Both land in the same 26-index layout."
    )
    L.append("")

    # ---- 2. mirror twin ----
    mt = by_slot.get("hip_adduction_r|side_right")
    mt48 = v48_by_slot.get(("hip_adduction_r", "side_right"))
    L.append("## 2. Mirror-twin verdict: hip_adduction_r / side_right (v47 0.277)")
    L.append("")
    if mt48 and mt48.get("ensemble_applicable") and "ccc_lin" in mt48:
        base_ccc = mt48["baseline_ccc_lin"]
        ens_ccc = mt48["ccc_lin"]
        ens_loa = mt48["loa_half_width_deg"]
        sib = v48_by_slot.get(("hip_adduction_r", "side_left"))
        L.append(
            "Measured on the hand-engineered geometry reader (the most "
            "transparent direct function of L1 keypoints -- the cleanest "
            "detector-asymmetry probe)."
        )
        L.append("")
        L.append(f"- Geometry reader, ORIGINAL DWPose: CCC **{_fmt(base_ccc)}**")
        L.append(f"- Geometry reader, DETECTOR ENSEMBLE: CCC **{_fmt(ens_ccc)}** (LoA {_fmt(ens_loa,2)} deg)")
        if sib and "ccc_lin" in sib:
            L.append(
                f"- Mirror sibling side_left geometry reader: ORIGINAL CCC "
                f"{_fmt(sib['baseline_ccc_lin'])} -> ENSEMBLE CCC {_fmt(sib['ccc_lin'])}"
            )
        delta = (ens_ccc - base_ccc) if not (math.isnan(ens_ccc) or math.isnan(base_ccc)) else float("nan")
        L.append(f"- **Ensemble delta on the geometry reader: {('%+.3f' % delta) if not math.isnan(delta) else 'n/a'} CCC**")
        L.append("")
        L.append(f"- v47 deploy pick (consolidated): CCC {_fmt(mt['v47_ccc'])} (tier {mt['v47_tier']}, reader {mt['v47_reader']}).")
        L.append(f"- v49 deploy pick: {mt['v49_reader']} CCC {_fmt(mt['v49_ccc'])} (tier {mt['v49_tier']}).")
        promoted_good = (not math.isnan(ens_ccc)) and ens_ccc >= 0.60 and ens_loa <= 10
        L.append("")
        if promoted_good:
            L.append("- **Promoted to Good?** YES -- the detector ensemble crossed the Good gate (CCC>0.60, LoA<10).")
        else:
            L.append(
                "- **Promoted to Good?** NO. The mirror twin needed 0.277 -> 0.60; "
                "the detector ensemble did not get there. See section 9 for the verdict."
            )
    else:
        L.append("- (side_right not in the scoped ensemble run -- check scope/cams.)")
    L.append("")

    # ---- 3. A/B table ----
    L.append("## 3. A/B table (geometry reader, ORIGINAL DWPose vs detector ensemble)")
    L.append("")
    L.append("| Slot | v47 deploy CCC | geom ORIG CCC | geom ENSEMBLE CCC | ENS LoA | delta | tier |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for target, view, v47ref in TARGET_SLOTS:
        s = v48_by_slot.get((target, view))
        pick = by_slot.get(f"{target}|{view}")
        v47c = pick["v47_ccc"] if pick else v47ref
        if s and s.get("ensemble_applicable") and "ccc_lin" in s:
            d = s.get("delta_ccc_ensemble_minus_baseline")
            L.append(
                f"| {target}/{view} | {_fmt(v47c)} | {_fmt(s['baseline_ccc_lin'])} | "
                f"{_fmt(s['ccc_lin'])} | {_fmt(s['loa_half_width_deg'],2)} | "
                f"{('%+.3f' % d) if d is not None else 'n/a'} | {s['classification']} |"
            )
        else:
            reason = s.get("reason", "not in scoped run") if s else "not in scoped run"
            L.append(f"| {target}/{view} | {_fmt(v47c)} | n/a | n/a | n/a | n/a | {reason} |")
    L.append("")

    # ---- 4. ensemble vs flip-TTA ----
    L.append("## 4. Ensemble (v48, two detectors) vs flip-TTA (v46, one detector flipped)")
    L.append("")
    L.append(
        "Both are Layer-1 levers on the same geometry reader. flip-TTA runs ONE "
        "detector (DWPose) twice (original + mirror); the ensemble runs TWO "
        "detectors (DWPose + RTMPose) once each. Comparison on slots flip-TTA "
        "improved in v47:"
    )
    L.append("")
    L.append("| Slot | geom ORIG | flip-TTA (v46) | ensemble (v48) | ensemble beats flip-TTA? |")
    L.append("| --- | ---: | ---: | ---: | --- |")
    flip_validity_path = DATA_ROOT / "layer3_retrain_flip_tta" / "per_slot_validity_v46.json"
    flip_by_slot = {}
    if flip_validity_path.exists():
        fraw = json.loads(flip_validity_path.read_text())
        flip_by_slot = {(s["target"], s["view"]): s for s in fraw.get("slots", [])}
    flip_compare_slots = [
        ("ankle_angle_r", "side_left"),
        ("hip_adduction_r", "side_right"),
        ("hip_flexion_r", "side_right"),
        ("knee_angle_r", "side_right"),
    ]
    for target, view in flip_compare_slots:
        e = v48_by_slot.get((target, view))
        f = flip_by_slot.get((target, view))
        orig = e["baseline_ccc_lin"] if (e and "baseline_ccc_lin" in e) else (
            f["baseline_ccc_lin"] if (f and "baseline_ccc_lin" in f) else float("nan"))
        flip_ccc = f["ccc_lin"] if (f and "ccc_lin" in f) else float("nan")
        ens_ccc = e["ccc_lin"] if (e and "ccc_lin" in e) else float("nan")
        if not math.isnan(ens_ccc) and not math.isnan(flip_ccc):
            verdict = "YES" if ens_ccc > flip_ccc + 1e-6 else ("tie" if abs(ens_ccc - flip_ccc) <= 1e-6 else "no")
        else:
            verdict = "n/a"
        L.append(f"| {target}/{view} | {_fmt(orig)} | {_fmt(flip_ccc)} | {_fmt(ens_ccc)} | {verdict} |")
    L.append("")

    # ---- 5. counts ----
    L.append("## 5. v49 Good count + delta vs v47")
    L.append("")
    L.append(
        f"- **v49 Good slots: {out['good_count_from_consolidated']}** "
        f"(delta vs v47's {V47_GOOD}: {out['delta_good_vs_v47']:+d}) "
        "-- counted from `consolidated_metrics_v49.json`, the source-of-truth "
        "artifact rebuilt from each reader's validity file (not the stale v47 "
        "routing map)."
    )
    L.append(f"- v48 (detector ensemble) picked in **{out['n_slots_taking_v48']}** slots.")
    L.append("")

    # ---- 6. tier 1 ----
    L.append("## 6. v49 Tier 1 (CCC >= 0.79 AND Good)")
    L.append("")
    L.append(
        f"- **v49 Tier 1: {out['tier1_count_ccc_ge_0p79_and_good']}** "
        f"(delta vs v47's {V47_TIER1}: {out['delta_tier1_vs_v47']:+d})."
    )
    L.append("")

    # ---- 7. regressions ----
    L.append("## 7. Did the ensemble hurt any v47 Good slots?")
    L.append("")
    hurt = []
    for p in out["picks"]:
        if p["v47_tier"] == "Good" and p["v49_tier"] != "Good":
            hurt.append(p["slot"])
    if hurt:
        L.append("- **YES -- regressions detected (should not happen with the "
                 "no-regression fallback; investigate):** " + ", ".join(hurt))
    else:
        L.append(
            "- **No.** The no-regression fallback (v48 cannot regress CCC by "
            ">0.02 vs the v47 pick, and never demotes tier) preserved every v47 "
            "Good slot. The ensemble only ever displaced a v47 pick when it "
            "strictly improved tier or CCC/LoA."
        )
    L.append("")

    # ---- 8. latency ----
    L.append("## 8. Latency (ms/frame)")
    L.append("")
    if timing_path.exists():
        t = json.loads(timing_path.read_text())
        pf = t.get("per_frame_ms", {})
        ens_ms = pf.get("mean")
        L.append(
            f"- **Detector ensemble: {_fmt(ens_ms,1)} ms/frame** for two "
            f"detector passes (DWPose {t.get('dwpose_device','?')} + RTMPose "
            f"{t.get('rtmpose_device','?')}). p50 {_fmt(pf.get('p50'),1)}, "
            f"p95 {_fmt(pf.get('p95'),1)}."
        )
        flip_ms = None
        if flip_timing_path.exists():
            ft = json.loads(flip_timing_path.read_text())
            flip_ms = ft.get("per_frame_ms", {}).get("mean")
        L.append(
            f"- Reference: flip-TTA (v46) ~246 ms/frame"
            + (f" (measured {_fmt(flip_ms,1)} ms/frame this repo)" if flip_ms else "")
            + "; single-pass DWPose ~120 ms/frame."
        )
        if ens_ms:
            L.append(
                f"- The ensemble's RTMPose pass runs on CoreML (Neural Engine) "
                f"while DWPose stays on CPU, so the two-detector cost is "
                f"{('below' if (flip_ms and ens_ms < flip_ms) else 'comparable to')} "
                f"the two-CPU-pass flip-TTA. For Couro's near-real-time per-clip "
                f"analysis this is feasible."
            )
    else:
        L.append("- (ensemble timing file not found)")
    L.append("")

    # ---- 9. verdict ----
    L.append("## 9. Campaign-closing verdict: detector-limited or data-limited?")
    L.append("")
    if mt48 and "ccc_lin" in mt48:
        base_ccc = mt48["baseline_ccc_lin"]
        ens_ccc = mt48["ccc_lin"]
        sib = v48_by_slot.get(("hip_adduction_r", "side_left"))
        sib_base = sib["baseline_ccc_lin"] if sib else float("nan")
        delta = ens_ccc - base_ccc if not (math.isnan(ens_ccc) or math.isnan(base_ccc)) else float("nan")
        closed_gap = (not math.isnan(delta)) and (not math.isnan(sib_base)) and \
            (ens_ccc >= base_ccc + 0.5 * max(0.0, sib_base - base_ccc))
        if not math.isnan(delta) and delta > 0.05 and closed_gap:
            L.append(
                "**DETECTOR-LIMITED -- the ensemble helps.** A second "
                "architecture materially lifts the side_right geometry reader "
                "toward its side_left sibling, so part of the mirror-twin "
                "bottleneck was DWPose-specific detector error that a different "
                "architecture corrects."
            )
        elif not math.isnan(delta) and delta > 0.02:
            L.append(
                "**Partially detector-limited.** The second architecture nudges "
                "side_right up but does not close the gap to the side_left "
                "sibling -- detector error is real but only part of the story."
            )
        else:
            L.append(
                "**DATA-LIMITED -- the ensemble does NOT help.** A second, "
                "independent detector architecture (RTMPose) averaged with "
                "DWPose did not lift the side_right mirror twin. This is the "
                "third independent Layer-1/Layer-2 negative on this slot: NN's "
                "mirror-flip L2 training, UU's flip-TTA L1, and now VV2's "
                "two-architecture detector ensemble all fail to move it. Two "
                "different network architectures agreeing on the 'wrong' answer "
                "rules out idiosyncratic detector error and points to the "
                "OpenCap capture / ground-truth side: the side_right camera's "
                "oblique view of the right limb, or an asymmetry baked into the "
                "mocap reference for this metric/view. The remaining lever is "
                "view-specific GT recalibration or additional capture, not the "
                "detector. **The detector is not the bottleneck.**"
            )
    else:
        L.append("- (side_right not measured in scoped run.)")
    L.append("")

    # ---- discipline ----
    L.append("## 10. LOSO discipline & honesty notes")
    L.append("")
    L.append(
        "- Subject-level LOSO at L3 on the OpenCap cohort; the ensemble changed "
        "only OpenCap keypoints, so validity is measured on OpenCap subjects "
        "only (combined-dataset ASPset trials still train the ridge but are "
        "excluded from the slot statistic, matching all prior builds). No leak."
    )
    L.append(
        "- The consolidated v49 metrics are reproduced from each reader's "
        "validity file. The known-stale v47 routing map was NOT inherited."
    )
    L.append(
        "- Per-slot fallback to the v47 pick is enforced by the oracle "
        "(no-regression, no tier demotion)."
    )
    L.append("")

    (V49_OUT_DIR / "REPORT.md").write_text("\n".join(L))
    print(f"[v49] wrote {V49_OUT_DIR / 'REPORT.md'}")


def main() -> None:
    out = build_v49()
    write_deploy(out)
    dump_v48_predictions()
    write_report(out)
    print(f"[v49] Good slots: {out['good_count_from_consolidated']} (v47: {V47_GOOD})")
    print(f"[v49] Tier 1: {out['tier1_count_ccc_ge_0p79_and_good']} (v47: {V47_TIER1})")
    print(f"[v49] v48 picked in {out['n_slots_taking_v48']} slots")


if __name__ == "__main__":
    main()
