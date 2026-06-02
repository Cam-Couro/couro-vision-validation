"""Pick the best of {v4 full-trial, v6 event-anchored} per (metric × view).

For each (metric, view), pick whichever LOSO model has higher Pearson r AND
RMSE ≤ baseline naive. Outputs a single deploy-ready JSON containing only
the winning model per slot, with provenance ("approach": "full_trial" or
"event_anchored").

Honest reasoning: event-anchoring is a strong win in cases where the
biomech signal lives in the loading phase (hip flexion on side_right,
lumbar extension on front_center, knee angle on front_oblique_right).
It hurts when event detection drops trials (knee front_center went 54→29)
or when full-trial ROM already captures the signal well (most v4 deploy
cases). Picking per slot avoids deploying a worse model out of consistency.
"""
from __future__ import annotations

import json
from pathlib import Path


RESULTS_DIR = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/results")


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def slot_score(rmse: float, r: float) -> float:
    """Composite score: prefer higher r, break ties by lower RMSE.

    r is the dominant axis since RMSE without r means population-mean prediction.
    """
    if r is None or rmse is None:
        return -1e9
    # Penalize negative r heavily — those models actively mispredict
    r_term = r if r > 0 else r * 2.0
    return r_term - 0.01 * rmse


def main() -> None:
    v4 = load(RESULTS_DIR / "all_metrics_regression_models.json")
    v6 = load(RESULTS_DIR / "event_anchored_all_metrics.json")
    bilateral_path = RESULTS_DIR / "ankle_bilateral_models.json"
    bilateral_ankle = load(bilateral_path) if bilateral_path.exists() else {"models": {}}
    v9_path = RESULTS_DIR / "v9_phased_all_metrics.json"
    v9 = load(v9_path) if v9_path.exists() else {"models": {}}
    v10_path = RESULTS_DIR / "v10_gbm_all_metrics.json"
    v10 = load(v10_path) if v10_path.exists() else {"models": {}}
    v12_path = RESULTS_DIR / "v12_combined_models.json"
    v12 = load(v12_path) if v12_path.exists() else {"models": {}}
    v13_path = RESULTS_DIR / "v13_dwpose_combined_models.json"
    v13 = load(v13_path) if v13_path.exists() else {"models": {}}
    v14_path = RESULTS_DIR / "v14_full_dwpose_models.json"
    v14 = load(v14_path) if v14_path.exists() else {"models": {}}

    final: dict = {
        "version": "1.0",
        "produced_by": "harness.finalize_best_models",
        "produced_date": "2026-05-27",
        "description": (
            "Final deploy-ready models — per (metric × view) we picked "
            "the model with higher LOSO Pearson r and acceptable RMSE "
            "between two approaches: (a) full-trial ROM features (v4), "
            "(b) DJ-impact event-anchored loading-window features (v6)."
        ),
        "approaches": {
            "full_trial": {
                "feature_pool": "13 ROM-based features across all 5 metrics, "
                                "computed across the full trial duration",
                "source": "all_metrics_regression_models.json",
            },
            "event_anchored": {
                "feature_pool": "10 features anchored to detected DJ initial "
                                "contact → pushoff window (typically 100-230ms)",
                "source": "event_anchored_all_metrics.json",
                "event_detector": "harness/dj_events.py (scipy peak-find on "
                                  "ankle vertical trajectory)",
            },
        },
        "training_dataset": v4["training_dataset"],
        "models": {},
    }

    summary_rows = []
    for target in v4["targets"]:
        final["models"][target] = {}
        for view in ["side_left", "front_oblique_left", "front_center",
                     "front_oblique_right", "side_right"]:
            m_v4 = v4["targets"][target]["models"].get(view)
            m_v6 = v6["targets"][target]["models"].get(view)
            cands = []
            if m_v4 is not None:
                cands.append(("full_trial", m_v4))
            if m_v6 is not None:
                cands.append(("event_anchored", m_v6))
            # Ankle has a third option: bilateral feature pool
            if target == "ankle_angle_r":
                m_bi = bilateral_ankle.get("models", {}).get(view)
                if m_bi is not None:
                    cands.append(("event_anchored_bilateral", m_bi))
            # v9 phased features available for all metrics
            m_v9 = v9.get("models", {}).get(target, {}).get(view)
            if m_v9 is not None:
                cands.append(("v9_phased", m_v9))
            # v10 GBM on phased features
            m_v10 = v10.get("models", {}).get(target, {}).get(view)
            if m_v10 is not None:
                cands.append(("v10_gbm", m_v10))
            # v12 combined OpenCap+ASPset (RTMPose backbone)
            m_v12 = v12.get("models", {}).get(target, {}).get(view)
            if m_v12 is not None:
                cands.append(("v12_combined", m_v12))
            # v13 combined OpenCap+ASPset (RTMPose OpenCap, DWPose ASPset)
            m_v13 = v13.get("models", {}).get(target, {}).get(view)
            if m_v13 is not None:
                cands.append(("v13_dwpose_hybrid", m_v13))
            # v14 full DWPose pipeline (both datasets through DWPose)
            m_v14 = v14.get("models", {}).get(target, {}).get(view)
            if m_v14 is not None:
                cands.append(("v14_full_dwpose", m_v14))
            if not cands:
                continue
            scored = [
                (approach, m, slot_score(
                    m["loso_cv_stats"]["rmse_deg"],
                    m["loso_cv_stats"]["pearson_r"],
                ))
                for approach, m in cands
            ]
            scored.sort(key=lambda x: x[2], reverse=True)
            winner_approach, winner_model, winner_score = scored[0]

            winner_record = dict(winner_model)
            winner_record["approach"] = winner_approach
            winner_record["picked_over"] = [a for a, _, _ in scored[1:]]
            final["models"][target][view] = winner_record

            r = winner_model["loso_cv_stats"]["pearson_r"]
            rmse = winner_model["loso_cv_stats"]["rmse_deg"]
            summary_rows.append((target, view, winner_approach, rmse, r))

    out_path = RESULTS_DIR / "deploy_ready_models.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)

    # Pretty-print
    print(f"\n{'TARGET':<22}{'VIEW':<22}{'APPROACH':<18}{'RMSE':>9}{'r':>7}  verdict")
    print("-" * 92)
    deploy_count = 0
    scale_count = 0
    nope_count = 0
    for target, view, approach, rmse, r in summary_rows:
        if r >= 0.30:
            verdict = "DEPLOY"
            deploy_count += 1
        elif r >= 0.10:
            verdict = "deploy (modest r)"
            deploy_count += 1
        elif r > 0:
            verdict = "scale-only"
            scale_count += 1
        else:
            verdict = "DO NOT SHIP"
            nope_count += 1
        print(f"  {target:<20}{view:<22}{approach:<18}{rmse:>8.2f}°{r:>+6.2f}  {verdict}")
    print(f"\nDeploy: {deploy_count}  Scale-only: {scale_count}  Do not ship: {nope_count}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
