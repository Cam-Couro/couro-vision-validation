"""Post-LOSO analysis for Agent HH's combined OpenCap+ASPset run.

Reads per_slot_results.json and produces:
  * per-source breakdown (OpenCap-held vs ASPset-held)
  * per-metric vs EE2 OpenCap-only baseline
  * per-fold table
  * Layer 3 ROM Bland-Altman + CCC per (source, metric)
  * v17 deploy-slot tier-change projection (qualitative)
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation")
HH = ROOT / "data" / "learned_layer2_combined" / "per_slot_results.json"
EE2 = ROOT / "data" / "learned_layer2_real_gt" / "per_slot_results.json"
V17 = ROOT / "data" / "biomech_validity_stats" / "per_slot_validity.json"

DEPLOY = ("hip_flexion_r", "hip_adduction_r", "knee_angle_r",
          "ankle_angle_r", "lumbar_extension")
COURO_BASELINE = {
    "hip_flexion_r":    0.644,
    "hip_adduction_r":  0.251,
    "knee_angle_r":     0.608,
    "ankle_angle_r":    0.522,
    "lumbar_extension": 0.546,
}
COURO_POOLED = 0.514


def _safe_mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _safe_std(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.std(xs)) if xs else float("nan")


def _ccc(a, b):
    a = np.asarray(a); b = np.asarray(b)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 3:
        return float("nan")
    a = a[f]; b = b[f]
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    return float((2 * r * sa * sb) / (sa**2 + sb**2 + (ma - mb)**2))


def _loa(a, b):
    diff = np.asarray(a) - np.asarray(b)
    f = np.isfinite(diff)
    if f.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    d = diff[f]
    bias = float(d.mean())
    sd = float(d.std())
    return bias, sd, float(1.96 * sd)


def main():
    with open(HH) as f:
        hh = json.load(f)
    with open(EE2) as f:
        ee2 = json.load(f)
    with open(V17) as f:
        v17 = json.load(f)

    print("\n=========================================")
    print(" HH combined trainer post-run analysis ")
    print("=========================================\n")
    print(f"HH version:  {hh.get('version')}")
    print(f"folds:       {len(hh.get('fold_results', []))}")
    print(f"intermediate flag: {hh.get('intermediate')}")
    print(f"epochs:      {hh.get('epochs')}")
    print(f"batch:       {hh.get('batch_size')}")

    # Determine overall struct (intermediate dump or final)
    overall = hh.get("overall")
    if overall is None:
        # Recompute from fold_results (intermediate)
        per_metric = {m: [] for m in DEPLOY}
        all_p = []
        for fr in hh["fold_results"]:
            for m, v in fr["aggregate"]["per_metric"].items():
                if math.isfinite(v["mean_abs_r"]):
                    per_metric[m].append(v["mean_abs_r"])
            p = fr["aggregate"]["pooled"]["mean_abs_r"]
            if math.isfinite(p):
                all_p.append(p)
        overall = {
            "pooled": _safe_mean(all_p),
            "per_metric": {m: _safe_mean(vs) for m, vs in per_metric.items()},
        }
        pooled_key = "pooled"
        per_m_key = "per_metric"
    else:
        pooled_key = "pooled_subject_mean_abs_r"
        per_m_key = "per_metric_subject_mean_abs_r"

    # ----- Headline -----
    print("\n--- Headline pooled |r| ---")
    if isinstance(overall.get(pooled_key), dict):
        pooled = overall[pooled_key]["mean_abs_r"]
        n_folds = overall[pooled_key]["n_folds"]
    else:
        pooled = overall[pooled_key]
        n_folds = len(hh["fold_results"])
    print(f"  HH combined (OpenCap + ASPset):  pooled |r| = {pooled:.4f}  ({n_folds} folds)")
    print(f"  EE2 OpenCap-only baseline:        pooled |r| = 0.6452  (9 folds)")
    print(f"  Couro hand-engineered baseline:   pooled |r| = {COURO_POOLED:.4f}")

    # ----- Per-metric -----
    print("\n--- Per-metric subject-mean |r| ---")
    print(f"{'metric':<22}{'HH':>8}{'EE2 OC-only':>13}{'Couro hand':>13}{'ΔEE2':>8}{'ΔCouro':>8}")
    ee2_pm = ee2["overall"]["per_metric_subject_mean_abs_r"]
    for m in DEPLOY:
        if isinstance(overall.get(per_m_key, {}).get(m), dict):
            hh_r = overall[per_m_key][m]["mean_abs_r"]
        else:
            hh_r = overall.get(per_m_key, {}).get(m, float("nan"))
        ee2_r = ee2_pm[m]["mean_abs_r"]
        couro_r = COURO_BASELINE[m]
        d_ee2 = hh_r - ee2_r
        d_couro = hh_r - couro_r
        print(f"  {m:<20}{hh_r:>+7.3f}{ee2_r:>+12.3f}{couro_r:>+12.3f}{d_ee2:>+8.3f}{d_couro:>+8.3f}")

    # ----- Per-source breakdown -----
    print("\n--- Per-source: held-out OpenCap vs held-out ASPset ---")
    by_src = {"opencap": [], "aspset": []}
    by_src_pm = {"opencap": {m: [] for m in DEPLOY}, "aspset": {m: [] for m in DEPLOY}}
    for fr in hh["fold_results"]:
        src = fr.get("source") or ("opencap" if fr["held_out"].startswith("opencap_") else "aspset")
        p = fr["aggregate"]["pooled"]["mean_abs_r"]
        if math.isfinite(p):
            by_src[src].append(p)
        for m, v in fr["aggregate"]["per_metric"].items():
            if math.isfinite(v["mean_abs_r"]):
                by_src_pm[src][m].append(v["mean_abs_r"])

    for src in ["opencap", "aspset"]:
        n = len(by_src[src])
        pooled_src = _safe_mean(by_src[src])
        print(f"  {src:<10s}  n_folds={n:<3d}  pooled |r| = {pooled_src:.4f}")

    print(f"\n--- Per-source per-metric |r| (OpenCap-held columns | ASPset-held columns) ---")
    print(f"{'metric':<22}{'OC-held':>10}{'ASP-held':>10}{'EE2(OC)':>10}")
    for m in DEPLOY:
        oc_r = _safe_mean(by_src_pm["opencap"][m])
        asp_r = _safe_mean(by_src_pm["aspset"][m])
        ee2_r = ee2_pm[m]["mean_abs_r"]
        print(f"  {m:<20}{oc_r:>+10.3f}{asp_r:>+10.3f}{ee2_r:>+10.3f}")

    # ----- Per-fold table -----
    print("\n--- Per-fold pooled |r| ---")
    print(f"{'fold':>4}  {'held_out':<26s}{'src':>9s}{'pooled':>9s}")
    for fr in hh["fold_results"]:
        held = fr["held_out"]
        src = fr.get("source") or ("opencap" if held.startswith("opencap_") else "aspset")
        p = fr["aggregate"]["pooled"]["mean_abs_r"]
        print(f"  {fr['fold_idx']:>2d}  {held:<26s}{src:>9s}{p:>+9.4f}")

    # ----- ROM analysis (per-subject CCC + LoA) -----
    print("\n--- Layer 3 ROM analysis (per subject, per metric) ---")
    rom_by_src_metric: dict[str, dict[str, dict[str, list[float]]]] = {
        "opencap": {m: {"pred": [], "gt": []} for m in DEPLOY},
        "aspset":  {m: {"pred": [], "gt": []} for m in DEPLOY},
    }
    for fr in hh["fold_results"]:
        src = fr.get("source") or ("opencap" if fr["held_out"].startswith("opencap_") else "aspset")
        agg = fr["aggregate"]
        if "rom_records" in agg:
            recs = agg["rom_records"]
        else:
            # Reconstruct from per_clip
            recs = []
            for clip in fr["per_clip"]:
                for metric, mm in clip["metrics"].items():
                    if math.isfinite(mm.get("r", float("nan"))):
                        # Don't have ROM without recomputing — skip
                        pass
        # Aggregate to per-subject ROM (per metric)
        subj_rom: dict[str, dict[str, dict[str, list[float]]]] = {}
        for r in recs:
            subj = r["subject"]
            m = r["metric"]
            subj_rom.setdefault(subj, {}).setdefault(m, {"pred": [], "gt": []})
            subj_rom[subj][m]["pred"].append(r["pred_rom"])
            subj_rom[subj][m]["gt"].append(r["gt_rom"])
        # Per-subject per-metric median ROM
        for subj, mm in subj_rom.items():
            for m, d in mm.items():
                if not d["gt"]:
                    continue
                rom_by_src_metric[src][m]["pred"].append(float(np.median(d["pred"])))
                rom_by_src_metric[src][m]["gt"].append(float(np.median(d["gt"])))

    print(f"{'metric':<22}{'source':<10s}{'n_subj':>7}{'pred_md':>9}{'gt_md':>9}{'bias':>8}{'sd':>7}{'LoA±':>8}{'CCC':>7}")
    for m in DEPLOY:
        for src in ["opencap", "aspset"]:
            preds = rom_by_src_metric[src][m]["pred"]
            gts = rom_by_src_metric[src][m]["gt"]
            if not preds:
                continue
            bias, sd, loa = _loa(preds, gts)
            ccc = _ccc(preds, gts)
            print(f"  {m:<20}{src:<10s}{len(preds):>7d}"
                  f"{np.median(preds):>+9.2f}{np.median(gts):>+9.2f}"
                  f"{bias:>+8.2f}{sd:>+7.2f}{loa:>+8.2f}{ccc:>+7.2f}")

    # ----- v17 deploy slot comparison (qualitative) -----
    print("\n--- v17 deploy-slot baseline (for comparison context) ---")
    v17_slots = v17.get("slots", [])
    print(f"  v17 has {len(v17_slots)} slots")
    v17_pm_mean = {m: [] for m in DEPLOY}
    for s in v17_slots:
        if s["target"] in v17_pm_mean and math.isfinite(s.get("pearson_r", float("nan"))):
            v17_pm_mean[s["target"]].append(s["pearson_r"])
    print(f"{'metric':<22}{'v17 mean':>10}{'v17 max':>10}{'HH |r|':>10}")
    for m in DEPLOY:
        v_mean = _safe_mean(v17_pm_mean[m])
        v_max = max(v17_pm_mean[m]) if v17_pm_mean[m] else float("nan")
        if isinstance(overall.get(per_m_key, {}).get(m), dict):
            hh_r = overall[per_m_key][m]["mean_abs_r"]
        else:
            hh_r = overall.get(per_m_key, {}).get(m, float("nan"))
        print(f"  {m:<20}{v_mean:>+10.3f}{v_max:>+10.3f}{hh_r:>+10.3f}")

    # Save derived analytics
    out = ROOT / "data" / "learned_layer2_combined" / "_hh_analysis.json"
    analysis = {
        "headline_pooled_r": pooled,
        "n_folds": n_folds,
        "couro_baseline_pooled": COURO_POOLED,
        "ee2_opencap_only_pooled": 0.6452,
        "per_metric": {
            m: (overall[per_m_key][m]["mean_abs_r"]
                if isinstance(overall.get(per_m_key, {}).get(m), dict)
                else overall.get(per_m_key, {}).get(m, float("nan")))
            for m in DEPLOY
        },
        "per_source_pooled": {
            "opencap_held": _safe_mean(by_src["opencap"]),
            "aspset_held": _safe_mean(by_src["aspset"]),
        },
        "per_source_per_metric": {
            src: {m: _safe_mean(by_src_pm[src][m]) for m in DEPLOY}
            for src in ["opencap", "aspset"]
        },
        "rom_analysis": {
            src: {
                m: {
                    "n_subj": len(rom_by_src_metric[src][m]["pred"]),
                    "ccc": _ccc(rom_by_src_metric[src][m]["pred"],
                                rom_by_src_metric[src][m]["gt"]),
                    "loa_half_width_deg": _loa(rom_by_src_metric[src][m]["pred"],
                                               rom_by_src_metric[src][m]["gt"])[2],
                    "bias_deg": _loa(rom_by_src_metric[src][m]["pred"],
                                     rom_by_src_metric[src][m]["gt"])[0],
                } for m in DEPLOY if rom_by_src_metric[src][m]["pred"]
            } for src in ["opencap", "aspset"]
        },
    }
    out.write_text(json.dumps(analysis, indent=2, default=str))
    print(f"\nwrote summary -> {out}")


if __name__ == "__main__":
    main()
