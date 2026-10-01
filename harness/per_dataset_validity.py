#!/usr/bin/env python3
"""Per-dataset validity check: does a slot's agreement survive WITHIN one dataset?

Why this exists
---------------
The validity pool mixes two public datasets (OpenCap, ASPset). For several slots their
ground-truth values sit in non-overlapping clusters -- e.g. hip_adduction_r/side_left is
8.6-15.6 deg in OpenCap and 41.8-49.2 deg in ASPset, a 26 deg gap with no subjects between.

Lin's CCC measures agreement ACROSS THE OBSERVED RANGE, so pooling two such cohorts manufactures
range and a reader that only knows "ASPset high, OpenCap low" scores near 1.0. Ten of the fourteen
slots that clear the deploy bar lose their CCC when recomputed within OpenCap alone, and the effect
reproduces across five independent reader dumps (v17/v23/v26/v31/v44), so it is structural rather
than reader-specific.

Bland-Altman LoA is range-independent -- absolute agreement in degrees -- so it does NOT inflate
with pooling, and in several slots it IMPROVES within-dataset. LoA is therefore the statistic to
lead with in coach- and investor-facing claims; pooled CCC answers a different question
(cross-population discrimination) and should be labelled as such, not hidden.

Usage
-----
    python3 harness/per_dataset_validity.py                     # all readers on disk
    python3 harness/per_dataset_validity.py --reader v44        # one reader
    python3 harness/per_dataset_validity.py --json out.json     # machine-readable

Reads only files already in the repo. No network, no GPU, no spend.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMPS = os.path.join(REPO, "data/per_slot_predictions/per_slot_predictions_v*.json")
DEPLOYED = os.path.join(REPO, "data/v49_selective_oracle/consolidated_metrics_v49.json")

CCC_BAR = 0.60
LOA_BAR = 10.0
MIN_N = 4          # below this, agreement statistics are not worth reporting


def ccc_lin(x: list[float], y: list[float]) -> float:
    """Lin's concordance correlation coefficient."""
    mx, my = st.mean(x), st.mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / len(x)
    return 2 * cov / (st.pvariance(x) + st.pvariance(y) + (mx - my) ** 2)


def loa_half(x: list[float], y: list[float]) -> float:
    """Bland-Altman 95% limits-of-agreement half-width, in degrees."""
    diffs = [a - b for a, b in zip(x, y)]
    return 1.96 * st.stdev(diffs)


def bias(x: list[float], y: list[float]) -> float:
    return st.mean([a - b for a, b in zip(x, y)])


def dataset_of(subject: str) -> str:
    return subject.split("_", 1)[0]


def analyse(slot: dict) -> dict:
    """Pooled vs per-dataset statistics for one slot."""
    records = slot["fold_records"]
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_ds[dataset_of(r["subject"])].append(r)

    def stats(rs: list[dict]) -> dict | None:
        if len(rs) < MIN_N:
            return None
        p = [r["pred"] for r in rs]
        g = [r["gt"] for r in rs]
        return {
            "n": len(rs),
            "ccc": round(ccc_lin(p, g), 3),
            "loa_half_deg": round(loa_half(p, g), 2),
            "bias_deg": round(bias(p, g), 2),
            "gt_min": round(min(g), 1),
            "gt_max": round(max(g), 1),
            "gt_sd": round(st.stdev(g), 2),
            "mean_abs_err": round(st.mean([abs(a - b) for a, b in zip(p, g)]), 2),
        }

    pooled = stats(records)
    per_ds = {ds: s for ds, s in ((d, stats(rs)) for d, rs in by_ds.items()) if s}

    # Cluster separation: the gap between dataset gt ranges, which is what inflates pooled CCC.
    gap = None
    if len(per_ds) >= 2:
        spans = sorted(((s["gt_min"], s["gt_max"], d) for d, s in per_ds.items()))
        gap = round(max(0.0, spans[-1][0] - spans[0][1]), 1)

    return {
        "metric": slot["target"],
        "view": slot["view"],
        "pooled": pooled,
        "per_dataset": per_ds,
        "cluster_gap_deg": gap,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", help="only this reader (e.g. v44)")
    ap.add_argument("--json", help="write machine-readable results here")
    args = ap.parse_args()

    deployed = {}
    if os.path.exists(DEPLOYED):
        deployed = {(d["metric"], d["view"]): d for d in json.load(open(DEPLOYED))}

    out: list[dict] = []
    for path in sorted(glob.glob(DUMPS)):
        dump = json.load(open(path))
        reader = dump.get("reader", os.path.basename(path))
        if args.reader and reader != args.reader:
            continue
        for slot in dump.get("slots", []):
            if "fold_records" not in slot:
                continue
            row = analyse(slot)
            row["reader"] = reader
            dep = deployed.get((row["metric"], row["view"]))
            if dep:
                row["deployed"] = {"ccc": dep["ccc"], "loa_half_deg": dep["loa_half_deg"],
                                   "reader": dep.get("reader"),
                                   "clears_bar": dep["ccc"] > CCC_BAR and dep["loa_half_deg"] < LOA_BAR}
            out.append(row)

    collapses = 0
    print(f"{'slot':<38} {'reader':<7} {'pooled CCC/LoA':>16} {'largest-dataset CCC/LoA':>24}  flag")
    for row in out:
        if not row["pooled"] or not row["per_dataset"]:
            continue
        biggest = max(row["per_dataset"].values(), key=lambda s: s["n"])
        p, b = row["pooled"], biggest
        flag = ""
        dep = row.get("deployed")
        if dep and dep["clears_bar"] and b["ccc"] < CCC_BAR:
            flag = "CCC COLLAPSES within-dataset"
            collapses += 1
        slot = f'{row["metric"]}/{row["view"]}'
        print(f'{slot:<38} {row["reader"]:<7} '
              f'{p["ccc"]:>+7.2f}/{p["loa_half_deg"]:<7.1f} '
              f'{b["ccc"]:>+14.2f}/{b["loa_half_deg"]:<7.1f}  {flag}')

    print(f"\n{collapses} deploy-bar-clearing slot/reader pairs lose CCC within a single dataset.")
    print("LoA is range-independent and is the statistic to lead with; pooled CCC measures "
          "cross-population discrimination and must be labelled as such.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
