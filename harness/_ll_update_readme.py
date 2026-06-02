"""Update README.md based on v25 selective oracle outcomes.

If the floor lifted to ≥0.80 across one or more of Cameron's bottom-5
slots, the headline range gets bumped and v25 is added as the
recommended deploy. Otherwise we keep v22 as the recommended deploy,
add v24 as a learning entry, and document why the floor didn't lift.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH: Final[Path] = REPO_ROOT / "README.md"
V25_PATH: Final[Path] = (
    REPO_ROOT / "results" / "deploy_ready_models_v25_selective.json"
)
V25_PICKS: Final[Path] = (
    REPO_ROOT / "data" / "v25_selective_oracle" / "per_slot_picks_v25.json"
)


def _fmt(v: object, prec: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


def main() -> None:
    v25 = json.loads(V25_PATH.read_text())
    picks = json.loads(V25_PICKS.read_text())

    tier_counts = v25["tier_counts"]
    reader_dist = v25["reader_distribution"]
    floor_lifted = v25.get("floor_lifted_to_0_80", False)

    good_slots = [p for p in picks["picks"] if p["tier"] == "Good"]
    good_slots_sorted = sorted(
        good_slots,
        key=lambda p: -(p["ccc"] if p["ccc"] is not None
                        and not (
                            isinstance(p["ccc"], float)
                            and math.isnan(p["ccc"])
                        ) else -1.0),
    )

    n_good = len(good_slots_sorted)
    if good_slots_sorted:
        min_ccc = min(
            p["ccc"] for p in good_slots_sorted
            if p["ccc"] is not None
            and not (
                isinstance(p["ccc"], float)
                and math.isnan(p["ccc"])
            )
        )
        max_ccc = max(
            p["ccc"] for p in good_slots_sorted
            if p["ccc"] is not None
            and not (
                isinstance(p["ccc"], float)
                and math.isnan(p["ccc"])
            )
        )
    else:
        min_ccc, max_ccc = 0.0, 0.0

    headline_range = f"CCC {min_ccc:.2f}–{max_ccc:.2f}"

    # Compose the new "Current state" block.
    if floor_lifted:
        # Use v25 as recommended deploy.
        rec_deploy = (
            "results/deploy_ready_models_v25_selective.json"
        )
        rec_label = (
            f"**Recommended:** per-slot oracle-best across "
            f"v17/v18/v20/v23/v24 (**{n_good} Good slots**, +{n_good - 8} "
            "vs v22). Floor lift achieved."
        )
        v22_label = (
            "Per-slot oracle-best across v17/v18/v20/v23 (8 Good slots; "
            "v22 prior recommendation)"
        )
    else:
        rec_deploy = (
            "results/deploy_ready_models_v22_selective.json"
        )
        rec_label = (
            "**Recommended:** per-slot oracle-best across v17/v18/v20/v23 "
            f"(8 Good slots). v25 ({n_good} Good slots) adds the v24 LL "
            "reader but the validated CCC floor stayed under 0.80. "
            "Honest recommendation: present the validated Good slots as "
            "**Tier 1 (CCC ≥ 0.79)** vs **supplementary (CCC 0.60–0.78)** "
            "rather than headline a higher floor."
        )
        v22_label = (
            "**Recommended:** per-slot oracle-best across v17/v18/v20/v23 "
            "(**8 Good slots**, +1 vs v21)"
        )

    state_lines = []
    state_lines.append(
        f"## Current state (as of 2026-06-02 / v25 selective oracle)"
    )
    state_lines.append("")
    state_lines.append(
        f"**{n_good} validated deploy slots** clearing the standard "
        "biomechanics validity bar (Lin's CCC > 0.60 AND Bland-Altman 95% "
        "LoA half-width < ±10°):"
    )
    state_lines.append("")
    state_lines.append("| Slot | CCC | LoA half | Reader |")
    state_lines.append("|---|---:|---:|---|")
    reader_desc = {
        "v17": "v17 hand-engineered",
        "v18": "v18 EE2 learned Layer 2",
        "v20": "v20 ROM-aware learned Layer 2",
        "v23": "v23 combined-cohort learned Layer 2",
        "v24": "v24 combined-cohort + ROM-aware learned Layer 2 (LL)",
    }
    for p in good_slots_sorted:
        slot_disp = p["slot"].replace("|", " / ")
        ccc_str = _fmt(p["ccc"])
        loa_str = f"±{_fmt(p['loa_half'])}°"
        state_lines.append(
            f"| {slot_disp} | {ccc_str} | {loa_str} | "
            f"{reader_desc.get(p['reader'], p['reader'])} |"
        )
    state_lines.append("")
    state_lines.append(
        f"**Range: {headline_range}, single phone camera, no calibration.** "
        f"v25 selective oracle ({n_good} Good slots). "
        f"Reader distribution: v17={reader_dist.get('v17', 0)}, "
        f"v18={reader_dist.get('v18', 0)}, v20={reader_dist.get('v20', 0)}, "
        f"v23={reader_dist.get('v23', 0)}, v24={reader_dist.get('v24', 0)}."
    )
    state_lines.append("")
    if not floor_lifted:
        state_lines.append(
            "> **Floor lift status (LL build):** the 0.80 CCC floor was "
            "**not cleared**. Of the 5 sub-0.80 slots Cameron called out "
            "in the LL brief, the closest were "
            "lumbar_extension/front_oblique_right and "
            "lumbar_extension/side_right — see "
            "`data/v25_selective_oracle/REPORT.md` for the slot-by-slot "
            "verdict and the proposed Tier 1 / supplementary marketing "
            "restructure."
        )
        state_lines.append("")
    else:
        state_lines.append(
            "> **Floor lift status (LL build):** the 0.80 CCC floor was "
            "**cleared** in at least one slot. See "
            "`data/v25_selective_oracle/REPORT.md`."
        )
        state_lines.append("")

    # Read existing README and splice the new section in.
    text = README_PATH.read_text()
    # Find the start of "## Current state" and the next "## " header.
    start_marker = "## Current state"
    next_marker = "## Headline build cycles"
    start = text.find(start_marker)
    end = text.find(next_marker)
    if start < 0 or end < 0:
        print("WARN: could not find current state markers in README")
        return
    new_section = "\n".join(state_lines) + "\n"
    new_text = text[:start] + new_section + text[end:]

    # Update the deploy table candidates section (replace recommendation).
    # Add v24 + v25 rows.
    deploy_marker = (
        "| `results/deploy_ready_models_v22_selective.json` | "
        "**Recommended:**"
    )
    if deploy_marker in new_text:
        # Replace v22 line entirely.
        before, _, rest = new_text.partition(deploy_marker)
        # Find end of v22 line.
        v22_end = rest.find("\n")
        new_text = before + (
            f"| `results/deploy_ready_models_v22_selective.json` | "
            f"{v22_label} |\n"
            "| `results/deploy_ready_models_v24_combined_rom_aware.json` | "
            "v17 base + v24 LL combined-cohort + ROM-aware learned-L2 "
            "ridge re-fit (Agent LL build) |\n"
            f"| `results/deploy_ready_models_v25_selective.json` | "
            f"{rec_label} |"
        ) + rest[v22_end:]

    # Append a v25 build cycle entry.
    build_cycle_marker = "## Headline build cycles"
    if build_cycle_marker in new_text:
        # Find the existing 2026-06-02 cycle line and add LL note before
        # "## Deploy table candidates".
        deploy_table_marker = "## Deploy table candidates"
        bc_start = new_text.find(build_cycle_marker)
        dt_start = new_text.find(deploy_table_marker)
        if bc_start > 0 and dt_start > bc_start:
            insertion = (
                "- **2026-06-02 LL build (this work):** LL combined-cohort "
                "+ ROM-aware learned Layer 2 (Agent LL) → v24 Phase B Layer "
                "3 retrain; v25 selective oracle adds v24 to the reader "
                f"pool. Verdict: {n_good} Good slots, floor "
                f"{'lifted to ≥0.80' if floor_lifted else 'NOT lifted to 0.80'}"
                f". See `data/v25_selective_oracle/REPORT.md`.\n\n"
            )
            new_text = (
                new_text[:dt_start] + insertion + new_text[dt_start:]
            )

    README_PATH.write_text(new_text)
    print(f"updated {README_PATH}")
    print(
        f"  floor_lifted={floor_lifted} n_good={n_good} "
        f"range={headline_range} recommended_deploy={rec_deploy}"
    )


if __name__ == "__main__":
    main()
