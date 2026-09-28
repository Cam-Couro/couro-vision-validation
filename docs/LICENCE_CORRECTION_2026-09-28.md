# Licence correction: VideoPose3D is CC BY-NC 4.0, not Apache 2.0 (2026-09-28)

**Status:** correction notice. Metric values in this repository are unchanged. Only licence labels were changed, and
research-only status notes were added.

## 1. The error

Several files in this repository described FAIR's **VideoPose3D** (Pavllo et al., CVPR 2019) as
"Apache 2.0", "commercial-clean", "licence-clean" or "commercially permissive". VideoPose3D is the 2D→3D pose lifter
used by:

- the view-aware blend (`harness/view_aware_blend.py`, `data/layer2_view_aware_blend/`), and
- the v44 reader "VideoPose3D L2" (`harness/learned_layer2_videopose3d.py`,
  `harness/layer3_retrain_on_videopose3d_l2.py`, `data/videopose3d_layer2/`, `results/deploy_ready_models_v44_videopose3d.json`).

Both load the pretrained checkpoint `pretrained_h36m_detectron_coco.bin`.

**That label was wrong.** The correct licence is **CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International)**, which
does not permit commercial use. The pretrained weights are also trained on **Human3.6M**, whose licence is
academic-only. So any result that depends on VideoPose3D is **research-only** and is not a basis for commercial product
use or commercial claims.

## 2. Evidence

| Source | What it says |
|---|---|
| https://github.com/facebookresearch/VideoPose3D/blob/main/LICENSE | First line: **"Attribution-NonCommercial 4.0 International"**. The file has one commit, `8229764` ("initial commit", 2018-11-02), and has not changed since |
| https://github.com/facebookresearch/VideoPose3D#license (README, "License" section) | *"This work is licensed under CC BY-NC. See LICENSE for details. Third-party datasets are subject to their respective licenses."* |
| Human3.6M (http://vision.imar.ro/human3.6m/) | The dataset the checkpoint was trained on (`h36m` in the file name). Its licence is academic-only |

The two VideoPose3D rows were re-checked on 2026-09-28 against the live repository: the raw `LICENSE`, the README, and
the GitHub commit history of `LICENSE`.

## 3. What changed in this repository

- **Licence strings corrected** from `Apache-2.0` / `Apache 2.0` / "commercial-clean" to
  `CC-BY-NC-4.0 (non-commercial; weights trained on Human3.6M, academic-only)` or an equivalent short form. This covers
  the JSON fields `model_license`, `l2_model_license`, `v44_l2_model_license`, `license` and `blend_config.lifter`.
  JSON stays valid. Only string values were edited, and no key was removed or renamed.
- **Harness sources corrected**, so a re-run does not re-emit the wrong label.
- **Five HTML validation documents from May–June 2026** now carry a correction banner at the top, plus inline
  corrections where the label appeared.
- **Three v49 slots are marked research-only.** `data/v49_selective_oracle/consolidated_metrics_v49.json` gains one
  additional key, `"licence_status": "research-only (CC BY-NC VideoPose3D reader)"`, on those rows. No existing key or
  value changed.
- The `"license": "Apache 2.0"` entries attached to the **OpenCap LabValidation dataset** (`training_dataset` blocks)
  were **not** changed by this correction. They are about a different artefact, not VideoPose3D, and are out of scope
  here.

## 4. The three affected v49 slots: RESEARCH-ONLY

These three slots are served in v49 by the v44 VideoPose3D reader. They are **research-only (not for commercial
product use)** until a licence-clean reader replaces them.

| Slot | v49 (VideoPose3D reader) | Licence-clean fallback (pre-v44 pick, from the v45 REPORT table) | Tier with fallback |
|---|---|---|---|
| hip_adduction_r / front_oblique_right | CCC 0.960, LoA ±8.51°, Good | v30: CCC 0.889, LoA ±13.80° | Moderate |
| hip_adduction_r / side_left | CCC 0.946, LoA ±9.32°, Good | ensemble v17+v23+v26+v31: CCC 0.944, LoA ±8.71° | Good |
| hip_adduction_r / front_center | CCC 0.897, LoA ±11.95°, Moderate | v30: CCC 0.793, LoA ±18.00° | Poor |

The fallback numbers are the v43 picks in the `data/v45_selective_oracle/REPORT.md` section 3 table. The tiers follow
`harness/biomech_validity_stats.py::classify`.

## 5. Scoreboard on a licence-clean basis: PROVISIONAL

| | Good | Moderate | Poor |
|---|---:|---:|---:|
| v49 as published (`consolidated_metrics_v49.json`) | 14 | 5 | 4 |
| Licence-clean, provisional (three slots swapped to their fallbacks) | **13** | **5** | **5** |

**13 / 5 / 5 is provisional.** It holds only if the fallback readers and the other deployed readers (v23, v26, v30, v31
and v33) do not themselves depend on VideoPose3D. That lineage has **not yet been audited**. One known open point:

- v17 adopted the VideoPose3D view-aware blend for `lumbar_extension / front_oblique_right`. Every deploy table from v17
  to v49 still carries that slot's `blend_metadata.blend_config.lifter` entry, including the v33 reader, which serves
  that slot in v49 (CCC 0.804, Good).
- Whether v33's fit actually uses VideoPose3D-derived features is **not verified**. If it does, that slot is also
  research-only, and the licence-clean Good count falls further.

The same caveat applies to results that used the view-aware blend in the May 2026 documents. Those include the pooled
Layer 2 |r| 0.581 and the v17 lumbar front-oblique slot, and both are research-only.

## 6. What this does not change

- Measurement-agreement statistics (CCC, Bland–Altman LoA) are unchanged. The fix changes what those numbers may be
  used for, not what was measured.
- These are measurement-agreement statistics against motion capture, not clinical statements about any person.

## 7. Next step

Replace the VideoPose3D reader with a licence-clean 3D lifter. The lifter would be written by Couro and trained only on
permissively licensed marker data. It would then be re-scored on the v49 reference before any of these three slots
return to the commercial scoreboard. Until then they stay research-only.
