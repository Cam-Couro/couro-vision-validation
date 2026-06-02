# Commercial-Clean Dataset Playbook — 2026-06-01

**Replaces:** the earlier "dataset expansion playbook" (deleted) that listed OpenCap / Motion-X / EMDB — all of which are non-commercial research-only and would put Couro in license violation if used for commercial training or customer-facing claims.

**Constraint:** Couro is a commercial company. Every dataset / model used to train or validate the shipped product must be either (a) explicitly commercial-clean, (b) commercially licensed for cost, or (c) self-collected. Research-only data can be used for **internal research validation only** and cannot back numbers in sales decks, marketing, or model weights that ship.

## The commercial-clean dataset universe (short list)

| Source | License | What it is | Commercial-OK? |
| --- | --- | --- | --- |
| **CMU MoCap** ([mocap.cs.cmu.edu](http://mocap.cs.cmu.edu)) | "may be copied, modified, or redistributed" | 2,500+ motion sequences, ASF/AMC + C3D format. Sports: basketball, baseball, soccer, golf, boxing, dance, walking, **running**. No video. | ✅ **Yes** — most permissive option |
| **SMPL-Body** (subset of SMPL) | **CC-BY 4.0** | Mesh + skeleton + pose blendshapes + dynamic blendshapes. **No shape betas** — one body shape only. | ✅ **Yes** — what your team's PIPELINE_PLAN.md picked |
| **Full SMPL / SMPL-X via Meshcapade** | Commercial license ($) | Includes shape betas (body diversity) | ✅ Yes, paid — `sales@meshcapade.com` |
| **STAR** body model | Variable (check per-release) | SMPL alternative from a different group | ⚠️ Verify per-release |
| **Adobe Mixamo** | Commercial-OK for avatars | Game-quality rigged characters | ⚠️ Not biomech-grade — animation, not realistic mechanics |
| **Self-collected Couro data** | Couro owns it | Phone-cam captures + your own GT | ✅ Yes, forever |
| Everything else surveyed (OpenCap / MPI-INF-3DHP / Fukuchi / Motion-X / EMDB / Human3.6M / AIST++ / BEDLAM / 3DPW / AMASS) | Non-commercial only | Various | ❌ **Internal research only** |

That's it. The commercial-clean universe is much smaller than the research universe.

## The actual commercial-clean training pipeline

Your team already designed this in [PIPELINE_PLAN.md](../synthetic_amass/PIPELINE_PLAN.md). It's just been called "the AMASS pipeline" colloquially — the commercial-clean version drops AMASS and substitutes CMU MoCap as the motion source.

```
                          COMMERCIAL-CLEAN STACK
                          ======================
   CMU MoCap (.c3d/.asf)          SMPL-Body (CC-BY 4.0)
   Motion source            +     Body model               ─→  Synthetic video
   2,500+ sequences               One shape, full pose          + 3D angle GT
   "may be copied,                (no betas; trade-off)         at arbitrary cameras
    modified,                                                   (any front, rear,
    redistributed"                                              oblique angle)
        │
        ▼
   Existing render_proof_frame.py
   + Halpe-26 mapping
   + GT angle extractor
   + virtual camera projection
   (all already built — Phase 0 of PIPELINE_PLAN)
```

**No academic registration required** — both CMU MoCap and SMPL-Body are direct downloads under their respective licenses. The previously documented "Phase 1 blocker" only applies if you want SMPL+H (hands) or shape diversity, both of which require either MPG academic registration OR Meshcapade commercial licensing.

## Three concrete actions, in order

### 1. Stand up the commercial-clean synthetic pipeline (this week)

a. Download CMU MoCap subset for the motion classes you need:
```bash
# Top-level zip (allasfamc.zip ~2.5 GB)
curl -L -o cmu_mocap_all.zip http://mocap.cs.cmu.edu/allasfamc.zip
unzip cmu_mocap_all.zip
# Subject index: http://mocap.cs.cmu.edu/search.php
# Subjects to pull first:
#   12-14, 16, 35, 91 (walking, running, jumping)
#   60, 75 (sports - basketball / boxing)
#   141, 143 (martial arts / sports)
```

b. Download SMPL-Body under CC-BY 4.0:
- Go to https://smpl.is.tue.mpg.de/ → register → accept CC-BY-Body terms (not the research-only Model terms)
- The CC-BY page is at https://smpl.is.tue.mpg.de/license.html (not modellicense.html which is research-only)
- Pull `basicmodel_neutral_smpl_body.pkl` or equivalent

c. Adapt `synthetic_amass/synthetic_skeleton.py` to ingest CMU MoCap ASF/AMC instead of (the academic-only) AMASS .npz. This is a different loader but the rest of the pipeline (Halpe-26 mapping, camera projection, GT angles) is unchanged.

d. Re-run `render_proof_frame.py` with a CMU MoCap input → verify proof artifacts match the existing AMASS-based proof.

**Estimated effort:** 3-5 engineering days (per the existing PIPELINE_PLAN Phase 1-2). Saad's branch, your review per the no-push-without-Saad-review rule.

### 2. Self-collect commercial-clean validation footage (this month)

The one gap synthetic can't close: **real video of athletes running rear+front with single-camera capture**, owned by Couro. Outline:

- 8-12 athletes (~mixed gender, mixed skill, mixed body type)
- Treadmill or overground running at 3 speeds (2.5, 3.5, 4.5 m/s — matches Fukuchi conventions for later comparison)
- 2 phone camera positions per session: rear-oblique (~−159°, matching your MPI-INF-3DHP cam7 setup) + front-center
- Soft GT: existing trusted side-camera RTMPose-x pipeline (your headline 0.83-0.94 CCC slots), measured simultaneously
- Bonus if budgeted: 1-2 sessions with a rented mocap setup (Vicon/Qualisys day-rate at a partner lab like Cal/AUSL) for gold GT validation

Output: Couro-owned dataset that backs the rear+front running CCC numbers in marketing / investor decks, with no license risk.

### 3. For shape diversity (later, only if synthetic-only proves insufficient)

If synthetic-only training plateaus and you need body-shape variation:
- Contact `sales@meshcapade.com` for SMPL/SMPL-X commercial license pricing
- Or: continue training on SMPL-Body single-shape + augment with self-collected diversity

## What this enables

| Goal | Path | Status |
| --- | --- | --- |
| Synthetic rear/front-view training data | CMU MoCap + SMPL-Body render | Unblocked once CMU MoCap is downloaded |
| Commercial-clean rear-view validation | Render Fukuchi-class running motion + project | Unblocked same path |
| Real running rear video, Couro-owned | Self-collect | Multi-week / month |
| Shape diversity in synthetic | Meshcapade SMPL commercial license | Cost question; not blocking |
| Customer-facing CCC claims on rear-view running | Self-collect → run pipeline → publish | The actual product blocker |

## What this still doesn't fix

- **Layer-3 ridge ceiling** (the 5-attempt failure documented in `data/lever_c_3d_ridge/REPORT.md`). More commercial-clean data may or may not help — Lever-(c) result on OpenCap showed +0.11 CCC was the best single-slot lift, and adding shape diversity from Meshcapade is unlikely to dramatically change this. Layer-3 progress is bottlenecked by *cohort size*, not just commercial-clean-ness.
- **The headline CCC 0.94 hip-adduction-side-left** number — measured on OpenCap data. Internally honest; cannot ship to customers as a quantitative claim. Need self-collected equivalent before that number appears in any sales material.

## What to do today

If Cameron has 30 min: run the CMU MoCap download to a local dir. The full set is ~2.5 GB. That single step unblocks the entire commercial-clean synthetic pipeline.

```bash
mkdir -p ~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/cmu_mocap
cd ~/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/cmu_mocap
curl -L -O http://mocap.cs.cmu.edu/allasfamc.zip
unzip allasfamc.zip
# Then check what's in there:
ls all_asfamc/subjects | head
```

After that, the SMPL-Body download (a few hundred MB) is the second step. Both together are the commercial-clean unblock.

## Apology

The previous playbook ignored Couro's own documented license constraint and would have gotten you into a license violation if followed. Sorry — that was my mistake.
