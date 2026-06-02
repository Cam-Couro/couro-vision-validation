# Rear-View Validation Dataset Survey

**Date:** 2026-05-28
**Goal:** Find a multi-view dataset with **rear-camera coverage + mocap GT** that we can pull tonight, no human-approved registration forms, to close Couro's rear-view validation gap.

## Summary

Two datasets are fully open and were actually downloaded tonight:

1. **CMU Panoptic Studio** — 31 HD cameras spanning full 360° dome (rear coverage confirmed), direct HTTP, no auth. Sample calibration pulled.
2. **MPI-INF-3DHP** — 14 cameras incl. wall-mounted rear angles, wget script with no registration. Metadata + S1/Seq1 camera calibration pulled.

Everything else either needs a human-approved form (Human3.6M, Fit3D, TotalCapture, HumanEva) or is offline (Berkeley MHAD server returns Cloudflare 1001 / no DNS).

## Full evaluation table

| # | Dataset | Rear cam? | Mocap GT? | License | Download method | Athletic? | Tonight? |
|---|---|---|---|---|---|---|---|
| 1 | Fit3D | Multi-view but exact rear coverage not stated on public pages | Vicon 3D skeletons | IMAR non-commercial, custom academic | Account signup required; download page is behind login wall (auto-signup may or may not auto-approve — unconfirmed) | Yes, 37 fitness exercises | **Risky** — login wall, unclear if instant |
| 2 | **CMU Panoptic Studio** | **Yes — 31 HD + 480 VGA cameras full 360° dome** | Triangulated 3D keypoints (no Vicon, but multi-view triangulation GT) | Non-commercial research only | **Direct HTTP, no auth.** `getData.sh` from CMU repo uses `curl/wget` against `http://domedb.perception.cs.cmu.edu/webdata/dataset/{seq}/...`. SNU mirror available. | Mixed — Range of Motion, pose, "Pose1" sequences, range of activities incl. dance/sports clips | **YES — actually pulled** |
| 3 | **MPI-INF-3DHP** | **Yes — 14 cameras incl. wall-mounted side/rear ("other_angled_cameras.zip")** | Multi-view markerless + commercial mocap | MPI non-commercial research | **Direct HTTP via wget, no registration.** `get_dataset.sh` pulls from `https://vcai.mpi-inf.mpg.de/3dhp-dataset/...`. Just flip `ready_to_download=1` in `conf.ig`. | Yes — sequence 2 has an explicit **Sports** activity (boxing, tennis, golf, soccer) and Exercise (lunges/pushups/stretch). | **YES — actually pulled** |
| 4 | Human3.6M | 4 cams at 0°/90°/180°/270° → 180° is true rear | Vicon mocap | IMAR academic, no commercial use | Requires account; signup form behind login. Unclear if auto-approved tonight. Signup page existed but tested POST returned the standard 404 page — likely behind a captcha/manual approval. | Daily activities (walking, sitting, posing); some sports-adjacent motions | **No** — gated |
| 5 | 3DPW | Outdoor handheld, mostly side/front, no controlled rear | IMU GT (no Vicon) | MPI license | Login wall + license-accept form | Mostly walking outdoors — low athletic value for our use case | No |
| 6 | TotalCapture | 8 cams 360° → includes rear | Vicon + IMU | Registration with human approval required | Explicit "the dataset requires registration" — needs email | Walking/acting/freestyle (some athletic) | No |
| 7 | Berkeley MHAD | 12 cams 360° (would include rear) | Mocap + IMU | Public | **Site dead** — `tele-immersion.citris-uc.org` returns Cloudflare 1001 / DNS failure | Calisthenics, jumping | No (dead host) |
| 8 | AIST++ | n/a | n/a | n/a | Owned by other agent | n/a | Skip |
| 9 | AMASS | n/a | n/a | n/a | Owned by other agent | n/a | Skip |
| 10 | **CMU MoCap (mocap.cs.cmu.edu)** | n/a (no video — just mocap) | Vicon mocap, ASF/AMC/C3D | **"Data may be copied, modified, or redistributed"** — most permissive option available | **Direct HTTP, no auth.** All-in-one zips linked from the FAQ page (`allasfamc.zip`, `allc3d_*.zip`, `allmpg/`, `allavi.zip`). | Yes — sports motions: basketball, boxing, golf, baseball, soccer, dance | **YES** — usable as synthetic rear-view source by projecting MoCap into a virtual rear camera |
| 11 | HumanEva | 7 cams (4 grayscale + 3 color) on cube around subject — includes back view | Vicon mocap | MPI non-commercial | Signup/login required; standard MPI signup form | Walking, jogging, gestures (limited athletic) | No |
| 12 | Other (papers-with-code etc.) | — | — | — | papers-with-code 3D pose dataset listing redirected to Hugging Face Papers (no actionable results) | — | — |

## Top 2 recommendations

### Recommendation #1: CMU Panoptic Studio  (highest fidelity, biggest engineering lift)
- **Why:** Full 360° dome means you can pick any "rear-quartile" HD camera (e.g. yaw ≈ ±150–180°) and pair it with the same subject's existing front/side captures. We computed world positions from `171204_pose1` calibration and confirmed HD cameras span the full yaw range (−175° through +178°).
- **Limits:** No Vicon GT — pose GT is multi-view triangulated 3D keypoints. Athletic motion content is mixed (pose, dance, social interaction); not pure sports. Single HD video is ~3 GB so we should pull 1 sequence + 1 rear-quartile camera, not bulk.
- **Next steps:** Use `panoptic-toolbox/scripts/getData.sh <seq> 0 1` to pull one HD camera + calibration + keypoint GT. Target `171204_pose1` or any sequence in their "Range of Motion" / dance categories for the closest analog to sports motion.

### Recommendation #2: MPI-INF-3DHP  (best athletic-content fit, smallest dataset)
- **Why:** Explicit **Sports** activity bucket (boxing, tennis, golf, soccer kicks). 14 cameras with the "other_angled_cameras.zip" subset containing wall-mounted angles that include behind-the-subject views. Pure wget, just edit `ready_to_download=1`. The whole training set is ~25 GB so a single subject + sequence is feasible tonight.
- **Limits:** Studio green-screen environment, not in-the-field. Non-commercial license.
- **Next steps:** Edit `data/mpi_inf_3dhp_meta/mpi_inf_3dhp/conf.ig` to set `subjects=(1)`, `download_extra_wall_cameras=1`, `ready_to_download=1`, then run `./get_dataset.sh`. Estimated single-subject pull is ~3–4 GB. Activity A6 (Sports) is in Sequence 2.

### Honorable mention: CMU MoCap (mocap.cs.cmu.edu)
- **Why mention:** Most permissive license of anything found ("may be copied, modified, or redistributed"). Pure mocap (no video) means we can synthesize a clean rear-view by projecting joint positions into a virtual camera placed behind the subject. Great as a **commercial-safe** unit-test layer even if we don't use the video datasets in production.

## Where downloaded samples landed

```
/Users/cameronvan/Documents/Claude/Projects/Couro/research-agent/multiview-validation/data/
├── cmu_panoptic_rear_sample/
│   └── calibration_171204_pose1.json        (252 KB — full 520-camera calibration; HD camera yaws span −175° to +178°)
├── mpi_inf_3dhp_meta/
│   └── mpi_inf_3dhp/
│       ├── README.txt
│       ├── license.txt
│       ├── conf.ig                          (edit ready_to_download=1 + download_extra_wall_cameras=1 to pull rear cams)
│       ├── get_dataset.sh                   (direct wget against vcai.mpi-inf.mpg.de — no auth)
│       ├── get_testset.sh
│       └── util/mpii_get_camera_set.m       (confirms 14 cams: 0–10 chest/knee/angled, 11–13 ceiling)
└── mpi_inf_3dhp_rear_sample/
    └── S1/Seq1/
        └── camera.calibration               (4.5 KB — Skeletool V1.0 format, all 14 camera intrinsics+extrinsics)
```

## Honest blockers

- **Human3.6M, Fit3D, HumanEva, TotalCapture, 3DPW** all require an account behind a form. Even if signup is "instant" in the click-through sense, none of these advertise instant-grant terms, and at least TotalCapture explicitly states human approval. Skipping per the no-outreach constraint.
- **Berkeley MHAD** host is down (Cloudflare 1001 error). No alternative mirror found in 10 min of probing.
- **License caveat for production:** All five accessible datasets (CMU Panoptic, MPI-INF-3DHP, CMU MoCap, plus the gated ones) are **non-commercial research only** except CMU MoCap, which permits redistribution. For validation reports that stay internal / pre-product this is fine; if any rear-view validation artifact ships in a customer-visible form, we should either (a) keep it inside the research/non-commercial scope, (b) lean on CMU MoCap-based synthetic projections, or (c) revisit with a commercial-license inquiry to MPI / IMAR.
- **Path forward tonight without further blockers:** flip the MPI-INF-3DHP `ready_to_download` flag and pull Subject 1, Sequence 2 with `download_extra_wall_cameras=1` to get rear cameras + Sports activity in one go. That gives us a real rear-view smoke test against the existing DWPose-L → Halpe-26 pipeline by tomorrow morning.
