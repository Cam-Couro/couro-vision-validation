# Source-media timing and frame-paired OpenCap training

## What changed

The OpenCap DWPose writer formerly assigned decoded ordinal / FPS to every
keypoint row. That discards media-clock gaps. The regression fixture has 60 Hz
AVI metadata, 170 declared frames, and 166 decoded frames with PTS
`0, 5, 6, ..., 169`. Ordinal time is 66.667 ms early from the second decoded
frame onward. The fixed writer captures `CAP_PROP_PTS` immediately after each
successful decode and carries it through the corresponding inference batch.

OpenCV reports this PTS in FPS time-base units. `source_pts` preserves that
reported value, and `timestamp_ms = source_pts / fps * 1000`, without resetting
the origin or rounding. This is decoder-reported media time, not a guarantee
that native packet timestamps were available or that a camera clock matches
motion capture. See [OpenCV VideoCapture properties](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html).

The `*_syncdWithMocap.avi` suffix is not synchronization evidence. The dataset
maintainer says the videos are unsynchronized in this
[OpenCap forum response](https://simtk.org/plugins/phpBB/viewtopicPhpbb.php?f=2385&t=25040&p=0&start=0&view=).

## Cache contract

New OpenCap outputs use `couro-dwpose-halpe26-v2-source-pts`. Each decoded
ordinal keeps its own `source_pts`, `timestamp_ms`, and `timestamp_status`.
Top-level `timing` records the timing schema (`source-media-pts-v1`),
`source_media` clock, OpenCV version, backend, property, FPS time base, source
video SHA-256, validity, issues with frame indices, and unvalidated mocap sync.

- Gaps and nonzero/negative starting PTS are preserved
- Duplicate, nonmonotonic, missing/nonfinite, unsupported, or empty timelines
  are explicitly invalid; no ordinal, POS_MSEC, guessed-FPS, sort, or dedup
  repair is applied
- Unsupported/missing values remain null; diagnostic caches may still hold
  keypoints but cannot be used as valid timed input
- Existing output paths, regardless of size/schema, are refused and untouched
- Legacy caches are labeled `legacy_unverified_timestamp_ms`; their old values
  are not relabeled as PTS or silently reinterpreted
- `load_couro_output` still reads legacy caches with their previous behavior
  and now exposes `timestamp_source`. It refuses invalid new PTS caches instead
  of invoking its legacy missing-time fallback. Declaring the new source-PTS
  schema requires timing provenance even when the timing block is missing;
  it cannot downgrade to legacy behavior

## Explicit video-to-mocap mapping

The three frame-paired OpenCap builders in `learned_layer2_real_gt.py`,
`learned_layer2_combined.py`, and `learned_layer2_persource_mirrorflip.py` now
require `<cache-directory>/sync/<keypoint-cache-stem>.sync.json` for each new
PTS cache. The dedicated subdirectory keeps sidecars out of existing
`*.json` clip discovery. Their
shared callers, including combined/per-source/ROM-aware training, inherit the
gate. A `TimingContractError` propagates rather than silently dropping a clip
into a different training cohort.

The mapping is affine: `mocap_seconds = scale * source_media_seconds +
offset_seconds`. Its schema is:

```json
{
  "schema_version": "source-media-to-mocap-v1",
  "source_clock": "source_media_seconds",
  "target_clock": "mocap_seconds",
  "keypoints_sha256": "EXACT_CACHE_SHA256",
  "mocap_sha256": "EXACT_MOT_SHA256",
  "scale": null,
  "offset_seconds": null,
  "valid_source_interval_seconds": [null, null],
  "validation": {
    "status": "needs_validation",
    "method": "",
    "evidence": "",
    "validated_by": "",
    "uncertainty_seconds": null,
    "accepted_uncertainty_seconds": null,
    "calibration_split": {
      "kind": "independent_sync_evidence",
      "description": ""
    }
  }
}
```

This deliberately unusable template must be populated from actual reviewed
synchronization evidence. The status becomes `validated` only after that
review. The code checks supplied provenance claims; it does not establish
independent scientific truth from a checkbox or evidence string. Do not infer
identity mapping from a filename, valid PTS, or a visually plausible overlay.

Use `independent_sync_evidence` for hardware or event synchronization that is
independent of the evaluated pose/angle targets. It may concern the same
recording. Describe the evidence and independence explicitly. Fitting a lag to
maximize the evaluated pose-to-mocap correlation does not qualify.

Alternatively, use `held_out_recordings` and provide nonempty,
disjoint `fit_recording_ids` and `evaluation_recording_ids` inside
`calibration_split`. Recording IDs are `{subject}_{trial}`, excluding camera;
the current recording must appear in the evaluation list. This records the
split, not evidence that a fitted clock mapping transfers across recordings.

The gate also requires finite positive scale; finite offset; nonnegative
uncertainty within the explicitly accepted tolerance; source coverage for all
frames; and exact cache/mocap hashes. Mapped frames must lie within the actual
finite, strictly increasing `.mot` time axis. No extrapolation or endpoint
clamping is allowed. Existing angle interpolation keeps out-of-support finite
angle values as NaN rather than inventing labels.

## Migration and limits

1. Keep original caches and historical results unchanged
2. Recapture PTS from the exact source media into a new output directory. Do
   not rewrite cached ordinal times using FPS or fill missing video frames
3. Obtain/review synchronization evidence and create the bound sidecar. No
   validated sidecar is supplied by this patch
4. Point the frame-paired workflow at the new cache directory only after the
   timing, mapping, uncertainty, and calibration/evaluation split are reviewed
5. Rerun training/evaluation separately under explicit authorization; this
   patch provides no new model, score, accuracy, or tracking-quality claim

This is an OpenCap source-clock/frame-pairing repair. It does not migrate
ASPset/MPI inference, historical derived-reader caches, generic historical
validation scripts, temporal-window sampling, bbox/frame correspondence,
camera calibration, or camera rotation math. Those need separate work. In
particular, native PTS does not itself make a temporal CNN gap-aware.

## Verification

Run `python -m pytest -q tests` with existing NumPy, pytest, and OpenCV. The real
AVI round-trip also uses an existing FFmpeg executable; it skips if that
prerequisite is absent and never installs anything. Model frameworks are not
required: writer tests use explicit fake decode/session adapters, and builder
regressions execute the actual loading function AST without importing Torch.
No model training or model inference runs in these tests.
