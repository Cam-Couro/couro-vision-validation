"""Deterministic source clock and explicit sync contract, without models."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.frame_timing import SourceMediaClock, TimingContractError, cache_timing_label, require_mocap_times, sha256_file
from harness.couro_keypoints import load_couro_output
from test_frame_timing_regressions import legacy_cache, test_real_builder_uses_explicit_mapping_not_media_clock as make_valid_fixture


@pytest.fixture
def paired(tmp_path):
    make_valid_fixture(tmp_path, "learned_layer2_real_gt.py", "build_clip_dataset", "load_opencap_clip")
    kp = tmp_path / "subject1_DJ1_Cam0.json"
    mot = tmp_path / "lab/subject1/OpenSimData/Mocap/IK/DJ1.mot"
    sidecar = kp.parent / "sync" / f"{kp.stem}.sync.json"
    return kp, mot, sidecar


def mutate(path, fn):
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_legacy_remains_readable_and_labeled_without_mutation(tmp_path):
    kp, mot = legacy_cache(tmp_path)
    before = kp.read_bytes()
    assert cache_timing_label(json.loads(before)) == "legacy_unverified_timestamp_ms"
    assert load_couro_output(kp).time == pytest.approx([0, 1 / 60, 2 / 60])
    with pytest.raises(TimingContractError, match="legacy_unverified"):
        require_mocap_times(kp, mot, [0.0, 5.0])
    assert kp.read_bytes() == before


@pytest.mark.parametrize("backend,property_present,fps,raw,reason", [
    ("FFMPEG", False, 60, 2, "unsupported_backend_or_pts_property"),
    ("GSTREAMER", True, 60, 2, "unsupported_backend_or_pts_property"),
    ("FFMPEG", True, 0, 2, "invalid_fps_time_base"),
    ("FFMPEG", True, float("nan"), 2, "invalid_fps_time_base"),
    ("FFMPEG", True, 60, None, "missing_pts"),
    ("FFMPEG", True, 60, float("inf"), "missing_pts"),
    ("FFMPEG", True, 1e-308, 1e308, "invalid_converted_pts"),
])
def test_clock_never_fabricates_fallback(backend, property_present, fps, raw, reason):
    cv = SimpleNamespace(__version__="test")
    if property_present:
        cv.CAP_PROP_PTS = 71
    cap = SimpleNamespace(getBackendName=lambda: backend, get=lambda _: raw)
    clock = SourceMediaClock(cap, cv, fps)
    frame = dict(frame_idx=0, **clock.read(cap))
    assert frame["timestamp_ms"] is None
    assert frame["timestamp_status"] == reason
    metadata = clock.metadata([frame], "a" * 64)
    assert metadata["validity"] == "invalid"
    json.dumps(dict(frame=frame, timing=metadata), allow_nan=False)


def test_pts_origin_is_not_zeroed():
    cv = SimpleNamespace(__version__="test", CAP_PROP_PTS=71)
    cap = SimpleNamespace(getBackendName=lambda: "FFMPEG", get=lambda _: 600)
    assert SourceMediaClock(cap, cv, 60).read(cap)["timestamp_ms"] == 10000


def test_empty_sequence_invalid():
    cv = SimpleNamespace(__version__="test", CAP_PROP_PTS=71)
    cap = SimpleNamespace(getBackendName=lambda: "FFMPEG")
    assert SourceMediaClock(cap, cv, 60).metadata([], "a" * 64)["validity"] == "invalid"


def test_native_pts_without_sync_refused(paired):
    kp, mot, sidecar = paired
    sidecar.unlink()
    with pytest.raises(TimingContractError, match="validated sync mapping required"):
        require_mocap_times(kp, mot, [0.0, 5.0])


@pytest.mark.parametrize("field", ["method", "evidence", "validated_by", "uncertainty_seconds", "accepted_uncertainty_seconds", "calibration_split"])
def test_missing_validation_evidence_refused(paired, field):
    kp, mot, sidecar = paired
    mutate(sidecar, lambda d: d["validation"].pop(field))
    with pytest.raises(TimingContractError):
        require_mocap_times(kp, mot, [0.0, 5.0])


@pytest.mark.parametrize("field,value", [
    ("schema_version", "unknown"), ("source_clock", "mocap_seconds"),
    ("scale", 0), ("scale", -1), ("scale", float("nan")), ("offset_seconds", float("inf")),
    ("valid_source_interval_seconds", [-1, 0.09]), ("valid_source_interval_seconds", [0.001, 1]),
    ("valid_source_interval_seconds", None), ("offset_seconds", 6),
])
def test_unsafe_mapping_refused(paired, field, value):
    kp, mot, sidecar = paired
    mutate(sidecar, lambda d: d.update({field: value}))
    with pytest.raises(TimingContractError):
        require_mocap_times(kp, mot, [0.0, 5.0])


def test_excess_uncertainty_refused(paired):
    kp, mot, sidecar = paired
    mutate(sidecar, lambda d: d["validation"].update(uncertainty_seconds=0.01))
    with pytest.raises(TimingContractError, match="uncertainty"):
        require_mocap_times(kp, mot, [0.0, 5.0])


@pytest.mark.parametrize("fit,evaluation,accept", [
    (["subject1_DJ1"], ["subject1_DJ1"], False),
    (["subject2_DJ1"], ["subject3_DJ1"], False),
    ([], ["subject1_DJ1"], False),
    (["subject2_DJ1"], ["subject1_DJ1"], True),
])
def test_calibration_split_is_explicit_and_disjoint(paired, fit, evaluation, accept):
    kp, mot, sidecar = paired
    split = dict(kind="held_out_recordings", description="Fixture only", fit_recording_ids=fit, evaluation_recording_ids=evaluation)
    mutate(sidecar, lambda d: d["validation"].update(calibration_split=split))
    if accept:
        assert require_mocap_times(kp, mot, [0.0, 5.0]) == pytest.approx([1, 1 + 5 / 60, 1.1])
    else:
        with pytest.raises(TimingContractError, match="disjoint"):
            require_mocap_times(kp, mot, [0.0, 5.0])


@pytest.mark.parametrize("which", ["cache", "mocap"])
def test_stale_binding_refused(paired, which):
    kp, mot, sidecar = paired
    path = kp if which == "cache" else mot
    path.write_text(path.read_text() + "\n")
    with pytest.raises(TimingContractError, match="stale or mismatched"):
        require_mocap_times(kp, mot, [0.0, 5.0])


@pytest.mark.parametrize("field,value", [("source_pts", 0), ("timestamp_ms", 17), ("frame_idx", 9), ("timestamp_status", "missing_pts")])
def test_forged_validity_flag_cannot_mask_invalid_timeline(paired, field, value):
    kp, mot, sidecar = paired
    mutate(kp, lambda d: d["keypoints_sequence"][1].update({field: value}))
    mutate(sidecar, lambda d: d.update(keypoints_sha256=sha256_file(kp)))
    with pytest.raises(TimingContractError, match="invalid source-media PTS"):
        require_mocap_times(kp, mot, [0.0, 5.0])


@pytest.mark.parametrize("gt", [[0, 0], [5, 0], [0, float("nan")], [0, 0.5]])
def test_bad_or_insufficient_actual_mocap_support_refused(paired, gt):
    kp, mot, _ = paired
    with pytest.raises(TimingContractError):
        require_mocap_times(kp, mot, gt)


def test_generic_reader_labels_source_clock_without_claiming_sync(paired):
    kp, _, _ = paired
    series = load_couro_output(kp)
    assert series.timestamp_source == "source_media_pts"
    assert series.time == pytest.approx([0, 5 / 60, 6 / 60])


def test_generic_reader_never_falls_back_for_missing_new_pts(paired):
    kp, _, _ = paired
    mutate(kp, lambda d: d["keypoints_sequence"][1].update(timestamp_ms=None))
    with pytest.raises(TimingContractError, match="invalid source-media PTS"):
        load_couro_output(kp)


@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_malformed_validation_is_actionable(paired, value):
    kp, mot, sidecar = paired
    mutate(sidecar, lambda d: d.update(validation=value))
    with pytest.raises(TimingContractError, match="explicit validated sync evidence"):
        require_mocap_times(kp, mot, [0.0, 5.0])


def test_invalid_fps_writer_persists_diagnostic_json(tmp_path, monkeypatch):
    from test_frame_timing_regressions import run_writer
    payload, *_ = run_writer(tmp_path, monkeypatch, [0, 1, 2], fps=float("nan"))
    assert payload["video_metadata"]["fps"] is None
    assert payload["timing"]["validity"] == "invalid"
    assert all(fr["timestamp_ms"] is None for fr in payload["keypoints_sequence"])


def test_real_opencv_ffmpeg_gap_round_trip(tmp_path):
    """Real decode/timestamp path, without a detector or inference session."""
    import shutil
    import subprocess
    import cv2
    if not shutil.which("ffmpeg") or not hasattr(cv2, "CAP_PROP_PTS"):
        pytest.skip("requires existing ffmpeg and OpenCV CAP_PROP_PTS; no install performed")
    video = tmp_path / "gap.avi"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=64x64:rate=60:duration=2.7666666667", "-frames:v", "166", "-vf",
        r"setpts=if(eq(N\,0)\,0\,N+4)/(60*TB)", "-fps_mode", "passthrough",
        "-c:v", "mjpeg", "-q:v", "4", str(video),
    ], check=True, timeout=30)
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 170
    fps = cap.get(cv2.CAP_PROP_FPS)
    clock = SourceMediaClock(cap, cv2, fps)
    frames = []
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frames.append(dict(frame_idx=len(frames), **clock.read(cap)))
    finally:
        cap.release()
    assert len(frames) == 166
    assert [fr["source_pts"] for fr in frames] == [0] + list(range(5, 170))
    assert [fr["timestamp_ms"] for fr in frames] == pytest.approx(np.array([0] + list(range(5, 170))) / 60 * 1000)
    assert clock.metadata(frames, sha256_file(video))["validity"] == "valid"


def test_sync_sidecars_are_not_discovered_as_keypoint_clips(paired):
    kp, _, sidecar = paired
    assert sidecar.exists()
    assert list(kp.parent.glob("*.json")) == [kp]


@pytest.mark.parametrize("remove_timestamps", [False, True])
def test_declared_pts_schema_cannot_downgrade_to_legacy_when_timing_missing(paired, remove_timestamps):
    kp, _, _ = paired
    def remove_provenance(payload):
        payload["schema_version"] = "couro-dwpose-halpe26-v2-source-pts"
        payload.pop("timing")
        if remove_timestamps:
            for frame in payload["keypoints_sequence"]:
                frame.pop("timestamp_ms")
    mutate(kp, remove_provenance)
    with pytest.raises(TimingContractError, match="declared_source_pts_missing_timing"):
        load_couro_output(kp)
