"""CPU-only regressions: fake decode/session adapters, no model inference.

TIMING_REPO_ROOT permits running the same regressions against the pinned
baseline. Builder tests compile the real data-loading functions (AST only) to
avoid importing Torch/model modules. No training/inference function is run.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(os.environ.get("TIMING_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))


class Capture:
    def __init__(self, pts, cv2, fps=60.0):
        self.pts, self.cv2, self.fps = pts, cv2, fps
        self.i, self.released = -1, False

    def isOpened(self):
        return True

    def getBackendName(self):
        return "FFMPEG"

    def get(self, prop):
        return {
            self.cv2.CAP_PROP_PTS: self.pts[self.i],
            self.cv2.CAP_PROP_FPS: self.fps,
            self.cv2.CAP_PROP_FRAME_WIDTH: 4,
            self.cv2.CAP_PROP_FRAME_HEIGHT: 4,
        }.get(prop, 0)

    def read(self):
        self.i += 1
        return (True, np.full((4, 4, 3), self.i, dtype=np.float32)) if self.i < len(self.pts) else (False, None)

    def release(self):
        self.released = True


def run_writer(tmp_path, monkeypatch, pts, batch_size=8, fps=60.0):
    # Baseline imports ORT eagerly; this empty stub is never used to load a model.
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace())
    driver = importlib.import_module("harness.opencap_infer_dwpose_gpu")
    cap = Capture(pts, driver.cv2, fps=fps)
    monkeypatch.setattr(driver.cv2, "VideoCapture", lambda _: cap)
    monkeypatch.setattr(driver, "preprocess_crop", lambda frame, _: (frame, None))
    monkeypatch.setattr(driver, "decode_simcc", lambda x, y: (x, y))
    monkeypatch.setattr(driver, "transform_back", lambda x, _: x)
    monkeypatch.setattr(driver, "dwpose_to_halpe26", lambda x, y: (x, y))
    # Preserve the decoded ordinal in fake keypoints to detect batch mispairing.
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="test_input")],
        run=lambda _, feeds: (feeds["test_input"], np.ones(len(feeds["test_input"]))),
    )
    video = tmp_path / "DJ1_syncdWithMocap.avi"
    video.write_bytes(b"synthetic decode adapter; not a real AVI")
    bbox = tmp_path / "boxes.csv"
    bbox.write_text("x1,y1,x2,y2\n")
    out = tmp_path / "keypoints.json"
    assert driver.process_video(video, bbox, session, out, batch_size=batch_size)
    assert cap.released
    return json.loads(out.read_text()), driver, out, video, bbox, session


def test_gapped_pts_stay_with_decoded_frames_across_batches(tmp_path, monkeypatch):
    pts = [0] + list(range(5, 170))
    payload, *_ = run_writer(tmp_path, monkeypatch, pts, batch_size=7)
    frames = payload["keypoints_sequence"]
    assert len(frames) == 166
    assert [f["timestamp_ms"] for f in frames] == pytest.approx(np.array(pts) / 60 * 1000)
    assert [f["keypoints"][0][0][0] for f in frames] == list(range(166))
    assert payload["timing"]["validity"] == "valid"
    assert payload["timing"]["mocap_sync"] == "unvalidated"
    assert frames[1]["timestamp_ms"] - 1000 / 60 == pytest.approx(66.6666666667)


@pytest.mark.parametrize("pts,reason", [
    ([0, 5, 5], "duplicate_pts"),
    ([0, 5, 4], "nonmonotonic_pts"),
    ([0, float("nan"), 6], "missing_pts"),
])
def test_invalid_pts_are_persisted_as_invalid_not_repaired(tmp_path, monkeypatch, pts, reason):
    payload, *_ = run_writer(tmp_path, monkeypatch, pts)
    assert payload["timing"]["validity"] == "invalid"
    assert reason in [x["reason"] for x in payload["timing"]["issues"]]
    if reason == "missing_pts":
        assert payload["keypoints_sequence"][1]["timestamp_ms"] is None


def test_existing_legacy_cache_is_not_overwritten_or_reported_as_fresh(tmp_path, monkeypatch):
    payload, driver, out, video, bbox, sess = run_writer(tmp_path, monkeypatch, [0, 1, 2])
    payload.pop("timing", None)
    payload["schema_version"] = "couro-dwpose-halpe26-v1"
    for frame in payload["keypoints_sequence"]:
        frame.pop("source_pts", None)
        frame.pop("timestamp_status", None)
    payload["padding"] = "x" * 1100
    out.write_text(json.dumps(payload))
    before = out.read_bytes()
    with pytest.raises(FileExistsError, match="legacy_unverified_timestamp_ms"):
        driver.process_video(video, bbox, sess, out)
    assert out.read_bytes() == before


BUILDERS = [
    ("learned_layer2_real_gt.py", "build_clip_dataset", "load_opencap_clip"),
    ("learned_layer2_combined.py", "build_opencap_clip", "_load_opencap_keypoints"),
    ("learned_layer2_persource_mirrorflip.py", "_load_opencap_clip_lr", "_load_opencap_keypoints"),
]


def compile_builder(filename, fn, loader, lab):
    from harness.parsers import parse_mot
    namespace = dict(np=np, json=json, Path=Path, LAB=lab, DEPLOY_METRICS=("knee_angle_r",),
                     DEPLOY_METRICS_L=("knee_angle_l",), log=lambda _: None)
    helper_path = ROOT / "harness/frame_timing.py"
    if helper_path.exists():
        from harness.frame_timing import require_mocap_times
        namespace["require_mocap_times"] = require_mocap_times
    namespace["parse_mot"] = parse_mot
    source = ast.parse((ROOT / "harness" / filename).read_text())
    shared = ast.parse((ROOT / "harness/learned_layer2_combined.py").read_text())
    for name in ("_normalize_kp", "_resample_gt", loader, fn):
        node = next((n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
        if node is None:
            node = next(n for n in shared.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "harness" / filename), "exec"), namespace)
    return namespace[fn]


def legacy_cache(tmp_path):
    kp = tmp_path / "subject1_DJ1_Cam0.json"
    frames = [dict(frame_idx=i, timestamp_ms=i / 60 * 1000, keypoints=np.zeros((26, 2)).tolist(),
                   keypoint_scores=np.ones(26).tolist()) for i in range(3)]
    kp.write_text(json.dumps(dict(video_metadata=dict(fps=60, width=4, height=4), keypoints_sequence=frames)))
    mot = tmp_path / "lab/subject1/OpenSimData/Mocap/IK/DJ1.mot"
    mot.parent.mkdir(parents=True)
    mot.write_text("inDegrees=yes\nendheader\ntime\tknee_angle_r\tknee_angle_l\n0\t0\t0\n5\t50\t100\n")
    return kp, mot


@pytest.mark.parametrize("filename,fn,loader", BUILDERS)
def test_real_builder_refuses_legacy_frame_pairing(tmp_path, filename, fn, loader):
    kp, _ = legacy_cache(tmp_path)
    builder = compile_builder(filename, fn, loader, tmp_path / "lab")
    with pytest.raises(ValueError, match="legacy_unverified_timestamp_ms"):
        builder(kp, np.arange(26))


@pytest.mark.parametrize("filename,fn,loader", BUILDERS)
def test_real_builder_uses_explicit_mapping_not_media_clock(tmp_path, filename, fn, loader):
    # The helper is imported from the repaired tree even in baseline regression
    # mode solely to create the same explicit input fixture for both versions.
    import importlib.util
    helper = Path(__file__).resolve().parents[1] / "harness/frame_timing.py"
    spec = importlib.util.spec_from_file_location("fixture_timing", helper)
    timing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing)
    kp, mot = legacy_cache(tmp_path)
    payload = json.loads(kp.read_text())
    frames = payload["keypoints_sequence"]
    for fr, pts in zip(frames, [0, 5, 6]):
        fr.update(source_pts=pts, timestamp_ms=pts / 60 * 1000, timestamp_status="reported")
    import cv2
    clock = timing.SourceMediaClock(SimpleNamespace(getBackendName=lambda: "FFMPEG"), cv2, 60)
    payload["timing"] = clock.metadata(frames, "a" * 64)
    kp.write_text(json.dumps(payload))
    sync = dict(schema_version=timing.SYNC_SCHEMA, source_clock="source_media_seconds", target_clock="mocap_seconds",
                keypoints_sha256=timing.sha256_file(kp), mocap_sha256=timing.sha256_file(mot), scale=1.0,
                offset_seconds=1.0, valid_source_interval_seconds=[0, 0.1],
                validation=dict(status="validated", method="synthetic known clock", evidence="test fixture",
                                validated_by="test", uncertainty_seconds=0.001, accepted_uncertainty_seconds=0.002,
                                calibration_split=dict(kind="independent_sync_evidence", description="Synthetic clock, no target fitting")))
    sidecar = kp.parent / "sync" / f"{kp.stem}.sync.json"
    sidecar.parent.mkdir(exist_ok=True)
    sidecar.write_text(json.dumps(sync))
    builder = compile_builder(filename, fn, loader, tmp_path / "lab")
    result = builder(kp, np.arange(26))
    assert result["t_sec"] == pytest.approx([1, 1 + 5 / 60, 1.1])
    assert result["angles"][:, 0] == pytest.approx([10, 10 + 50 / 60, 11])
    if fn == "_load_opencap_clip_lr":
        assert result["angles_l"][:, 0] == pytest.approx([20, 20 + 100 / 60, 22])
