"""Source-media timing and the explicit boundary to frame-paired mocap data.

OpenCV/FFmpeg reports CAP_PROP_PTS in FPS time-base units, not decoded-frame
ordinals. These decoder-reported timestamps are not a synchronization claim.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

PTS_KEYPOINT_SCHEMA = "couro-dwpose-halpe26-v2-source-pts"
TIMING_SCHEMA = "source-media-pts-v1"
SYNC_SCHEMA = "source-media-to-mocap-v1"


class TimingContractError(ValueError):
    """A cache cannot safely be used for frame-paired mocap training."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class SourceMediaClock:
    """Capture each frame's reported PTS immediately after a successful read.

    Unsupported/missing values stay null; there is no ordinal, POS_MSEC, or
    guessed-FPS fallback. Sequence validity is checked separately at finalization.
    """

    def __init__(self, cap, cv2, fps: float):
        self.fps = fps
        self.backend = cap.getBackendName()
        self.opencv_version = cv2.__version__
        self.property = getattr(cv2, "CAP_PROP_PTS", None)
        self.supported = self.backend == "FFMPEG" and self.property is not None

    def read(self, cap) -> dict:
        raw = cap.get(self.property) if self.supported else None
        pts = float(raw) if _finite(raw) else None
        valid_fps = _finite(self.fps) and self.fps > 0
        status = "reported"
        if not self.supported:
            status = "unsupported_backend_or_pts_property"
        elif pts is None:
            status = "missing_pts"
        elif not valid_fps:
            status = "invalid_fps_time_base"
        ms = pts / self.fps * 1000.0 if pts is not None and valid_fps else None
        if ms is not None and not _finite(ms):
            ms, status = None, "invalid_converted_pts"
        return {
            "source_pts": pts,
            "timestamp_ms": ms,
            "timestamp_status": status,
        }

    def metadata(self, frames: list[dict], source_sha256: str) -> dict:
        issues = _timeline_issues(frames, self.fps)
        return {
            "schema_version": TIMING_SCHEMA,
            "clock": "source_media",
            "source": "opencv_CAP_PROP_PTS",
            "backend": self.backend,
            "opencv_version": self.opencv_version,
            "pts_units": "fps_time_base",
            "fps_time_base": self.fps if _finite(self.fps) and self.fps > 0 else None,
            "source_video_sha256": source_sha256,
            "validity": "invalid" if issues else "valid",
            "issues": issues,
            "mocap_sync": "unvalidated",
        }


def _timeline_issues(frames: list[dict], fps) -> list[dict]:
    issues = []
    if not isinstance(frames, list) or not frames:
        return [{"frame_idx": None, "reason": "empty_or_invalid_sequence"}]
    previous = None
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            issues.append({"frame_idx": i, "reason": "invalid_frame"})
            continue
        pts, ms = frame.get("source_pts"), frame.get("timestamp_ms")
        reason = None
        if frame.get("frame_idx") != i:
            reason = "decoded_frame_index_mismatch"
        elif frame.get("timestamp_status") != "reported":
            reason = frame.get("timestamp_status") or "missing_pts_status"
        elif not _finite(fps) or fps <= 0:
            reason = "invalid_fps_time_base"
        elif not _finite(pts) or not _finite(ms):
            reason = "missing_pts"
        elif not math.isclose(ms, pts / fps * 1000.0, rel_tol=1e-12, abs_tol=1e-9):
            reason = "timestamp_pts_mismatch"
        elif previous is not None and pts <= previous:
            reason = "duplicate_pts" if pts == previous else "nonmonotonic_pts"
        if reason:
            issues.append({"frame_idx": i, "reason": reason})
        if _finite(pts):
            previous = pts
    return issues


def cache_timing_label(payload: dict) -> str:
    """Label legacy caches without changing, upgrading, or inferring their clock."""
    if not isinstance(payload, dict):
        return "invalid_cache"
    timing = payload.get("timing")
    if timing is None:
        if payload.get("schema_version") == PTS_KEYPOINT_SCHEMA:
            return "declared_source_pts_missing_timing"
        return "legacy_unverified_timestamp_ms"
    if not isinstance(timing, dict) or timing.get("schema_version") != TIMING_SCHEMA:
        return "unknown_timing_schema"
    return "source_media_pts"


def require_source_media_times(payload: dict, cache_name: str = "cache") -> list[float]:
    """Read a structurally valid source clock; this does not validate mocap sync."""
    label = cache_timing_label(payload)
    if label != "source_media_pts":
        raise TimingContractError(f"{cache_name}: {label}; recapture PTS into a new cache before frame pairing")
    timing = payload["timing"]
    frames = payload.get("keypoints_sequence", [])
    if (timing.get("clock") != "source_media"
            or timing.get("source") != "opencv_CAP_PROP_PTS"
            or timing.get("backend") != "FFMPEG"
            or timing.get("pts_units") != "fps_time_base"
            or timing.get("validity") != "valid"
            or _timeline_issues(frames, timing.get("fps_time_base"))):
        raise TimingContractError(f"{cache_name}: invalid source-media PTS")
    source_hash = timing.get("source_video_sha256", "")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        raise TimingContractError(f"{cache_name}: missing source-video provenance")
    return [frame["timestamp_ms"] / 1000.0 for frame in frames]


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (ValueError, UnicodeError) as exc:
        raise TimingContractError(f"{path.name}: invalid timing/sync JSON") from exc
    if not isinstance(value, dict):
        raise TimingContractError(f"{path.name}: timing/sync JSON object required")
    return value


def require_mocap_times(kp_path: Path, mot_path: Path, mocap_time) -> list[float]:
    """Return explicitly validated mocap seconds, or fail closed.

    `sync/<cache>.sync.json` must bind the exact cache and mocap file bytes, record
    validation evidence, and provide an affine mapping. Filename suffixes and
    PTS validity never establish video/mocap synchronization. Validation of the
    scientific alignment is a separate reviewed step, not performed here.
    """
    kp_path, mot_path = Path(kp_path), Path(mot_path)
    payload = _read_json_object(kp_path)
    source_times = require_source_media_times(payload, kp_path.name)
    sync_path = kp_path.parent / "sync" / f"{kp_path.stem}.sync.json"
    if not sync_path.exists():
        raise TimingContractError(f"{kp_path.name}: validated sync mapping required at {sync_path}; source PTS are not mocap time")
    sync = _read_json_object(sync_path)
    validation = sync.get("validation", {})
    if (not isinstance(validation, dict)
            or sync.get("schema_version") != SYNC_SCHEMA
            or sync.get("source_clock") != "source_media_seconds"
            or sync.get("target_clock") != "mocap_seconds"
            or validation.get("status") != "validated"
            or any(not isinstance(validation.get(k), str) or not validation[k].strip()
                   for k in ("method", "evidence", "validated_by"))):
        raise TimingContractError(f"{sync_path.name}: explicit validated sync evidence required")
    if sync.get("keypoints_sha256") != sha256_file(kp_path) or sync.get("mocap_sha256") != sha256_file(mot_path):
        raise TimingContractError(f"{sync_path.name}: stale or mismatched cache/mocap binding")
    scale, offset = sync.get("scale"), sync.get("offset_seconds")
    if not _finite(scale) or scale <= 0 or not _finite(offset):
        raise TimingContractError(f"{sync_path.name}: finite positive scale and finite offset_seconds required")
    uncertainty = validation.get("uncertainty_seconds")
    accepted = validation.get("accepted_uncertainty_seconds")
    if not _finite(uncertainty) or uncertainty < 0 or not _finite(accepted) or not 0 <= uncertainty <= accepted:
        raise TimingContractError(f"{sync_path.name}: declared uncertainty must be within the accepted tolerance")
    split = validation.get("calibration_split", {})
    if not isinstance(split, dict) or not isinstance(split.get("description"), str) or not split["description"].strip():
        raise TimingContractError(f"{sync_path.name}: calibration/evaluation split description required")
    if split.get("kind") == "held_out_recordings":
        fit, evaluation = split.get("fit_recording_ids"), split.get("evaluation_recording_ids")
        if (not isinstance(fit, list) or not fit or not isinstance(evaluation, list) or not evaluation
                or any(not isinstance(x, str) or not x.strip() for x in fit + evaluation)
                or set(fit) & set(evaluation)
                or kp_path.stem.rsplit("_", 1)[0] not in evaluation):
            raise TimingContractError(f"{sync_path.name}: disjoint fit/evaluation recording IDs including this recording required")
    elif split.get("kind") != "independent_sync_evidence":
        raise TimingContractError(f"{sync_path.name}: declare held-out recordings or independent synchronization evidence")
    # The independent-evidence declaration is a provenance claim requiring
    # review, not proof: fitting lag on evaluated pose/angle targets is invalid.
    coverage = sync.get("valid_source_interval_seconds")
    if (not isinstance(coverage, list) or len(coverage) != 2
            or not all(_finite(x) for x in coverage) or coverage[0] >= coverage[1]
            or source_times[0] < coverage[0] or source_times[-1] > coverage[1]):
        raise TimingContractError(f"{sync_path.name}: validated source interval must cover all frames; extrapolation forbidden")
    times = [scale * t + offset for t in source_times]
    if any(not _finite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise TimingContractError(f"{sync_path.name}: invalid mapped mocap timeline")
    gt_times = list(mocap_time)
    if (len(gt_times) < 2 or any(not _finite(t) for t in gt_times)
            or any(b <= a for a, b in zip(gt_times, gt_times[1:]))):
        raise TimingContractError(f"{mot_path.name}: finite strictly increasing mocap times required")
    if times[0] < gt_times[0] or times[-1] > gt_times[-1]:
        raise TimingContractError(f"{sync_path.name}: mapped frames exceed mocap support; extrapolation/clamping forbidden")
    return times
