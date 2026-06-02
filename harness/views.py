"""Classify camera positions into Couro's three-view bucket: front, side, rear.

Couro uses 0° (anterior), 90° (sagittal/side), 180° (posterior/rear).
OpenCap cameras live on a frontal arc (designed for two-phone clinical capture
at ±45°); pure rear-view cameras are absent from the OpenCap lab dataset.
This module reports honestly when a view bucket has no cameras.
"""
from __future__ import annotations

from dataclasses import dataclass

from .parsers import CameraExtrinsics


FRONT_TOLERANCE_DEG = 30.0
SIDE_MIN_DEG = 60.0
SIDE_MAX_DEG = 120.0
REAR_MIN_DEG = 150.0


@dataclass(frozen=True)
class ViewClassification:
    cam_id: str
    yaw_deg: float
    bucket: str            # "front" | "front-oblique" | "side" | "rear" | "unclassified"


def classify_camera(cam: CameraExtrinsics) -> ViewClassification:
    yaw_abs = abs(cam.yaw_deg)
    if yaw_abs <= FRONT_TOLERANCE_DEG:
        bucket = "front"
    elif FRONT_TOLERANCE_DEG < yaw_abs < SIDE_MIN_DEG:
        bucket = "front-oblique"
    elif SIDE_MIN_DEG <= yaw_abs <= SIDE_MAX_DEG:
        bucket = "side"
    elif yaw_abs >= REAR_MIN_DEG:
        bucket = "rear"
    else:
        bucket = "unclassified"
    return ViewClassification(cam_id=cam.cam_id, yaw_deg=cam.yaw_deg, bucket=bucket)


def bucket_summary(classifications: list[ViewClassification]) -> dict[str, list[str]]:
    """Group cam IDs by view bucket; missing buckets map to empty list."""
    summary: dict[str, list[str]] = {
        "front": [],
        "front-oblique": [],
        "side": [],
        "rear": [],
        "unclassified": [],
    }
    for c in classifications:
        summary[c.bucket].append(c.cam_id)
    return summary
