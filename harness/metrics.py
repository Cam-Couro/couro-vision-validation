"""Couro-relevant scalar extraction from time-series joint angles.

Each metric extracts a single number per trial (e.g., peak knee flexion at landing).
The reference implementation here works on any OpenSim .mot file regardless of source
(Vicon mocap, OpenPose 2-cam, HRNet 5-cam, or eventually Couro CV) so the same metric
extraction code runs on both sides of the comparison.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .parsers import MotionData


@dataclass(frozen=True)
class MetricSpec:
    name: str
    body_region: str
    plane: str                          # "sagittal" | "frontal" | "transverse"
    extractor: Callable[[MotionData], float]
    units: str
    description: str


def _peak(col: str, sign: int = 1):
    def f(m: MotionData) -> float:
        if not m.has(col):
            return float("nan")
        return float(sign * np.nanmax(sign * m.column(col)))
    return f


def _mean(col: str):
    def f(m: MotionData) -> float:
        if not m.has(col):
            return float("nan")
        return float(np.nanmean(m.column(col)))
    return f


def _rom(col: str):
    """Range of motion: max - min."""
    def f(m: MotionData) -> float:
        if not m.has(col):
            return float("nan")
        v = m.column(col)
        return float(np.nanmax(v) - np.nanmin(v))
    return f


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "peak_knee_flexion_r", "knee", "sagittal", _peak("knee_angle_r", sign=1),
        "deg", "Peak right knee flexion angle during trial",
    ),
    MetricSpec(
        "peak_knee_flexion_l", "knee", "sagittal", _peak("knee_angle_l", sign=1),
        "deg", "Peak left knee flexion angle during trial",
    ),
    MetricSpec(
        "peak_hip_flexion_r", "hip", "sagittal", _peak("hip_flexion_r", sign=1),
        "deg", "Peak right hip flexion angle during trial",
    ),
    MetricSpec(
        "peak_hip_flexion_l", "hip", "sagittal", _peak("hip_flexion_l", sign=1),
        "deg", "Peak left hip flexion angle during trial",
    ),
    MetricSpec(
        "peak_hip_adduction_r", "hip", "frontal", _peak("hip_adduction_r", sign=1),
        "deg", "Peak right hip adduction (pelvic-drop proxy)",
    ),
    MetricSpec(
        "peak_hip_adduction_l", "hip", "frontal", _peak("hip_adduction_l", sign=1),
        "deg", "Peak left hip adduction",
    ),
    MetricSpec(
        "peak_ankle_flexion_r", "ankle", "sagittal", _peak("ankle_angle_r", sign=1),
        "deg", "Peak right ankle dorsiflexion",
    ),
    MetricSpec(
        "peak_ankle_flexion_l", "ankle", "sagittal", _peak("ankle_angle_l", sign=1),
        "deg", "Peak left ankle dorsiflexion",
    ),
    MetricSpec(
        "trunk_lean_max", "trunk", "sagittal", _peak("lumbar_extension", sign=-1),
        "deg", "Peak forward trunk lean (negative lumbar extension)",
    ),
    MetricSpec(
        "pelvis_tilt_mean", "pelvis", "sagittal", _mean("pelvis_tilt"),
        "deg", "Mean pelvis tilt over trial",
    ),
    MetricSpec(
        "knee_flexion_rom_r", "knee", "sagittal", _rom("knee_angle_r"),
        "deg", "Knee flexion range of motion (right)",
    ),
    MetricSpec(
        "hip_flexion_rom_r", "hip", "sagittal", _rom("hip_flexion_r"),
        "deg", "Hip flexion range of motion (right)",
    ),
)


def extract_all(m: MotionData) -> dict[str, float]:
    """Extract every defined metric scalar from a single MotionData."""
    return {spec.name: spec.extractor(m) for spec in METRICS}
