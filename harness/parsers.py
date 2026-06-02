"""OpenSim .mot / .sto and Vicon .trc parsers. Pure stdlib + numpy."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionData:
    """A parsed OpenSim .mot / .sto motion file."""
    columns: tuple[str, ...]
    time: np.ndarray           # (n_frames,)
    values: np.ndarray         # (n_frames, n_cols - 1)
    in_degrees: bool

    def column(self, name: str) -> np.ndarray:
        idx = self.columns.index(name) - 1  # time is column 0
        return self.values[:, idx]

    def has(self, name: str) -> bool:
        return name in self.columns


def parse_mot(path: Path) -> MotionData:
    """Parse an OpenSim .mot or .sto file.

    Format: header block ending with 'endheader', then a tab-separated table
    whose first row is column names and remaining rows are numeric.
    """
    in_degrees = True
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("indegrees="):
                in_degrees = line.split("=", 1)[1].strip().lower() == "yes"
            if line.lower() == "endheader":
                break
        header = f.readline().strip().split("\t")
        rows = []
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            rows.append([float(x) for x in parts])
    arr = np.asarray(rows, dtype=np.float64)
    return MotionData(
        columns=tuple(header),
        time=arr[:, 0],
        values=arr[:, 1:],
        in_degrees=in_degrees,
    )


@dataclass(frozen=True)
class TrcData:
    """Parsed Vicon .trc marker data."""
    marker_names: tuple[str, ...]
    time: np.ndarray                       # (n_frames,)
    positions: np.ndarray                  # (n_frames, n_markers, 3)
    rate: float


def parse_trc(path: Path) -> TrcData:
    """Parse a Vicon .trc file. mm-scaled XYZ positions per marker per frame."""
    with open(path) as f:
        lines = f.readlines()
    # Header is 5 lines: PathFileType, DataFormat, DataValues, MarkerNames row, XYZ-component row.
    rate = float(lines[2].split("\t")[0])
    marker_names_row = lines[3].rstrip("\n").split("\t")
    # Marker name appears every 3 columns starting at column 2 (after Frame#, Time).
    marker_names = tuple(n for i, n in enumerate(marker_names_row[2:]) if i % 3 == 0 and n.strip())
    data_start = 5
    rows = []
    for line in lines[data_start:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        try:
            rows.append([float(x) if x.strip() else np.nan for x in parts])
        except ValueError:
            continue
    arr = np.asarray(rows, dtype=np.float64)
    time = arr[:, 1]
    # Reshape (n_frames, n_markers * 3) -> (n_frames, n_markers, 3)
    marker_data = arr[:, 2:].reshape(arr.shape[0], -1, 3)
    return TrcData(
        marker_names=marker_names,
        time=time,
        positions=marker_data,
        rate=rate,
    )


@dataclass(frozen=True)
class CameraExtrinsics:
    cam_id: str
    translation_mm: np.ndarray             # (3,) world translation
    rotation_euler_rad: np.ndarray         # (3,) euler angles (radians)
    yaw_deg: float                         # convenience: rotation around vertical axis


def parse_camera_pickle(path: Path, cam_id: str) -> CameraExtrinsics:
    with open(path, "rb") as f:
        data = pickle.load(f)
    t = np.asarray(data["translation"], dtype=np.float64).flatten()
    eul = np.asarray(data["rotation_EulerAngles"], dtype=np.float64).flatten()
    # OpenCap convention: euler[1] is yaw around vertical axis (Y).
    yaw_deg = float(np.degrees(eul[1]))
    return CameraExtrinsics(
        cam_id=cam_id,
        translation_mm=t,
        rotation_euler_rad=eul,
        yaw_deg=yaw_deg,
    )
