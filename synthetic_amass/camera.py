"""Virtual camera placement and projection for synthetic AMASS rendering.

Conventions:
- World: right-handed, +X right, +Y up, +Z forward (toward viewer / out of mocap floor).
- Subject is placed at origin.
- Camera placed in spherical coords (azimuth, elevation, distance) around subject,
  looking at the subject's hip_center (root).
- Azimuth=0 -> camera in front (+Z), looking toward -Z.
- Azimuth=180 -> camera behind subject (rear view).
- Elevation=0 -> level with hip_center. Positive elevation -> looking down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


DEFAULT_IMAGE_WIDTH: Final[int] = 1280
DEFAULT_IMAGE_HEIGHT: Final[int] = 720
# Reasonable focal length for ~45 deg horizontal FOV at 1280px wide.
DEFAULT_FOCAL_PX: Final[float] = 1500.0


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def default(cls) -> "CameraIntrinsics":
        return cls(
            fx=DEFAULT_FOCAL_PX,
            fy=DEFAULT_FOCAL_PX,
            cx=DEFAULT_IMAGE_WIDTH / 2.0,
            cy=DEFAULT_IMAGE_HEIGHT / 2.0,
            width=DEFAULT_IMAGE_WIDTH,
            height=DEFAULT_IMAGE_HEIGHT,
        )


@dataclass(frozen=True)
class CameraPose:
    """Camera-to-world transformation, parameterized by spherical placement."""

    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    look_at_world: np.ndarray  # shape (3,)


def _build_camera_extrinsic(pose: CameraPose) -> np.ndarray:
    """Return 4x4 world->camera transform.

    Camera looks at pose.look_at_world from the spherical position determined
    by azimuth/elevation/distance. Up vector is world +Y.
    """
    az = np.deg2rad(pose.azimuth_deg)
    el = np.deg2rad(pose.elevation_deg)
    d = pose.distance_m

    # Camera position in world coordinates.
    # az=0  -> +Z axis (in front);  az=180 -> -Z axis (behind).
    cam_x = d * np.cos(el) * np.sin(az)
    cam_y = d * np.sin(el) + pose.look_at_world[1]
    cam_z = d * np.cos(el) * np.cos(az)

    cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

    # Forward: from camera toward look_at.
    forward = pose.look_at_world - cam_pos
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # World-to-camera rotation (rows are camera axes in world coords).
    # Camera convention: +X right, +Y down, +Z forward (OpenCV-style).
    rot_w2c = np.stack([right, -up, forward], axis=0)
    trans_w2c = -rot_w2c @ cam_pos

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot_w2c
    extrinsic[:3, 3] = trans_w2c
    return extrinsic


def project_points(
    points_world: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world-coords to Nx2 pixel coords.

    Returns:
        pixels: (N, 2) float pixel coordinates.
        depths: (N,) depth in camera frame (positive = in front).
    """
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got {points_world.shape}")

    extrinsic = _build_camera_extrinsic(pose)
    n = points_world.shape[0]
    points_h = np.concatenate([points_world, np.ones((n, 1))], axis=1)
    cam_coords = (extrinsic @ points_h.T).T[:, :3]

    depths = cam_coords[:, 2]
    # Avoid divide-by-zero for points exactly at the camera plane.
    safe_z = np.where(np.abs(depths) < 1e-6, 1e-6, depths)
    x_norm = cam_coords[:, 0] / safe_z
    y_norm = cam_coords[:, 1] / safe_z

    u = intrinsics.fx * x_norm + intrinsics.cx
    v = intrinsics.fy * y_norm + intrinsics.cy
    pixels = np.stack([u, v], axis=1)
    return pixels, depths
