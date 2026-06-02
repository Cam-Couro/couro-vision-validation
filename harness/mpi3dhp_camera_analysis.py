"""Parse MPI-INF-3DHP camera.calibration (Skeletool format) and compute world
yaw for each camera so we can identify front, side, oblique, and rear cameras.

The studio subject lives near the world origin facing roughly +X (we infer
front from the camera with smallest yaw toward the origin). Yaw is computed
from the camera's optical axis projected onto the floor plane.

Skeletool extrinsics are stored as a row-major 4x4 with rotation R and
translation t such that p_cam = R * p_world + t. The camera position in
world coords is C = -R^T * t, and the camera viewing direction in world
coords is the third row of R (the +Z axis of camera coords expressed in
world coords).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Camera:
    cam_id: int
    intrinsic: tuple
    extrinsic: tuple

    @property
    def R(self) -> list:
        e = self.extrinsic
        return [
            [e[0], e[1], e[2]],
            [e[4], e[5], e[6]],
            [e[8], e[9], e[10]],
        ]

    @property
    def t(self) -> list:
        e = self.extrinsic
        return [e[3], e[7], e[11]]

    @property
    def center_world(self) -> list:
        # C = -R^T * t
        R, t = self.R, self.t
        c = [0.0, 0.0, 0.0]
        for i in range(3):
            for j in range(3):
                c[i] += -R[j][i] * t[j]
        return c

    @property
    def view_dir_world(self) -> list:
        # camera +Z axis in camera frame is the optical axis, expressed in
        # world coords by R^T (third row of R^T = third column of R = third row of R^T)
        # Equivalently: world dir = R[2,:] when R maps world->camera in row-major.
        R = self.R
        # Camera looks along +Z_cam. In world: d_world = R^T * [0,0,1] = R[:,2]
        return [R[0][2], R[1][2], R[2][2]]


def parse_calibration(path: Path) -> list[Camera]:
    cams: list[Camera] = []
    cur_id: int | None = None
    cur_int: tuple | None = None
    cur_ext: tuple | None = None
    name_re = re.compile(r"^\s*name\s+(\d+)")
    intr_re = re.compile(r"^\s*intrinsic\s+(.*)$")
    extr_re = re.compile(r"^\s*extrinsic\s+(.*)$")
    with open(path) as f:
        for line in f:
            m = name_re.match(line)
            if m:
                if cur_id is not None and cur_int is not None and cur_ext is not None:
                    cams.append(Camera(cur_id, cur_int, cur_ext))
                cur_id = int(m.group(1))
                cur_int = cur_ext = None
                continue
            m = intr_re.match(line)
            if m:
                vals = [float(v) for v in m.group(1).split() if v.strip()]
                cur_int = tuple(vals)
                continue
            m = extr_re.match(line)
            if m:
                vals = [float(v) for v in m.group(1).split() if v.strip()]
                cur_ext = tuple(vals)
        if cur_id is not None and cur_int is not None and cur_ext is not None:
            cams.append(Camera(cur_id, cur_int, cur_ext))
    return cams


def classify_cameras(cams: list[Camera]) -> dict:
    """Determine each camera's angle relative to the subject.

    Approach: subject is roughly at world origin. We compute the bearing
    from the subject to each camera (azimuth in the floor plane). Then
    we identify front (closest to subject's facing direction) and rear
    (opposite). MPI-INF-3DHP world: Y is up. Floor plane is XZ.

    Subject orientation is inferred by treating camera 0 as the canonical
    "front camera" (per VNect convention) — then any camera positioned
    opposite to camera 0 across the studio is "rear".
    """
    summary = []
    for c in cams:
        C = c.center_world
        # bearing from origin to camera in floor plane (azimuth in XZ)
        bearing_deg = math.degrees(math.atan2(C[2], C[0]))
        height = C[1]  # Y is up
        radius = math.hypot(C[0], C[2])
        summary.append(
            {
                "cam_id": c.cam_id,
                "world_xyz": [round(v, 1) for v in C],
                "azimuth_deg": round(bearing_deg, 1),
                "radius_mm": round(radius, 1),
                "height_mm": round(height, 1),
            }
        )

    # Reference azimuth: cam 0
    cam0_az = next(s["azimuth_deg"] for s in summary if s["cam_id"] == 0)

    def rel_az(s):
        d = (s["azimuth_deg"] - cam0_az + 540) % 360 - 180
        return d

    for s in summary:
        s["rel_az_from_cam0"] = round(rel_az(s), 1)
        a = abs(s["rel_az_from_cam0"])
        if a < 20:
            cat = "front"
        elif a < 70:
            cat = "front-oblique"
        elif a < 110:
            cat = "side"
        elif a < 160:
            cat = "rear-oblique"
        else:
            cat = "rear"
        s["category"] = cat

    return {"reference_az_cam0_deg": cam0_az, "cameras": summary}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("calib_path")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cams = parse_calibration(Path(args.calib_path))
    result = classify_cameras(cams)
    print(f"Parsed {len(cams)} cameras from {args.calib_path}")
    print(f"Reference (cam 0) azimuth: {result['reference_az_cam0_deg']}\xb0")
    print()
    print(f"{'cam':>4} {'category':>14} {'rel_az':>8} {'height':>10} {'radius':>10}")
    for s in result["cameras"]:
        print(
            f"{s['cam_id']:>4} {s['category']:>14} {s['rel_az_from_cam0']:>8.1f}"
            f" {s['height_mm']:>10.1f} {s['radius_mm']:>10.1f}"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
