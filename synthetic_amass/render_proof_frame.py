"""End-to-end proof-of-concept: render one synthetic frame at azimuth=180 (rear).

This script demonstrates the full pipeline architecture:
    1. Generate 3D skeleton in smplx 144-joint layout
       (synthetic for proof; will be replaced by SMPL+H forward pass once
       the model file is registered and downloaded).
    2. Compute ground-truth joint angles from the 3D positions.
    3. Map Halpe-26 keypoints from the 144-joint array.
    4. Synthesize head_top (the only Halpe-26 point not present in smplx output).
    5. Place virtual camera (azimuth=180, elevation=15, distance=4m, rear view).
    6. Project Halpe-26 3D positions to 2D pixel coordinates.
    7. Save (xy keypoints + GT angles) to proof_frame.json.
    8. Render a visualization PNG with the projected skeleton.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np

from camera import CameraIntrinsics, CameraPose, project_points
from halpe26_mapping import HALPE26_NAMES, HALPE26_TO_SMPLX_INDEX, get_mapping_info
from joint_angles import compute_joint_angles
from synthetic_skeleton import SyntheticPose, generate_synthetic_joints


OUTPUT_DIR: Final[Path] = Path(__file__).resolve().parent
PROOF_FRAME_JSON: Final[Path] = OUTPUT_DIR / "proof_frame.json"
PROOF_VIZ_PNG: Final[Path] = OUTPUT_DIR / "proof_frame_viz.png"


def map_to_halpe26(joints_144: np.ndarray) -> np.ndarray:
    """Build (26, 3) Halpe-26 3D keypoints from the (144, 3) smplx output."""
    out = np.zeros((26, 3), dtype=np.float64)
    for i, name in enumerate(HALPE26_NAMES):
        idx = HALPE26_TO_SMPLX_INDEX[name]
        if idx is not None:
            out[i] = joints_144[idx]
        elif name == "head_top":
            # Synthesize head_top: head joint (15) + skull height offset along +Y.
            head = joints_144[15]
            out[i] = head + np.array([0.0, 0.13, 0.0])
        else:
            raise ValueError(f"No source defined for Halpe-26 point: {name}")
    return out


def render_visualization(
    pixels_2d: np.ndarray,
    image_size: tuple[int, int],
    output_path: Path,
) -> bool:
    """Render Halpe-26 projected keypoints onto a blank canvas. Returns True if saved."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    width, height = image_size
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # flip Y so pixel (0,0) is top-left
    ax.set_aspect("equal")
    ax.set_facecolor("#101820")

    # Plot points + labels
    for i, name in enumerate(HALPE26_NAMES):
        x, y = pixels_2d[i]
        if not (0 <= x < width and 0 <= y < height):
            color = "#cc4444"  # off-screen
        else:
            color = "#22c55e"
        ax.scatter([x], [y], s=30, c=color, zorder=3)
        ax.annotate(
            name, (x, y), xytext=(4, -4), textcoords="offset points",
            color="white", fontsize=6,
        )

    # Draw a skeleton (limb edges).
    edges = [
        ("hip_center", "neck"),
        ("neck", "head_top"),
        ("neck", "L_sho"), ("neck", "R_sho"),
        ("L_sho", "L_elb"), ("L_elb", "L_wri"),
        ("R_sho", "R_elb"), ("R_elb", "R_wri"),
        ("hip_center", "L_hip"), ("hip_center", "R_hip"),
        ("L_hip", "L_kne"), ("L_kne", "L_ank"),
        ("R_hip", "R_kne"), ("R_kne", "R_ank"),
        ("L_ank", "L_heel"), ("L_ank", "L_big_toe"),
        ("R_ank", "R_heel"), ("R_ank", "R_big_toe"),
    ]
    name_to_idx = {n: i for i, n in enumerate(HALPE26_NAMES)}
    for a, b in edges:
        ia, ib = name_to_idx[a], name_to_idx[b]
        ax.plot(
            [pixels_2d[ia, 0], pixels_2d[ib, 0]],
            [pixels_2d[ia, 1], pixels_2d[ib, 1]],
            color="#60a5fa", linewidth=1.2, zorder=2,
        )

    ax.set_title("Synthetic AMASS-style proof frame — Rear view (az=180°)",
                 color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, facecolor="#101820")
    plt.close(fig)
    return True


def main() -> int:
    # Realistic pose: athletic squat-like stance.
    # Hip flexion ~45 deg, knee flexion ~80 deg, slight forward lean.
    pose = SyntheticPose(
        hip_flex_L=np.deg2rad(45.0),
        hip_flex_R=np.deg2rad(45.0),
        knee_flex_L=np.deg2rad(80.0),
        knee_flex_R=np.deg2rad(80.0),
        ankle_dorsiflex_L=np.deg2rad(10.0),
        ankle_dorsiflex_R=np.deg2rad(10.0),
        hip_adduct_L=np.deg2rad(0.0),
        hip_adduct_R=np.deg2rad(0.0),
        lumbar_extension=np.deg2rad(15.0),
    )

    # 1-3. Generate 3D skeleton + GT angles.
    joints_3d = generate_synthetic_joints(pose)
    gt_angles = compute_joint_angles(joints_3d)
    print(f"3D skeleton shape: {joints_3d.shape}")
    print(f"GT joint angles (deg): {gt_angles.to_dict()}")

    # 4. Map to Halpe-26 3D points.
    halpe26_3d = map_to_halpe26(joints_3d)
    mapping_info = get_mapping_info()
    print(
        f"Halpe-26 mapping: {mapping_info.direct_count} direct, "
        f"{mapping_info.synthesized_count} synthesized, total {mapping_info.total}"
    )

    # 5. Place rear-view camera.
    intr = CameraIntrinsics.default()
    cam_pose = CameraPose(
        azimuth_deg=180.0,
        elevation_deg=15.0,
        distance_m=4.0,
        look_at_world=np.array([0.0, 1.0, 0.0]),  # waist height
    )
    print(
        f"Camera: az={cam_pose.azimuth_deg}, el={cam_pose.elevation_deg}, "
        f"d={cam_pose.distance_m}m, intr=({intr.width}x{intr.height}, fx={intr.fx})"
    )

    # 6. Project to pixels.
    pixels_2d, depths = project_points(halpe26_3d, cam_pose, intr)
    in_frame = int(
        np.sum(
            (pixels_2d[:, 0] >= 0)
            & (pixels_2d[:, 0] < intr.width)
            & (pixels_2d[:, 1] >= 0)
            & (pixels_2d[:, 1] < intr.height)
        )
    )
    print(f"Projected {len(pixels_2d)} points, {in_frame}/{len(pixels_2d)} in frame")
    print(f"Depth range (m): {depths.min():.3f} to {depths.max():.3f}")

    # 7. Save proof_frame.json.
    proof_payload = {
        "schema_version": "1.0",
        "source": {
            "kind": "synthetic_procedural_skeleton",
            "note": (
                "Procedural skeleton in smplx 144-joint layout. "
                "Replace with smplx.create('smplh', ...) forward pass once "
                "SMPL+H model file is registered/downloaded."
            ),
        },
        "camera": {
            "azimuth_deg": cam_pose.azimuth_deg,
            "elevation_deg": cam_pose.elevation_deg,
            "distance_m": cam_pose.distance_m,
            "look_at_world": cam_pose.look_at_world.tolist(),
            "image_width": intr.width,
            "image_height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.cx,
            "cy": intr.cy,
        },
        "halpe26_keypoints_2d": [
            {
                "name": HALPE26_NAMES[i],
                "x": float(pixels_2d[i, 0]),
                "y": float(pixels_2d[i, 1]),
                "confidence": 1.0,
                "depth_m": float(depths[i]),
            }
            for i in range(len(HALPE26_NAMES))
        ],
        "halpe26_keypoints_3d_world": [
            {
                "name": HALPE26_NAMES[i],
                "x": float(halpe26_3d[i, 0]),
                "y": float(halpe26_3d[i, 1]),
                "z": float(halpe26_3d[i, 2]),
            }
            for i in range(len(HALPE26_NAMES))
        ],
        "gt_joint_angles_deg": gt_angles.to_dict(),
        "input_pose_params_deg": {
            "hip_flex_L": float(np.rad2deg(pose.hip_flex_L)),
            "hip_flex_R": float(np.rad2deg(pose.hip_flex_R)),
            "knee_flex_L": float(np.rad2deg(pose.knee_flex_L)),
            "knee_flex_R": float(np.rad2deg(pose.knee_flex_R)),
            "ankle_dorsiflex_L": float(np.rad2deg(pose.ankle_dorsiflex_L)),
            "ankle_dorsiflex_R": float(np.rad2deg(pose.ankle_dorsiflex_R)),
            "hip_adduct_L": float(np.rad2deg(pose.hip_adduct_L)),
            "hip_adduct_R": float(np.rad2deg(pose.hip_adduct_R)),
            "lumbar_extension": float(np.rad2deg(pose.lumbar_extension)),
        },
    }
    PROOF_FRAME_JSON.write_text(json.dumps(proof_payload, indent=2))
    print(f"Wrote: {PROOF_FRAME_JSON}")

    # 8. Render visualization.
    viz_ok = render_visualization(
        pixels_2d, (intr.width, intr.height), PROOF_VIZ_PNG
    )
    if viz_ok:
        print(f"Wrote: {PROOF_VIZ_PNG}")
    else:
        print("Visualization skipped (matplotlib unavailable).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
