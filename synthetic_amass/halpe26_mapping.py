"""Halpe-26 keypoint mapping from the smplx extended joint set (144 joints).

The smplx package, when called without a joint_mapper, returns 144 joints in
JOINT_NAMES order. The first 22 are SMPL+H body joints (pelvis..wrists). After
the hand joints, indices 55-65 are facial landmarks and feet derived from
specific mesh vertex IDs (see smplx.vertex_ids).

This module documents the mapping of Halpe-26 keypoints to indices in the
smplx output joints array, so the proof-of-concept and full pipeline use the
same mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# Halpe-26 keypoint order (matches the order used by the Couro pipeline).
HALPE26_NAMES: Final[tuple[str, ...]] = (
    "nose",
    "L_eye",
    "R_eye",
    "L_ear",
    "R_ear",
    "L_sho",
    "R_sho",
    "L_elb",
    "R_elb",
    "L_wri",
    "R_wri",
    "L_hip",
    "R_hip",
    "L_kne",
    "R_kne",
    "L_ank",
    "R_ank",
    "head_top",
    "neck",
    "hip_center",
    "L_big_toe",
    "R_big_toe",
    "L_small_toe",
    "R_small_toe",
    "L_heel",
    "R_heel",
)


# Mapping: Halpe-26 name -> index into smplx 144-joint output array.
# When index is None, synthesis is required (e.g., head_top from head joint).
# These indices come from smplx.joint_names.JOINT_NAMES.
HALPE26_TO_SMPLX_INDEX: Final[dict[str, int | None]] = {
    "nose": 55,
    "L_eye": 57,
    "R_eye": 56,
    "L_ear": 59,
    "R_ear": 58,
    "L_sho": 16,
    "R_sho": 17,
    "L_elb": 18,
    "R_elb": 19,
    "L_wri": 20,
    "R_wri": 21,
    "L_hip": 1,
    "R_hip": 2,
    "L_kne": 4,
    "R_kne": 5,
    "L_ank": 7,
    "R_ank": 8,
    "head_top": None,       # synthesized: head joint + ~0.13m along skull axis
    "neck": 12,
    "hip_center": 0,        # pelvis
    "L_big_toe": 60,
    "R_big_toe": 63,
    "L_small_toe": 61,
    "R_small_toe": 64,
    "L_heel": 62,
    "R_heel": 65,
}


@dataclass(frozen=True)
class JointMappingInfo:
    """Summary of how each Halpe-26 point is sourced."""

    direct_count: int
    synthesized_count: int
    total: int


def get_mapping_info() -> JointMappingInfo:
    direct = sum(1 for v in HALPE26_TO_SMPLX_INDEX.values() if v is not None)
    synth = sum(1 for v in HALPE26_TO_SMPLX_INDEX.values() if v is None)
    return JointMappingInfo(
        direct_count=direct,
        synthesized_count=synth,
        total=direct + synth,
    )
