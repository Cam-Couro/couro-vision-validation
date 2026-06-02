"""Minimal VideoPose3D (FAIR) inference module — 2D Halpe-26 keypoints → 3D H36M-17.

Reproduces the `TemporalModel` architecture from facebookresearch/VideoPose3D
so we can load the public pretrained checkpoint and run a 2D→3D lifting pass
on Couro's existing keypoint outputs. Pure PyTorch, runs on MPS or CPU.

Receptive field: 243 frames (kernel=3, dilations=[1,3,9,27,81]).
Input shape: (batch, n_frames, 17, 2) — normalized 2D keypoints
Output shape: (batch, n_outputs, 17, 3) — 3D joint positions in metres
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """Faithful reproduction of facebookresearch/VideoPose3D TemporalModel.

    Architecture:
        expand_conv (kernel=3, dilation=1)  →  BN  →  ReLU  →  Dropout
        for d in [3, 9, 27, 81]:
            conv (kernel=3, dilation=d)  →  BN  →  ReLU  →  Dropout
            conv (kernel=1)              →  BN  →  ReLU  →  Dropout
            + skip
        shrink (kernel=1)  →  17 × 3 output
    """

    def __init__(
        self,
        num_joints_in: int = 17,
        in_features: int = 2,
        num_joints_out: int = 17,
        filter_widths: tuple[int, ...] = (3, 3, 3, 3, 3),
        causal: bool = False,
        dropout: float = 0.25,
        channels: int = 1024,
    ):
        super().__init__()
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.pad = [filter_widths[0] // 2]

        self.expand_conv = nn.Conv1d(
            num_joints_in * in_features, channels, filter_widths[0], bias=False
        )
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)

        layers_conv = []
        layers_bn = []
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            layers_conv.append(
                nn.Conv1d(
                    channels, channels, filter_widths[i],
                    dilation=next_dilation, bias=False,
                )
            )
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, J, 2) → (B, J*2, T) for Conv1d
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        B, T = x.shape[0], x.shape[1]
        x = x.view(B, T, -1).permute(0, 2, 1).contiguous()  # (B, J*2, T)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            res = x[:, :, pad:x.shape[-1] - pad]
            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(
                self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x)))
            )
        x = self.shrink(x)  # (B, J*3, T_out)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.num_joints_out, 3)
        return x


def load_pretrained(checkpoint_path: Path, device: str = "cpu") -> TemporalModel:
    """Load the public pretrained_h36m_detectron_coco.bin checkpoint."""
    model = TemporalModel()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_pos"])
    model.to(device)
    model.eval()
    return model


# ============================================================================
# Halpe-26 → H36M-17 keypoint mapping
# ============================================================================

HALPE26 = {
    "nose": 0, "L_eye": 1, "R_eye": 2, "L_ear": 3, "R_ear": 4,
    "L_sho": 5, "R_sho": 6, "L_elb": 7, "R_elb": 8, "L_wri": 9, "R_wri": 10,
    "L_hip": 11, "R_hip": 12, "L_kne": 13, "R_kne": 14, "L_ank": 15, "R_ank": 16,
    "head_top": 17, "neck": 18, "hip_center": 19,
}

# H36M-17 names in order (Detectron/VideoPose3D convention)
H36M17_NAMES = (
    "hip", "R_hip", "R_kne", "R_ank",
    "L_hip", "L_kne", "L_ank",
    "spine", "thorax", "neck",
    "head", "L_sho", "L_elb", "L_wri",
    "R_sho", "R_elb", "R_wri",
)


def halpe26_to_h36m17(kp_halpe: np.ndarray) -> np.ndarray:
    """Map (..., 26, 2) Halpe-26 keypoints to (..., 17, 2) H36M-17 layout.

    H36M-17 indices use Couro's Halpe-26 keypoints directly where possible.
    For spine/neck (H36M intermediate points), interpolate from existing
    landmarks the same way the Detectron/CPN→H36M conversion does:
      spine  = midpoint(hip_center, neck)
      thorax = neck (Halpe26 has it directly)
      neck   = midpoint(neck, head_top)   (H36M "neck" sits between thorax and head)
    """
    h = lambda name: kp_halpe[..., HALPE26[name], :]
    out = np.stack([
        h("hip_center"),                                      # 0 hip
        h("R_hip"),                                            # 1
        h("R_kne"),                                            # 2
        h("R_ank"),                                            # 3
        h("L_hip"),                                            # 4
        h("L_kne"),                                            # 5
        h("L_ank"),                                            # 6
        (h("hip_center") + h("neck")) / 2,                    # 7 spine
        h("neck"),                                             # 8 thorax
        (h("neck") + h("head_top")) / 2,                      # 9 neck-H36M
        h("head_top"),                                         # 10 head
        h("L_sho"),                                            # 11
        h("L_elb"),                                            # 12
        h("L_wri"),                                            # 13
        h("R_sho"),                                            # 14
        h("R_elb"),                                            # 15
        h("R_wri"),                                            # 16
    ], axis=-2)
    return out


def normalize_screen_coords(kp: np.ndarray, w: int, h: int) -> np.ndarray:
    """VideoPose3D normalization: scale so image fills [-1, +1] in width."""
    out = kp.astype(np.float64).copy()
    out[..., :2] = out[..., :2] / w * 2.0 - np.array([1.0, h / w])
    return out


def lift_sequence(
    model: TemporalModel,
    kp_2d: np.ndarray,
    width: int,
    height: int,
    *,
    receptive_field: int = 243,
    device: str = "cpu",
) -> np.ndarray:
    """Run the lifter on a single video's keypoints.

    Args:
        kp_2d: (T, 26, 2) Halpe-26 pixel coords from RTMPose output
        width, height: image dimensions in pixels
    Returns:
        (T, 17, 3) 3D positions in metres, root (hip) at origin
    """
    kp_h36m_2d = halpe26_to_h36m17(kp_2d)  # (T, 17, 2)
    kp_norm = normalize_screen_coords(kp_h36m_2d, width, height)  # (T, 17, 2)

    T = kp_norm.shape[0]
    pad = receptive_field // 2
    # Replicate-pad along time axis so each input frame yields one output
    padded = np.concatenate([
        np.repeat(kp_norm[0:1], pad, axis=0),
        kp_norm,
        np.repeat(kp_norm[-1:], pad, axis=0),
    ], axis=0)  # (T + 2*pad, 17, 2)

    x = torch.from_numpy(padded).float().unsqueeze(0).to(device)  # (1, T+2*pad, 17, 2)
    with torch.no_grad():
        y = model(x)  # (1, T, 17, 3)
    return y.squeeze(0).cpu().numpy()
