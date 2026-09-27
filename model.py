"""
Day 3 — model definition: reduced 3D U-Net.

Owns:
    - UNet3D: contracting path (double 3x3x3 conv + instance norm +
      leaky relu, then max-pool downsampling), a bottleneck, an expanding
      path (transposed-conv upsampling with skip-connection concatenation
      at matching depths), and a final 1x1x1 conv. Depth and base_filters
      come from config.yaml's `model` block — 4 levels / 24 base filters,
      deliberately smaller than the canonical U-Net (5 levels / 64 filters)
      for CPU/MPS feasibility on a 1-week timeline.
    - The model outputs raw logits (no final sigmoid) — combine with
      BCEWithLogitsLoss for numerical stability; use predict_proba() or
      apply torch.sigmoid() explicitly for Dice/IoU computation and
      inference.

    UNet2p5D (the 2.5D fallback) is NOT implemented — Day 1's compute
    check came back "full 3D feasible" (~0.1 min/epoch estimated on MPS),
    so the fallback was never needed. dataset.py made the same call.

Usage:
    from model import UNet3D
    model = UNet3D(in_channels=2, out_channels=1, base_filters=24, depth=4)
"""

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """Two 3x3x3 conv + instance norm + leaky relu blocks — the repeated
    unit at every depth of both the contracting and expanding paths."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """Reduced 3D U-Net.

    Channel progression for depth=4, base_filters=24:
        encoder:    in_channels -> 24 -> 48 -> 96 -> 192
        bottleneck: 192 -> 384
        decoder:    384 -> 192 -> 96 -> 48 -> 24 -> out_channels

    Filters double per downsampling level and halve per upsampling level,
    matching the standard U-Net pattern — just fewer levels and fewer
    filters per level than the canonical (depth=5, base=64) version.
    """

    def __init__(self, in_channels=2, out_channels=1, base_filters=24, depth=4):
        super().__init__()
        self.depth = depth

        enc_channels = [base_filters * (2 ** i) for i in range(depth)]
        bottleneck_channels = base_filters * (2 ** depth)

        # Contracting path
        self.enc_blocks = nn.ModuleList()
        prev_ch = in_channels
        for ch in enc_channels:
            self.enc_blocks.append(DoubleConv3D(prev_ch, ch))
            prev_ch = ch
        self.pool = nn.MaxPool3d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(enc_channels[-1], bottleneck_channels)

        # Expanding path
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        prev_ch = bottleneck_channels
        for ch in reversed(enc_channels):
            self.up_convs.append(nn.ConvTranspose3d(prev_ch, ch, kernel_size=2, stride=2))
            self.dec_blocks.append(DoubleConv3D(ch * 2, ch))  # *2 for the concatenated skip
            prev_ch = ch

        # Final 1x1x1 conv -> raw logits (no activation; see module docstring)
        self.final_conv = nn.Conv3d(enc_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc_block in self.enc_blocks:
            x = enc_block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up_conv, dec_block, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            x = up_conv(x)
            x = self._pad_to_match(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        return self.final_conv(x)

    @staticmethod
    def _pad_to_match(x, skip):
        """Patch dimensions aren't always evenly divisible by 2**depth (e.g.
        a padded-to-64 axis is fine, but real data won't always be this
        clean once full-volume sliding-window inference is added in Day 5).
        Pads x to match skip's spatial shape rather than assuming exact
        divisibility — cheap insurance against an off-by-one crash later."""
        diffs = [s - xs for s, xs in zip(skip.shape[2:], x.shape[2:])]
        if any(d != 0 for d in diffs):
            pad = []
            for d in reversed(diffs):  # F.pad expects last-dim-first ordering
                pad.extend([d // 2, d - d // 2])
            x = nn.functional.pad(x, pad)
        return x

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check — run directly: python model.py
# ---------------------------------------------------------------------------

def _sanity_check():
    from utils import load_config

    cfg = load_config()
    patch_size = tuple(cfg["patching"]["patch_size"])
    in_channels = cfg["model"]["in_channels"]
    out_channels = cfg["model"]["out_channels"]
    base_filters = cfg["model"]["base_filters"]
    depth = cfg["model"]["depth"]

    model = UNet3D(in_channels, out_channels, base_filters, depth)
    x = torch.randn(1, in_channels, *patch_size)
    out = model(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape[2:] == x.shape[2:], "output spatial dims don't match input — check padding logic"
    print("Output spatial dims match input — OK")

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,} ({n_params / 1e6:.2f}M)")


if __name__ == "__main__":
    _sanity_check()
