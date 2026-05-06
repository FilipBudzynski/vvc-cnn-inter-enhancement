"""
STENet (2024): Space-Time Enhancement Network — clean implementation.

Based on:
  "Joint Reference Frame Synthesis and Post Filter Enhancement for VVC"
  (arXiv:2404.18058)

Two-stage design:
  1. RFS (Reference Frame Synthesis) — given a temporal triplet
     (prev, curr, next) plus VVC decoder metadata, synthesise a
     "virtual reference" — i.e. a clean estimate of curr drawn from
     temporal neighbours.
  2. PFE (Post-Filter Enhancement) — concatenate the original
     reconstructed curr with the synthesised reference and metadata,
     and learn a residual that improves quality.

Forward signature follows the rest of the codebase:
    model(curr, prev, next, metadata) -> enhanced
where prev/curr/next are [B,3,H,W] in [0,1] and metadata is
[B,9,H,W] (the same 9-channel stack used by Martell / Snow-Wide).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.conv2(F.relu(self.conv1(x))), inplace=False)


class RFS(nn.Module):
    """Reference Frame Synthesis — produce a clean estimate of the current
    frame from its temporal neighbours."""

    def __init__(self, base_channels: int = 64, metadata_channels: int = 9, num_blocks: int = 4):
        super().__init__()
        in_channels = 3 + 3 + 3 + metadata_channels  # prev + curr + next + meta
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_blocks)])
        self.out = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, curr, prev, nxt, meta):
        x = torch.cat([prev, curr, nxt, meta], dim=1)
        f = F.relu(self.head(x), inplace=False)
        f = self.body(f)
        # Residual on the current reconstruction so the network only has
        # to learn the correction toward "clean curr".
        return (curr + self.out(f)).clamp(0, 1)


class PFE(nn.Module):
    """Post-Filter Enhancement — fuse the reconstructed current frame
    with the RFS-synthesised reference to produce the final output."""

    def __init__(self, base_channels: int = 64, metadata_channels: int = 9, num_blocks: int = 6):
        super().__init__()
        in_channels = 3 + 3 + metadata_channels  # curr + synth + meta
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_blocks)])
        self.out = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, curr, synth, meta):
        x = torch.cat([curr, synth, meta], dim=1)
        f = F.relu(self.head(x), inplace=False)
        f = self.body(f)
        return (curr + self.out(f)).clamp(0, 1)


class STENet2024(nn.Module):
    def __init__(self, config):
        super().__init__()
        base = getattr(config, "base_channels", 64)
        meta = getattr(config, "metadata_channels", 9)
        self.rfs = RFS(base_channels=base, metadata_channels=meta, num_blocks=4)
        self.pfe = PFE(base_channels=base, metadata_channels=meta, num_blocks=6)

    def forward(self, curr, prev, nxt, meta):
        synth = self.rfs(curr, prev, nxt, meta)
        enhanced = self.pfe(curr, synth, meta)
        # Returning (enhanced, synth) lets the trainer apply a second loss
        # on the synthesis stage. evaluate_bd.py only needs `enhanced`, so
        # we pack it as a tuple — caller picks the first element.
        return enhanced, synth


if __name__ == "__main__":
    class Cfg:
        base_channels = 64
        metadata_channels = 9

    m = STENet2024(Cfg())
    print(f"STENet2024 params: {sum(p.numel() for p in m.parameters()):,}")
    prev = torch.randn(2, 3, 132, 132)
    curr = torch.randn(2, 3, 132, 132)
    nxt = torch.randn(2, 3, 132, 132)
    meta = torch.randn(2, 9, 132, 132)
    enh, synth = m(curr, prev, nxt, meta)
    print(f"enhanced {enh.shape}  synth {synth.shape}")
