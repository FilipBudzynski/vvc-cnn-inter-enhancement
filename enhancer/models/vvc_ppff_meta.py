"""VVC-PPFF-Meta: VVC-PPFF backbone enriched with Martell-style 9-channel
decoder metadata via mid-stream MetadataAttention.
"""

import torch
import torch.nn as nn
from torch import Tensor


class FeatureExtractionBlock(nn.Module):
    """3x3 conv + BN + PReLU, identical to vvc_ppff.FeatureExtractionBlock."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.prelu(self.bn(self.conv(x)))


class MetadataAttention(nn.Module):
    """
    Martell's metadata-attention pattern, ported to VVC-PPFF channel
    width. Projects 9-ch metadata -> C-ch features, concats with
    incoming features, learns a [B,C,H,W] gate via 1x1 convs, returns
    features * gate.
    """

    def __init__(self, metadata_channels: int, feature_channels: int):
        super().__init__()
        self.metadata_conv = nn.Sequential(
            nn.Conv2d(metadata_channels, feature_channels, 1),
            nn.PReLU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.PReLU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 1),
            nn.PReLU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
        )
        # Identity at init: gate = 2 * sigmoid(0) = 1
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.act = nn.Sigmoid()

    def forward(self, features: Tensor, metadata: Tensor) -> Tensor:
        meta_feat = self.metadata_conv(metadata)
        combined = torch.cat([features, meta_feat], dim=1)
        gate = 2.0 * self.act(self.fusion(combined))  # ≈ 1 at init
        return features * gate


class VVCPPFFMeta(nn.Module):
    """
    forward(curr, metadata) -> enhanced [B,3,H,W]
    The (prev, next) frames from the triplet are accepted by the eval
    harness but ignored: VVC-PPFF is single-frame.
    """

    def __init__(self, in_channels: int = 3, metadata_channels: int = 9,
                 base_channels: int = 128, num_blocks: int = 16,
                 attention_every: int = 4):
        super().__init__()
        self.num_blocks = num_blocks
        self.metadata_channels = metadata_channels
        self.attention_every = attention_every

        # Paper-faithful 4-ch input: YUV + QP map (chan 0 of metadata).
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, base_channels, 1),
            nn.PReLU(),
        )

        self.feature_blocks = nn.ModuleList(
            [FeatureExtractionBlock(base_channels) for _ in range(num_blocks)]
        )
        self.fusion_layers = nn.ModuleList(
            [nn.Conv2d(base_channels, base_channels, 1) for _ in range(num_blocks - 1)]
        )

        # MetadataAttention after every attention_every-th block (3, 7, 11, 15)
        n_attn = num_blocks // attention_every
        self.attention_blocks = nn.ModuleList(
            [MetadataAttention(metadata_channels, base_channels) for _ in range(n_attn)]
        )

        self.final_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        # Residual head (same as VVC-PPFF)
        self.output_conv_pre = nn.Conv2d(base_channels, base_channels, 1)
        self.tanh = nn.Tanh()
        self.output_conv_post = nn.Conv2d(base_channels, in_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Skip layers we deliberately zero-inited for identity start.
                if m.weight.numel() > 0 and float(m.weight.detach().abs().sum()) == 0.0:
                    continue
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, metadata: Tensor) -> Tensor:
        if metadata.dim() == 3:
            metadata = metadata.unsqueeze(0)
        original = x

        # Input: YUV + QP (paper-faithful)
        qp_map = metadata[:, 0:1, :, :]
        x = torch.cat([x, qp_map], dim=1)
        x = self.input_conv(x)

        # Feature extraction with progressive fusion + mid-stream metadata attention.
        prev_sum = x
        attn_idx = 0
        for i, block in enumerate(self.feature_blocks):
            x = block(x)
            if i < self.num_blocks - 1:
                refined = self.fusion_layers[i](x)
                x = refined + prev_sum
                prev_sum = prev_sum + refined
            # Apply metadata attention after every `attention_every` blocks.
            if (i + 1) % self.attention_every == 0 and attn_idx < len(self.attention_blocks):
                x = self.attention_blocks[attn_idx](x, metadata)
                attn_idx += 1

        x = self.final_conv(x)

        # Residual head (same as VVC-PPFF)
        x = self.output_conv_pre(x)
        x = self.tanh(x)
        residual = self.output_conv_post(x)

        return (original + residual).clamp(0, 1)


if __name__ == "__main__":
    m = VVCPPFFMeta()
    n = sum(p.numel() for p in m.parameters())
    print(f"VVC-PPFF-Meta params: {n:,}")
    x = torch.randn(2, 3, 132, 132)
    meta = torch.randn(2, 9, 132, 132)
    y = m(x, meta)
    print(f"output {y.shape}")
