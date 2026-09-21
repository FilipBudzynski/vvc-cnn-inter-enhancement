"""SnowWideEnhancer-UNet: encoder-decoder variant with spatial bottleneck."""

import torch
import torch.nn as nn
from torch import Tensor

from enhancer.models.snow_wide import (
    AlignmentModule, AttentionFusion, FeatureExtractionModule,
    MetadataAttention, ResBlock,
)


class DownBlock(nn.Module):
    """Stride-2 conv + BN + PReLU, halves spatial dims."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    """ConvTranspose 4x4 stride 2 + BN + PReLU, doubles spatial dims."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class SnowWideEnhancerUNet(nn.Module):
    """
    Same outer skeleton as SnowWideEnhancer:
      1. Feature extraction (shared, 3 frames)
      2. Alignment (DCN-style) at full res
      3. Attention fusion at full res
    Replaces flat 12×ResBlock reconstruction with:
      4. U-Net encoder: 64 -> 96 (1/2) -> 128 (1/4)
      5. Bottleneck: 4× ResBlock @ 128, 1/4 res
      6. U-Net decoder with skip concat, back to 64 @ full res
      7. MetadataAttention at FULL res (preserve block-level precision)
      8. Output: 64 -> 32 -> 3, then residual add.
    """

    def __init__(self, config):
        super().__init__()
        c0 = config.base_channels       # full res, 64
        c1 = getattr(config, "unet_mid_channels", 96)   # 1/2 res
        c2 = getattr(config, "unet_bottom_channels", 128)  # 1/4 res
        n_bottleneck = getattr(config, "unet_bottleneck_blocks", 4)
        metadata_channels = 9

        # 1. Feature extraction (full res)
        self.feature_extractor = FeatureExtractionModule(3, c0)

        # 2. Alignment (full res)
        self.align_prev = AlignmentModule(c0)
        self.align_next = AlignmentModule(c0)

        # 3. Fusion of 3 temporally-aligned features (full res)
        self.attention_fusion = AttentionFusion(c0)

        # 4. Encoder: 2 downsampling levels
        self.down1 = DownBlock(c0, c1)      # full -> 1/2 res
        self.down2 = DownBlock(c1, c2)      # 1/2 -> 1/4 res

        # 5. Bottleneck @ 1/4 res
        self.bottleneck = nn.Sequential(*[ResBlock(c2) for _ in range(n_bottleneck)])

        # 6. Decoder
        self.up2 = UpBlock(c2, c1)          # 1/4 -> 1/2 res, joins enc1 (c1)
        self.dec1_fuse = nn.Conv2d(c1 + c1, c1, kernel_size=1)
        self.dec1_act = nn.PReLU()
        self.dec1_block = ResBlock(c1)

        self.up1 = UpBlock(c1, c0)          # 1/2 -> full res, joins enc0 (c0)
        self.dec0_fuse = nn.Conv2d(c0 + c0, c0, kernel_size=1)
        self.dec0_act = nn.PReLU()
        self.dec0_block = ResBlock(c0)

        # 7. Metadata attention at full res
        self.metadata_attention = MetadataAttention(metadata_channels, c0)

        # 8. Output (zero-init last conv so init is identity)
        self.output = nn.Sequential(
            nn.Conv2d(c0, c0 // 2, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(c0 // 2, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, current: Tensor, prev: Tensor, next: Tensor, metadata: Tensor) -> Tensor:
        feat_curr = self.feature_extractor(current)
        feat_prev = self.feature_extractor(prev)
        feat_next = self.feature_extractor(next)

        aligned_prev = self.align_prev(feat_curr, feat_prev)
        aligned_next = self.align_next(feat_curr, feat_next)

        enc0 = self.attention_fusion(aligned_prev, feat_curr, aligned_next)  # [B,c0,H,W]

        enc1 = self.down1(enc0)     # [B,c1,H/2,W/2]
        enc2 = self.down2(enc1)     # [B,c2,H/4,W/4]

        bottle = self.bottleneck(enc2)  # [B,c2,H/4,W/4]

        d1 = self.up2(bottle)       # [B,c1,H/2,W/2]
        d1 = self.dec1_act(self.dec1_fuse(torch.cat([d1, enc1], dim=1)))
        d1 = self.dec1_block(d1)

        d0 = self.up1(d1)           # [B,c0,H,W]
        d0 = self.dec0_act(self.dec0_fuse(torch.cat([d0, enc0], dim=1)))
        d0 = self.dec0_block(d0)

        guided = self.metadata_attention(d0, metadata)
        residual = self.output(guided)
        return current + residual
