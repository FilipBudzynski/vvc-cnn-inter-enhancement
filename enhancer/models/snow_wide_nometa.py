"""SnowWideEnhancerNoMeta: ablation of Martell-Hybrid that removes
the MetadataAttention block entirely. Same architecture otherwise.
"""

import torch
import torch.nn as nn
from torch import Tensor

from enhancer.models.snow_wide import (
    AlignmentModule,
    AttentionFusion,
    FeatureExtractionModule,
    ResBlock,
    WideContextModule,
)


class SnowWideEnhancerNoMeta(nn.Module):
    def __init__(self, config):
        super().__init__()
        base_channels = config.base_channels

        self.feature_extractor = FeatureExtractionModule(3, base_channels)
        self.wide_context = WideContextModule(base_channels)
        self.align_prev = AlignmentModule(base_channels)
        self.align_next = AlignmentModule(base_channels)
        self.attention_fusion = AttentionFusion(base_channels)

        self.reconstruction = nn.Sequential(
            ResBlock(base_channels), ResBlock(base_channels), ResBlock(base_channels),
            ResBlock(base_channels), ResBlock(base_channels), ResBlock(base_channels),
            ResBlock(base_channels), ResBlock(base_channels), ResBlock(base_channels),
            ResBlock(base_channels),
            WideContextModule(base_channels),
            ResBlock(base_channels), ResBlock(base_channels),
        )

        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels // 2, 3, 3, padding=1),
        )

    def forward(self, current: Tensor, prev: Tensor, next: Tensor,
                metadata: Tensor | None = None) -> Tensor:
        feat_curr = self.feature_extractor(current)
        feat_prev = self.feature_extractor(prev)
        feat_next = self.feature_extractor(next)

        wide_curr = self.wide_context(feat_curr)
        aligned_prev = self.align_prev(feat_curr, feat_prev)
        aligned_next = self.align_next(feat_curr, feat_next)
        fused = self.attention_fusion(aligned_prev, wide_curr, aligned_next)

        reconstructed = self.reconstruction(fused)
        residual = self.output(reconstructed)
        return current + residual


if __name__ == "__main__":
    class Cfg:
        base_channels = 64
    m = SnowWideEnhancerNoMeta(Cfg())
    n_nometa = sum(p.numel() for p in m.parameters())
    print(f"SnowWideEnhancerNoMeta params: {n_nometa:,}")

    from enhancer.models.snow_wide import SnowWideEnhancer
    class Cfg2:
        base_channels = 64
        metadata_channels = 9
    m2 = SnowWideEnhancer(Cfg2())
    n_full = sum(p.numel() for p in m2.parameters())
    print(f"SnowWideEnhancer (with meta) params: {n_full:,}")
    print(f"Difference (params in MetadataAttention): {n_full - n_nometa:,}")

    curr = torch.randn(2, 3, 132, 132)
    prev = torch.randn(2, 3, 132, 132)
    nxt  = torch.randn(2, 3, 132, 132)
    out = m(curr, prev, nxt)
    print(f"output: {out.shape}")
