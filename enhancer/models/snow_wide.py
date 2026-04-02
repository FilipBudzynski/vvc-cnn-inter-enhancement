"""
Snow-Wide - Snow with Wide Context Path
- Adds 7x7 depthwise conv for larger VVC block context
- Additional pathway for global context
- Designed for larger patches (256x256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.PReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + x)


class WideContextModule(nn.Module):
    """7x7 depthwise convolution with dilation for wide context"""
    def __init__(self, channels: int):
        super().__init__()
        # Depthwise 7x7 with dilation=2 - captures larger structures (VVC blocks)
        # Effective receptive field: 7 + (7-1)*(2-1) = 13 pixels
        self.dw_conv = nn.Conv2d(channels, channels, 7, padding=6, groups=channels, dilation=2)
        self.bn = nn.BatchNorm2d(channels)
        self.conv_1x1 = nn.Conv2d(channels, channels, 1)
        self.relu = nn.PReLU()
        
        # Zero-init residual branch so it starts near identity
        # Prevents amplification of random features at initialization
        nn.init.zeros_(self.conv_1x1.weight)
        nn.init.zeros_(self.conv_1x1.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.dw_conv(x)
        out = self.bn(out)
        out = self.conv_1x1(out)
        return self.relu(x + out)  # residual


class FeatureExtractionModule(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.PReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.PReLU(),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class AlignmentModule(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )
        self.dcn = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, current: Tensor, reference: Tensor) -> Tensor:
        offset_input = torch.cat([current, reference], dim=1)
        offset = self.offset_conv(offset_input)
        aligned = self.dcn(reference + offset)
        return aligned


class AttentionFusion(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.PReLU(),
            nn.Conv2d(channels, 3, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, prev: Tensor, curr: Tensor, next: Tensor) -> Tensor:
        combined = torch.cat([prev, curr, next], dim=1)
        weights = self.attention(combined)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        w_prev, w_curr, w_next = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        result = w_prev * prev + w_curr * curr + w_next * next
        
        return curr + self.gamma * result


class MetadataAttention(nn.Module):
    def __init__(self, metadata_channels: int = 19, feature_channels: int = 64):
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
        
    def forward(self, features: Tensor, metadata: Tensor) -> Tensor:
        meta_feat = self.metadata_conv(metadata)
        combined = torch.cat([features, meta_feat], dim=1)
        attn = self.fusion(combined)
        return features * attn


class SnowWideEnhancer(nn.Module):
    """
    Snow-Wide - Snow with additional Wide Context path
    - Feature Extraction (same as Snow)
    - Wide Context (7x7 depthwise) for larger structures
    - Alignment (same as Snow)
    - Attention Fusion (same as Snow)
    - Deep Reconstruction (larger than Snow)
    """
    def __init__(self, config):
        super().__init__()
        base_channels = config.base_channels
        metadata_channels = config.metadata_channels
        
        # 1. Feature Extraction (same as original Snow)
        self.feature_extractor = FeatureExtractionModule(3, base_channels)
        
        # 2. NEW: Wide Context Module (7x7 depthwise)
        self.wide_context = WideContextModule(base_channels)
        
        # 3. Alignment (same as original Snow)
        self.align_prev = AlignmentModule(base_channels)
        self.align_next = AlignmentModule(base_channels)
        
        # 4. Attention Fusion (same as original Snow)
        self.attention_fusion = AttentionFusion(base_channels)
        
        # 5. Metadata Attention (same as original Snow)
        self.metadata_attention = MetadataAttention(metadata_channels, base_channels)
        
        # 6. Deep Reconstruction - MORE blocks than original Snow
        self.reconstruction = nn.Sequential(
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            WideContextModule(base_channels),  # Add wide context in middle too
            ResBlock(base_channels),
            ResBlock(base_channels),
        )
        
        # 7. Output
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels // 2, 3, 3, padding=1),
        )
        
    def forward(self, current: Tensor, prev: Tensor, next: Tensor, metadata: Tensor) -> Tensor:
        # 1. Feature extraction
        feat_curr = self.feature_extractor(current)
        feat_prev = self.feature_extractor(prev)
        feat_next = self.feature_extractor(next)
        
        # 2. Wide context (captures larger VVC block patterns)
        wide_curr = self.wide_context(feat_curr)
        
        # 3. Alignment
        aligned_prev = self.align_prev(feat_curr, feat_prev)
        aligned_next = self.align_next(feat_curr, feat_next)
        
        # 4. Attention fusion (with wide context added)
        fused = self.attention_fusion(aligned_prev, wide_curr, aligned_next)
        
        # 5. Metadata
        guided = self.metadata_attention(fused, metadata)
        
        # 6. Reconstruction
        reconstructed = self.reconstruction(guided)
        
        # 7. Output
        residual = self.output(reconstructed)
        return current + residual
