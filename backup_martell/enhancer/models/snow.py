"""
Snow Model - Enhanced Blackfyre with:
1. Feature Extraction Module - extract features from Ft, Ft-1, Ft+1
2. Alignment Module (DCN v2) - align neighboring features to current frame
3. Attention Fusion - learned weighting of frames
4. Reconstruction (Residual Blocks) - deep residual blocks

Named after House Stark from Game of Thrones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock(nn.Module):
    """Residual block with BatchNorm"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.PReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DenseBlock(nn.Module):
    """Dense block for feature extraction"""
    def __init__(self, channels: int, growth_rate: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.relu = nn.PReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(torch.cat([x, out1], dim=1))))
        return torch.cat([x, out1, out2], dim=1)


class FeatureExtractionModule(nn.Module):
    """Extract features from each frame independently"""
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        # Shared feature extractor for all frames
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
    """DCN v2-style alignment - deformable convolution to align features"""
    def __init__(self, channels: int = 64):
        super().__init__()
        # Offset prediction network
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )
        # Deformable convolution
        self.dcn = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, current: Tensor, reference: Tensor) -> Tensor:
        """
        Args:
            current: features from current frame [B, C, H, W]
            reference: features from neighboring frame [B, C, H, W]
        Returns:
            aligned features of reference frame
        """
        # Predict offset based on difference
        offset_input = torch.cat([current, reference], dim=1)
        offset = self.offset_conv(offset_input)
        
        # Apply deformable convolution (simplified - using regular conv + offset)
        # For full DCN we'd use deformable_conv2d from torchvision.ops
        aligned = self.dcn(reference + offset)
        
        return aligned


class AttentionFusion(nn.Module):
    """Learn which frame to focus on - pixel-wise attention"""
    def __init__(self, channels: int = 64):
        super().__init__()
        # Attention network
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.PReLU(),
            nn.Conv2d(channels, 3, 1),  # 3 weights: prev, curr, next
            nn.Sigmoid(),
        )
        # Learnable gamma for residual connection
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, prev: Tensor, curr: Tensor, next: Tensor) -> Tensor:
        """
        Args:
            prev: [B, C, H, W] - features from Ft-1
            curr: [B, C, H, W] - features from Ft
            next: [B, C, H, W] - features from Ft+1
        Returns:
            fused features with attention
        """
        combined = torch.cat([prev, curr, next], dim=1)
        weights = self.attention(combined)
        
        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Apply attention
        w_prev, w_curr, w_next = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        
        result = w_prev * prev + w_curr * curr + w_next * next
        
        # Residual connection
        return curr + self.gamma * result


class MetadataAttention(nn.Module):
    """Use metadata (QP, Depth) to modulate features"""
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


class SnowEnhancer(nn.Module):
    """
    Snow - Enhanced Blackfyre with:
    - Feature Extraction for each frame
    - DCN v2 Alignment for motion compensation
    - Attention Fusion for frame weighting
    - Deep Reconstruction with residual blocks
    """
    def __init__(self, config):
        super().__init__()
        base_channels = config.base_channels
        metadata_channels = config.metadata_channels
        
        # 1. Feature Extraction Module (shared encoder for all frames)
        self.feature_extractor = FeatureExtractionModule(3, base_channels)
        
        # 2. Alignment Module (align neighboring frames to current)
        self.align_prev = AlignmentModule(base_channels)
        self.align_next = AlignmentModule(base_channels)
        
        # 3. Attention Fusion
        self.attention_fusion = AttentionFusion(base_channels)
        
        # 4. Metadata Attention (use VVC decoder metadata)
        self.metadata_attention = MetadataAttention(metadata_channels, base_channels)
        
        # 5. Deep Reconstruction (more ResBlocks than original Blackfyre)
        self.reconstruction = nn.Sequential(
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
        )
        
        # 6. Output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(base_channels // 2, 3, 3, padding=1),
        )
        
    def forward(self, current: Tensor, prev: Tensor, next: Tensor, metadata: Tensor) -> Tensor:
        """
        Args:
            current: [B, 3, H, W] - current frame YUV
            prev: [B, 3, H, W] - previous frame
            next: [B, 3, H, W] - next frame
            metadata: [B, 19, H, W] - VVC decoder metadata
        Returns:
            enhanced frame [B, 3, H, W]
        """
        # 1. Extract features from all frames
        feat_curr = self.feature_extractor(current)
        feat_prev = self.feature_extractor(prev)
        feat_next = self.feature_extractor(next)
        
        # 2. Align neighboring features to current frame
        aligned_prev = self.align_prev(feat_curr, feat_prev)
        aligned_next = self.align_next(feat_curr, feat_next)
        
        # 3. Attention fusion - learned weighting
        fused = self.attention_fusion(aligned_prev, feat_curr, aligned_next)
        
        # 4. Apply metadata attention
        guided = self.metadata_attention(fused, metadata)
        
        # 5. Deep reconstruction
        reconstructed = self.reconstruction(guided)
        
        # 6. Output with residual connection
        residual = self.output(reconstructed)
        return current + residual
