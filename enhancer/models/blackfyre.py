"""
Blackfyre Model - Temporal Attention with Enhanced Decoder Metadata
Named after House Blackfyre from Game of Thrones

Uses:
- Temporal Attention to learn which frames to focus on
- Enhanced metadata (QP, Depth, SkipFlag, etc.)
- Metadata-guided attention weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock(nn.Module):
    """Residual block"""
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


class TemporalAttention(nn.Module):
    """Learn which frames to attend to - simplified version"""
    
    def __init__(self, channels: int = 64):
        super().__init__()
        
        # Conv layers to compute attention weights
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.PReLU(),
            nn.Conv2d(channels, 3, 1),  # 3 weights for 3 frames
            nn.Sigmoid(),
        )
        
        # Learnable weight
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, current: Tensor, prev: Tensor, next: Tensor) -> Tensor:
        """
        Args:
            current: [B, C, H, W]
            prev: [B, C, H, W]
            next: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        # Concatenate all frame features
        combined = torch.cat([prev, current, next], dim=1)  # [B, 3C, H, W]
        
        # Compute attention weights for each frame
        weights = self.attention_conv(combined)  # [B, 3, H, W]
        
        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Split weights
        w_prev, w_curr, w_next = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        
        # Apply weights
        result = w_prev * prev + w_curr * current + w_next * next
        
        # Residual
        return current + self.gamma * result


class MetadataAttention(nn.Module):
    """Use metadata (QP, Depth) to modulate features"""
    
    def __init__(self, metadata_channels: int = 16, feature_channels: int = 64):
        super().__init__()
        
        # Process metadata
        self.metadata_conv = nn.Conv2d(metadata_channels, feature_channels, 1)
        
        # Compute attention from combined features
        self.attention = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 1),
            nn.PReLU(),
            nn.Conv2d(feature_channels, 1, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, features: Tensor, metadata: Tensor) -> Tensor:
        """
        Args:
            features: [B, C, H, W]
            metadata: [B, M, H, W]
        Returns:
            [B, C, H, W]
        """
        # Transform metadata
        meta_feat = self.metadata_conv(metadata)  # [B, C, H, W]
        
        # Combine
        combined = torch.cat([features, meta_feat], dim=1)
        
        # Compute attention
        attn_weights = self.attention(combined)  # [B, 1, H, W]
        
        return features * attn_weights


class BlackfyreNet(nn.Module):
    """
    Blackfyre - Temporal Attention Network with Enhanced Metadata
    """
    
    def __init__(
        self,
        num_frames: int = 3,
        metadata_channels: int = 16,
        base_channels: int = 64,
    ):
        super().__init__()
        
        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, base_channels, 3, padding=1),
            nn.PReLU(),
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(channels=base_channels)
        
        # Metadata attention
        self.metadata_attention = MetadataAttention(metadata_channels if metadata_channels else 8, base_channels)
        
        # Main processing
        self.blocks = nn.Sequential(
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.PReLU(),
        )
        
        # Output
        self.output_conv = nn.Conv2d(base_channels // 2, 3, 3, padding=1)
        
    def forward(self, frames: list, metadata: Tensor) -> Tensor:
        prev_frame, curr_frame, next_frame = frames
        
        # Encode frames
        prev_feat = self.frame_encoder(prev_frame)
        curr_feat = self.frame_encoder(curr_frame)
        next_feat = self.frame_encoder(next_frame)
        
        # Temporal attention
        temporal_out = self.temporal_attention(curr_feat, prev_feat, next_feat)
        
        # Metadata attention
        enhanced = self.metadata_attention(temporal_out, metadata)
        
        # Process
        out = self.blocks(enhanced)
        
        # Output residual
        residual = self.output_conv(out)
        
        return curr_frame + residual


class BlackfyreEnhancer(nn.Module):
    """Wrapper for Blackfyre"""
    def __init__(self, config):
        super().__init__()
        
        metadata_channels = 19
        base_channels = getattr(config, 'base_channels', 64)
        
        self.model = BlackfyreNet(
            num_frames=3,
            metadata_channels=metadata_channels,
            base_channels=base_channels,
        )
        
    def forward(self, current_frame: Tensor, prev_frame: Tensor = None, 
               next_frame: Tensor = None, metadata: Tensor = None) -> Tensor:
        if prev_frame is None:
            prev_frame = current_frame
        if next_frame is None:
            next_frame = current_frame
        if metadata is None:
            metadata = torch.zeros(
                current_frame.shape[0], 16, 
                current_frame.shape[2], current_frame.shape[3],
                device=current_frame.device
            )
        
        return self.model([prev_frame, current_frame, next_frame], metadata)
