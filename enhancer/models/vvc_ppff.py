"""
VVC-PPFF: Versatile Video Coding-Post Processing Feature Fusion
Implemented per paper: "Versatile Video Coding-Post Processing Feature Fusion"
- 4-channel input (YUV 3ch + QP map 1ch)
- 16 Feature Extraction blocks with 128 channels each
- Progressive feature fusion combining early + deep layers
- Residual learning with skip connection
- Trained per QP (22, 27, 32, 37, 42)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeatureExtractionBlock(nn.Module):
    """3x3 conv + PReLU - single block"""
    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.prelu(self.bn(self.conv(x)))


class VVCPPFF(nn.Module):
    """
    VVC-PPFF Model - Post Processing Feature Fusion
    Paper architecture:
    - Input: YUV (3ch) + QP map (1ch) = 4 channels total
    - Initial: 1x1 conv -> 128 channels + PReLU
    - 16 Feature Extraction blocks: 3x3 conv, 128 ch, PReLU
    - Feature Fusion: Progressive (early + deep layers combined)
    - Output: 1x1 conv + Tanh (residual)
    - Skip connection: input + residual
    """
    def __init__(self, in_channels: int = 4, base_channels: int = 128, num_blocks: int = 16):
        super().__init__()
        self.num_blocks = num_blocks
        
        # Initial: 1x1 conv to compress/expand
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.PReLU(),
        )
        
        # 16 Feature Extraction Blocks
        self.feature_blocks = nn.ModuleList([
            FeatureExtractionBlock(base_channels) for _ in range(num_blocks)
        ])
        
        # Feature Fusion layers (1x1 conv to combine features)
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels, 1) for _ in range(num_blocks - 1)
        ])
        
        # Final feature aggregation: 3x3 conv
        self.final_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        
        # Output: 1x1 conv to get residual, Tanh activation
        # Final 1x1 conv + Tanh (per paper line 519)
        self.output_conv_pre = nn.Conv2d(base_channels, base_channels, 1)
        self.tanh = nn.Tanh()
        self.output_conv_post = nn.Conv2d(base_channels, 3, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor, metadata: Tensor = None) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] - YUV frame
            metadata: [B, N, H, W] - QP map and other features (use ch0=QP)
        Returns:
            enhanced: [B, 3, H, W] - Enhanced YUV frame
        """
        original = x
        
        # If metadata provided, use QP map as 4th channel
        if metadata is not None:
            # Handle [N,H,W] or [B,N,H,W]
            if metadata.dim() == 3:
                metadata = metadata.unsqueeze(0)  # Add batch dim
            # Extract QP from metadata (channel 0)
            qp_map = metadata[:, 0:1, :, :]
            x = torch.cat([x, qp_map], dim=1)
        else:
            # Create default QP map (mid-value = 0.5 for QP 32)
            qp_map = torch.full_like(x[:, 0:1, :, :], 0.5)
            x = torch.cat([x, qp_map], dim=1)
        
        # Initial convolution
        x = self.input_conv(x)  # [B, 128, H, W]
        
        # Sequential fusion per block (Eq 2-3, OPTIMIZED with running sum)
        prev_sum = x
        for i, block in enumerate(self.feature_blocks):
            x = block(x)  # Process block
            
            # Eq 2: refine current, then Eq 3: add to accumulated previous
            if i < self.num_blocks - 1:
                refined = self.fusion_layers[i](x)  # Eq 2
                x = refined + prev_sum  # Eq 3: add accumulated
                prev_sum = prev_sum + refined  # Update accumulator (efficient!)
        
        # Final aggregation
        x = self.final_conv(x)
        
        # Output residual: 1x1 conv + Tanh (per paper line 519)
        x = self.output_conv_pre(x)
        x = self.tanh(x)
        residual = self.output_conv_post(x)
        
        # Skip connection: add residual to input
        enhanced = original + residual
        
        return enhanced.clamp(0, 1)


# Default config
def create_vvc_ppff():
    return VVCPPFF(in_channels=4, base_channels=128, num_blocks=16)
