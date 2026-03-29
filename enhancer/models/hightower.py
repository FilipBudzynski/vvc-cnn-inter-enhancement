"""
Hightower Model - STDF-style Inter-Frame Enhancement
Named after House Hightower from Game of Thrones

Takes multiple neighboring frames (F-1, F0, F+1) and motion vectors as input,
uses temporal fusion to enhance the current frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HightowerBlock(nn.Module):
    """Residual block"""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.PReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class HightowerNet(nn.Module):
    """
    Hightower - Inter-frame enhancement network
    
    Input: [F-1, F0, F+1] YUV frames + motion vectors + metadata
    Output: Enhanced F0
    """
    def __init__(
        self,
        num_frames: int = 3,  # F-1, F0, F+1
        yuv_channels: int = 3,
        mv_channels: int = 4,
        metadata_channels: int = 4,
        base_channels: int = 64,
    ):
        super().__init__()
        self.num_frames = num_frames
        
        # Input: 3 frames * 3 channels + 4 MV + 4 metadata = 17 channels
        total_input_channels = (num_frames * yuv_channels) + mv_channels + metadata_channels
        
        # Initial feature extraction - process all channels together
        self.input_conv = nn.Sequential(
            nn.Conv2d(total_input_channels, base_channels, kernel_size=7, padding=3),
            nn.PReLU(),
        )
        
        # Main processing blocks
        self.blocks = nn.Sequential(
            HightowerBlock(base_channels, kernel_size=3),
            HightowerBlock(base_channels, kernel_size=3),
            HightowerBlock(base_channels, kernel_size=3),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        
        # Output - residual learning
        self.output_conv = nn.Conv2d(base_channels // 2, yuv_channels, kernel_size=3, padding=1)
        
    def forward(self, frames: list, motion_vectors: Tensor, metadata: Tensor, current_frame_idx: int = 1) -> Tensor:
        """
        frames: list of [B, 3, H, W] tensors [F-1, F0, F+1]
        motion_vectors: [B, 4, H, W] 
        metadata: [B, 4, H, W]
        current_frame_idx: which frame is the target (default 1 = F0)
        """
        # Get target frame size
        target_h, target_w = frames[current_frame_idx].shape[2], frames[current_frame_idx].shape[3]
        
        # Resize all inputs to match target frame size
        resized_frames = []
        for frame in frames:
            if frame.shape[2:] != (target_h, target_w):
                frame = F.interpolate(frame, size=(target_h, target_w), mode='bilinear', align_corners=False)
            resized_frames.append(frame)
        
        # Concatenate all frames
        x = torch.cat(resized_frames, dim=1)  # [B, 9, H, W]
        
        # Resize motion vectors and metadata if needed
        if motion_vectors.shape[2:] != (target_h, target_w):
            motion_vectors = F.interpolate(motion_vectors, size=(target_h, target_w), mode='bilinear', align_corners=False)
        if metadata.shape[2:] != (target_h, target_w):
            metadata = F.interpolate(metadata, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Concatenate: 9 + 4 + 4 = 17 channels
        x = torch.cat([x, motion_vectors, metadata], dim=1)
        
        # Initial features
        x = self.input_conv(x)
        
        # Main processing
        x = self.blocks(x)
        
        # Output residual
        residual = self.output_conv(x)
        
        # Add to current frame (F0)
        return frames[current_frame_idx] + residual


class HightowerEnhancer(nn.Module):
    """Wrapper for Hightower that handles metadata encoding"""
    def __init__(self, config):
        super().__init__()
        self.model = HightowerNet(
            num_frames=3,  # F-1, F0, F+1
            yuv_channels=3,
            mv_channels=4,  # MVL0_X, MVL0_Y, MVL1_X, MVL1_Y
            metadata_channels=4,  # QP, Depth, PredMode, Boundary
            base_channels=getattr(config, 'base_channels', 64),
        )
        
    def forward(self, current_frame: Tensor, prev_frame: Tensor = None, 
               next_frame: Tensor = None, motion_vectors: Tensor = None,
               metadata: Tensor = None) -> Tensor:
        """
        current_frame: [B, 3, H, W] - F0 (current frame to enhance)
        prev_frame: [B, 3, H, W] - F-1 (previous frame)
        next_frame: [B, 3, H, W] - F+1 (next frame)
        motion_vectors: [B, 4, H, W]
        metadata: [B, 4, H, W]
        """
        # Use current frame only if neighbors not available
        if prev_frame is None:
            prev_frame = current_frame
        if next_frame is None:
            next_frame = current_frame
        if motion_vectors is None:
            motion_vectors = torch.zeros(current_frame.shape[0], 4, *current_frame.shape[2:], 
                                       device=current_frame.device)
        if metadata is None:
            metadata = torch.zeros(current_frame.shape[0], 4, *current_frame.shape[2:],
                                 device=current_frame.device)
        
        return self.model([prev_frame, current_frame, next_frame], motion_vectors, metadata)
