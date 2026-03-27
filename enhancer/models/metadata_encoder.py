"""
SOTA Metadata Encoder - treats metadata as guidance, not input channels
Based on MetaBit (WACV 2024) and CPGA (CVPR 2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetadataEncoderSOTA(nn.Module):
    """Process metadata into guidance signal"""
    def __init__(
        self,
        metadata_size: int = 8,
        guidance_channels: int = 32,
        use_mv_guidance: bool = True,
    ):
        super().__init__()
        
        self.use_mv_guidance = use_mv_guidance
        self.guidance_channels = guidance_channels
        
        # Simple projection from 8 -> guidance_channels
        self.proj = nn.Sequential(
            nn.Conv2d(metadata_size, guidance_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(guidance_channels, guidance_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """Process metadata into guidance signal"""
        return self.proj(metadata)
