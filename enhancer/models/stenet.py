"""
STENet: Space-Time Enhancement Network
Based on: "Joint Reference Frame Synthesis and Post Filter Enhancement for VVC" (2024)
arXiv: 2404.18058

Key differences from Martell/Snow-Wide:
- Uses 2 frames (not 3)
- Has Synthesis pipeline (for RFS) and Enhancement pipeline (for PFE)
- Optical flow between frames (or use metadata motion vectors)
- Joint training/inference capability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalFlowEstimator(nn.Module):
    """Fast optical flow estimation (IFRNet-based)"""
    
    def __init__(self, in_channels=3, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.flow_head = nn.Conv2d(hidden_channels, 2, 3, padding=1)  # 2D flow field
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, frame1, frame2):
        """
        Args:
            frame1: (B, 3, H, W)
            frame2: (B, 3, H, W)
        Returns:
            flow: (B, 2, H, W) - flow from frame1 to frame2
        """
        x = torch.cat([frame1, frame2], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        flow = self.flow_head(x)
        return flow


class EnhancementPipeline(nn.Module):
    """Enhancement Pipeline for PFE - enhances both input frames"""
    
    def __init__(self, in_channels=6, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_conv(in_channels, base_channels)
        self.enc2 = self._make_conv(base_channels, base_channels * 2, stride=2)
        self.enc3 = self._make_conv(base_channels * 2, base_channels * 4, stride=2)
        
        # Decoder with skip connections
        self.dec1 = self._make_conv(base_channels * 4, base_channels * 2)
        self.dec2 = self._make_conv(base_channels * 2 * 2, base_channels)  # *2 for skip
        self.dec3 = self._make_conv(base_channels * 2, base_channels)  # *2 for skip
        
        # Output
        self.output = nn.Conv2d(base_channels, 3, 3, padding=1)
        
        # Upsample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def _make_conv(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, I0, I1):
        """
        Args:
            I0: (B, 3, H, W) - First input frame
            I1: (B, 3, H, W) - Second input frame
        Returns:
            I0_enh, I1_enh: Enhanced frames
        """
        x = torch.cat([I0, I1], dim=1)  # (B, 6, H, W)
        
        # Encoder
        e1 = self.enc1(x)  # (B, 64, H, W)
        e2 = self.enc2(e1)  # (B, 128, H/2, W/2)
        e3 = self.enc3(e2)  # (B, 256, H/4, W/4)
        
        # Decoder with skip connections
        d1 = self.dec1(self.up(e3))  # (B, 128, H/2, W/2)
        d1 = torch.cat([d1, e2], dim=1)  # Skip connection
        
        d2 = self.dec2(d1)  # (B, 64, H/2, W/2)
        d2 = torch.cat([self.up(d2), e1], dim=1)  # Skip + upsample
        
        d3 = self.dec3(d2)  # (B, 64, H, W)
        
        # Output residual
        residual = self.output(d3)
        
        # Enhanced frames (residual + input)
        I0_enh = I0 + residual
        I1_enh = I1 + residual
        
        return I0_enh.clamp(0, 1), I1_enh.clamp(0, 1)


class SynthesisPipeline(nn.Module):
    """Synthesis Pipeline for RFS - synthesizes intermediate virtual reference frame"""
    
    def __init__(self, in_channels=6, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Conv2d(base_channels, 3, 3, padding=1)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, I0, I1):
        """
        Args:
            I0: (B, 3, H, W)
            I1: (B, 3, H, W)
        Returns:
            I_syn: Synthesized intermediate frame
        """
        x = torch.cat([I0, I1], dim=1)
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        b = self.bottleneck(e2)
        
        d1 = self.dec1(self.up(b))
        I_syn = self.dec2(d1)
        
        return I_syn.clamp(0, 1)


class STENet(nn.Module):
    """
    Space-Time Enhancement Network (STENet)
    For VVC enhancement - uses 2 frames (not 3 like Martell)
    
    Input: I0, I1 (2 compressed frames)
    Output: I0_enh, I_syn, I1_enh
    """
    
    def __init__(self, base_channels=64, use_optical_flow=True):
        super().__init__()
        
        self.use_optical_flow = use_optical_flow
        
        if use_optical_flow:
            self.flow_estimator = OpticalFlowEstimator(in_channels=3)
        
        self.enhancement_pipeline = EnhancementPipeline(
            in_channels=6,  # 2 frames * 3 channels
            base_channels=base_channels
        )
        
        self.synthesis_pipeline = SynthesisPipeline(
            in_channels=6,
            base_channels=base_channels
        )
    
    def forward(self, I0, I1, metadata=None):
        """
        Args:
            I0: (B, 3, H, W) - First frame
            I1: (B, 3, H, W) - Second frame
            metadata: Optional metadata (QP, frame type, etc.)
        Returns:
            I0_enh: Enhanced I0
            I_syn: Synthesized intermediate frame
            I1_enh: Enhanced I1
        """
        # Optional: use optical flow for alignment
        if self.use_optical_flow:
            flow_0to1 = self.flow_estimator(I0, I1)
            flow_1to0 = self.flow_estimator(I1, I0)
            # Could warp frames here if needed
        
        # Enhancement pipeline (PFE)
        I0_enh, I1_enh = self.enhancement_pipeline(I0, I1)
        
        # Synthesis pipeline (RFS)
        I_syn = self.synthesis_pipeline(I0, I1)
        
        return I0_enh, I_syn, I1_enh


if __name__ == "__main__":
    # Test
    model = STENet(base_channels=64)
    I0 = torch.randn(1, 3, 64, 64)
    I1 = torch.randn(1, 3, 64, 64)
    
    I0_enh, I_syn, I1_enh = model(I0, I1)
    print(f"I0_enh shape: {I0_enh.shape}")
    print(f"I_syn shape: {I_syn.shape}")
    print(f"I1_enh shape: {I1_enh.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
