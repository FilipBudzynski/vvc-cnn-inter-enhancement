"""
VTM Reference Enhancer
======================

Reference-style DenseNet enhancer with VTM metadata integration
Based on vvc-gan-decode-enhacement repository architecture
Enhanced with VTM decoder block statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetadataEncoder(nn.Module):
    """Encodes VTM metadata (7 feature maps) to feature space"""
    
    def __init__(self, metadata_channels=7, metadata_features=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(metadata_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, metadata_features, 3, padding=1),
            nn.BatchNorm2d(metadata_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, metadata):
        return self.encoder(metadata)


class DenseLayer(nn.Module):
    """Dense layer: bottleneck -> 3x3 conv with concatenation"""
    
    def __init__(self, in_channels, growth_rate=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, 1),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1)
        )
    
    def forward(self, x):
        return torch.cat([x, self.net(x)], dim=1)


class DenseBlock(nn.Module):
    """Block of dense layers"""
    
    def __init__(self, in_channels, growth_rate=16, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate) 
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VTMReferenceEnhancer(nn.Module):
    """
    Reference-style DenseNet enhancer with VTM metadata integration
    
    Architecture based on reference project vvc-gan-decode-enhacement
    Enhanced with VTM decoder block statistics
    """
    
    def __init__(self, input_channels=3, metadata_channels=7, metadata_features=32):
        super().__init__()
        
        # Metadata encoder for VTM features
        self.metadata_encoder = MetadataEncoder(metadata_channels, metadata_features)
        
        # Initial convolution (large receptive field like reference)
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels + metadata_features, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Dense blocks (simplified from reference but maintains key properties)
        self.block1 = DenseBlock(64, growth_rate=16, num_layers=2)   # 96
        self.block2 = DenseBlock(96, growth_rate=16, num_layers=2)   # 128
        self.block3 = DenseBlock(128, growth_rate=16, num_layers=2)  # 160
        self.block4 = DenseBlock(160, growth_rate=16, num_layers=2)  # 192
        
        # Output layers
        self.output = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()  # Output in [0,1] range
        )
    
    def forward(self, yuv_input, metadata):
        """
        Args:
            yuv_input: [B, 3, H, W] - decoded YUV frame
            metadata: [B, 7, H, W] - VTM feature maps
        
        Returns:
            enhanced: [B, 3, H, W] - enhanced YUV frame
        """
        # Encode VTM metadata
        metadata_encoded = self.metadata_encoder(metadata)
        
        # Concatenate YUV + encoded metadata
        combined_input = torch.cat([yuv_input, metadata_encoded], dim=1)
        
        # Process through DenseNet backbone
        x = self.initial(combined_input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output(x)
        
        return x