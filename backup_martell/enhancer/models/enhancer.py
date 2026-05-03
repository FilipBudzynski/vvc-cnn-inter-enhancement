import torch
import torch.nn as nn
from torch import Tensor
from enhancer.models.dense import DenseNet
from enhancer.models.res import ResNet
from enhancer.models.conv import ConvNet
from enhancer.models.metadata_encoder import MetadataEncoderSOTA


class Enhancer(nn.Module):
    """SOTA-style Enhancer - treats metadata as guidance"""

    def __init__(self, config) -> None:
        super().__init__()
        self.with_mask = config.with_mask
        self.use_sota_metadata = getattr(config, 'use_sota_metadata', False)
        self.guidance_channels = config.metadata_features
        
        if self.use_sota_metadata:
            self.metadata_encoder = MetadataEncoderSOTA(
                metadata_size=config.metadata_size,
                guidance_channels=config.metadata_features,
            )
            # Model expects 3 + 32 = 35 channels
            input_channels = 3 + config.metadata_features
        else:
            input_channels = config.input_shape[0]
            
        self.model = self._build_base_model(config, initial_features=input_channels)

    def _build_base_model(self, config, initial_features: int):
        impl = config.implementation.value if hasattr(config.implementation, 'value') else config.implementation
        
        return {
            "dense": DenseNet,
            "res": ResNet,
            "conv": ConvNet,
        }[impl](config, initial_features=initial_features)

    def forward(self, x: Tensor, metadata: Tensor = None) -> Tensor:
        if self.use_sota_metadata:
            # SOTA mode: x is YUV (3 channels), metadata is separate
            if metadata is not None:
                # Generate guidance from metadata, concat with YUV
                guidance = self.metadata_encoder(metadata)
                combined = torch.cat([x, guidance], dim=1)
                residual = self.model(combined)
            else:
                # No metadata provided - use YUV only
                residual = self.model(x)
        else:
            # Legacy mode: x might be YUV+metadata concatenated
            if x.shape[1] > 3:
                # Concatenated format
                residual = self.model(x)
            else:
                residual = self.model(x)
        
        if self.with_mask:
            return torch.add(x, residual)
        return residual
