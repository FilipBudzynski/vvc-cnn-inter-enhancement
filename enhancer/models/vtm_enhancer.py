import torch
import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments

from .dense import DenseNet
from .res import ResNet
from .conv import ConvNet
from ..config import NetworkConfig, NetworkImplementation, EnhancerConfig


class VTMFeatureEncoder(nn.Module):
    """Encodes VTM decoder feature maps to feature space"""
    
    def __init__(
        self, 
        input_features: int,  
        output_features: int = 32,
        size: int = 132
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_features, output_features, 3, padding=1),
            nn.BatchNorm2d(output_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_features, output_features, 3, padding=1),
            nn.BatchNorm2d(output_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class MetadataEncoder(nn.Module):
    """
    Encoder of basic metadata (QP, intra flag)
    """
    
    @validate_arguments
    def __init__(
        self,
        metadata_size: int = 2,  # [QP, is_intra]
        metadata_features: int = 6,
        size: int = 132,
        num_of_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.size = size
        self.encoder = nn.Sequential(
            nn.Conv2d(metadata_size, metadata_features, 3, padding=1),
            nn.BatchNorm2d(metadata_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(metadata_features, metadata_features, 3, padding=1),
            nn.BatchNorm2d(metadata_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor, size: None | int | tuple[int, int] = None) -> Tensor:
        if size is None:
            size = self.size

        x = torch.nn.functional.interpolate(x, size=size)
        return x


class Enhancer(nn.Module):
    """
    Enhanced VVC enhancer with VTM decoder statistics integration
    """

    def __init__(
        self,
        config: EnhancerConfig,
        use_vtm_features: bool = True,
        vtm_feature_channels: int = 10,  # Number of VTM feature channels
    ) -> None:
        super().__init__()

        self.with_mask = config.with_mask
        self.use_vtm_features = use_vtm_features

        # Basic metadata encoder
        self.metadata_encoder = MetadataEncoder(
            metadata_size=2,  # [QP, is_intra]
            metadata_features=config.metadata_features,
            size=config.input_shape[0],
        )
        
        # VTM feature encoder
        self.vtm_feature_encoder = None
        if self.use_vtm_features:
            self.vtm_feature_encoder = VTMFeatureEncoder(
                input_features=vtm_feature_channels,
                output_features=config.metadata_features,
                size=config.input_shape[0],
            )

        # Calculate input channels
        num_features = config.input_shape[2]  # RGB channels
        num_features += config.metadata_features  # Basic metadata
        
        if self.use_vtm_features and self.vtm_feature_encoder:
            num_features += config.metadata_features  # VTM features

        self.model = {
            NetworkImplementation.DENSE: DenseNet,
            NetworkImplementation.RES: ResNet,
            NetworkImplementation.CONV: ConvNet,
        }[config.implementation](
            config,
            initial_features=num_features,
        )

    def forward(self, input_: Tensor, metadata: Tensor, vtm_features: Tensor = None) -> Tensor:
        shape = input_.shape[2:]
        
        # Encode basic metadata
        encoded_metadata = self.metadata_encoder(metadata, shape)
        
        # Combine input with basic metadata
        data = torch.cat((input_, encoded_metadata), 1)
        
        # Add VTM features if available
        if self.use_vtm_features and self.vtm_feature_encoder and vtm_features is not None:
            encoded_vtm = self.vtm_feature_encoder(vtm_features)
            data = torch.cat((data, encoded_vtm), 1)

        result = self.model(data)

        if self.with_mask:
            with_mask = torch.add(input_, result)
            return with_mask

        return result


if __name__ == "__main__":
    from torchsummary import summary
    import sys
    from .config import Config

    config = Config.load(sys.argv[1])

    # Test with VTM features
    g = Enhancer(config.enhancer, use_vtm_features=True, vtm_feature_channels=10)
    
    # Mock inputs
    rgb_input = torch.rand((1, 3, 132, 132))
    metadata = torch.rand((1, 2, 1, 1))  # QP, is_intra
    vtm_features = torch.rand((1, 10, 132, 132))  # 10 VTM feature channels
    
    result = g(rgb_input, metadata, vtm_features)
    print(f"Result shape: {result.shape}")

    summary(g, [(3, 132, 132), (2, 1, 1), (10, 132, 132)], device="cpu", depth=10)
