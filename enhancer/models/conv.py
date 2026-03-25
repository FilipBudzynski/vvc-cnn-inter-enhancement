import torch.nn as nn
from pydantic import validate_arguments
from typing import Optional
from ..config import NetworkConfig


def get_activation(activation: str) -> Optional[nn.Module]:
    """Helper to return the requested activation function."""
    if activation == "prelu":
        return nn.PReLU()
    elif activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == "none":
        return None
    else:
        raise ValueError(f"Unknown activation: {activation}")


class ConvLayer(nn.Sequential):
    """The fundamental building block: Pad -> Conv -> BN -> Activation."""

    @validate_arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        reflect_padding: bool = True,
        activation: str = "prelu",
    ) -> None:
        super().__init__()

        # Use Reflection padding to avoid edge artifacts in video
        if reflect_padding and padding > 0:
            self.add_module("pad", nn.ReflectionPad2d(padding))

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if not reflect_padding else 0,
                bias=False,  # Bias is handled by BatchNorm
            ),
        )

        self.add_module("bn", nn.BatchNorm2d(out_channels))

        act = get_activation(activation)
        if act:
            self.add_module("activation", act)

        if dropout > 0:
            self.add_module("dropout", nn.Dropout2d(p=dropout))


class ConvBlock(nn.Sequential):
    """A series of ConvLayers."""

    @validate_arguments
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        reflect_padding: bool = True,
        activation: str = "prelu",
    ) -> None:
        super().__init__()

        # First layer handles the transition from in_channels to out_channels
        self.add_module(
            "conv0",
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout=dropout,
                reflect_padding=reflect_padding,
                activation=activation,
            ),
        )

        # Subsequent layers maintain the channel count
        for i in range(1, num_layers):
            self.add_module(
                f"conv{i}",
                ConvLayer(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dropout=dropout,
                    reflect_padding=reflect_padding,
                    activation=activation,
                ),
            )


class OutputBlock(nn.Sequential):
    """The final layer that maps hidden features to 3-channel YUV output."""

    @validate_arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,  # Usually 3 for YUV
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        reflect_padding: bool = True,
        tanh: bool = False,
    ) -> None:
        super().__init__()

        if reflect_padding and padding > 0:
            self.add_module("pad", nn.ReflectionPad2d(padding))

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if not reflect_padding else 0,
                bias=True,  # Final layer needs bias
            ),
        )

        if tanh:
            self.add_module("tanh", nn.Tanh())

        conv_module = self._modules["conv"]
        if isinstance(conv_module, nn.Conv2d):
            nn.init.zeros_(conv_module.weight)
            if conv_module.bias is not None:
                nn.init.zeros_(conv_module.bias)

class Features(ConvLayer):
    """The initial feature extraction layer."""

    @validate_arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,  # Changed from 2 to 1 to keep resolution
        padding: int = 3,
        reflect_padding: bool = True,
        activation: str = "prelu",
        pool: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            reflect_padding=reflect_padding,
            activation=activation,
        )

        if pool:
            self.add_module("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class ConvNet(nn.Sequential):
    """Standard CNN structure."""

    @validate_arguments
    def __init__(
        self,
        config: NetworkConfig,
        initial_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        num_features = initial_features or config.input_shape[0]

        # 1. Feature Extraction
        if config.features:
            self.add_module(
                "features",
                Features(
                    in_channels=num_features,
                    out_channels=config.features.features,
                    kernel_size=config.features.kernel_size,
                    stride=config.features.stride,
                    padding=config.features.padding,
                    reflect_padding=config.reflect_padding,
                    activation=config.activation,
                    pool=config.features.pool,
                ),
            )
            num_features = config.features.features

        # 2. Main Processing Blocks
        for i, block_config in enumerate(config.structure.blocks):
            self.add_module(
                f"block{i}",
                ConvBlock(
                    num_layers=block_config.num_layers,
                    in_channels=num_features,
                    out_channels=block_config.features,
                    kernel_size=block_config.kernel_size,
                    stride=block_config.stride,
                    padding=block_config.padding,
                    dropout=block_config.dropout,
                    reflect_padding=config.reflect_padding,
                    activation=config.activation,
                ),
            )
            num_features = block_config.features

        # 3. Output Reconstruction
        if config.output_block:
            self.add_module(
                "output_block",
                OutputBlock(
                    in_channels=num_features,
                    out_channels=config.output_block.features,
                    kernel_size=config.output_block.kernel_size,
                    stride=config.output_block.stride,
                    padding=config.output_block.padding,
                    reflect_padding=config.reflect_padding,
                    tanh=config.output_block.tanh,
                ),
            )
