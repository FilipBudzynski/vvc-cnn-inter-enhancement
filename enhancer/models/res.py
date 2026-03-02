import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments
from typing import Optional

# Absolute imports for consistency
from enhancer.config import NetworkConfig
from enhancer.models.conv import ConvLayer, OutputBlock, get_activation


def downsample_layer(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    """Creates a 1x1 conv to match dimensions for the residual shortcut."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    )


class ResLayer(nn.Module):
    """A standard Residual Block: (Conv -> BN -> Act -> Conv -> BN) + Shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        reflect_padding: bool = True,
        activation: str = "prelu",
    ) -> None:
        super().__init__()

        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            reflect_padding=reflect_padding,
            activation=activation,
        )

        self.conv2 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            reflect_padding=reflect_padding,
            activation="none",  # Act comes after the addition
        )

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = downsample_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            )

        self.activation = get_activation(activation)

    def forward(self, input: Tensor) -> Tensor:
        identity = input

        out = self.conv1(input)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(input)

        out += identity
        if self.activation:
            out = self.activation(out)

        return out


class ResBlock(nn.Sequential):
    """A series of ResLayers."""

    @validate_arguments
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        reflect_padding: bool = True,
        activation: str = "prelu",
    ) -> None:
        super().__init__()

        curr_in = in_channels
        for i in range(num_layers):
            layer = ResLayer(
                in_channels=curr_in,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride if i == 0 else 1,  # Stride only on first layer
                padding=padding,
                reflect_padding=reflect_padding,
                activation=activation,
            )
            self.add_module(f"res_layer_{i}", layer)
            curr_in = out_channels


class ResFeatures(nn.Module):
    """Initial feature extraction for ResNet."""

    @validate_arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,  # Default to 1 to preserve detail
        padding: int = 3,
        reflect_padding: bool = True,
        activation: str = "prelu",
        pool: bool = False,
        res: bool = False,
    ) -> None:
        super().__init__()

        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            reflect_padding=reflect_padding,
            activation="none",
        )

        self.downsample = None
        if res and (in_channels != out_channels or stride != 1):
            self.downsample = downsample_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            )

        self.activation = get_activation(activation)
        self.pool = nn.MaxPool2d(3, 2, 1) if pool else None
        self.res_enabled = res

    def forward(self, input: Tensor) -> Tensor:
        output = self.conv(input)

        if self.res_enabled:
            identity = self.downsample(input) if self.downsample else input
            output += identity

        if self.activation:
            output = self.activation(output)
        if self.pool:
            output = self.pool(output)
        return output


class ResNet(nn.Sequential):
    """The ResNet-based model for VVC enhancement."""

    @validate_arguments
    def __init__(
        self,
        config: NetworkConfig,
        initial_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        # FIX: Index [0] for Channels
        num_channels = initial_features or config.input_shape[0]

        if config.features:
            self.add_module(
                "features_head",
                ResFeatures(
                    in_channels=num_channels,
                    out_channels=config.features.features,
                    kernel_size=config.features.kernel_size,
                    stride=config.features.stride,
                    padding=config.features.padding,
                    reflect_padding=config.reflect_padding,
                    activation=config.activation,
                    pool=config.features.pool,
                    res=config.features.res,
                ),
            )
            num_channels = config.features.features

        for i, block_config in enumerate(config.structure.blocks):
            self.add_module(
                f"block{i}",
                ResBlock(
                    num_layers=block_config.num_layers,
                    in_channels=num_channels,
                    out_channels=block_config.features,
                    kernel_size=block_config.kernel_size,
                    stride=block_config.stride,
                    padding=block_config.padding,
                    reflect_padding=config.reflect_padding,
                    activation=config.activation,
                ),
            )
            num_channels = block_config.features

        if config.output_block:
            self.add_module(
                "output_block",
                OutputBlock(
                    in_channels=num_channels,
                    out_channels=config.output_block.features,
                    kernel_size=config.output_block.kernel_size,
                    stride=config.output_block.stride,
                    padding=config.output_block.padding,
                    reflect_padding=config.reflect_padding,
                    tanh=config.output_block.tanh,
                ),
            )
