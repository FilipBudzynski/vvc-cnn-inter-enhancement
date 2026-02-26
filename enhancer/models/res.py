import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments
from typing import Optional
from .config import NetworkConfig
from .dense import Classifier
from .conv import ConvLayer, OutputBlock, get_activation


def downsample_layer(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    """downsample_layer.

    :param in_channels:
    :type in_channels: int
    :param out_channels:
    :type out_channels: int
    :rtype: nn.Module
    """
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
    """ResLayer."""

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
            activation="none",
        )

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = downsample_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            )

        self.activation = get_activation(activation)

    def forward(self, _input: Tensor) -> Tensor:
        """forward.

        :param _input:
        :type _input: Tensor
        :rtype: Tensor
        """
        identity = _input

        out = self.conv1(_input)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(_input)

        out += identity
        out = self.activation(out)

        return out


class ResBlock(nn.Sequential):
    """ResBlock."""

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

        num_channels = in_channels

        for i in range(num_layers):
            layer = ResLayer(
                in_channels=num_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride if i == 0 else 1,
                padding=padding,
                reflect_padding=reflect_padding,
                activation=activation,
            )
            num_channels = out_channels
            self.add_module(f"dense_layer_{i}", layer)


class ResFeatures(nn.Module):
    @validate_arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
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

        if pool:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = None

        self._res = res

    def forward(self, _input: Tensor) -> Tensor:
        """forward.

        :param _input:
        :type _input: Tensor
        :rtype: Tensor
        """
        output = self.conv(_input)

        if self._res:
            identity = _input

            if self.downsample is not None:
                identity = self.downsample(_input)

            output += identity

        output = self.activation(output)

        if self.pool is not None:
            output = self.pool(output)

        return output


class ResNet(nn.Sequential):
    """
    ResNet-based network structure
    """

    @validate_arguments
    def __init__(
        self,
        config: NetworkConfig,
        initial_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        num_channels = initial_features or config.input_shape[2]

        if config.features:
            self.add_module(
                "channels",
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
            block = ResBlock(
                num_layers=block_config.num_layers,
                in_channels=num_channels,
                out_channels=block_config.features,
                kernel_size=block_config.kernel_size,
                stride=block_config.stride,
                padding=block_config.padding,
                reflect_padding=config.reflect_padding,
                activation=config.activation,
            )
            num_channels = block_config.features
            self.add_module(f"block{i}", block)

        if config.classifier:
            self.add_module(
                "classifier",
                Classifier(
                    in_channels=num_channels,
                    out_channels=config.classifier.features,
                    sigmoid=config.classifier.sigmoid,
                ),
            )

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
