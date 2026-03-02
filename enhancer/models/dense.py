import torch
import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments
from typing import Optional
from enhancer.config import NetworkConfig, TransitionMode
from enhancer.models.conv import ConvLayer, OutputBlock, Features, get_activation


class DenseLayer(nn.Sequential):
    """
    A single layer within a DenseBlock.
    It uses a bottleneck (1x1) to reduce computation before the 3x3 conv.
    """

    @validate_arguments
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        reflect_padding: bool = True,
        activation: str = "prelu",
        bn_size: float = 2.0,  # Multiplier for bottleneck channels
    ) -> None:
        super().__init__()

        # 1. Bottleneck layer: Reduces input depth to save memory
        self.add_module(
            "bottleneck",
            ConvLayer(
                in_channels=in_channels,
                out_channels=int(bn_size * growth_rate),
                kernel_size=1,
                stride=1,
                padding=0,
                reflect_padding=reflect_padding,
                activation=activation,
            ),
        )

        # 2. Main Convolution: Actually extracts the new features
        self.add_module(
            "conv",
            ConvLayer(
                in_channels=int(bn_size * growth_rate),
                out_channels=growth_rate,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout=dropout,
                reflect_padding=reflect_padding,
                activation=activation,
            ),
        )

    def forward(self, input: Tensor) -> Tensor:
        # The 'Dense' part: Concatenate input with the newly learned features
        output = super().forward(input)
        return torch.cat((input, output), 1)


class DenseBlock(nn.Sequential):
    """A sequence of DenseLayers where each layer sees everything before it."""

    @validate_arguments
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        reflect_padding: bool = True,
        activation: str = "prelu",
        bn_size: float = 2.0,
    ) -> None:
        super().__init__()

        num_features = in_channels
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels=num_features,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout=dropout,
                reflect_padding=reflect_padding,
                activation=activation,
                bn_size=bn_size,
            )
            self.add_module(f"dense_layer_{i}", layer)
            # Channel count grows by the growth_rate at every layer
            num_features += growth_rate


class Transition(nn.Sequential):
    """Reduces the number of channels between DenseBlocks to keep the model small."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        reflect_padding: bool = True,
        activation: str = "prelu",
        mode: TransitionMode = TransitionMode.same,
    ):
        super().__init__()
        out_channels = in_channels // 2

        self.add_module(
            "conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        if mode == TransitionMode.down:
            self.add_module("rescale", nn.AvgPool2d(kernel_size=2, stride=2))

        self.add_module("bn", nn.BatchNorm2d(out_channels))
        self.add_module("activation", get_activation(activation))


class DenseFeatures(Features):
    """Initial feature extraction for DenseNet."""

    def forward(self, _input: Tensor) -> Tensor:
        output = super().forward(_input)
        if hasattr(self, "_dense") and self._dense:
            return torch.cat((_input, output), 1)
        return output


class DenseNet(nn.Sequential):
    """The complete DenseNet architecture."""

    @validate_arguments
    def __init__(
        self,
        config: NetworkConfig,
        initial_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        # FIXED: Index [0] is Channels. Piotr had [2] (Width).
        num_features = initial_features or config.input_shape[0]

        # 1. Initial Feature Extraction
        if config.features:
            self.add_module(
                "features",
                DenseFeatures(
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
            # Set internal flag for concatenation
            self.features._dense = config.features.dense

            if config.features.dense:
                num_features += config.features.features
            else:
                num_features = config.features.features

        # 2. Add Dense Blocks and Transitions
        for i, block_config in enumerate(config.structure.blocks):
            block = DenseBlock(
                num_layers=block_config.num_layers,
                in_channels=num_features,
                growth_rate=block_config.features,
                kernel_size=block_config.kernel_size,
                stride=block_config.stride,
                padding=block_config.padding,
                dropout=block_config.dropout,
                reflect_padding=config.reflect_padding,
                activation=config.activation,
                bn_size=config.bn_size,
            )
            self.add_module(f"block{i}", block)
            num_features += block_config.features * block_config.num_layers

            # Optional Transition layer to shrink channel count
            if block_config.transition is not None:
                transition = Transition(
                    in_channels=num_features,
                    activation=config.activation,
                    mode=block_config.transition.mode,
                )
                num_features = num_features // 2
                self.add_module(f"transition{i}", transition)

        # 3. Final Output Block (Maps back to 3-channel YUV)
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
