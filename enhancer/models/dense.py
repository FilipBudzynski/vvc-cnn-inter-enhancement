import torch
import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments
from typing import Optional
from .config import NetworkConfig, TransitionMode
from .conv import ConvLayer, OutputBlock, Features, get_activation


class DenseLayer(nn.Sequential):
    """DenseLayer."""

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
        bn_size: float = 2.0,
    ) -> None:
        super().__init__()
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

    def forward(self, _input: Tensor) -> Tensor:
        """forward.

        :param _input:
        :type _input: Tensor
        :rtype: Tensor
        """
        output = super().forward(_input)
        return torch.cat((_input, output), 1)


class DenseBlock(nn.Sequential):
    """DenseBlock."""

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
            num_features += growth_rate
            self.add_module(f"dense_layer_{i}", layer)


class Transition(nn.Sequential):
    """Transition."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 1,
        reflect_padding: bool = True,
        activation: str = "prelu",
        mode: TransitionMode = TransitionMode.same,
    ):
        super().__init__()
        out_channels = in_channels // 2

        if reflect_padding and padding > 0:
            self.add_module(
                "pad",
                nn.ReflectionPad2d(
                    padding,
                ),
            )

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if not reflect_padding else 0,
                bias=False,
            ),
        )

        if mode == TransitionMode.down:
            self.add_module(
                "rescale",
                nn.AvgPool2d(kernel_size=2, stride=2),
            )

        self.add_module(
            "bn",
            nn.BatchNorm2d(out_channels),
        )

        self.add_module("activation", get_activation(activation))


class Classifier(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sigmoid: bool = False,
    ) -> None:
        super().__init__()

        self.add_module(
            "pool2d",
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.add_module(
            "flatten",
            nn.Flatten(1),
        )

        self.add_module(
            "linear",
            nn.Linear(
                in_channels,
                out_channels,
            ),
        )

        if sigmoid:
            self.add_module(
                "sigmoid",
                nn.Sigmoid(),
            )


class DenseFeatures(Features):
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
        dense: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            reflect_padding=reflect_padding,
            activation=activation,
            pool=pool,
        )

        self._dense = dense

    def forward(self, _input: Tensor) -> Tensor:
        """forward.

        :param _input:
        :type _input: Tensor
        :rtype: Tensor
        """
        output = super().forward(_input)

        if self._dense:
            return torch.cat((_input, output), 1)

        return output


class DenseNet(nn.Sequential):
    """
    DenseNet-based network structure
    """

    def __init__(
        self,
        config: NetworkConfig,
        initial_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        num_features = initial_features or config.input_shape[2]

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
                    dense=config.features.dense,
                ),
            )

            if config.features.dense:
                num_features += config.features.features
            else:
                num_features = config.features.features

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
            num_features += block_config.features * block_config.num_layers
            self.add_module(f"block{i}", block)

            if block_config.transition is not None:
                transition = Transition(
                    in_channels=num_features,
                    kernel_size=block_config.transition.kernel_size,
                    stride=block_config.transition.stride,
                    padding=block_config.transition.padding,
                    reflect_padding=config.reflect_padding,
                    activation=config.activation,
                    mode=block_config.transition.mode,
                )
                num_features = num_features // 2
                self.add_module(f"transition{i}", transition)

        if config.classifier:
            self.add_module(
                "classifier",
                Classifier(
                    in_channels=num_features,
                    out_channels=config.classifier.features,
                    sigmoid=config.classifier.sigmoid,
                ),
            )

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
