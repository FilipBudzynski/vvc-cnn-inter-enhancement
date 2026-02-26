import torch
import torch.nn as nn
from torch import Tensor
from pydantic import validate_arguments

from .dense import DenseNet
from .res import ResNet
from .conv import ConvNet
from .config import EnhancerConfig, NetworkImplementation


class MetadataEncoder(nn.Module):
    """
    Encoder of metadata
    """

    @validate_arguments
    def __init__(
        self,
        metadata_size: int = 6,
        metadata_features: int = 64,
        size: int = 132,
        num_of_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.size = size

    def forward(self, x: Tensor, size: None | int | tuple[int, int] = None) -> Tensor:
        if size is None:
            size = self.size

        x = torch.nn.functional.interpolate(x, size=size)
        return x


class Enhancer(nn.Module):
    """
    Enhancer network
    """

    @validate_arguments
    def __init__(
        self,
        config: EnhancerConfig,
    ) -> None:
        super().__init__()

        self.with_mask = config.with_mask

        self.metadata_encoder = MetadataEncoder(
            metadata_size=config.metadata_size,
            metadata_features=config.metadata_features,
            size=config.input_shape[0],
        )

        num_features = config.metadata_features + config.input_shape[2]

        self.model = {
            NetworkImplementation.DENSE: DenseNet,
            NetworkImplementation.RES: ResNet,
            NetworkImplementation.CONV: ConvNet,
        }[config.implementation](
            config,
            initial_features=num_features,
        )

    def forward(self, input_: Tensor, metadata: Tensor) -> Tensor:
        shape = input_.shape[2:]
        encoded_metadata = self.metadata_encoder(metadata, shape)
        data = torch.cat((input_, encoded_metadata), 1)
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

    g = Enhancer(config.enhancer)
    result = g(torch.rand((1, 3, 132, 132)), torch.rand((1, 6, 1, 1)))
    print(result.shape)

    summary(g, [(3, 132, 132), (6, 1, 1)], device="cpu", depth=10)
    # summary(g, [(3, 1920, 1080), (6, 1, 1)], device="cpu", depth=10)
