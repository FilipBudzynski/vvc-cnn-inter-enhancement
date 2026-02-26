import torch.nn as nn
import torch
from .dense import DenseNet
from .res import ResNet
from .conv import ConvNet
from .config import DiscriminatorConfig, NetworkImplementation


class Discriminator(nn.Module):
    def __init__(
        self,
        config: DiscriminatorConfig,
    ):
        super().__init__()

        self.model = {
            NetworkImplementation.DENSE: DenseNet,
            NetworkImplementation.RES: ResNet,
            NetworkImplementation.CONV: ConvNet,
        }[config.implementation](
            config,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    from torchsummary import summary
    import sys
    from .config import Config

    config = Config.load(sys.argv[1])

    g = Discriminator(config.discriminator)
    random_image = torch.rand((132, 3, 132, 132))
    print(g(random_image).shape)

    summary(g, (3, 132, 132), device="cpu", depth=10)
