import torch
import torch.nn as nn
from torch import Tensor
from enhancer.models.dense import DenseNet
# from .res import ResNet
from enhancer.models.conv import ConvNet


class Enhancer(nn.Module):
    """
    Enhancer network specifically tuned for VVC 10-channel input.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.with_mask = config.with_mask

        # In datamodoule, we have 3 (YUV) + 7 (Metadata) = 10 channels.
        # We need to tell the sub-network (Dense/Res) exactly how many channels to expect.
        total_in_channels = config.input_shape[0]

        self.model = {
            "dense": DenseNet,
            "res": ResNet,
            "conv": ConvNet,
        }[config.implementation](
            config,
            initial_features=total_in_channels,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [Batch, 10, H, W] (Merged YUV and Metadata)
        """
        # 1. Separate the degraded YUV from the metadata for the residual connection
        # NOTE: We assume the first 3 channels are Y, U, V
        yuv_degraded = x[:, :3, :, :]

        # 2. Pass the full 10-channel stack through the CNN
        # The model 'sees' the metadata maps and uses them to decide where to fix the pixels.
        residual = self.model(x)

        # 3. Residual Learning (The 'Mask')
        # We learn the delta. Final = Degraded + Cleaned_Delta
        if self.with_mask:
            # Result = Original Pixels + Predicted Enhancement
            # This is mathematically: $$Y = X_{degraded} + f(X_{merged}, \theta)$$
            return torch.add(yuv_degraded, residual)

        return residual
