import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Robust L1 loss that handles outliers better than MSE."""
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.eps = epsilon

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

class WeightedYUVLoss(nn.Module):
    """VVC-specific loss: Weights Y more than U/V channels."""
    def __init__(self, weights=[1.0, 0.5, 0.5]):
        super().__init__()
        self.weights = weights
        self.criterion = CharbonnierLoss()

    def forward(self, enhanced, target):
        loss = 0
        for i in range(3):
            loss += self.weights[i] * self.criterion(enhanced[:, [i]], target[:, [i]])
        return loss
