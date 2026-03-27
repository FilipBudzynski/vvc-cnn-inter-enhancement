import torch
import torch.nn as nn
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    Based on Piotr's thesis: L1 loss + Perceptual loss (VGG19)
    """
    def __init__(self):
        super().__init__()
        
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:21]  # Up to relu4_3
        self.vgg = vgg.eval()
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def yuv_to_rgb(self, yuv):
        """Convert YUV to RGB"""
        y, u, v = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
        r = y + 1.402 * (v - 0.5)
        g = y - 0.344136 * (u - 0.5) - 0.714136 * (v - 0.5)
        b = y + 1.772 * (u - 0.5)
        return torch.cat([r, g, b], dim=1)

    def forward(self, pred, target):
        """
        pred/target: [B, 3, H, W] in YUV format
        """
        if pred.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {pred.shape[1]}")
        if target.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {target.shape[1]}")
        
        pred_rgb = self.yuv_to_rgb(pred)
        target_rgb = self.yuv_to_rgb(target)
        
        pred_rgb = (pred_rgb - self.mean) / self.std
        target_rgb = (target_rgb - self.mean) / self.std
        
        pred_features = self.vgg(pred_rgb)
        target_features = self.vgg(target_rgb)
        
        return nn.functional.l1_loss(pred_features, target_features)
