"""
Loss Functions Module
==================

Multi-component loss functions for VTM enhancement training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from configs.training_config import LossWeights


class MultiComponentLoss(nn.Module):
    """
    Multi-component loss function combining MSE, L1, SSIM, and MS-SSIM
    Based on reference project loss combination
    """
    
    def __init__(self, weights: LossWeights):
        super().__init__()
        self.weights = weights
        
        # Basic losses
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # For SSIM losses, we'll use simple implementations for now
        # Could integrate the complex SSIM from training/ssim.py later
    
    def forward(self, enhanced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate multi-component loss
        
        Args:
            enhanced: Enhanced output [B, C, H, W]
            target: Target output [B, C, H, W]
        
        Returns:
            Combined loss tensor
        """
        # MSE loss
        mse = self.mse_loss(enhanced, target)
        
        # L1 loss
        l1 = self.l1_loss(enhanced, target)
        
        # Simple SSIM loss (1 - SSIM)
        ssim = self._simple_ssim(enhanced, target)
        ssim_loss = 1.0 - ssim
        
        # Simple MS-SSIM loss (average of multiple scales)
        mssim = self._simple_multiscale_ssim(enhanced, target)
        mssim_loss = 1.0 - mssim
        
        # Combine losses with weights
        total_loss = (
            self.weights.mse * mse +
            self.weights.l1 * l1 +
            self.weights.ssim * ssim_loss +
            self.weights.msssim * mssim_loss
        )
        
        return total_loss
    
    def _simple_ssim(self, enhanced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple SSIM implementation"""
        mu1 = enhanced.mean()
        mu2 = target.mean()
        sigma1 = enhanced.std()
        sigma2 = target.std()
        
        sigma12 = ((enhanced - mu1) * (target - mu2)).mean()
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
        )
        
        return ssim
    
    def _simple_multiscale_ssim(self, enhanced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple multi-scale SSIM implementation"""
        # Calculate SSIM at multiple scales
        scales = [1, 2, 4]
        ssim_values = []
        
        for scale in scales:
            if scale > 1:
                # Downsample
                size = (enhanced.shape[-2] // scale, enhanced.shape[-1] // scale)
                enhanced_scaled = F.interpolate(enhanced, size=size, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
            else:
                enhanced_scaled = enhanced
                target_scaled = target
            
            ssim_val = self._simple_ssim(enhanced_scaled, target_scaled)
            ssim_values.append(ssim_val)
        
        # Average across scales
        return torch.stack(ssim_values).mean()


class ChannelAwareLoss(nn.Module):
    """
    Channel-aware loss with different weights for Y, U, V channels
    Based on reference project's channel-wise gradients
    """
    
    def __init__(
        self, 
        weights: LossWeights,
        channel_weights: Optional[list] = None
    ):
        super().__init__()
        self.weights = weights
        self.channel_weights = channel_weights or [2/3, 1/6, 1/6]  # Y, U, V
        
        self.multi_loss = MultiComponentLoss(weights)
    
    def forward(self, enhanced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate channel-aware loss
        
        Args:
            enhanced: Enhanced output [B, C, H, W]
            target: Target output [B, C, H, W]
        
        Returns:
            Channel-weighted combined loss
        """
        # Calculate loss for each channel separately
        channel_losses = []
        for i in range(enhanced.shape[1]):
            channel_loss = self.multi_loss(
                enhanced[:, i:i+1], 
                target[:, i:i+1]
            )
            channel_losses.append(channel_loss * self.channel_weights[i])
        
        # Combine channel losses
        total_loss = sum(channel_losses)
        
        return total_loss


class ReferenceStyleLoss(nn.Module):
    """
    Reference project style loss combination with exact channel weighting
    Matches vvc-gan-decode-enhacement implementation exactly
    """
    def __init__(self, channels_grad_scales: tuple = (2/3, 1/6, 1/6)):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.channels_grad_scales = channels_grad_scales
        
    def forward(self, enhanced: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate reference-style losses with exact channel weighting
        
        Args:
            enhanced: Enhanced frame tensor [B, C, H, W]
            target: Target frame tensor [B, C, H, W]
            
        Returns:
            Dictionary with individual loss components
        """
        def split_channels(x):
            """Split Y, U, V channels"""
            return x[:, [0]], x[:, [1]], x[:, [2]]
        
        def to_tensor(x):
            """Convert to tensor on same device"""
            return torch.tensor(x, device=enhanced.device)
        
        # Split channels for per-channel calculation
        eY, eU, eV = split_channels(enhanced)
        oY, oU, oV = split_channels(target)
        
        channels_grad_scales = to_tensor(self.channels_grad_scales)
        
        # Calculate per-channel MSE loss
        mseY = self.mse_loss(eY, oY)
        mseU = self.mse_loss(eU, oU)
        mseV = self.mse_loss(eV, oV)
        mse_loss = (channels_grad_scales * to_tensor([mseY, mseU, mseV])).sum()
        
        # Calculate per-channel L1 loss
        l1Y = self.l1_loss(eY, oY)
        l1U = self.l1_loss(eU, oU)
        l1V = self.l1_loss(eV, oV)
        l1_loss = (channels_grad_scales * to_tensor([l1Y, l1U, l1V])).sum()
        
        # Calculate per-channel SSIM loss (1 - SSIM)
        ssimY = 1.0 - self._simple_ssim(eY, oY)
        ssimU = 1.0 - self._simple_ssim(eU, oU)
        ssimV = 1.0 - self._simple_ssim(eV, oV)
        ssim_loss = (channels_grad_scales * to_tensor([ssimY, ssimU, ssimV])).sum()
        
        # Calculate per-channel MS-SSIM loss (1 - MS-SSIM)
        mssimY = 1.0 - self._simple_multiscale_ssim(eY, oY)
        mssimU = 1.0 - self._simple_multiscale_ssim(eU, oU)
        mssimV = 1.0 - self._simple_multiscale_ssim(eV, oV)
        mssim_loss = (channels_grad_scales * to_tensor([mssimY, mssimU, mssimV])).sum()
        
        # Reference combination: 0.1*msssim + 0.1*ssim + mse + 0.5*l1
        total_loss = 0.1 * mssim_loss + 0.1 * ssim_loss + mse_loss + 0.5 * l1_loss
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'l1': l1_loss,
            'ssim_loss': ssim_loss,
            'mssim_loss': mssim_loss,
            'mseY': mseY,
            'mseU': mseU,
            'mseV': mseV,
            'l1Y': l1Y,
            'l1U': l1U,
            'l1V': l1V,
            'ssimY': ssimY,
            'ssimU': ssimU,
            'ssimV': ssimV,
            'mssimY': mssimY,
            'mssimU': mssimU,
            'mssimV': mssimV
        }
    
    def _simple_ssim(self, enhanced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple SSIM implementation"""
        mu1 = enhanced.mean()
        mu2 = target.mean()
        sigma1 = enhanced.std()
        sigma2 = target.std()
        
        sigma12 = ((enhanced - mu1) * (target - mu2)).mean()
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
        )
        
        return ssim
    
    def _simple_multiscale_ssim(self, enhanced: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple multi-scale SSIM implementation"""
        scales = [1, 2, 4]
        ssim_values = []
        
        for scale in scales:
            if scale > 1:
                size = (enhanced.shape[-2] // scale, enhanced.shape[-1] // scale)
                enhanced_scaled = F.interpolate(enhanced, size=size, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
            else:
                enhanced_scaled = enhanced
                target_scaled = target
            
            ssim_val = self._simple_ssim(enhanced_scaled, target_scaled)
            ssim_values.append(ssim_val)
        
        return torch.stack(ssim_values).mean()


def create_loss_function(loss_type: str, weights: LossWeights, channels_grad_scales: tuple = (2/3, 1/6, 1/6)):
    """
    Factory function to create loss functions
    
    Args:
        loss_type: Type of loss ('multi_component', 'channel_aware', 'reference_style', 'reference_plf')
        weights: Loss weights configuration
        channels_grad_scales: Channel gradient scales for reference-style loss (Y, U, V)
    
    Returns:
        Loss function module
    """
    if loss_type == "multi_component":
        return MultiComponentLoss(weights)
    elif loss_type == "channel_aware":
        return ChannelAwareLoss(weights)
    elif loss_type == "reference_style":
        return ReferenceStyleLoss(channels_grad_scales)
    elif loss_type == "reference_plf":
        from loss.reference_plf import ReferencePLF
        return ReferencePLF()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")