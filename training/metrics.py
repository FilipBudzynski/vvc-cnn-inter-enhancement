"""
Metrics Module
===============

PSNR and SSIM evaluation utilities for VTM enhancement models
"""

import torch
import torch.nn.functional as F
from typing import Optional


def calculate_psnr(
    enhanced: torch.Tensor, 
    original: torch.Tensor, 
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        enhanced: Enhanced tensor [B, C, H, W]
        original: Original tensor [B, C, H, W]  
        max_val: Maximum possible pixel value (default 1.0 for normalized data)
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(enhanced, original, reduction='mean')
    if mse == 0:
        return torch.tensor(float('inf'))
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


def calculate_psnr_channelwise(
    enhanced: torch.Tensor, 
    original: torch.Tensor, 
    channel_weights: Optional[list] = None
) -> dict:
    """
    Calculate PSNR for each channel separately (Y, U, V)
    
    Args:
        enhanced: Enhanced tensor [B, C, H, W]
        original: Original tensor [B, C, H, W]
        channel_weights: Weights for each channel [Y, U, V]
    
    Returns:
        Dictionary with channel-wise PSNR values
    """
    if channel_weights is None:
        # Default weights from reference project: Y=2/3, U=1/6, V=1/6
        channel_weights = [2/3, 1/6, 1/6]
    
    psnr_values = {}
    for i, channel in enumerate(['Y', 'U', 'V']):
        psnr_val = calculate_psnr(enhanced[:, i:i+1], original[:, i:i+1])
        psnr_values[channel] = psnr_val.item()
    
    # Weighted average (like reference project)
    weighted_psnr = sum(w * psnr_values[ch] for w, ch in zip(channel_weights, ['Y', 'U', 'V']))
    psnr_values['weighted'] = weighted_psnr
    
    return psnr_values


def calculate_ssim(
    enhanced: torch.Tensor,
    original: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5
) -> torch.Tensor:
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        enhanced: Enhanced tensor [B, C, H, W]
        original: Original tensor [B, C, H, W]
        window_size: Window size for SSIM calculation
        sigma: Standard deviation for Gaussian window
    
    Returns:
        SSIM value [0, 1]
    """
    # Simple SSIM implementation (for now, could use the more complex version from trainer_module.py)
    mu1 = F.avg_pool2d(enhanced, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(original, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(enhanced * enhanced, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(original * original, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(enhanced * original, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def calculate_ssim_channelwise(
    enhanced: torch.Tensor,
    original: torch.Tensor,
    channel_weights: Optional[list] = None
) -> dict:
    """
    Calculate SSIM for each channel separately (Y, U, V)
    
    Args:
        enhanced: Enhanced tensor [B, C, H, W]
        original: Original tensor [B, C, H, W]
        channel_weights: Weights for each channel [Y, U, V]
    
    Returns:
        Dictionary with channel-wise SSIM values
    """
    if channel_weights is None:
        channel_weights = [2/3, 1/6, 1/6]
    
    ssim_values = {}
    for i, channel in enumerate(['Y', 'U', 'V']):
        if i < enhanced.shape[1]:  # Make sure channel exists
            ssim_val = calculate_ssim(enhanced[:, i:i+1], original[:, i:i+1])
            ssim_values[channel] = ssim_val.item()
    
    # Weighted average
    if len(ssim_values) == 3:
        weighted_ssim = sum(w * ssim_values[ch] for w, ch in zip(channel_weights, ['Y', 'U', 'V']))
        ssim_values['weighted'] = weighted_ssim
    
    return ssim_values


def batch_evaluate(
    enhanced: torch.Tensor,
    original: torch.Tensor
) -> dict:
    """
    Comprehensive evaluation with multiple metrics
    
    Args:
        enhanced: Enhanced tensor [B, C, H, W]
        original: Original tensor [B, C, H, W]
    
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {}
    
    # Overall PSNR and SSIM
    results['psnr'] = calculate_psnr(enhanced, original).item()
    results['ssim'] = calculate_ssim(enhanced, original).item()
    
    # Channel-wise metrics
    psnr_channels = calculate_psnr_channelwise(enhanced, original)
    ssim_channels = calculate_ssim_channelwise(enhanced, original)
    
    results.update({f'psnr_{ch}': val for ch, val in psnr_channels.items()})
    results.update({f'ssim_{ch}': val for ch, val in ssim_channels.items()})
    
    return results


def compare_with_baseline(
    enhanced: torch.Tensor,
    original: torch.Tensor,
    decoded: torch.Tensor,
    channel_weights: Optional[list] = None
) -> dict:
    """
    Compare enhanced results against baseline (decoded) and original
    
    Args:
        enhanced: Enhanced tensor [B, C, H, W]
        original: Original tensor [B, C, H, W]
        decoded: Decoded (baseline) tensor [B, C, H, W]
        channel_weights: Weights for each channel
    
    Returns:
        Dictionary with comparison metrics
    """
    if channel_weights is None:
        channel_weights = [2/3, 1/6, 1/6]
    
    results = {}
    
    # Enhanced vs Original
    enhanced_metrics = batch_evaluate(enhanced, original)
    results.update({f'enhanced_{k}': v for k, v in enhanced_metrics.items()})
    
    # Decoded vs Original (baseline)
    baseline_metrics = batch_evaluate(decoded, original)
    results.update({f'baseline_{k}': v for k, v in baseline_metrics.items()})
    
    # Improvement metrics
    improvement_psnr = enhanced_metrics['psnr'] - baseline_metrics['psnr']
    improvement_ssim = enhanced_metrics['ssim'] - baseline_metrics['ssim']
    
    results['psnr_improvement'] = improvement_psnr
    results['ssim_improvement'] = improvement_ssim
    
    return results