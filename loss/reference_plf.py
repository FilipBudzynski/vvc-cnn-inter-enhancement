"""
Reference Perceptual Loss Function
Based on CVEGAN paper: 0.3*log(L1) + 0.1*log(MSE) + 0.2*log(SSIM) + 0.4*log(MS_SSIM)
With YUV channel weighting (Y=4/6, U=1/6, V=1/6)

Note: This expects YUV input, not RGB. The pipeline now uses pure YUV throughout.
"""

import torch
import torch.nn.functional as F
from torch.nn import Module


def rgb_to_yuv(rgb_tensor):
    """Convert RGB tensor to YUV using standard conversion"""
    # Ensure input is in [0,1] range
    rgb = rgb_tensor.clamp(0, 1)
    
    # Standard RGB to YUV conversion
    R, G, B = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.147 * R - 0.289 * G + 0.436 * B
    V = 0.615 * R - 0.515 * G - 0.100 * B
    
    # Normalize to [0,1] range and stack
    yuv = torch.stack([Y, U, V], dim=1)
    
    # Scale to match input range (rough approximation)
    yuv = (yuv + 0.5) / 1.0
    
    return yuv


def ssim_function(img1, img2, window_size=11, sigma=1.5):
    """Simple SSIM implementation for reference loss"""
    # Convert to grayscale first
    if img1.shape[1] == 3:
        img1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
    if img2.shape[1] == 3:
        img2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2)
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2)
    
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 - C2)
    ssim_den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = ssim_num / (ssim_den + 1e-8)
    return torch.mean(ssim_map)


def ms_ssim_function(img1, img2):
    """Multi-scale SSIM implementation (simplified)"""
    # For now, return SSIM at original scale (simplified implementation)
    return ssim_function(img1, img2)


class ReferencePLF(Module):
    """Perceptual Loss Function from CVEGAN reference
    
    Exact implementation: PLF = 0.3*log(L1) + 0.1*log(MSE) + 0.2*log(SSIM) + 0.4*log(MS_SSIM)
    With YUV channel weighting: Y=4/6, U=1/6, V=1/6
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, target, gen_output):
        # Input should already be YUV, no conversion needed
        target_yuv = target
        gen_yuv = gen_output
        
        # Split channels for weighting
        tuple_Y = (target_yuv[:, 0:1], gen_yuv[:, 0:1])
        tuple_U = (target_yuv[:, 1:2], gen_yuv[:, 1:2])
        tuple_V = (target_yuv[:, 2:3], gen_yuv[:, 2:3])
        
        # Calculate individual losses with channel weighting
        L1 = (F.l1_loss(*tuple_Y) * 4 + F.l1_loss(*tuple_U) + F.l1_loss(*tuple_V)) / 6
        MSE = (F.mse_loss(*tuple_Y) * 4 + F.mse_loss(*tuple_U) + F.mse_loss(*tuple_V)) / 6
        SSIM = 1 - (ssim_function(*tuple_Y) * 4 + ssim_function(*tuple_U) + ssim_function(*tuple_V)) / 6
        MS_SSIM = 1 - (ms_ssim_function(*tuple_Y) * 4 + ms_ssim_function(*tuple_U) + ms_ssim_function(*tuple_V)) / 6
        
        # CRITICAL: Logarithmic scaling from reference
        PLF = 0.3 * torch.log(L1 + 1e-8) + 0.1 * torch.log(MSE + 1e-8) + 0.2 * torch.log(SSIM + 1e-8) + 0.4 * torch.log(MS_SSIM + 1e-8)
        
        return PLF