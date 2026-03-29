#!/usr/bin/env python3
"""
Visualize Blackfyre model results - generate side-by-side comparisons
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from enhancer.models.blackfyre import BlackfyreEnhancer
from enhancer.dataset_blackfyre import BlackfyreDataset
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 132

class BlackfyreConfig:
    base_channels = 64
    metadata_channels = 19

model = BlackfyreEnhancer(BlackfyreConfig()).to(DEVICE)

# Load best checkpoint
checkpoint_path = "checkpoints/blackfyre_epoch_170.pt"  # Best epoch
print(f"Loading: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()

# Dataset
val_dataset = BlackfyreDataset(data_dir="data/precomputed", patch_size=PATCH_SIZE, split="val")
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# Metrics
psnr_metric_enhanced = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
psnr_metric_input = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

def yuv_to_rgb(yuv):
    """Convert YUV to RGB"""
    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]
    
    r = y + 1.402 * (v - 0.5)
    g = y - 0.344136 * (u - 0.5) - 0.714136 * (v - 0.5)
    b = y + 1.772 * (u - 0.5)
    
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 1)

def to_rgb(tensor):
    """Convert YUV tensor to RGB numpy array"""
    img = tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    return yuv_to_rgb(img)

def visualize_batch(batch_idx=0, num_samples=8):
    """Visualize predictions for a batch"""
    
    total_psnr_gain = 0
    total_ssim = 0
    count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx > 0:
                break
                
            (prev_frames, curr_frames, next_frames), original, features, info = batch
            
            prev_frames = prev_frames.to(DEVICE)
            curr_frames = curr_frames.to(DEVICE)
            next_frames = next_frames.to(DEVICE)
            original = original.to(DEVICE)
            features = features.to(DEVICE)
            
            enhanced = model(curr_frames, prev_frames, next_frames, features).clamp(0, 1)
            
            # Create figure
            fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
            
            for i in range(min(num_samples, len(original))):
                orig_img = to_rgb(original[i])
                inp_img = to_rgb(curr_frames[i])
                enh_img = to_rgb(enhanced[i])
                
                psnr_in = psnr_metric_input(curr_frames[i:i+1], original[i:i+1]).item()
                psnr_enh = psnr_metric_enhanced(enhanced[i:i+1], original[i:i+1]).item()
                ssim = ssim_metric(enhanced[i:i+1], original[i:i+1]).item()
                
                total_psnr_gain += (psnr_enh - psnr_in)
                total_ssim += ssim
                count += 1
                
                axes[i, 0].imshow(orig_img)
                axes[i, 0].set_title(f'Original\nPSNR: {psnr_enh:.2f}', fontsize=10)
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(inp_img)
                axes[i, 1].set_title(f'VVC Input\nPSNR: {psnr_in:.2f}', fontsize=10)
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(enh_img)
                axes[i, 2].set_title(f'Enhanced\nPSNR: {psnr_enh:.2f}', fontsize=10)
                axes[i, 2].axis('off')
                
                # Difference map
                diff = np.abs(enh_img - orig_img)
                axes[i, 3].imshow(diff, cmap='hot')
                axes[i, 3].set_title(f'Diff (×10)\nΔ: {psnr_enh-psnr_in:+.2f} dB', fontsize=10)
                axes[i, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'blackfyre_visualization_batch{batch_idx}.png', dpi=150)
            print(f"Saved: blackfyre_visualization_batch{batch_idx}.png")
            plt.close()
    
    avg_psnr_gain = total_psnr_gain / count
    avg_ssim = total_ssim / count
    print(f"\nAverage PSNR Gain: {avg_psnr_gain:+.3f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    visualize_batch()
