#!/usr/bin/env python3
"""
Visualize Snow model results - generate side-by-side comparisons
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from enhancer.models.snow import SnowEnhancer
from enhancer.dataset_blackfyre import BlackfyreDataset
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 132

class SnowConfig:
    base_channels = 64
    metadata_channels = 19

model = SnowEnhancer(SnowConfig()).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/snow_epoch_490.pt", map_location=DEVICE))
model.eval()

test_dataset = BlackfyreDataset(data_dir="data/precomputed", patch_size=PATCH_SIZE, split="test")
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

psnr_metric_enhanced = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
psnr_metric_input = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

def yuv_to_rgb(yuv):
    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]
    r = y + 1.402 * (v - 0.5)
    g = y - 0.344136 * (u - 0.5) - 0.714136 * (v - 0.5)
    b = y + 1.772 * (u - 0.5)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 1)

def to_rgb(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return yuv_to_rgb(img)

# Process ALL test samples for proper statistics
total_gain = 0
total_ssim = 0
num_samples = 0
sample_images = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        decoded, original, features, info = batch
        prev_frames, curr_frames, next_frames = decoded
        
        prev_frames = prev_frames.to(DEVICE)
        curr_frames = curr_frames.to(DEVICE)
        next_frames = next_frames.to(DEVICE)
        original = original.to(DEVICE)
        features = features.to(DEVICE)
        
        enhanced = model(curr_frames, prev_frames, next_frames, features).clamp(0, 1)
        
        for i in range(len(original)):
            psnr_in = psnr_metric_input(curr_frames[i:i+1], original[i:i+1]).item()
            psnr_enh = psnr_metric_enhanced(enhanced[i:i+1], original[i:i+1]).item()
            ssim = ssim_metric(enhanced[i:i+1], original[i:i+1]).item()
            
            total_gain += (psnr_enh - psnr_in)
            total_ssim += ssim
            num_samples += 1
            
            # Store first 8 samples for visualization
            if len(sample_images) < 8:
                sample_images.append({
                    'orig': to_rgb(original[i]),
                    'inp': to_rgb(curr_frames[i]),
                    'enh': to_rgb(enhanced[i]),
                    'psnr_in': psnr_in,
                    'psnr_enh': psnr_enh,
                    'gain': psnr_enh - psnr_in
                })
            
            psnr_metric_enhanced.reset()
            psnr_metric_input.reset()

# Create visualization
fig, axes = plt.subplots(8, 4, figsize=(16, 32))

for i, s in enumerate(sample_images):
    axes[i, 0].imshow(s['orig'])
    axes[i, 0].set_title(f"Original\nPSNR: {s['psnr_enh']:.2f}", fontsize=10)
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(s['inp'])
    axes[i, 1].set_title(f"VVC Input\nPSNR: {s['psnr_in']:.2f}", fontsize=10)
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(s['enh'])
    axes[i, 2].set_title(f"Snow Enhanced\nPSNR: {s['psnr_enh']:.2f}", fontsize=10)
    axes[i, 2].axis('off')
    
    diff = np.abs(s['enh'] - s['orig'])
    axes[i, 3].imshow(diff, cmap='hot')
    axes[i, 3].set_title(f"Diff\nΔ: {s['gain']:+.2f} dB", fontsize=10)
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig('snow_visualization.png', dpi=150, bbox_inches='tight')
print(f"Saved: snow_visualization.png")

print(f"\n=== Snow Test Results ({num_samples} samples) ===")
print(f"Average PSNR Gain: {total_gain/num_samples:+.3f} dB")
print(f"Average SSIM: {total_ssim/num_samples:.4f}")
