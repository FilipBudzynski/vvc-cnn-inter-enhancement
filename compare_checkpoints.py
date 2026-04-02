#!/usr/bin/env python3
"""
Compare two Snow-Wide checkpoints: old (epoch 360, L1+SSIM) vs new (epoch 110, L1+SSIM+Gradient)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from enhancer.models.snow_wide import SnowWideEnhancer
from enhancer.dataset_blackfyre import BlackfyreDataset
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    base_channels = 64
    metadata_channels = 19

config = Config()

def to_rgb(tensor):
    t = tensor.cpu()
    yuv = t.permute(1, 2, 0).numpy()
    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]
    r = y + 1.402 * (v - 0.5)
    g = y - 0.344136 * (u - 0.5) - 0.714136 * (v - 0.5)
    b = y + 1.772 * (u - 0.5)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1)

def compute_grad_magnitude(img):
    gray = np.mean(img, axis=2)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    return np.mean(dx) + np.mean(dy)

# Load both models
print("Loading old model (epoch 360, L1+SSIM)...")
old_model = SnowWideEnhancer(config).to(DEVICE)
old_model.load_state_dict(torch.load("checkpoints/snow_wide_epoch_360.pt", map_location=DEVICE))
old_model.eval()

print("Loading new model (epoch 110, L1+SSIM+Gradient)...")
new_model = SnowWideEnhancer(config).to(DEVICE)
new_model.load_state_dict(torch.load("checkpoints/snow_wide_epoch_140.pt", map_location=DEVICE))
new_model.eval()

# Load test data
ds = BlackfyreDataset(data_dir="data/precomputed", patch_size=132, split="test")
loader = DataLoader(ds, batch_size=1, shuffle=True)

psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

results = {"old": [], "new": []}

print("\nComparing on test samples...")
for i, batch in enumerate(loader):
    if i >= 50:  # Compare on 50 samples
        break
    
    decoded, original, features, info = batch
    prev, curr, next_f = decoded
    meta = features
    
    prev = prev.to(DEVICE)
    curr = curr.to(DEVICE)
    next_f = next_f.to(DEVICE)
    original = original.to(DEVICE)
    meta = meta.to(DEVICE)
    
    with torch.no_grad():
        old_enh = old_model(curr, prev, next_f, meta).clamp(0, 1)
        new_enh = new_model(curr, prev, next_f, meta).clamp(0, 1)
    
    # PSNR
    psnr_in = psnr_metric(curr, original).item()
    psnr_old = psnr_metric(old_enh, original).item()
    psnr_new = psnr_metric(new_enh, original).item()
    
    results["old"].append(psnr_old - psnr_in)
    results["new"].append(psnr_new - psnr_in)
    
    psnr_metric.reset()
    
    if i % 10 == 0:
        print(f"  Processed {i+1}/50 samples...")

old_gain = np.mean(results["old"])
new_gain = np.mean(results["new"])

print(f"\n=== COMPARISON RESULTS (50 samples) ===")
print(f"Old Model (epoch 360, L1+SSIM):     {old_gain:+.4f} dB")
print(f"New Model (epoch 110, L1+SSIM+Grad): {new_gain:+.4f} dB")
print(f"Improvement: {new_gain - old_gain:+.4f} dB")

# Visualize some examples
print("\nGenerating visualization...")
fig, axes = plt.subplots(4, 5, figsize=(20, 16))

sample_idx = 0
for batch in loader:
    if sample_idx >= 4:
        break
    
    decoded, original, features, info = batch
    prev, curr, next_f = decoded
    meta = features
    
    prev = prev.to(DEVICE)
    curr = curr.to(DEVICE)
    next_f = next_f.to(DEVICE)
    original = original.to(DEVICE)
    meta = meta.to(DEVICE)
    
    with torch.no_grad():
        old_enh = old_model(curr, prev, next_f, meta).clamp(0, 1)
        new_enh = new_model(curr, prev, next_f, meta).clamp(0, 1)
    
    orig_np = to_rgb(original[0])
    curr_np = to_rgb(curr[0])
    old_np = to_rgb(old_enh[0])
    new_np = to_rgb(new_enh[0])
    
    psnr_in = psnr_metric(curr, original).item()
    psnr_old = psnr_metric(old_enh, original).item()
    psnr_new = psnr_metric(new_enh, original).item()
    psnr_metric.reset()
    
    # Row: Original, Input, Old, New
    axes[sample_idx, 0].imshow(orig_np)
    axes[sample_idx, 0].set_title(f'Original\nPSNR: {psnr_old:.2f}', fontsize=10)
    axes[sample_idx, 0].axis('off')
    
    axes[sample_idx, 1].imshow(curr_np)
    axes[sample_idx, 1].set_title(f'VVC Input\nPSNR: {psnr_in:.2f}', fontsize=10)
    axes[sample_idx, 1].axis('off')
    
    axes[sample_idx, 2].imshow(old_np)
    axes[sample_idx, 2].set_title(f'Old (epoch 360)\nPSNR: {psnr_old:.2f} ({psnr_old-psnr_in:+.2f})', fontsize=10)
    axes[sample_idx, 2].axis('off')
    
    axes[sample_idx, 3].imshow(new_np)
    axes[sample_idx, 3].set_title(f'New (epoch 110)\nPSNR: {psnr_new:.2f} ({psnr_new-psnr_in:+.2f})', fontsize=10)
    axes[sample_idx, 3].axis('off')
    
    # Gradient magnitude comparison
    curr_grad = compute_grad_magnitude(curr_np)
    old_grad = compute_grad_magnitude(old_np)
    new_grad = compute_grad_magnitude(new_np)
    
    axes[sample_idx, 4].bar(['Input', 'Old', 'New'], [curr_grad, old_grad, new_grad])
    axes[sample_idx, 4].set_title(f'Gradient Mag\n(Detail level)', fontsize=10)
    axes[sample_idx, 4].set_ylabel('Magnitude')
    
    sample_idx += 1

plt.suptitle('Comparison: Old (L1+SSIM) vs New (L1+SSIM+Gradient Loss)', fontsize=16)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: model_comparison.png")
