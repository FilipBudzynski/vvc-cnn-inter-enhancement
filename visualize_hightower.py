#!/usr/bin/env python3
"""
Visualize Hightower model results - generate side-by-side comparisons
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

from enhancer.models.hightower import HightowerEnhancer
from enhancer.dataset_hightower import HightowerDataset
from enhancer.ssim import SSIM, MS_SSIM
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 132

# Wandb
wandb.init(project="vvc-cnn-inter", name="hightower-visualization", mode="online")

# Model
class HightowerConfig:
    base_channels = 64

model = HightowerEnhancer(HightowerConfig()).to(DEVICE)

# Load checkpoint (latest)
checkpoint_path = "checkpoints/hightower_epoch_90.pt"  # or use the latest
print(f"Loading: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()

# Dataset
val_dataset = HightowerDataset(data_dir="data/precomputed", patch_size=PATCH_SIZE, split="val")
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# Metrics
psnr_metric_enhanced = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
psnr_metric_input = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

def to_y_channel(tensor):
    """Convert RGB to Y channel (grayscale)"""
    img = tensor.permute(1, 2, 0).numpy()
    y = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    return np.clip(y, 0, 1)

# Generate visualizations
print("Generating visualizations...")

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= 5:  # Process 5 batches
            break
            
        (prev_frames, curr_frames, next_frames), original, motion_vectors, metadata, info = batch
        
        prev_frames = prev_frames.to(DEVICE)
        curr_frames = curr_frames.to(DEVICE)
        next_frames = next_frames.to(DEVICE)
        original = original.to(DEVICE)
        motion_vectors = motion_vectors.to(DEVICE)
        metadata = metadata.to(DEVICE)
        
        # Enhance
        enhanced = model(curr_frames, prev_frames, next_frames, motion_vectors, metadata).clamp(0, 1)
        
        # Process each sample in batch
        for i in range(min(4, original.shape[0])):
            orig = original[i].cpu()
            inp = curr_frames[i].cpu()
            enh = enhanced[i].cpu()
            
            # Calculate PSNR
            psnr_input = psnr_metric_input(inp.unsqueeze(0), orig.unsqueeze(0)).item()
            psnr_enh = psnr_metric_enhanced(enh.unsqueeze(0), orig.unsqueeze(0)).item()
            
            # Convert to Y channel
            y_orig = to_y_channel(orig)
            y_input = to_y_channel(inp)
            y_enh = to_y_channel(enh)
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(y_orig, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f'Ground Truth\n(PSNR ref)', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(y_input, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'VVC Compressed\nPSNR: {psnr_input:.2f} dB', fontsize=12)
            axes[1].axis('off')
            
            axes[2].imshow(y_enh, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f'Hightower Enhanced\nPSNR: {psnr_enh:.2f} dB (Δ +{psnr_enh-psnr_input:.2f})', fontsize=12)
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({
                f"val_sample_{batch_idx}_{i}": wandb.Image(fig),
                "step": batch_idx * 4 + i
            })
            
            plt.close(fig)
            
            print(f"Batch {batch_idx}, Sample {i}: Input PSNR={psnr_input:.2f}, Enhanced PSNR={psnr_enh:.2f}, Gain={psnr_enh-psnr_input:.2f}")
        
        psnr_metric_enhanced.reset()
        psnr_metric_input.reset()

print("Done! Check wandb for visualizations.")
