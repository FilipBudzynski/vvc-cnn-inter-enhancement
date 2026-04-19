#!/usr/bin/env python3
"""
Hightower Training Script - Inter-frame Enhancement
Named after House Hightower from Game of Thrones

Uses:
- Neighboring frames (F-1, F0, F+1) for temporal context
- Motion vectors from VVC
- Improved loss: SSIM + MS-SSIM + L1 + Gradient
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import wandb
from enhancer.models.hightower import HightowerEnhancer
from enhancer.ssim import SSIM, MS_SSIM
from enhancer.config import Config

BATCH_SIZE = 8
NUM_EPOCHS = 500
LEARNING_RATE = 1e-4
PATCH_SIZE = 132
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_gradient(x):
    """Compute image gradient for sharpness"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    sobel_x = sobel_x.view(1, 1, 3, 3).to(x.device)
    sobel_y = sobel_y.view(1, 1, 3, 3).to(x.device)
    
    # Apply to each channel
    grad_x = F.conv2d(x, sobel_x.repeat(3, 1, 1, 1), groups=3, padding=1)
    grad_y = F.conv2d(x, sobel_y.repeat(3, 1, 1, 1), groups=3, padding=1)
    
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)


def compute_loss(enhanced, original, ssim_mod, msssim_mod):
    """
    Improved loss function:
    - 0.1 * MS-SSIM (structural similarity)
    - 0.1 * SSIM  
    - 0.5 * L1 (pixel-wise)
    - 0.3 * L2 (smoothness)
    - 0.1 * Gradient (sharpness)
    """
    # L1 loss
    l1_loss = F.l1_loss(enhanced, original)
    
    # L2 (MSE) loss
    l2_loss = F.mse_loss(enhanced, original)
    
    # SSIM loss
    ssim_val = ssim_mod(enhanced, original)
    ssim_loss = 1 - ssim_val
    
    # MS-SSIM loss
    msssim_val = msssim_mod(enhanced, original)
    msssim_loss = 1 - msssim_val
    
    # Gradient loss (edge preservation)
    grad_enh = compute_gradient(enhanced)
    grad_orig = compute_gradient(original)
    grad_loss = F.l1_loss(grad_enh, grad_orig)
    
    # Combined loss (Piotr's weights + gradient)
    loss = (
        0.1 * msssim_loss +
        0.1 * ssim_loss +
        0.5 * l1_loss +
        0.3 * l2_loss +
        0.1 * grad_loss
    )
    
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", action="store_true", help="Precompute features first")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", "-e", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    
    if args.precompute:
        print("Precomputing features...")
        import scripts.precompute_features as precomp
        precomp.main()
        return
    
    # Load config
    config = Config.load(args.config)
    
    # Override model config for Hightower
    class HightowerConfig:
        base_channels = 64
    
    hightower_config = HightowerConfig()
    
    wandb.init(project="vvc-cnn-inter", name="hightower-v1", mode="online")
    
    # Import Hightower dataset
    from enhancer.dataset_hightower import HightowerDataset
    
    train_dataset = HightowerDataset(data_dir="data/precomputed", patch_size=PATCH_SIZE, split="train")
    val_dataset = HightowerDataset(data_dir="data/precomputed", patch_size=PATCH_SIZE, split="val")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Model - Hightower
    model = HightowerEnhancer(hightower_config).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 300])
    
    # Metrics
    psnr_metric_enhanced = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    psnr_metric_input = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    # SSIM modules for loss
    ssim_mod = SSIM(data_range=1.0, win_size=9).to(DEVICE)
    msssim_mod = MS_SSIM(data_range=1.0, win_size=9).to(DEVICE)
    
    print("Starting Hightower training...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            (prev_frames, curr_frames, next_frames), original, motion_vectors, metadata, info = batch
            
            prev_frames = prev_frames.to(DEVICE)
            curr_frames = curr_frames.to(DEVICE)
            next_frames = next_frames.to(DEVICE)
            original = original.to(DEVICE)
            motion_vectors = motion_vectors.to(DEVICE)
            metadata = metadata.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass - use temporal frames
            enhanced = model(
                current_frame=curr_frames,
                prev_frame=prev_frames,
                next_frame=next_frames,
                motion_vectors=motion_vectors,
                metadata=metadata
            )
            enhanced = enhanced.clamp(0, 1)
            
            # Compute loss
            loss = compute_loss(enhanced, original, ssim_mod, msssim_mod)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_psnr_gain = 0
        val_psnr_input = 0
        val_psnr_enhanced = 0
        val_ssim = 0
        
        with torch.no_grad():
            for batch in val_loader:
                (prev_frames, curr_frames, next_frames), original, motion_vectors, metadata, info = batch
                
                prev_frames = prev_frames.to(DEVICE)
                curr_frames = curr_frames.to(DEVICE)
                next_frames = next_frames.to(DEVICE)
                original = original.to(DEVICE)
                motion_vectors = motion_vectors.to(DEVICE)
                metadata = metadata.to(DEVICE)
                
                enhanced = model(curr_frames, prev_frames, next_frames, motion_vectors, metadata).clamp(0, 1)
                
                psnr_enhanced = psnr_metric_enhanced(enhanced, original)
                psnr_input = psnr_metric_input(curr_frames, original)
                
                val_psnr_gain += (psnr_enhanced - psnr_input).item()
                val_psnr_input += psnr_input.item()
                val_psnr_enhanced += psnr_enhanced.item()
                
                ssim_val = ssim_metric(enhanced, original)
                val_ssim += ssim_val.item()
                
                psnr_metric_enhanced.reset()
                psnr_metric_input.reset()
        
        n_val = len(val_loader)
        print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val PSNR Gain: {val_psnr_gain/n_val:.4f}, Input: {val_psnr_input/n_val:.4f}, "
              f"Enhanced: {val_psnr_enhanced/n_val:.4f}, SSIM: {val_ssim/n_val:.4f}")
        
        # Log images for visual comparison (every 10 epochs)
        if epoch % 10 == 0:
            # Get a sample for visualization
            with torch.no_grad():
                sample_idx = 0  # First sample in batch
                sample_original = original[sample_idx:sample_idx+1].cpu()
                sample_input = curr_frames[sample_idx:sample_idx+1].cpu()
                sample_enhanced = enhanced[sample_idx:sample_idx+1].cpu()
                
                # Create side-by-side comparison (Y channel only for visualization)
                def to_y_channel(tensor):
                    # Y = 0.299*R + 0.587*G + 0.114*B
                    img = tensor[0].permute(1, 2, 0).numpy()  # [H, W, 3]
                    y = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
                    return y
                
                y_orig = to_y_channel(sample_original)
                y_input = to_y_channel(sample_input)
                y_enh = to_y_channel(sample_enhanced)
                
                # Create comparison image
                import numpy as np
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Ground Truth (Original)
                im1 = axes[0].imshow(y_orig, cmap='gray', vmin=0, vmax=1)
                axes[0].set_title(f'Ground Truth\n(PSNR: {float(psnr_metric_input(sample_input, sample_original)):.2f} dB)')
                axes[0].axis('off')
                
                # VVC Compressed (Input)
                im2 = axes[1].imshow(y_input, cmap='gray', vmin=0, vmax=1)
                axes[1].set_title(f'VVC Compressed\n(PSNR: {float(psnr_metric_input(sample_input, sample_original)):.2f} dB)')
                axes[1].axis('off')
                
                # Enhanced
                im3 = axes[2].imshow(y_enh, cmap='gray', vmin=0, vmax=1)
                axes[2].set_title(f'Enhanced (Hightower)\n(PSNR: {float(psnr_metric_enhanced(sample_enhanced, sample_original)):.2f} dB)')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Log to wandb
                wandb.log({
                    "val_images": wandb.Image(fig)},
                    step=epoch)
                
                plt.close(fig)
                
                # Reset metrics after image logging
                psnr_metric_enhanced.reset()
                psnr_metric_input.reset()
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss/len(train_loader),
            "val_psnr_gain": val_psnr_gain/n_val,
            "val_psnr_input": val_psnr_input/n_val,
            "val_psnr_enhanced": val_psnr_enhanced/n_val,
            "val_ssim": val_ssim/n_val,
            "lr": optimizer.param_groups[0]["lr"],
            "step": epoch
        })
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/hightower_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
