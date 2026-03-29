#!/usr/bin/env python3
"""
Blackfyre Training Script - Temporal Attention with Enhanced Metadata
Named after House Blackfyre from Game of Thrones

Uses:
- Temporal Attention (learns which frames to focus on)
- Enhanced metadata (16 channels: QP, Depth, SkipFlag, etc.)
- Metadata-guided attention
- Color wandb logging every 10 epochs
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from enhancer.models.blackfyre import BlackfyreEnhancer
from enhancer.ssim import SSIM, MS_SSIM

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
    
    grad_x = F.conv2d(x, sobel_x.repeat(3, 1, 1, 1), groups=3, padding=1)
    grad_y = F.conv2d(x, sobel_y.repeat(3, 1, 1, 1), groups=3, padding=1)
    
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)


def compute_loss(enhanced, original, ssim_mod, msssim_mod):
    """Loss: 0.1*MS-SSIM + 0.1*SSIM + 0.5*L1 + 0.3*L2 + 0.1*Gradient"""
    l1_loss = F.l1_loss(enhanced, original)
    l2_loss = F.mse_loss(enhanced, original)
    
    ssim_val = ssim_mod(enhanced, original)
    ssim_loss = 1 - ssim_val
    
    msssim_val = msssim_mod(enhanced, original)
    msssim_loss = 1 - msssim_val
    
    grad_enh = compute_gradient(enhanced)
    grad_orig = compute_gradient(original)
    grad_loss = F.l1_loss(grad_enh, grad_orig)
    
    loss = (
        0.1 * msssim_loss +
        0.1 * ssim_loss +
        0.5 * l1_loss +
        0.3 * l2_loss +
        0.1 * grad_loss
    )
    
    return loss


def log_color_images(epoch, original, enhanced, curr_frames, psnr_metric_enh, psnr_metric_in):
    """Log color side-by-side comparison to wandb"""
    import wandb
    
    with torch.no_grad():
        sample_idx = 0
        orig = original[sample_idx:1].cpu()
        enh = enhanced[sample_idx:1].cpu()
        inp = curr_frames[sample_idx:1].cpu()
        
        # Calculate PSNR
        psnr_input = psnr_metric_in(inp, orig).item()
        psnr_enh = psnr_metric_enh(enh, orig).item()
        
        # Convert to numpy [H, W, 3] for display
        orig_np = orig[0].permute(1, 2, 0).numpy()
        enh_np = enh[0].permute(1, 2, 0).numpy()
        inp_np = inp[0].permute(1, 2, 0).numpy()
        
        # Clip to [0, 1]
        orig_np = np.clip(orig_np, 0, 1)
        enh_np = np.clip(enh_np, 0, 1)
        inp_np = np.clip(inp_np, 0, 1)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(orig_np)
        axes[0].set_title(f'Ground Truth\n(Original)', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(inp_np)
        axes[1].set_title(f'VVC Compressed\nPSNR: {psnr_input:.2f} dB', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(enh_np)
        axes[2].set_title(f'Blackfyre Enhanced\nPSNR: {psnr_enh:.2f} dB (Δ +{psnr_enh-psnr_input:.2f})', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        wandb.log({"val_images": wandb.Image(fig)}, step=epoch)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", action="store_true", help="Precompute features first")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", "-e", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    import random
    random.seed(42)
    import numpy as np
    np.random.seed(42)
    
    if args.precompute:
        print("Precomputing features...")
        import scripts.precompute_features as precomp
        precomp.main()
        return
    
    # Wandb - online mode for real-time syncing
    wandb.init(project="vvc-cnn-inter", name="blackfyre-v1", mode="offline")
    
    # Config for model
    class BlackfyreConfig:
        metadata_channels = 19  # Enhanced metadata
        base_channels = 64
    
    config = BlackfyreConfig()
    
    # Dataset
    from enhancer.dataset_blackfyre import BlackfyreDataset
    
    train_dataset = BlackfyreDataset(data_dir="data/precomputed", patch_size=PATCH_SIZE, split="train")
    val_dataset = BlackfyreDataset(data_dir="data/precomputed", patch_size=PATCH_SIZE, split="val")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Model
    model = BlackfyreEnhancer(config).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 300])
    
    # Metrics
    psnr_metric_enhanced = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    psnr_metric_input = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    # SSIM for loss
    ssim_mod = SSIM(data_range=1.0, win_size=9).to(DEVICE)
    msssim_mod = MS_SSIM(data_range=1.0, win_size=9).to(DEVICE)
    
    print("Starting Blackfyre training...")
    print("Features: 16 channels (enhanced metadata)")
    print("Model: Temporal Attention + Metadata-Guided Attention")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            (prev_frames, curr_frames, next_frames), original, features, info = batch
            
            prev_frames = prev_frames.to(DEVICE)
            curr_frames = curr_frames.to(DEVICE)
            next_frames = next_frames.to(DEVICE)
            original = original.to(DEVICE)
            features = features.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass - Blackfyre with temporal attention
            enhanced = model(
                current_frame=curr_frames,
                prev_frame=prev_frames,
                next_frame=next_frames,
                metadata=features
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
                (prev_frames, curr_frames, next_frames), original, features, info = batch
                
                prev_frames = prev_frames.to(DEVICE)
                curr_frames = curr_frames.to(DEVICE)
                next_frames = next_frames.to(DEVICE)
                original = original.to(DEVICE)
                features = features.to(DEVICE)
                
                enhanced = model(curr_frames, prev_frames, next_frames, features).clamp(0, 1)
                
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
        
        # Log images every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                for val_batch in val_loader:
                    (prev_frames, curr_frames, next_frames), original, features, info = val_batch
                    prev_frames = prev_frames.to(DEVICE)
                    curr_frames = curr_frames.to(DEVICE)
                    next_frames = next_frames.to(DEVICE)
                    original = original.to(DEVICE)
                    features = features.to(DEVICE)
                    
                    enhanced = model(curr_frames, prev_frames, next_frames, features).clamp(0, 1)
                    
                    # Log color images
                    log_color_images(
                        epoch, original, enhanced, curr_frames,
                        psnr_metric_enhanced, psnr_metric_input
                    )
                    break
        
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
            torch.save(model.state_dict(), f"checkpoints/blackfyre_epoch_{epoch}.pt")
            wandb.save(f"checkpoints/blackfyre_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
