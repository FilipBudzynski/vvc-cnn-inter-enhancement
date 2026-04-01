#!/usr/bin/env python3
"""
Snow-Wide Fine-tuning - start from pre-trained Snow weights
With Charbonnier loss for better edge preservation
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
import wandb
import os

BATCH_SIZE = 4
NUM_EPOCHS = 200
LEARNING_RATE = 5e-5
PATCH_SIZE = 256  # Now works with mirror padding!
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def yuv_to_rgb(yuv):
    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]
    r = y + 1.402 * (v - 0.5)
    g = y - 0.344136 * (u - 0.5) - 0.714136 * (v - 0.5)
    b = y + 1.772 * (u - 0.5)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1)


def charbonnier_loss(pred, target, eps=1e-3):
    """Charbonnier loss - better for edges than L1"""
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)
    return loss.mean()


def compute_loss(enhanced, original):
    loss = charbonnier_loss(enhanced, original)
    return loss, loss.item()


def load_pretrained_snow(model, checkpoint_path):
    snow_ckpt = torch.load(checkpoint_path, map_location='cpu')
    snow_state = model.state_dict()
    
    transferred = 0
    skipped = []
    
    new_state = {}
    for name, param in model.named_parameters():
        if name in snow_ckpt:
            if snow_ckpt[name].shape == param.shape:
                new_state[name] = snow_ckpt[name]
                transferred += 1
            else:
                skipped.append(name)
        else:
            skipped.append(name)
    
    model.load_state_dict(new_state, strict=False)
    print(f"Transferred {transferred} layers from pretrained Snow")
    print(f"Skipped {len(skipped)} layers (new in Snow-Wide)")
    
    return model


def log_images(epoch, original, enhanced, curr_frames, psnr_metric_enh, psnr_metric_in):
    import matplotlib.pyplot as plt
    
    with torch.no_grad():
        sample_idx = 0
        orig = original[sample_idx:1].cpu()
        enh = enhanced[sample_idx:1].cpu()
        inp = curr_frames[sample_idx:1].cpu()
        
        psnr_input = psnr_metric_in(inp, orig).item()
        psnr_enh = psnr_metric_enh(enh, orig).item()
        
        orig_np = yuv_to_rgb(orig[0].permute(1, 2, 0).numpy())
        enh_np = yuv_to_rgb(enh[0].permute(1, 2, 0).numpy())
        inp_np = yuv_to_rgb(inp[0].permute(1, 2, 0).numpy())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(orig_np)
        axes[0].set_title(f'Ground Truth\nPSNR: {psnr_metric_enh(orig, orig).item():.2f} dB', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(inp_np)
        axes[1].set_title(f'VVC Compressed\nPSNR: {psnr_input:.2f} dB', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(enh_np)
        axes[2].set_title(f'Snow-Wide FT\nPSNR: {psnr_enh:.2f} dB (Δ +{psnr_enh-psnr_input:.2f})', fontsize=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        wandb.log({"val_images": wandb.Image(fig)}, step=epoch)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", "-e", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", "-r", type=str, default="checkpoints/snow_epoch_490.pt")
    args = parser.parse_args()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    import random
    random.seed(42)
    np.random.seed(42)
    
    wandb.init(project="vvc-cnn-inter", name="snow-wide-ft", mode="offline")
    
    class SnowWideConfig:
        metadata_channels = 19
        base_channels = 64
    
    config = SnowWideConfig()
    
    # Use fixed dataset with mirror padding!
    from enhancer.dataset_blackfyre_fixed import BlackfyreDataset
    
    train_dataset = BlackfyreDataset(data_dir="data/precomputed", patch_size=PATCH_SIZE, split="train")
    val_dataset = BlackfyreDataset(data_dir="data/precomputed", patch_size=PATCH_SIZE, split="val")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Loading pretrained from: {args.resume}")
    print("Using Charbonnier loss for better edge preservation")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    from enhancer.models.snow_wide import SnowWideEnhancer
    
    model = SnowWideEnhancer(config).to(DEVICE)
    model = load_pretrained_snow(model, args.resume)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150])
    
    psnr_metric_enhanced = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    psnr_metric_input = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    print("\nStarting Snow-Wide Fine-tuning...")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("Loss: Charbonnier (better for edges)")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            decoded, original, metadata, info = batch
            prev_frame, curr_frame, next_frame = decoded
            
            prev_frame = prev_frame.to(DEVICE)
            curr_frame = curr_frame.to(DEVICE)
            next_frame = next_frame.to(DEVICE)
            original = original.to(DEVICE)
            metadata = metadata.to(DEVICE)
            
            optimizer.zero_grad()
            
            enhanced = model(curr_frame, prev_frame, next_frame, metadata).clamp(0, 1)
            
            loss, loss_val = compute_loss(enhanced, original)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss_val
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss_val:.4f}")
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_psnr_gain = 0
        val_psnr_input = 0
        val_psnr_enhanced = 0
        val_ssim = 0
        n_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                decoded, original, metadata, info = batch
                prev_frame, curr_frame, next_frame = decoded
                
                prev_frame = prev_frame.to(DEVICE)
                curr_frame = curr_frame.to(DEVICE)
                next_frame = next_frame.to(DEVICE)
                original = original.to(DEVICE)
                metadata = metadata.to(DEVICE)
                
                enhanced = model(curr_frame, prev_frame, next_frame, metadata).clamp(0, 1)
                
                for i in range(enhanced.shape[0]):
                    psnr_enh = psnr_metric_enhanced(enhanced[i:i+1], original[i:i+1])
                    psnr_in = psnr_metric_input(curr_frame[i:i+1], original[i:i+1])
                    
                    val_psnr_gain += (psnr_enh - psnr_in).item()
                    val_psnr_input += psnr_in.item()
                    val_psnr_enhanced += psnr_enh.item()
                    
                    ssim_val = ssim_metric(enhanced[i:i+1], original[i:i+1])
                    val_ssim += ssim_val.item()
                    n_val += 1
                    
                    psnr_metric_enhanced.reset()
                    psnr_metric_input.reset()
        
        print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val PSNR Gain: {val_psnr_gain/n_val:.4f}")
        
        wandb.log({
            "train_loss": train_loss/len(train_loader),
            "val_psnr_gain": val_psnr_gain/n_val,
            "val_psnr_input": val_psnr_input/n_val,
            "val_psnr_enhanced": val_psnr_enhanced/n_val,
            "val_ssim": val_ssim/n_val,
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)
        
        if epoch % 10 == 0:
            log_images(epoch, original, enhanced, curr_frame, psnr_metric_enhanced, psnr_metric_input)
        
        if epoch % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/snow_wide_ft_epoch_{epoch}.pt")
            wandb.save(f"checkpoints/snow_wide_ft_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
