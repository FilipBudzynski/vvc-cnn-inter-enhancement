#!/usr/bin/env python3
"""
Generalized Training Script for VVC CNN Inter-Frame Enhancement
Supports interchangeable models: snow, snow_wide, hightower, vvc_ppff
"""

import argparse
import random
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
from enhancer.config import Config

BATCH_SIZE = 8
NUM_EPOCHS = 500
LEARNING_RATE = 1e-4
PATCH_SIZE = 132
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODELS = {
    "snow": ("enhancer.models.snow.SnowEnhancer", "enhancer.dataset_blackfyre.BlackfyreDataset"),
    "snow_wide": ("enhancer.models.snow_wide.SnowWideEnhancer", "enhancer.dataset_blackfyre.BlackfyreDataset"),
    "hightower": ("enhancer.models.hightower.HightowerEnhancer", "enhancer.dataset_hightower.HightowerDataset"),
    "vvc_ppff": ("enhancer.models.vvc_ppff.VVCPPFF", "enhancer.dataset_blackfyre.BlackfyreDataset"),
}


def compute_loss(enhanced, original):
    """L2 (MSE) loss - as per VVC-PPFF paper"""
    return F.mse_loss(enhanced, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", "-e", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--data-dir", default="data/precomputed")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
    
    # Initialize wandb - offline by default, online if --wandb flag
    wandb.init(project="vvc-cnn-inter", name="martell", mode="offline")
    
    if args.model == "vvc_ppff":
        from enhancer.models.vvc_ppff import VVCPPFF
        model = VVCPPFF().to(DEVICE)
    else:
        class ConfigModel:
            base_channels = 64
            metadata_channels = 19
        model = None
    
    from enhancer.dataset_blackfyre import BlackfyreDataset
    train_dataset = BlackfyreDataset(data_dir=args.data_dir, patch_size=args.patch_size, split="train")
    val_dataset = BlackfyreDataset(data_dir=args.data_dir, patch_size=args.patch_size, split="val")
    
    print(f"Model: {args.model}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    if model is None:
        if args.model == "snow":
            from enhancer.models.snow import SnowEnhancer
            model = SnowEnhancer(ConfigModel()).to(DEVICE)
        elif args.model == "snow_wide":
            from enhancer.models.snow_wide import SnowWideEnhancer
            model = SnowWideEnhancer(ConfigModel()).to(DEVICE)
        elif args.model == "hightower":
            from enhancer.models.hightower import HightowerEnhancer
            model = HightowerEnhancer(ConfigModel()).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 300])
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    print(f"Starting {args.model} training...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            frames = batch[0]
            prev_frame = frames[0].to(DEVICE)
            curr_frame = frames[1].to(DEVICE)
            next_frame = frames[2].to(DEVICE)
            original = batch[1].to(DEVICE)
            metadata = batch[2].to(DEVICE)
            
            optimizer.zero_grad()
            
            if args.model == "vvc_ppff":
                enhanced = model(curr_frame, metadata).clamp(0, 1)
            else:
                enhanced = model(curr_frame, prev_frame, next_frame, metadata).clamp(0, 1)
            
            loss = compute_loss(enhanced, original)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            if args.quick_test and batch_idx >= 0:
                break
        
        scheduler.step()
        
        if args.quick_test:
            print("Quick test passed!")
            wandb.finish()
            return
        
        model.eval()
        val_psnr_gain = 0
        val_ssim = 0
        n_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                frames = batch[0]
                curr_frame = frames[1].to(DEVICE)
                original = batch[1].to(DEVICE)
                metadata = batch[2].to(DEVICE)
                
                if args.model == "vvc_ppff":
                    enhanced = model(curr_frame, metadata).clamp(0, 1)
                else:
                    enhanced = model(curr_frame, frames[0].to(DEVICE), frames[2].to(DEVICE), metadata).clamp(0, 1)
                
                for i in range(enhanced.shape[0]):
                    psnr_enh = psnr_metric(enhanced[i:i+1], original[i:i+1])
                    psnr_in = psnr_metric(curr_frame[i:i+1], original[i:i+1])
                    val_psnr_gain += (psnr_enh - psnr_in).item()
                    ssim_val = ssim_metric(enhanced[i:i+1], original[i:i+1])
                    val_ssim += ssim_val.item()
                    n_val += 1
        
        print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val PSNR Gain: {val_psnr_gain/n_val:.4f}, SSIM: {val_ssim/n_val:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss/len(train_loader),
            "val_psnr_gain": val_psnr_gain/n_val,
            "val_ssim": val_ssim/n_val,
            "lr": optimizer.param_groups[0]["lr"],
        })
        
        if epoch % 10 == 0:
            import os
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{args.model}_epoch_{epoch}.pt")
            wandb.save(f"checkpoints/{args.model}_epoch_{epoch}.pt")
    
    # After training - ask to sync to wandb
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    sync = input("Sync results to wandb? (y/N): ").strip().lower()
    if sync == 'y':
        wandb.sync()
        print("Synced to wandb!")
    else:
        print("Results saved locally in ./wandb/")
        print("To sync later: wandb sync ./wandb/offline-run-*")
    
    wandb.finish()


if __name__ == "__main__":
    main()
