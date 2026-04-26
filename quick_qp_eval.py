#!/usr/bin/env python3
"""
Szybka ewaluacja na istniejących danych QP22
"""

import os
import re
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pytorch_msssim import ssim
import numpy as np
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_info(info_path):
    content = info_path.read_text()
    width = int(re.search(r"Width\s+:\s+(\d+)", content).group(1))
    height = int(re.search(r"Height\s+:\s+(\d+)", content).group(1))
    return width, height

def prepare_qp22():
    """Przygotuje dataset z QP22"""
    precomputed_dir = "./data_qp22_test"
    if os.path.exists(precomputed_dir):
        return precomputed_dir
    
    encoded_dir = "./output_qp22/encoded"
    os.makedirs(precomputed_dir, exist_ok=True)
    
    recon_files = sorted(Path(encoded_dir).glob("*_rec.yuv"))
    
    for recon_file in tqdm(recon_files, desc="Preparing QP22"):
        stem = recon_file.stem.replace("_QP22_rec", "")
        orig_file = Path("data") / f"{stem}.yuv"
        
        if not orig_file.exists():
            continue
        
        try:
            width, height = parse_info(list(Path("data").glob(f"{stem}*.info"))[0])
        except:
            continue
        
        video_dir = Path(precomputed_dir) / stem
        video_dir.mkdir(exist_ok=True)
        
        try:
            from features_generator.features_generator import FeaturesGenerator
            fg = FeaturesGenerator(str(recon_file), str(orig_file), width, height)
            frames = fg.generate()
            
            for i, frame_data in enumerate(frames[:20]):
                torch.save(frame_data, video_dir / f"poc_{i:04d}.pt")
        except Exception as e:
            print(f"Błąd {stem}: {e}")
    
    return precomputed_dir

class TestDataset(Dataset):
    def __init__(self, data_dir, max_samples=50):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        for video_dir in self.data_dir.iterdir():
            if video_dir.is_dir():
                frames = sorted(video_dir.glob("poc_*.pt"), 
                             key=lambda x: int(x.stem.split("_")[1]))
                for i in range(1, min(len(frames) - 1, max_samples + 1)):
                    self.samples.append((frames[i-1], frames[i], frames[i+1]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        prev_pt, curr_pt, next_pt = self.samples[idx]
        prev_data = torch.load(prev_pt, weights_only=True)
        curr_data = torch.load(curr_pt, weights_only=True)
        next_data = torch.load(next_pt, weights_only=True)
        
        return {
            "prev": prev_data["decoded"],
            "curr": curr_data["decoded"],
            "next": next_data["decoded"],
            "original": curr_data["original"],
            "features": curr_data["features"],
        }

class ResNetNoTemporal(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.metadata_encoder = nn.Sequential(
            nn.Conv2d(19, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 1),
            nn.PReLU(),
        )
        self.features = nn.Sequential(
            nn.Conv2d(3 + 32, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.PReLU(),
        )
        self.blocks = nn.Sequential(*[self._make_block(base_channels) for _ in range(4)])
        self.output = nn.Conv2d(base_channels, 3, 3, padding=1)
    
    def _make_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )
    
    def forward(self, curr, features):
        meta = self.metadata_encoder(features)
        x = torch.cat([curr, meta], dim=1)
        x = self.features(x)
        x = self.blocks(x)
        return curr + self.output(x)

def main():
    print(f"Device: {DEVICE}")
    
    # Przygotuj dataset
    print("\nPrzygotowanie datasetu...")
    data_dir = prepare_qp22()
    
    # Load models
    models = {}
    
    models["ResNet_Intra"] = ResNetNoTemporal(base_channels=64)
    try:
        ckpt = torch.load("experiments/enhancer/vtm_resnet_v6.pth", map_location=DEVICE)
        models["ResNet_Intra"].load_state_dict(ckpt, strict=False)
    except: pass
    
    try:
        from enhancer.models.snow import SnowEnhancer
        class C: base_channels=64; metadata_channels=19
        models["Snow"] = SnowEnhancer(C()).to(DEVICE)
        models["Snow"].load_state_dict(torch.load("checkpoints/snow_epoch_490.pt", map_location=DEVICE))
    except: pass
    
    try:
        from enhancer.models.snow_wide import SnowWideEnhancer
        class C: base_channels=64; metadata_channels=19
        models["Snow_Wide"] = SnowWideEnhancer(C()).to(DEVICE)
        models["Snow_Wide"].load_state_dict(torch.load("checkpoints/snow_wide_epoch_460.pt", map_location=DEVICE))
    except: pass
    
    for name, model in models.items():
        model.eval()
        model.to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} params")
    
    # Dataset
    dataset = TestDataset(data_dir)
    print(f"Test samples: {len(dataset)}")
    
    # Evaluate
    results = {name: {"psnr_gains": [], "ssim": []} for name in models.keys()}
    baseline = {"psnr": [], "ssim": []}
    
    with torch.no_grad():
        for batch in dataset:
            prev = batch["prev"].unsqueeze(0).to(DEVICE)
            curr = batch["curr"].unsqueeze(0).to(DEVICE)
            next_f = batch["next"].unsqueeze(0).to(DEVICE)
            original = batch["original"].unsqueeze(0).to(DEVICE)
            features = batch["features"].unsqueeze(0).to(DEVICE)
            
            psnr_base = 10 * torch.log10(1.0 / (torch.nn.functional.mse_loss(curr, original) + 1e-10) + 1e-10)
            baseline["psnr"].append(psnr_base.item())
            baseline["ssim"].append(ssim(curr, original, data_range=1.0).item())
            
            for name, model in models.items():
                if name == "ResNet_Intra":
                    enhanced = model(curr, features).clamp(0, 1)
                else:
                    enhanced = model(curr, prev, next_f, features).clamp(0, 1)
                
                psnr_enh = 10 * torch.log10(1.0 / (torch.nn.functional.mse_loss(enhanced, original) + 1e-10) + 1e-10)
                
                results[name]["psnr_gains"].append((psnr_enh - psnr_base).item())
                results[name]["ssim"].append(ssim(enhanced, original, data_range=1.0).item())
    
    # Results
    print("\n" + "="*60)
    print(f"WYNIKI EWALUACJI (QP=22)")
    print("="*60)
    print(f"\n{'Model':<15} {'PSNR Gain':<12} {'SSIM':<10} {'Params':<12}")
    print("-"*50)
    
    for name in models.keys():
        avg_gain = np.mean(results[name]["psnr_gains"])
        avg_ssim = np.mean(results[name]["ssim"])
        params = sum(p.numel() for p in models[name].parameters())
        print(f"{name:<15} {avg_gain:+.4f} dB    {avg_ssim:.4f}      {params:,}")
    
    print(f"\nBaseline PSNR: {np.mean(baseline['psnr']):.2f} dB")
    print(f"Baseline SSIM: {np.mean(baseline['ssim']):.4f}")

if __name__ == "__main__":
    main()
