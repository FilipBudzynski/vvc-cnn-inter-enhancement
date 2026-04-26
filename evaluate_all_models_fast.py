#!/usr/bin/env python3
"""
Szybka ewaluacja modeli na próbce zbioru testowego
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
from pytorch_msssim import ssim
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, data_dir="data/precomputed", max_samples=200):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        for video_dir in self.data_dir.iterdir():
            if video_dir.is_dir():
                frames = sorted(video_dir.glob("poc_*.pt"), 
                             key=lambda x: int(x.stem.split("_")[1]))
                for i in range(1, len(frames) - 1):
                    if len(self.samples) < max_samples:
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
    
    # Load models
    models = {}
    
    # ResNet Intra-only
    models["ResNet_Intra"] = ResNetNoTemporal(base_channels=64)
    try:
        ckpt = torch.load("experiments/enhancer/vtm_resnet_v6.pth", map_location=DEVICE)
        models["ResNet_Intra"].load_state_dict(ckpt, strict=False)
    except Exception as e:
        print(f"ResNet load error: {e}")
    
    # Snow
    try:
        from enhancer.models.snow import SnowEnhancer
        class C: base_channels=64; metadata_channels=19
        models["Snow"] = SnowEnhancer(C()).to(DEVICE)
        models["Snow"].load_state_dict(torch.load("checkpoints/snow_epoch_490.pt", map_location=DEVICE))
    except Exception as e:
        print(f"Snow load error: {e}")
    
    # Snow-Wide
    try:
        from enhancer.models.snow_wide import SnowWideEnhancer
        class C: base_channels=64; metadata_channels=19
        models["Snow_Wide"] = SnowWideEnhancer(C()).to(DEVICE)
        models["Snow_Wide"].load_state_dict(torch.load("checkpoints/snow_wide_epoch_460.pt", map_location=DEVICE))
    except Exception as e:
        print(f"Snow_Wide load error: {e}")
    
    for name, model in models.items():
        model.eval()
        model.to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} params")
    
    # Dataset
    dataset = TestDataset(max_samples=200)
    print(f"Test samples: {len(dataset)}")
    
    # Results
    results = {name: {"psnr_gains": [], "ssim": []} for name in models.keys()}
    baseline = {"psnr": [], "ssim": []}
    
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            prev = batch["prev"].unsqueeze(0).to(DEVICE)
            curr = batch["curr"].unsqueeze(0).to(DEVICE)
            next_f = batch["next"].unsqueeze(0).to(DEVICE)
            original = batch["original"].unsqueeze(0).to(DEVICE)
            features = batch["features"].unsqueeze(0).to(DEVICE)
            
            # Baseline
            psnr_base = 10 * torch.log10(1.0 / (F.mse_loss(curr, original) + 1e-10) + 1e-10)
            ssim_base = ssim(curr, original, data_range=1.0)
            baseline["psnr"].append(psnr_base.item())
            baseline["ssim"].append(ssim_base.item())
            
            for name, model in models.items():
                if name == "ResNet_Intra":
                    enhanced = model(curr, features).clamp(0, 1)
                else:
                    enhanced = model(curr, prev, next_f, features).clamp(0, 1)
                
                psnr_enh = 10 * torch.log10(1.0 / (F.mse_loss(enhanced, original) + 1e-10) + 1e-10)
                ssim_enh = ssim(enhanced, original, data_range=1.0)
                
                results[name]["psnr_gains"].append((psnr_enh - psnr_base).item())
                results[name]["ssim"].append(ssim_enh.item())
    
    # Summary
    print("\n" + "=" * 70)
    print("WYNIKI EWALUACJI (200 próbek)")
    print("=" * 70)
    print(f"\n{'Model':<15} {'PSNR Gain':<12} {'SSIM':<10} {'Parametry':<12}")
    print("-" * 50)
    
    summary = {}
    for name in models.keys():
        avg_gain = np.mean(results[name]["psnr_gains"])
        avg_ssim = np.mean(results[name]["ssim"])
        params = sum(p.numel() for p in models[name].parameters())
        print(f"{name:<15} {avg_gain:+.4f} dB    {avg_ssim:.4f}      {params:,}")
        summary[name] = {"psnr_gain": avg_gain, "ssim": avg_ssim, "params": params}
    
    avg_base_psnr = np.mean(baseline["psnr"])
    avg_base_ssim = np.mean(baseline["ssim"])
    print(f"\nBaseline PSNR: {avg_base_psnr:.2f} dB, SSIM: {avg_base_ssim:.4f}")
    
    with open("evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


if __name__ == "__main__":
    main()
