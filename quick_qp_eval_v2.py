#!/usr/bin/env python3
"""
Szybka ewaluacja modeli na danych QP22
Bezpośrednio wczytuje YUV bez features_generator
"""

import os
import re
import struct
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pytorch_msssim import ssim
from tqdm import tqdm
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_yuv_frames(filepath, width, height, num_frames=20):
    """Czyta ramki YUV z pliku"""
    frames = []
    frame_size = width * height * 3 // 2  # YUV420
    
    with open(filepath, 'rb') as f:
        for _ in range(num_frames):
            data = f.read(frame_size)
            if len(data) < frame_size:
                break
            
            y = np.frombuffer(data[:width*height], dtype=np.uint8).reshape(height, width)
            u = np.frombuffer(data[width*height:width*height + width*height//4], dtype=np.uint8).reshape(height//2, width//2)
            v = np.frombuffer(data[width*height + width*height//4:], dtype=np.uint8).reshape(height//2, width//2)
            
            # Konwersja do [0,1] float
            y = y.astype(np.float32) / 255.0
            u = u.astype(np.float32) / 255.0
            v = v.astype(np.float32) / 255.0
            
            # Upsample U, V do pełnej rozdzielczości (YUV jako 3 kanały)
            u_full = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
            v_full = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
            
            yuv = np.stack([y, u_full, v_full], axis=0).astype(np.float32)
            frames.append(yuv)
    
    return frames


def parse_info(info_path):
    content = Path(info_path).read_text()
    width = int(re.search(r"Width\s+:\s+(\d+)", content).group(1))
    height = int(re.search(r"Height\s+:\s+(\d+)", content).group(1))
    return width, height


class YUVTestDataset(Dataset):
    def __init__(self, rec_dir, orig_dir, max_frames_per_video=10, max_videos=10):
        self.samples = []
        
        rec_files = sorted(Path(rec_dir).glob("*_QP22_rec.yuv"))[:max_videos]
        
        for rec_file in rec_files:
            stem = rec_file.stem.replace("_QP22_rec", "")
            orig_file = Path(orig_dir) / f"{stem}.yuv"
            
            if not orig_file.exists():
                continue
            
            try:
                info_files = list(Path(orig_dir).glob(f"{stem}*.info"))
                if not info_files:
                    continue
                width, height = parse_info(info_files[0])
            except:
                continue
            
            try:
                rec_frames = read_yuv_frames(str(rec_file), width, height, max_frames_per_video + 2)
                orig_frames = read_yuv_frames(str(orig_file), width, height, max_frames_per_video + 2)
            except:
                continue
            
            for i in range(1, min(len(rec_frames) - 1, max_frames_per_video)):
                self.samples.append({
                    "prev": torch.from_numpy(rec_frames[i-1]),
                    "curr": torch.from_numpy(rec_frames[i]),
                    "next": torch.from_numpy(rec_frames[i+1]),
                    "original": torch.from_numpy(orig_frames[i]),
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class SimpleModel(nn.Module):
    """Model który tylko dodaje residuum (degraded input = output)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, curr, prev, next_f, features=None):
        return curr  # Identity - brak poprawy


class ResNetNoTemporal(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, padding=3),
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
    
    def forward(self, curr, prev=None, next_f=None, features=None):
        x = self.features(curr)
        x = self.blocks(x)
        return curr + self.output(x)


def main():
    print(f"Device: {DEVICE}")
    
    # Dataset
    print("\nŁadowanie danych...")
    dataset = YUVTestDataset(
        rec_dir="./output_qp22/encoded",
        orig_dir="./data",
        max_frames_per_video=15,
        max_videos=5
    )
    print(f"Test samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Brak danych!")
        return
    
    # Load models
    models = {}
    
    # Baseline (identity)
    models["Baseline"] = SimpleModel()
    
    # ResNet (bez metadata)
    models["ResNet_Intra"] = ResNetNoTemporal(base_channels=64)
    try:
        ckpt = torch.load("experiments/enhancer/vtm_resnet_v6.pth", map_location=DEVICE)
        # Próbuj załadować
        try:
            models["ResNet_Intra"].load_state_dict(ckpt, strict=False)
        except:
            pass  # Ignoruj błędy ładowania
    except:
        pass
    
    # Snow
    try:
        from enhancer.models.snow import SnowEnhancer
        class C: base_channels=64; metadata_channels=19
        models["Snow"] = SnowEnhancer(C()).to(DEVICE)
        models["Snow"].load_state_dict(torch.load("checkpoints/snow_epoch_490.pt", map_location=DEVICE))
    except:
        pass
    
    # Snow-Wide
    try:
        from enhancer.models.snow_wide import SnowWideEnhancer
        class C: base_channels=64; metadata_channels=19
        models["Snow_Wide"] = SnowWideEnhancer(C()).to(DEVICE)
        models["Snow_Wide"].load_state_dict(torch.load("checkpoints/snow_wide_epoch_460.pt", map_location=DEVICE))
    except:
        pass
    
    for name, model in models.items():
        model.eval()
        model.to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} params")
    
    # Evaluate
    print("\nEwaluacja...")
    results = {name: {"psnr_gains": [], "ssim": []} for name in models.keys()}
    baseline_psnr = []
    baseline_ssim = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset, desc="Testing")):
            curr = batch["curr"].unsqueeze(0).to(DEVICE)
            original = batch["original"].unsqueeze(0).to(DEVICE)
            
            # Baseline
            psnr_base = 10 * torch.log10(1.0 / (torch.nn.functional.mse_loss(curr, original) + 1e-10) + 1e-10)
            ssim_base = ssim(curr, original, data_range=1.0)
            baseline_psnr.append(psnr_base.item())
            baseline_ssim.append(ssim_base.item())
            
            # Models
            for name, model in models.items():
                if name == "Snow" or name == "Snow_Wide":
                    prev = batch["prev"].unsqueeze(0).to(DEVICE)
                    next_f = batch["next"].unsqueeze(0).to(DEVICE)
                    # Twórz dummy features (19 kanałów)
                    features = torch.zeros(1, 19, curr.shape[2], curr.shape[3], device=DEVICE)
                    enhanced = model(curr, prev, next_f, features).clamp(0, 1)
                elif name == "ResNet_Intra":
                    enhanced = model(curr).clamp(0, 1)
                else:
                    enhanced = curr
                
                psnr_enh = 10 * torch.log10(1.0 / (torch.nn.functional.mse_loss(enhanced, original) + 1e-10) + 1e-10)
                ssim_enh = ssim(enhanced, original, data_range=1.0)
                
                results[name]["psnr_gains"].append((psnr_enh - psnr_base).item())
                results[name]["ssim"].append(ssim_enh.item())
    
    # Results
    print("\n" + "="*70)
    print(f"WYNIKI EWALUACJI (QP=22)")
    print("="*70)
    print(f"\n{'Model':<15} {'PSNR Gain':<14} {'SSIM':<12} {'Parametry':<12}")
    print("-"*55)
    
    for name in models.keys():
        avg_gain = np.mean(results[name]["psnr_gains"])
        avg_ssim = np.mean(results[name]["ssim"])
        params = sum(p.numel() for p in models[name].parameters())
        print(f"{name:<15} {avg_gain:+.4f} dB       {avg_ssim:.4f}       {params:,}")
    
    print(f"\nBaseline PSNR: {np.mean(baseline_psnr):.2f} dB")
    print(f"Baseline SSIM: {np.mean(baseline_ssim):.4f}")
    
    # Save JSON
    all_results = {
        "qp": 22,
        "baseline_psnr": float(np.mean(baseline_psnr)),
        "baseline_ssim": float(np.mean(baseline_ssim)),
        "models": {name: {
            "psnr_gain": float(np.mean(results[name]["psnr_gains"])),
            "ssim": float(np.mean(results[name]["ssim"])),
            "params": sum(p.numel() for p in models[name].parameters()),
        } for name in models.keys()}
    }
    
    with open("qp22_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
