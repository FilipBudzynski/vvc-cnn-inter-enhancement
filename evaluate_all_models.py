#!/usr/bin/env python3
"""
Ewaluacja wszystkich modeli na zbiorze testowym
Porównanie PSNR dla różnych QP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import json
from pytorch_msssim import ssim
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    """Dataset testowy - ładuje pełne ramki bez cropowania"""
    
    def __init__(self, data_dir="data/precomputed"):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        for video_dir in self.data_dir.iterdir():
            if video_dir.is_dir():
                frames = sorted(video_dir.glob("poc_*.pt"), 
                             key=lambda x: int(x.stem.split("_")[1]))
                for i in range(1, len(frames) - 1):
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
            "video": curr_data["video"],
            "poc": curr_data["poc"],
        }


class ResNetNoTemporal(nn.Module):
    """ResNet bez ramek temporalnych - tylko aktualna ramka + metadata"""
    
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
        
        self.blocks = nn.Sequential(
            self._make_block(base_channels, base_channels),
            self._make_block(base_channels, base_channels),
            self._make_block(base_channels, base_channels),
            self._make_block(base_channels, base_channels),
        )
        
        self.output = nn.Conv2d(base_channels, 3, 3, padding=1)
    
    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        )
    
    def forward(self, curr, features):
        meta = self.metadata_encoder(features)
        x = torch.cat([curr, meta], dim=1)
        x = self.features(x)
        x = self.blocks(x)
        return curr + self.output(x)


def load_models():
    """Ładuje wszystkie modele"""
    models = {}
    
    class SnowConfig:
        base_channels = 64
        metadata_channels = 19
    
    class SnowWideConfig:
        base_channels = 64
        metadata_channels = 19
    
    # 1. ResNet bez temporal (Intra-only)
    models["ResNet_Intra"] = ResNetNoTemporal(base_channels=64)
    try:
        ckpt = torch.load("experiments/enhancer/vtm_resnet_v6.pth", map_location=DEVICE)
        # Adapt checkpoint keys if needed
        new_state = {}
        for k, v in ckpt.items():
            new_k = k.replace("resnet.", "metadata_encoder.") if "metadata" in k else k
            new_k = new_k.replace("model.", "") if "model" in k else new_k
            new_state[new_k] = v
        try:
            models["ResNet_Intra"].load_state_dict(new_state, strict=False)
        except:
            models["ResNet_Intra"].load_state_dict(ckpt, strict=False)
        print("Loaded ResNet_Intra from vtm_resnet_v6.pth")
    except Exception as e:
        print(f"Warning: Could not load ResNet_Intra: {e}")
    
    # 2. Snow (temporal)
    try:
        from enhancer.models.snow import SnowEnhancer
        models["Snow"] = SnowEnhancer(SnowConfig()).to(DEVICE)
        models["Snow"].load_state_dict(torch.load("checkpoints/snow_epoch_490.pt", map_location=DEVICE))
        print("Loaded Snow from snow_epoch_490.pt")
    except Exception as e:
        print(f"Warning: Could not load Snow: {e}")
    
    # 3. Snow-Wide (temporal + wide context)
    try:
        from enhancer.models.snow_wide import SnowWideEnhancer
        models["Snow_Wide"] = SnowWideEnhancer(SnowWideConfig()).to(DEVICE)
        models["Snow_Wide"].load_state_dict(torch.load("checkpoints/snow_wide_epoch_460.pt", map_location=DEVICE))
        print("Loaded Snow_Wide from snow_wide_epoch_460.pt")
    except Exception as e:
        print(f"Warning: Could not load Snow_Wide: {e}")
    
    for name, model in models.items():
        model.eval()
        model.to(DEVICE)
    
    return models


def count_parameters(models):
    """Liczy parametry każdego modelu"""
    counts = {}
    for name, model in models.items():
        counts[name] = sum(p.numel() for p in model.parameters())
    return counts


def evaluate_models(models, dataset):
    """Ewaluuje wszystkie modele"""
    results = {name: {"psnr_gain": [], "ssim": [], "psnr_before": [], "psnr_after": []} 
               for name in models.keys()}
    results["baseline"] = {"psnr": [], "ssim": []}
    
    sample_idx = 0
    
    for batch in tqdm(DataLoader(dataset, batch_size=1, shuffle=False), desc="Evaluating"):
        prev = batch["prev"].to(DEVICE)
        curr = batch["curr"].to(DEVICE)
        next_f = batch["next"].to(DEVICE)
        original = batch["original"].to(DEVICE)
        features = batch["features"].to(DEVICE)
        
        with torch.no_grad():
            # Baseline PSNR (przed enhancement)
            for i in range(curr.shape[0]):
                mse_before = F.mse_loss(curr[i], original[i]).item()
                psnr_before = 10 * np.log10(1.0 / mse_before) if mse_before > 0 else 40
                results["baseline"]["psnr"].append(psnr_before)
                results["baseline"]["ssim"].append(ssim(curr[i:i+1], original[i:i+1], data_range=1.0).item())
                
                # Evaluate each model
                for name, model in models.items():
                    if name == "ResNet_Intra":
                        enhanced = model(curr, features).clamp(0, 1)
                    else:
                        enhanced = model(curr, prev, next_f, features).clamp(0, 1)
                    
                    mse_after = F.mse_loss(enhanced[i], original[i]).item()
                    psnr_after = 10 * np.log10(1.0 / mse_after) if mse_after > 0 else 40
                    ssim_val = ssim(enhanced[i:i+1], original[i:i+1], data_range=1.0).item()
                    psnr_gain = psnr_after - psnr_before
                    
                    results[name]["psnr_gain"].append(psnr_gain)
                    results[name]["psnr_before"].append(psnr_before)
                    results[name]["psnr_after"].append(psnr_after)
                    results[name]["ssim"].append(ssim_val)
    
    return results


def main():
    print(f"Using device: {DEVICE}")
    
    # Ładowanie modeli
    print("\n=== Ładowanie modeli ===")
    models = load_models()
    
    # Liczenie parametrów
    print("\n=== Liczba parametrów ===")
    param_counts = count_parameters(models)
    for name, count in param_counts.items():
        print(f"{name}: {count:,} parametrów")
    
    # Ewaluacja
    print("\n=== Ewaluacja na zbiorze testowym ===")
    dataset = TestDataset()
    print(f"Liczba próbek testowych: {len(dataset)}")
    
    results = evaluate_models(models, dataset)
    
    # Podsumowanie wyników
    print("\n" + "=" * 70)
    print("WYNIKI EWALUACJI")
    print("=" * 70)
    print(f"\n{'Model':<20} {'PSNR przed':<12} {'PSNR po':<12} {'PSNR Gain':<12} {'SSIM':<10} {'Parametry':<15}")
    print("-" * 80)
    
    summary = {}
    for name in list(models.keys()) + ["baseline"]:
        if name not in results:
            continue
        psnr_gains = results[name]["psnr_gain"] if name != "baseline" else [0]
        ssims = results[name]["ssim"]
        psnr_before = results[name]["psnr_before"] if name != "baseline" else results["baseline"]["psnr"]
        psnr_after = results[name]["psnr_after"] if name != "baseline" else results["baseline"]["psnr"]
        
        if psnr_gains or name == "baseline":
            avg_psnr_gain = np.mean(psnr_gains) if psnr_gains else 0
            avg_ssim = np.mean(ssims) if ssims else 0
            avg_psnr_before = np.mean(psnr_before) if psnr_before else 0
            avg_psnr_after = np.mean(psnr_after) if psnr_after else 0
            
            summary[name] = {
                "psnr_gain": float(avg_psnr_gain),
                "ssim": float(avg_ssim),
                "psnr_before": float(avg_psnr_before),
                "psnr_after": float(avg_psnr_after),
                "parametry": param_counts.get(name, 0),
            }
            
            if name == "baseline":
                print(f"{name:<20} {avg_psnr_before:<12.2f} {avg_psnr_after:<12.2f} {'--':<12} {avg_ssim:<10.4f} {'--':<15}")
            else:
                params = param_counts.get(name, 0)
                print(f"{name:<20} {avg_psnr_before:<12.2f} {avg_psnr_after:<12.2f} {avg_psnr_gain:+.4f}{'':>6} {avg_ssim:<10.4f} {params:,}")
    
    # Zapisz wyniki do JSON
    with open("evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nWyniki zapisane do evaluation_results.json")
    
    return summary


if __name__ == "__main__":
    main()
