#!/usr/bin/env python3
"""
Ewaluacja modeli dla różnych QP
"""

import os
import subprocess
import re
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim
import numpy as np
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QP_VALUES = [22, 27, 32, 37, 42]
FRAMES_PER_VIDEO = 64

print(f"Device: {DEVICE}")


def parse_info(info_path):
    """Parsuje plik .info"""
    content = info_path.read_text()
    width = int(re.search(r"Width\s+:\s+(\d+)", content).group(1))
    height = int(re.search(r"Height\s+:\s+(\d+)", content).group(1))
    fps_match = re.search(r"FrameRate_Num\s+:\s+(\d+)", content)
    fps_denom_match = re.search(r"FrameRate_Den\s+:\s+(\d+)", content)
    if fps_match and fps_denom_match:
        fps = int(fps_match.group(1)) / int(fps_denom_match.group(1))
    else:
        fps = 30.0
    return width, height, fps


def encode_videos(qp):
    """Koduje wszystkie wideo dla danego QP"""
    print(f"\n{'='*60}")
    print(f"Kodowanie dla QP = {qp}")
    print(f"{'='*60}")
    
    output_dir = f"./output_qp{qp}/encoded"
    os.makedirs(output_dir, exist_ok=True)
    
    yuv_files = list(Path("data").glob("*.yuv"))
    
    for yuv_file in tqdm(yuv_files, desc=f"QP{qp}"):
        info_files = list(Path("data").glob(f"{yuv_file.stem}*.info"))
        if not info_files:
            continue
        
        width, height, fps = parse_info(info_files[0])
        stem = yuv_file.stem
        bitstream = f"{output_dir}/{stem}_QP{qp}.vvc"
        recon = f"{output_dir}/{stem}_QP{qp}_rec.yuv"
        
        if not os.path.exists(recon):
            cmd = [
                "./bin/vvenc/bin/release-static/vvencFFapp",
                "-i", str(yuv_file),
                "-s", f"{width}x{height}",
                "-fr", str(int(fps)),
                "-f", str(FRAMES_PER_VIDEO),
                "-q", str(qp),
                "-b", bitstream,
                "-o", recon,
                "--preset", "fast",
                "--alf", "0",
                "--sao", "0",
                "--LoopFilterDisable", "1",
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            except Exception as e:
                print(f"Błąd kodowania {stem}: {e}")


class TestDataset(Dataset):
    def __init__(self, data_dir, max_samples=100):
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


def load_models():
    models = {}
    
    models["ResNet_Intra"] = ResNetNoTemporal(base_channels=64)
    try:
        ckpt = torch.load("experiments/enhancer/vtm_resnet_v6.pth", map_location=DEVICE)
        models["ResNet_Intra"].load_state_dict(ckpt, strict=False)
    except:
        pass
    
    try:
        from enhancer.models.snow import SnowEnhancer
        class C: base_channels=64; metadata_channels=19
        models["Snow"] = SnowEnhancer(C()).to(DEVICE)
        models["Snow"].load_state_dict(torch.load("checkpoints/snow_epoch_490.pt", map_location=DEVICE))
    except:
        pass
    
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
    
    return models


def prepare_precomputed_from_decoded(qp, max_samples_per_video=10):
    """Przygotowuje precomputed z zakodowanych plików"""
    precomputed_dir = f"./data_qp{qp}_test"
    encoded_dir = f"./output_qp{qp}/encoded"
    
    if os.path.exists(precomputed_dir):
        print(f"Dataset QP{qp} już istnieje ({len(list(Path(precomputed_dir).glob('*/*.pt')))} plików)")
        return True
    
    os.makedirs(precomputed_dir, exist_ok=True)
    
    recon_files = sorted(Path(encoded_dir).glob("*_rec.yuv"))
    total_frames = 0
    
    for recon_file in tqdm(recon_files, desc=f"Prepare QP{qp}"):
        stem = recon_file.stem.replace(f"_QP{qp}_rec", "")
        orig_file = Path("data") / f"{stem}.yuv"
        
        if not orig_file.exists():
            continue
        
        try:
            width, height, fps = parse_info(list(Path("data").glob(f"{stem}*.info"))[0])
        except:
            continue
        
        video_dir = Path(precomputed_dir) / stem
        video_dir.mkdir(exist_ok=True)
        
        try:
            from features_generator.features_generator import FeaturesGenerator
            fg = FeaturesGenerator(str(recon_file), str(orig_file), width, height)
            frames = fg.generate()
            
            for i, frame_data in enumerate(frames[:max_samples_per_video + 2]):
                if len(list(video_dir.glob("*.pt"))) >= max_samples_per_video + 2:
                    break
                torch.save(frame_data, video_dir / f"poc_{i:04d}.pt")
                total_frames += 1
        except Exception as e:
            print(f"Błąd {stem}: {e}")
    
    print(f"Przygotowano {total_frames} ramek dla QP{qp}")
    return total_frames > 0


def evaluate_qp(qp, models, dataset):
    """Ewaluuje modele dla danego QP"""
    results = {name: {"psnr_gains": [], "ssim": []} for name in models.keys()}
    baseline = {"psnr": [], "ssim": []}
    
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            prev = batch["prev"].unsqueeze(0).to(DEVICE)
            curr = batch["curr"].unsqueeze(0).to(DEVICE)
            next_f = batch["next"].unsqueeze(0).to(DEVICE)
            original = batch["original"].unsqueeze(0).to(DEVICE)
            features = batch["features"].unsqueeze(0).to(DEVICE)
            
            psnr_base = 10 * torch.log10(1.0 / (torch.nn.functional.mse_loss(curr, original) + 1e-10) + 1e-10)
            ssim_base = ssim(curr, original, data_range=1.0)
            baseline["psnr"].append(psnr_base.item())
            baseline["ssim"].append(ssim_base.item())
            
            for name, model in models.items():
                if name == "ResNet_Intra":
                    enhanced = model(curr, features).clamp(0, 1)
                else:
                    enhanced = model(curr, prev, next_f, features).clamp(0, 1)
                
                psnr_enh = 10 * torch.log10(1.0 / (torch.nn.functional.mse_loss(enhanced, original) + 1e-10) + 1e-10)
                ssim_enh = ssim(enhanced, original, data_range=1.0)
                
                results[name]["psnr_gains"].append((psnr_enh - psnr_base).item())
                results[name]["ssim"].append(ssim_enh.item())
    
    return {
        "baseline_psnr": np.mean(baseline["psnr"]),
        "baseline_ssim": np.mean(baseline["ssim"]),
        "samples": len(dataset),
        "models": {name: {
            "psnr_gain": np.mean(results[name]["psnr_gains"]),
            "ssim": np.mean(results[name]["ssim"]),
        } for name in models.keys()}
    }


def main():
    # Krok 1: Kodowanie
    for qp in QP_VALUES:
        encode_videos(qp)
    
    # Krok 2: Przygotowanie datasetów
    for qp in QP_VALUES:
        prepare_precomputed_from_decoded(qp)
    
    # Krok 3: Ładowanie modeli
    print("\n" + "="*60)
    print("Ładowanie modeli")
    print("="*60)
    models = load_models()
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} params")
    
    # Krok 4: Ewaluacja
    print("\n" + "="*60)
    print("Ewaluacja")
    print("="*60)
    
    all_results = {}
    for qp in QP_VALUES:
        precomputed_dir = f"./data_qp{qp}_test"
        if not os.path.exists(precomputed_dir):
            continue
            
        dataset = TestDataset(precomputed_dir, max_samples=100)
        if len(dataset) == 0:
            continue
        
        print(f"\n--- QP = {qp} ({len(dataset)} próbek) ---")
        result = evaluate_qp(qp, models, dataset)
        all_results[qp] = result
        
        print(f"Baseline PSNR: {result['baseline_psnr']:.2f} dB")
        for name, res in result["models"].items():
            print(f"  {name}: PSNR Gain = {res['psnr_gain']:+.4f} dB, SSIM = {res['ssim']:.4f}")
    
    # Zapisz
    with open("qp_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Markdown
    md = "# Porównanie modeli dla różnych QP\n\n"
    md += "| QP | Baseline PSNR | Model | PSNR Gain | SSIM | Parametry |\n"
    md += "|-----|---------------|-------|-----------|------|-----------|\n"
    
    for qp in QP_VALUES:
        if qp not in all_results:
            continue
        first = True
        for name, res in all_results[qp]["models"].items():
            params = sum(p.numel() for p in models[name].parameters())
            if first:
                md += f"| **{qp}** | {all_results[qp]['baseline_psnr']:.2f} | {name} | {res['psnr_gain']:+.4f} | {res['ssim']:.4f} | {params:,} |\n"
                first = False
            else:
                md += f"| | | {name} | {res['psnr_gain']:+.4f} | {res['ssim']:.4f} | |\n"
    
    md += "\n## Architektury\n\n"
    md += "### ResNet (Intra-only)\n"
    md += "- Parametry: 414,702\n"
    md += "- Input: YUV (3 kanały) + Metadata (19 kanałów)\n"
    md += "- Brak ramek temporalnych\n\n"
    
    md += "### Snow\n"
    md += "- Parametry: 981,594\n"
    md += "- Input: F-1, F0, F+1 + Metadata\n"
    md += "- Feature Extraction + Alignment + Attention Fusion\n\n"
    
    md += "### Snow-Wide\n"
    md += "- Parametry: 1,293,024\n"
    md += "- Input: F-1, F0, F+1 + Metadata\n"
    md += "- Wide Context Module (7×7 depthwise, dilation=2)\n\n"
    
    md += "## Funkcje straty\n\n"
    md += "| Model | Loss |\n"
    md += "|-------|------|\n"
    md += "| ResNet | CharbonnierLoss |\n"
    md += "| Snow | L1 Loss |\n"
    md += "| Snow-Wide | 0.5×L1 + 0.15×MS-SSIM + 0.2×Gradient + 0.15×Laplacian |\n"
    
    with open("QP_COMPARISON.md", "w") as f:
        f.write(md)
    
    print("\nWyniki zapisane do QP_COMPARISON.md")


if __name__ == "__main__":
    main()
