#!/usr/bin/env python3
"""
Ewaluacja dla wszystkich QP
"""

import os
import re
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
    frames = []
    frame_size = width * height * 3 // 2
    with open(filepath, 'rb') as f:
        for _ in range(num_frames):
            data = f.read(frame_size)
            if len(data) < frame_size:
                break
            y = np.frombuffer(data[:width*height], dtype=np.uint8).reshape(height, width).astype(np.float32) / 255.0
            u = np.frombuffer(data[width*height:width*height + width*height//4], dtype=np.uint8).reshape(height//2, width//2).astype(np.float32) / 255.0
            v = np.frombuffer(data[width*height + width*height//4:], dtype=np.uint8).reshape(height//2, width//2).astype(np.float32) / 255.0
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


class YUVDataset(Dataset):
    def __init__(self, rec_dir, orig_dir, qp, max_frames=10, max_videos=15):
        self.samples = []
        rec_files = sorted(Path(rec_dir).glob(f"*_QP{qp}_rec.yuv"))[:max_videos]
        
        for rec_file in rec_files:
            stem = rec_file.stem.replace(f"_QP{qp}_rec", "")
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
                rec_frames = read_yuv_frames(str(rec_file), width, height, max_frames + 2)
                orig_frames = read_yuv_frames(str(orig_file), width, height, max_frames + 2)
            except:
                continue
            for i in range(1, min(len(rec_frames) - 1, max_frames)):
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


class ResNetSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.blocks = nn.Sequential(*[self._make_block(64) for _ in range(4)])
        self.output = nn.Conv2d(64, 3, 3, padding=1)
    
    def _make_block(self, ch):
        return nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(),
        )
    
    def forward(self, curr, prev=None, next_f=None, features=None):
        x = self.features(curr)
        x = self.blocks(x)
        return curr + self.output(x)


def main():
    print(f"Device: {DEVICE}")
    
    # Load models
    models = {}
    models["ResNet_Intra"] = ResNetSimple()
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
    
    # Evaluate all QP
    QP_VALUES = [22, 27, 32, 37, 42]
    all_results = {}
    
    print("\n" + "="*70)
    
    for qp in QP_VALUES:
        print(f"\n=== QP = {qp} ===")
        dataset = YUVDataset("./output_qp" + str(qp) + "/encoded", "./data", qp, max_frames=10, max_videos=15)
        print(f"Samples: {len(dataset)}")
        
        if len(dataset) == 0:
            continue
        
        results = {name: {"psnr_gains": [], "ssim": []} for name in models.keys()}
        baseline_psnr = []
        
        with torch.no_grad():
            for batch in dataset:
                curr = batch["curr"].unsqueeze(0).to(DEVICE)
                original = batch["original"].unsqueeze(0).to(DEVICE)
                
                psnr_base = 10 * torch.log10(1.0 / (torch.nn.functional.mse_loss(curr, original) + 1e-10) + 1e-10)
                baseline_psnr.append(psnr_base.item())
                
                for name, model in models.items():
                    if name in ["Snow", "Snow_Wide"]:
                        prev = batch["prev"].unsqueeze(0).to(DEVICE)
                        next_f = batch["next"].unsqueeze(0).to(DEVICE)
                        features = torch.zeros(1, 19, curr.shape[2], curr.shape[3], device=DEVICE)
                        enhanced = model(curr, prev, next_f, features).clamp(0, 1)
                    else:
                        enhanced = model(curr).clamp(0, 1)
                    
                    psnr_enh = 10 * torch.log10(1.0 / (torch.nn.functional.mse_loss(enhanced, original) + 1e-10) + 1e-10)
                    results[name]["psnr_gains"].append((psnr_enh - psnr_base).item())
        
        baseline_avg = np.mean(baseline_psnr)
        all_results[qp] = {"baseline_psnr": float(baseline_avg), "models": {}}
        
        print(f"Baseline PSNR: {baseline_avg:.2f} dB")
        for name in models.keys():
            avg_gain = np.mean(results[name]["psnr_gains"])
            all_results[qp]["models"][name] = {"psnr_gain": float(avg_gain)}
            print(f"  {name}: {avg_gain:+.4f} dB")
    
    # Save results
    with open("all_qp_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate markdown table
    md = "# Porównanie Modeli dla Różnych QP\n\n"
    md += "| QP | Baseline PSNR | "
    for name in models.keys():
        md += f"{name} PSNR Gain | "
    md += "Parametry |\n"
    md += "|-----|---------------|"
    for _ in models.keys():
        md += "-----------|"
    md += "-----------|\n"
    
    for qp in QP_VALUES:
        if qp not in all_results:
            continue
        md += f"| **{qp}** | {all_results[qp]['baseline_psnr']:.2f} dB |"
        for name in models.keys():
            if name in all_results[qp]["models"]:
                md += f" {all_results[qp]['models'][name]['psnr_gain']:+.4f} dB |"
            else:
                md += " -- |"
        params = sum(p.numel() for p in models[name].parameters())
        md += f" {params:,} |\n"
    
    with open("FINAL_QP_COMPARISON.md", "w") as f:
        f.write(md)
    
    print("\n\nZapisano: all_qp_results.json, FINAL_QP_COMPARISON.md")
    print("\n" + md)


if __name__ == "__main__":
    main()
