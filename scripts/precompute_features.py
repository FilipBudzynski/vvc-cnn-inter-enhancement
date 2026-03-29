#!/usr/bin/env python3
"""Pre-compute VVC features and save as .pt files for fast loading."""

import os
import re
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from features_parser.parser import VTMParser
from features_generator.generator import FeatureMapGenerator

FEATURE_ORDER = ["QP", "PredMode", "Depth", "Boundary", "MVL0_X", "MVL0_Y", "MVL1_X", "MVL1_Y"]

def parse_video_info(csv_path):
    width, height = 0, 0
    with open(csv_path, "r") as f:
        for line in f:
            if "# Sequence size:" in line:
                dims = re.findall(r"\d+", line)
                if len(dims) >= 2:
                    width, height = int(dims[0]), int(dims[1])
                    break
    return width, height

def parse_pocs(csv_path):
    with open(csv_path, "r") as f:
        content = f.read()
        found = re.findall(r"POC (\d+)", content)
        if found:
            return sorted(set(int(p) for p in found))
    return []

def read_yuv_frame(path, poc, width, height):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    with open(path, "rb") as f:
        f.seek(poc * (y_size + 2 * uv_size))
        y = np.frombuffer(f.read(y_size), dtype=np.uint8).copy()
        u = np.frombuffer(f.read(uv_size), dtype=np.uint8).copy()
        v = np.frombuffer(f.read(uv_size), dtype=np.uint8).copy()
    
    # Reshape and convert to [0, 1]
    y_t = torch.from_numpy(y.reshape(height, width)).float() / 255.0
    u_t = torch.from_numpy(u.reshape(height//2, width//2)).float() / 255.0
    v_t = torch.from_numpy(v.reshape(height//2, width//2)).float() / 255.0
    
    # Upsample chroma - use squeeze to remove extra dim
    u_up = F.interpolate(u_t.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    v_up = F.interpolate(v_t.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    
    return torch.stack([y_t, u_up, v_up], dim=0)

def normalize(name, data):
    t = torch.from_numpy(data).float()
    if name == "QP": return (t / 63.0).clamp(0, 1)
    elif name == "Depth": return (t / 7.0).clamp(0, 1)
    elif name == "PredMode": return (t / 3.0).clamp(0, 1)
    elif name == "Boundary": return (t / 1.0).clamp(0, 1)
    elif "MV" in name: return (t / 128.0).clamp(-1, 1)
    return t

def process_video(dec_yuv, orig_yuv, csv_path, output_dir, force=False):
    video_name = Path(csv_path).stem.split("_QP")[0]
    width, height = parse_video_info(csv_path)
    if not width: return None
    
    pocs = parse_pocs(csv_path)
    if not pocs: return None
    
    cache = Path(csv_path).with_suffix(".pkl")
    if cache.exists():
        with open(cache, "rb") as f: tokens = pickle.load(f)
    else:
        tokens = VTMParser().parse_file(csv_path)
        with open(cache, "wb") as f: pickle.dump(tokens, f)
    
    gen = FeatureMapGenerator(width, height)
    out_dir = output_dir / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for poc in tqdm(pocs, desc=f"  {video_name}", leave=False):
        out_file = out_dir / f"poc_{poc:04d}.pt"
        if out_file.exists() and not force: 
            count += 1
            continue
        
        try:
            decoded = read_yuv_frame(dec_yuv, poc, width, height)
            original = read_yuv_frame(orig_yuv, poc, width, height)
        except Exception as e:
            print(f"Error reading {video_name} POC {poc}: {e}")
            continue
        
        maps_dict = gen.generate_maps_for_frame(tokens.get(poc, []))
        features = torch.stack([normalize(n, maps_dict.get(n, np.zeros((height, width), np.float32))) for n in FEATURE_ORDER])
        
        torch.save({"decoded": decoded, "original": original, "features": features, "poc": poc, "video": video_name}, out_file)
        count += 1
    
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoded-dir", default="output/decoded")
    parser.add_argument("--original-dir", default="data")
    parser.add_argument("--output-dir", default="data/precomputed")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dec_dir = Path(args.decoded_dir)
    csv_files = list(dec_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} videos")
    
    total = 0
    for csv_path in tqdm(csv_files):
        base = csv_path.stem
        vid = base.split("_QP")[0]
        
        # Find decoded YUV
        dec_yuv = dec_dir / f"{base}_vtm_rec.yuv"
        if not dec_yuv.exists(): dec_yuv = dec_dir / f"{base}_rec.yuv"
        if not dec_yuv.exists(): continue
        
        # Find original YUV
        orig = Path(args.original_dir) / f"{vid}.yuv"
        if not orig.exists(): continue
        
        c = process_video(str(dec_yuv), str(orig), str(csv_path), out_dir, args.force)
        if c: total += c
    
    print(f"Done! Saved {total} frames to {out_dir}")

if __name__ == "__main__": main()
