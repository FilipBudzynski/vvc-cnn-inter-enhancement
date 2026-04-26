#!/usr/bin/env python3
"""Encode videos with QP and extract bitrate for BD-rate calculation."""

import subprocess
import re
import json
import os
import glob
from pathlib import Path

# Videos in data directory
def get_videos():
    """Get all yuv files from data dir."""
    videos = []
    for f in sorted(glob.glob("data/*.yuv")):
        name = Path(f).stem
        # Try to find resolution in existing precomputed data
        precomp_dirs = list(glob.glob(f"data/precomputed/{name}_*")) + list(glob.glob(f"data/precomputed/{name}"))
        if precomp_dirs:
            # Check first file for dimensions
            pt_files = list(Path(precomp_dirs[0]).glob("poc_*.pt"))
            if pt_files:
                import torch
                d = torch.load(pt_files[0], weights_only=True)
                # Assume YUV 3-channel, shape in H,W
                _, h, w = d['decoded'].shape
                videos.append((name, w, h))
                continue
        # Fallback - try to guess from name
        if 'qcif' in name:
            videos.append((name, 176, 144))
        elif 'cif' in name:
            videos.append((name, 352, 288))
        elif '4cif' in name:
            videos.append((name, 704, 576))
        elif '1080p' in name:
            videos.append((name, 1920, 1080))
        elif '720p' in name:
            videos.append((name, 1280, 720))
        elif 'sif' in name:
            videos.append((name, 352, 240))
        else:
            videos.append((name, 352, 288))
    return videos

QP_VALUES = [22, 27, 32, 37, 42]
ENCODER = "./bin/vvenc/bin/release-static/vvencFFapp"

def encode_video(name, width, height, qp):
    """Encode video and return bitrate."""
    output_dir = f"./output_qp{qp}/encoded"
    os.makedirs(output_dir, exist_ok=True)
    
    input_file = f"data/{name}.yuv"
    bitstream_out = f"{output_dir}/{name}_QP{qp}.vvc"
    recon_out = f"{output_dir}/{name}_QP{qp}_rec.yuv"
    
    cmd = [
        ENCODER,
        "-i", input_file,
        "-s", f"{width}x{height}",
        "-fr", "30",  # Use 30 fps (avoid NTSC issues)
        "-f", "64",  # 64 frames
        "-q", str(qp),
        "-b", bitstream_out,
        "-o", recon_out,
        "--preset", "fast",
        "--alf", "0",
        "--sao", "0",
        "--LoopFilterDisable", "1",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
        return None
    
    # Extract bitrate from output
    for line in result.stderr.split('\n'):
        if 'avg_bitrate' in line and 'kbps' in line:
            match = re.search(r'avg_bitrate[=\s]+(\d+\.?\d*)\s+kbps', line)
            if match:
                return float(match.group(1))
    
    return None

def main():
    videos = get_videos()
    print(f"Found {len(videos)} videos")
    
    results = {}
    
    for qp in QP_VALUES:
        print(f"\n=== Encoding QP={qp} ===")
        results[qp] = {}
        
        for name, width, height in videos:
            # Check if already encoded - try to reuse
            output_dir = f"./output_qp{qp}/encoded"
            bitstream = f"{output_dir}/{name}_QP{qp}.vvc"
            recon = f"{output_dir}/{name}_QP{qp}_rec.yuv"
            
            skip = os.path.exists(recon)
            
            if skip:
                # Re-encode to get fresh bitrate
                pass
            
            print(f"  {name}...", end=" ")
            bitrate = encode_video(name, width, height, qp)
            
            if bitrate:
                results[qp][name] = bitrate
                print(f"{bitrate:.1f} kbps")
            else:
                results[qp][name] = None
                print("FAILED")
        
        # Save intermediate
        with open("bitrate_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print("\n=== Summary ===")
    for qp, vids in results.items():
        rates = [r for r in vids.values() if r]
        if rates:
            avg = sum(rates) / len(rates)
            print(f"QP{qp}: avg {avg:.1f} kbps ({len(rates)}/{len(videos)} videos)")
    
    with open("bitrate_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nSaved to bitrate_results.json")

if __name__ == "__main__":
    main()
