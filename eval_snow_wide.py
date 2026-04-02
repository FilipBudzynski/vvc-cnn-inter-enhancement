#!/usr/bin/env python3
"""
Evaluate Snow-Wide on test split (patches) and full frames
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from pytorch_msssim import ssim, ms_ssim
import numpy as np
from pathlib import Path
from tqdm import tqdm

PATCH_SIZE = 132
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def yuv_to_rgb(yuv):
    y = yuv[:, :, 0]
    u = yuv[:, :, 1]
    v = yuv[:, :, 2]
    r = y + 1.402 * (v - 0.5)
    g = y - 0.344136 * (u - 0.5) - 0.714136 * (v - 0.5)
    b = y + 1.772 * (u - 0.5)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1)


def get_model(checkpoint):
    from enhancer.models.snow_wide import SnowWideEnhancer
    
    class Config:
        metadata_channels = 19
        base_channels = 64
    
    config = Config()
    model = SnowWideEnhancer(config).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def eval_patches(checkpoint, data_dir="data/precomputed"):
    """Evaluate on test split (132x132 patches)"""
    print("=" * 60)
    print("EVALUATING ON TEST SPLIT (132x132 patches)")
    print("=" * 60)
    
    from enhancer.dataset_blackfyre import BlackfyreDataset
    
    test_dataset = BlackfyreDataset(data_dir=data_dir, patch_size=PATCH_SIZE, split="test")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    model = get_model(checkpoint)
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    
    psnr_input_total = 0
    psnr_enhanced_total = 0
    ssim_total = 0
    n_samples = 0
    
    video_stats = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing patches"):
            decoded, original, metadata, info = batch
            prev_frame, curr_frame, next_frame = decoded
            
            prev_frame = prev_frame.to(DEVICE)
            curr_frame = curr_frame.to(DEVICE)
            next_frame = next_frame.to(DEVICE)
            original = original.to(DEVICE)
            metadata = metadata.to(DEVICE)
            
            enhanced = model(curr_frame, prev_frame, next_frame, metadata).clamp(0, 1)
            
            for i in range(enhanced.shape[0]):
                video = info["video"][i]
                
                psnr_in = psnr_metric(curr_frame[i:i+1], original[i:i+1]).item()
                psnr_enh = psnr_metric(enhanced[i:i+1], original[i:i+1]).item()
                ssim_val = ssim_metric(enhanced[i:i+1], original[i:i+1]).item()
                
                psnr_input_total += psnr_in
                psnr_enhanced_total += psnr_enh
                ssim_total += ssim_val
                n_samples += 1
                
                if video not in video_stats:
                    video_stats[video] = {"psnr_in": [], "psnr_enh": [], "ssim": []}
                video_stats[video]["psnr_in"].append(psnr_in)
                video_stats[video]["psnr_enh"].append(psnr_enh)
                video_stats[video]["ssim"].append(ssim_val)
                
                psnr_metric.reset()
                ssim_metric.reset()
    
    print(f"\nTest samples: {n_samples}")
    print(f"Average Input PSNR:  {psnr_input_total/n_samples:.4f} dB")
    print(f"Average Enhanced PSNR: {psnr_enhanced_total/n_samples:.4f} dB")
    print(f"Average PSNR Gain:   {(psnr_enhanced_total - psnr_input_total)/n_samples:.4f} dB")
    print(f"Average SSIM:        {ssim_total/n_samples:.4f}")
    
    print(f"\nPer-video results:")
    print(f"{'Video':<35} {'Input PSNR':>12} {'Enhanced PSNR':>14} {'Gain':>8} {'SSIM':>8}")
    print("-" * 80)
    for video in sorted(video_stats.keys()):
        stats = video_stats[video]
        n = len(stats["psnr_in"])
        psnr_in = sum(stats["psnr_in"]) / n
        psnr_enh = sum(stats["psnr_enh"]) / n
        ssim_val = sum(stats["ssim"]) / n
        gain = psnr_enh - psnr_in
        print(f"{video:<35} {psnr_in:>12.4f} {psnr_enh:>14.4f} {gain:>+8.4f} {ssim_val:>8.4f}")
    
    return psnr_input_total/n_samples, psnr_enhanced_total/n_samples, ssim_total/n_samples


def eval_full_frames(checkpoint, data_dir="data/precomputed"):
    """Evaluate on full frames using tile-based inference"""
    print("\n" + "=" * 60)
    print("EVALUATING ON FULL FRAMES (tile-based inference)")
    print("=" * 60)
    
    from enhancer.dataset_blackfyre import BlackfyreDataset
    
    model = get_model(checkpoint)
    
    # Load all frames from test videos (we'll use the same videos as test split)
    test_dataset = BlackfyreDataset(data_dir=data_dir, patch_size=PATCH_SIZE, split="test")
    
    # Get unique test videos
    test_videos = set()
    for video_name, _, _, _ in test_dataset.samples:
        test_videos.add(video_name)
    
    print(f"Test videos: {len(test_videos)}")
    
    psnr_input_total = 0
    psnr_enhanced_total = 0
    ssim_total = 0
    n_frames = 0
    
    video_results = {}
    
    tile_size = 132
    overlap = 4  # pixels overlap between tiles
    
    for video_name in tqdm(sorted(test_videos), desc="Processing videos"):
        video_dir = Path(data_dir) / video_name
        frames = sorted(video_dir.glob("poc_*.pt"), key=lambda x: int(x.stem.split("_")[1]))
        
        if len(frames) < 3:
            continue
        
        video_psnr_in = []
        video_psnr_enh = []
        video_ssim = []
        
        for frame_path in frames:
            data = torch.load(frame_path, weights_only=True)
            original = data["original"]  # [3, H, W]
            decoded = data["decoded"]    # [3, H, W]
            features = data["features"]  # [19, H, W]
            
            _, H, W = decoded.shape
            
            # Need prev and next frames
            poc = data["poc"]
            prev_path = video_dir / f"poc_{poc-1:04d}.pt"
            next_path = video_dir / f"poc_{poc+1:04d}.pt"
            
            if not prev_path.exists() or not next_path.exists():
                continue
            
            prev_data = torch.load(prev_path, weights_only=True)
            next_data = torch.load(next_path, weights_only=True)
            
            prev_frame = prev_data["decoded"]
            next_frame = next_data["decoded"]
            
            # Tile-based inference
            enhanced = torch.zeros_like(decoded)
            weight_map = torch.zeros_like(decoded)
            
            for y in range(0, H - tile_size + 1, tile_size - overlap):
                for x in range(0, W - tile_size + 1, tile_size - overlap):
                    y_end = min(y + tile_size, H)
                    x_end = min(x + tile_size, W)
                    y_start = y_end - tile_size
                    x_start = x_end - tile_size
                    
                    # Extract tiles
                    curr_tile = decoded[:, y_start:y_end, x_start:x_end].unsqueeze(0)
                    prev_tile = prev_frame[:, y_start:y_end, x_start:x_end].unsqueeze(0)
                    next_tile = next_frame[:, y_start:y_end, x_start:x_end].unsqueeze(0)
                    meta_tile = features[:, y_start:y_end, x_start:x_end].unsqueeze(0)
                    
                    with torch.no_grad():
                        enh_tile = model(
                            curr_tile.to(DEVICE),
                            prev_tile.to(DEVICE),
                            next_tile.to(DEVICE),
                            meta_tile.to(DEVICE)
                        ).clamp(0, 1).cpu()
                    
                    enhanced[:, y_start:y_end, x_start:x_end] += enh_tile[0]
                    weight_map[:, y_start:y_end, x_start:x_end] += 1
            
            enhanced = enhanced / weight_map
            
            # Compute PSNR on full frame
            psnr_in = 10 * torch.log10(1.0 / F.mse_loss(decoded, original)).item()
            psnr_enh = 10 * torch.log10(1.0 / F.mse_loss(enhanced, original)).item()
            
            # SSIM on full frame (may need to handle large frames)
            try:
                ssim_val = ssim(enhanced.unsqueeze(0), original.unsqueeze(0), data_range=1.0, size_average=True, win_size=7).item()
            except:
                # For very large frames, compute on center crop
                cy, cx = H // 2, W // 2
                crop = 256
                ssim_val = ssim(
                    enhanced[:, cy-crop:cy+crop, cx-crop:cx+crop].unsqueeze(0),
                    original[:, cy-crop:cy+crop, cx-crop:cx+crop].unsqueeze(0),
                    data_range=1.0, size_average=True, win_size=7
                ).item()
            
            psnr_input_total += psnr_in
            psnr_enhanced_total += psnr_enh
            ssim_total += ssim_val
            n_frames += 1
            
            video_psnr_in.append(psnr_in)
            video_psnr_enh.append(psnr_enh)
            video_ssim.append(ssim_val)
        
        if video_psnr_in:
            video_results[video_name] = {
                "psnr_in": sum(video_psnr_in) / len(video_psnr_in),
                "psnr_enh": sum(video_psnr_enh) / len(video_psnr_enh),
                "ssim": sum(video_ssim) / len(video_ssim),
                "frames": len(video_psnr_in)
            }
    
    if n_frames == 0:
        print("No full frames could be evaluated")
        return None, None, None
    
    print(f"\nFull frames evaluated: {n_frames}")
    print(f"Average Input PSNR:  {psnr_input_total/n_frames:.4f} dB")
    print(f"Average Enhanced PSNR: {psnr_enhanced_total/n_frames:.4f} dB")
    print(f"Average PSNR Gain:   {(psnr_enhanced_total - psnr_input_total)/n_frames:.4f} dB")
    print(f"Average SSIM:        {ssim_total/n_frames:.4f}")
    
    print(f"\nPer-video results:")
    print(f"{'Video':<35} {'Frames':>8} {'Input PSNR':>12} {'Enhanced PSNR':>14} {'Gain':>8} {'SSIM':>8}")
    print("-" * 90)
    for video in sorted(video_results.keys()):
        stats = video_results[video]
        gain = stats["psnr_enh"] - stats["psnr_in"]
        print(f"{video:<35} {stats['frames']:>8} {stats['psnr_in']:>12.4f} {stats['psnr_enh']:>14.4f} {gain:>+8.4f} {stats['ssim']:>8.4f}")
    
    return psnr_input_total/n_frames, psnr_enhanced_total/n_frames, ssim_total/n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/precomputed")
    parser.add_argument("--mode", type=str, choices=["patches", "full", "both"], default="both")
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    if args.mode in ["patches", "both"]:
        eval_patches(args.checkpoint, args.data_dir)
    
    if args.mode in ["full", "both"]:
        eval_full_frames(args.checkpoint, args.data_dir)


if __name__ == "__main__":
    main()
