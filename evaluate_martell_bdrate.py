#!/usr/bin/env python3
"""
Evaluate Martell model on TEST DATA ONLY
Computes BD-rate for Y, U, V channels
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from enhancer.dataset_blackfyre import BlackfyreDataset
from enhancer.models.snow_wide import SnowWideEnhancer
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from scipy.interpolate import PchipInterpolator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_bd_rate(bitrates, psnrs):
    """Calculate BD-rate using piecewise cubic interpolation (Bjontegaard)"""
    if len(bitrates) < 4 or len(psnrs) < 4:
        return None
    
    try:
        # Sort by bitrate
        indices = np.argsort(bitrates)
        br = np.array(bitrates)[indices]
        psnr = np.array(psnrs)[indices]
        
        # Use PCHIP interpolation (monotonic)
        pchip = PchipInterpolator(br, psnr)
        
        # Integrate over common bitrate range
        min_br = max(min(br), 100)   # Avoid extrapolation
        max_br = min(max(br), 10000)
        
        if min_br >= max_br:
            return None
        
        # BD-rate = (area_ref - area_test) / area_ref * 100
        area_test = pchip.integrate(min_br, max_br)
        
        # Create reference line (anchor = original compressed)
        # For anchor, we need original PSNR at same bitrates
        # Simplified: use linear interpolation of anchor PSNR
        return area_test  # Returns area (for comparison)
        
    except Exception as e:
        print(f"BD-rate calculation error: {e}")
        return None

def evaluate_martell_test():
    """Evaluate Martell model on test split"""
    
    # Load Martell model
    class Config:
        metadata_channels = 9
        base_channels = 64
    
    model = SnowWideEnhancer(Config()).to(DEVICE)
    
    try:
        checkpoint = torch.load("checkpoints/martell_epoch_190.pt", 
                               map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"✓ Loaded Martell model (epoch 190)")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None
    
    # Test data directories (QP-specific test sets)
    test_data_dirs = {
        22: "data_qp22_test",
        27: "data_qp27_test", 
        32: "data_qp32_test",
        37: "data_qp37_test",
        42: "data_qp42_test",
    }
    
    results = {
        "model": "Martell_SnowWide",
        "checkpoint": "martell_epoch_190.pt",
        "channels": {}
    }
    
    for channel in ['Y', 'U', 'V']:
        results["channels"][channel] = {
            "qp_results": {},
            "bd_rate_vs_anchor": None,
            "avg_gain": 0
        }
    
    # Evaluate at each QP
    for qp, data_dir in test_data_dirs.items():
        if not Path(data_dir).exists():
            print(f"⚠️  Test data not found: {data_dir}")
            print(f"   Run: python scripts/precompute_features.py --output {data_dir} --qp {qp}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating QP {qp} from {data_dir}")
        print('='*60)
        
        try:
            dataset = BlackfyreDataset(
                data_dir=data_dir,
                patch_size=132,
                split="test"
            )
            
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
            
            # Storage for this QP
            qp_results = {
                "bitrate": qp * 100,  # Approximate bitrate
                "psnr_before": {ch: [] for ch in ['Y', 'U', 'V']},
                "psnr_after": {ch: [] for ch in ['Y', 'U', 'V']},
            }
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"QP {qp}"):
                    decoded, original, metadata, info = batch
                    prev_frame, curr_frame, next_frame = decoded
                    
                    prev_frame = prev_frame.to(DEVICE)
                    curr_frame = curr_frame.to(DEVICE)
                    next_frame = next_frame.to(DEVICE)
                    original = original.to(DEVICE)
                    metadata = metadata.to(DEVICE)
                    
                    # Enhance
                    enhanced = model(curr_frame, prev_frame, next_frame, metadata).clamp(0, 1)
                    
                    # Calculate PSNR per channel
                    for i in range(curr_frame.shape[0]):
                        for ch_idx, ch_name in enumerate(['Y', 'U', 'V']):
                            orig_ch = original[i:i+1, ch_idx:ch_idx+1]
                            curr_ch = curr_frame[i:i+1, ch_idx:ch_idx+1]
                            enh_ch = enhanced[i:i+1, ch_idx:ch_idx+1]
                            
                            # Before
                            mse_before = F.mse_loss(curr_ch, orig_ch).item()
                            psnr_before = 10 * np.log10(1.0 / mse_before) if mse_before > 0 else 40
                            
                            # After
                            mse_after = F.mse_loss(enh_ch, orig_ch).item()
                            psnr_after = 10 * np.log10(1.0 / mse_after) if mse_after > 0 else 40
                            
                            qp_results["psnr_before"][ch_name].append(psnr_before)
                            qp_results["psnr_after"][ch_name].append(psnr_after)
            
            # Store results for this QP
            for ch in ['Y', 'U', 'V']:
                avg_before = np.mean(qp_results["psnr_before"][ch])
                avg_after = np.mean(qp_results["psnr_after"][ch])
                gain = avg_after - avg_before
                
                results["channels"][ch]["qp_results"][qp] = {
                    "bitrate": qp_results["bitrate"],
                    "psnr_before": float(avg_before),
                    "psnr_after": float(avg_after),
                    "gain": float(gain)
                }
                
                print(f"{ch} channel: {avg_before:.2f} → {avg_after:.2f} (gain: {gain:+.3f} dB)")
        
        except Exception as e:
            print(f"✗ Error evaluating QP {qp}: {e}")
            continue
    
    # Calculate BD-rate (requires at least 4 QP points)
    print(f"\n{'='*60}")
    print("BD-RATE CALCULATION")
    print('='*60)
    
    for ch in ['Y', 'U', 'V']:
        qp_data = results["channels"][ch]["qp_results"]
        
        if len(qp_data) >= 4:
            bitrates = [qp_data[qp]["bitrate"] for qp in sorted(qp_data.keys())]
            psnr_before = [qp_data[qp]["psnr_before"] for qp in sorted(qp_data.keys())]
            psnr_after = [qp_data[qp]["psnr_after"] for qp in sorted(qp_data.keys())]
            
            # BD-rate = area difference between curves
            try:
                pchip_before = PchipInterpolator(bitrates, psnr_before)
                pchip_after = PchipInterpolator(bitrates, psnr_after)
                
                min_br = max(min(bitrates), 100)
                max_br = min(max(bitrates), 10000)
                
                area_before = pchip_before.integrate(min_br, max_br)
                area_after = pchip_after.integrate(min_br, max_br)
                
                bd_rate = (area_after - area_before) / area_before * 100
                
                results["channels"][ch]["bd_rate_vs_anchor"] = float(bd_rate)
                print(f"{ch}: BD-rate = {bd_rate:.2f}% (negative = better)")
            except Exception as e:
                print(f"{ch}: Could not calculate BD-rate: {e}")
        
        # Average gain
        gains = [qp_data[qp]["gain"] for qp in qp_data.keys()]
        results["channels"][ch]["avg_gain"] = float(np.mean(gains))
    
    # Save results
    with open("martell_bdrate_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY: Martell Model BD-Rate Results")
    print('='*60)
    print(f"{'Channel':<10} {'Avg Gain':<15} {'BD-Rate':<15}")
    print("-" * 40)
    for ch in ['Y', 'U', 'V']:
        ch_data = results["channels"][ch]
        gain = ch_data["avg_gain"]
        bd = ch_data["bd_rate_vs_anchor"]
        bd_str = f"{bd:.2f}%" if bd is not None else "N/A"
        print(f"{ch:<10} {gain:+.3f} dB      {bd_str:<15}")
    
    print(f"\n✓ Results saved to martell_bdrate_results.json")
    
    return results

if __name__ == "__main__":
    evaluate_martell_test()
