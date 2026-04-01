# Snow - VVC Video Enhancement CNN

Master's thesis: Improving quality of VVC-encoded video using CNNs.

## Results

| Metric | Value |
|--------|-------|
| **PSNR Gain** | **+0.502 dB** |
| SSIM | 0.9553 |
| Input PSNR | ~37.18 dB |
| Enhanced PSNR | ~37.68 dB |

## Architecture

Snow uses:
- **Feature Extraction Module** - separate encoder for each frame (F-1, F0, F+1)
- **Alignment Module (DCN v2)** - align neighboring frames to current
- **Attention Fusion** - learned pixel-wise weighting of frames
- **Deep Reconstruction** - 8 residual blocks
- **Metadata Attention** - use VVC decoder metadata (QP, Depth, etc.)

## Quick Start

```bash
uv run python train_snow.py
```

## Files

- `train_snow.py` - Training script
- `visualize_snow.py` - Test & visualization
- `enhancer/models/snow.py` - Model architecture
- `checkpoints/snow_epoch_*.pt` - Trained checkpoints

## Test

```bash
uv run python visualize_snow.py
```
