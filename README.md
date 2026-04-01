# Snow-Wide - VVC Video Enhancement CNN

Master's thesis: Improving quality of VVC-encoded video using CNNs.

## Results

| Model | PSNR Gain | SSIM | Notes |
|-------|-----------|------|-------|
| Snow (132px) | +0.502 dB | 0.9553 | Original |
| Snow (720p) | +0.239 dB | 0.9405 | Direct inference |

## Architecture

Snow-Wide uses:
- **Feature Extraction Module** - encode each frame
- **Wide Context (7x7 depthwise)** - captures VVC block patterns
- **Alignment Module** - align neighboring frames
- **Attention Fusion** - learned pixel-wise weighting
- **Deep Reconstruction** - 13 residual blocks
- **Metadata Attention** - use VVC decoder metadata
- **Charbonnier Loss** - better edge preservation than L1

## Quick Start

### Fine-tuning (recommended)
```bash
uv run python train_snow_wide_finetune.py --epochs 200
```

### Training from scratch
```bash
uv run python train_snow_wide.py --epochs 500
```

## Test
```bash
uv run python visualize_snow.py
```

## Files
- `train_snow_wide.py` - Training from scratch
- `train_snow_wide_finetune.py` - Fine-tune from pretrained Snow
- `enhancer/models/snow_wide.py` - Model architecture
- `enhancer/dataset_blackfyre_fixed.py` - Dataset with 256x256 patches
- `checkpoints/` - Trained checkpoints
