# Blackfyre - VVC Video Enhancement with Temporal Attention

Master's thesis: Improving quality of VVC-encoded video using CNNs.

## Results

| Metric | Value |
|--------|-------|
| **PSNR Gain** | **+0.42 dB** |
| SSIM | 0.955 |
| Input PSNR | ~36 dB |
| Enhanced PSNR | ~36.4 dB |

## Architecture

Blackfyre uses **Temporal Attention** to learn pixel-wise which neighboring frame (F-1, F0, F+1) to focus on for enhancement.

Key features:
- 3-frame temporal context (F-1, F0, F+1)
- 19-channel decoder metadata (QP, Depth, SkipFlag, etc.)
- Pixel-wise attention weights learned during training
- Metadata-guided attention for compression-aware enhancement

See [BLACKFYRE.md](BLACKFYRE.md) for full details.

## Quick Start

```bash
source .venv/bin/activate
python train_blackfyre.py
```

## Files

- `train_blackfyre.py` - Training script
- `enhancer/models/blackfyre.py` - Model architecture
- `enhancer/dataset_blackfyre.py` - Dataset loader
- `visualize_blackfyre.py` - Test & visualization
- `checkpoints/blackfyre_epoch_*.pt` - Trained checkpoints

## Next: Targaryen

Targaryen (three-stream with motion compensation) is in separate worktree:
```bash
cd ../vvc-cnn-three-stream
python train_targaryen.py
```
