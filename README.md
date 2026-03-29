# VVC Video Enhancement with CNN

Master's thesis project: Improving quality of VVC-encoded video using convolutional neural networks.

## Models

| Model | Architecture | PSNR Gain | SSIM | Metadata | Temporal | Motion Comp. |
|-------|--------------|-----------|------|----------|----------|--------------|
| **Hightower** | Simple concat | +0.33 dB | 0.95 | 8 ch | No | No |
| **Blackfyre** | Temporal Attention | +0.42 dB | 0.96 | 19 ch | 3 frames | No |
| **Targaryen** | Three-Stream | TBD | TBD | 8 ch | 3 frames | Yes |

### Model Details

- **Hightower** - Baseline: concatenates F-1, F0, F+1 frames with 8-channel metadata
- **Blackfyre** - Adds temporal attention to learn pixel-wise frame weights
- **Targaryen** - Three-stream architecture with motion compensation using VVC motion vectors

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Train Blackfyre
python train_blackfyre.py

# Or train Targaryen (in three-stream worktree)
cd ../vvc-cnn-three-stream
python train_targaryen.py
```

## Dataset

- Precomputed features in `data/precomputed/`
- 19 metadata channels from VVC decoder
- Frame triplets: F-1, F0, F+1

## Documentation

- [Blackfyre Model](BLACKFYRE.md)
- [Targaryen Plan](MULTI_STREAM_PLAN.md)

## Results

Best result: **Blackfyre +0.423 dB** PSNR gain over VVC-compressed input.
