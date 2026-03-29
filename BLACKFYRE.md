# Blackfyre - Temporal Attention Network

**Named after House Blackfyre from Game of Thrones**

## Architecture

Blackfyre uses **Temporal Attention** to learn which neighboring frames to focus on for video enhancement. Unlike simple concatenation (Hightower), it learns pixel-wise attention weights to dynamically weight F-1, F0, and F+1 frames.

### Network Structure

```
Input:
├── Frames: F-1, F0, F+1 (3 channels each)
└── Metadata: 19 channels (QP, Depth, SkipFlag, MergeFlag, etc.)

Processing:
┌─────────────────────────────────────────────────────────────┐
│ 1. Frame Encoder                                            │
│    Conv2d(3, 32) → PReLU → Conv2d(32, 64) → PReLU         │
│    Output: [B, 64, H, W]                                   │
├─────────────────────────────────────────────────────────────┤
│ 2. Temporal Attention (pixel-wise)                         │
│    • Concatenate F-1, F0, F+1 features → [B, 192, H, W]   │
│    • Learn attention weights → [B, 3, H, W]                │
│    • Normalize weights (softmax)                             │
│    • Apply: w_prev*F-1 + w_curr*F0 + w_next*F+1            │
│    • Residual: F0 + gamma * attended_features               │
├─────────────────────────────────────────────────────────────┤
│ 3. Metadata-Guided Attention                               │
│    • Transform metadata (19 ch) → [B, 64, H, W]            │
│    • Concatenate with frame features                        │
│    • Learn attention weights per pixel                      │
│    • Multiply: features * attention                          │
├─────────────────────────────────────────────────────────────┤
│ 4. Processing Blocks                                       │
│    • 3x ResBlock (64 channels)                             │
│    • Conv2d(64, 32) → PReLU                                │
├─────────────────────────────────────────────────────────────┤
│ 5. Output                                                  │
│    • Conv2d(32, 3) → Residual connection                   │
│    • Output: F0 + residual                                 │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### Temporal Attention
- Learns **pixel-wise** which frame to focus on
- Each pixel can weight F-1, F0, F+1 differently
- `gamma` parameter (starts at 0) learns how much attention to apply

#### Metadata Attention  
- Uses 19 decoder metadata channels
- Learns where compression artifacts are likely (high QP, high depth)
- Applies attention to highlight/restore those regions

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base channels | 64 |
| Metadata channels | 19 |
| Patch size | 132 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| Scheduler | MultiStepLR [50, 100, 150, 200, 300] |
| Weight decay | 1e-4 |

### Loss Function

```
Loss = 0.1 × MS-SSIM + 0.1 × SSIM + 0.5 × L1 + 0.3 × L2 + 0.1 × Gradient
```

## Results

### Training Progress

| Epoch | Train Loss | Val PSNR Gain | Input PSNR | Enhanced PSNR | SSIM |
|-------|------------|---------------|------------|---------------|------|
| 0 | 0.0326 | -0.94 dB | 35.82 | 34.87 | 0.930 |
| 10 | 0.0163 | +0.10 dB | 35.88 | 35.98 | 0.948 |
| 50 | 0.0150 | +0.32 dB | 35.78 | 36.10 | 0.953 |
| 72 | 0.0147 | +0.37 dB | 35.92 | 36.29 | 0.953 |
| 100 | 0.0145 | +0.39 dB | 35.72 | 36.11 | 0.953 |
| 168 | 0.0144 | **+0.42 dB** | 35.93 | 36.36 | 0.955 |
| 170 | 0.0145 | +0.41 dB | 35.92 | 36.33 | 0.954 |

### Best Result

| Metric | Value |
|--------|-------|
| **Best PSNR Gain** | **+0.423 dB** |
| Best Epoch | ~168 |
| Final SSIM | 0.955 |
| Final Train Loss | 0.0144 |

### Comparison

| Model | PSNR Gain | Notes |
|-------|-----------|-------|
| Hightower | +0.33 dB | Simple concatenation |
| **Blackfyre** | **+0.42 dB** | Temporal Attention |

**Improvement over Hightower: +0.09 dB (+27%)**

## Observations

1. **Temporal attention works** - +0.42 dB vs Hightower's +0.33 dB
2. **Loss plateau** - Training loss stuck around 0.0145
3. **Metadata helps** - 19 channels provide guidance for compression artifacts
4. **Oscillation** - PSNR fluctuates ±0.05 dB around plateau

## Files

- `enhancer/models/blackfyre.py` - Model definition
- `enhancer/dataset_blackfyre.py` - Dataset loader
- `train_blackfyre.py` - Training script
- `checkpoints/blackfyre_epoch_*.pt` - Saved checkpoints
- `train_blackfyre.log` - Training log
- `wandb/` - Weights & Biases logs

## Future Improvements

1. **Motion Compensation** - Use MV to warp frames before attention (see Targaryen)
2. **More Frames** - Add F-2, F+2 for longer temporal context
3. **Reduce Metadata** - Test with fewer channels (8 instead of 19)
4. **Larger Model** - Increase base_channels to 96 or 128
5. **Different Scheduler** - Try cosine annealing
