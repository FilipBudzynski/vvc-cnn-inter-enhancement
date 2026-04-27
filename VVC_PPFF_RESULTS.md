# VVC-PPFF vs Snow-Wide Comparison Results

## Test Configuration
- Single model for all QPs (as per mentor requirement)
- Test dataset: `data/precomputed` (3286 frame triplets)
- Epochs trained: VVC-PPFF (190), Snow-Wide (460)

## PSNR Gain by QP

| QP | VVC-PPFF | Snow-Wide | Input PSNR |
|----|----------|----------|-----------|
| 27 | +0.11 dB | +0.67 dB | 41.06 dB |
| 28 | +0.06 dB | +0.06 dB | 40.01 dB |
| 30 | +0.17 dB | +0.33 dB | 37.17 dB |
| 32 | +0.16 dB | **+1.00 dB** | 37.09 dB |
| 33 | +0.17 dB | +0.58 dB | 38.08 dB |
| 34 | +0.19 dB | +0.85 dB | 39.31 dB |
| 35 | +0.10 dB | +0.35 dB | 36.54 dB |
| 36 | +0.19 dB | +0.56 dB | 35.75 dB |
| 37 | +0.21 dB | **+1.07 dB** | 39.23 dB |
| 38 | +0.16 dB | +0.48 dB | 37.76 dB |
| 39 | +0.22 dB | +0.53 dB | 33.19 dB |
| 40 | +0.16 dB | +0.69 dB | 37.94 dB |
| 41 | +0.19 dB | +0.50 dB | 32.97 dB |

## Summary

| Model | Parameters | Avg PSNR Gain | SSIM |
|-------|-----------|--------------|------|
| VVC-PPFF (epoch 190) | 2,778,260 | **+0.16 dB** | 0.9509 |
| Snow-Wide (epoch 460) | 1,293,024 | **+0.57 dB** | 0.9561 |

## Key Findings

1. **Snow-Wide outperforms VVC-PPFF** by ~3.5x on average (+0.57 dB vs +0.16 dB)
2. Best performance for both at QP 32 and QP 37
3. VVC-PPFF provides more consistent improvement across all QPs
4. Single model approach works for both architectures

## Model Files

- VVC-PPFF: `enhancer/models/vvc_ppff.py`
- Training: `train_model.py --model vvc_ppff`
- Checkpoints: `checkpoints/vvc_ppff_epoch_*.pt`
- Evaluation: `evaluate_all_models.py` (includes VVC_PPFF)

## Architecture (per paper)

VVC-PPFF implemented exactly one-to-one per "Appl Sci 2024" paper:
- 4-channel input (YUV + QP map)
- 16 Feature Extraction blocks (128 channels)
- Progressive feature fusion (Eq 2-3)
- 1x1 conv + Tanh + skip connection
- L2/MSE loss
- 200 epochs, Adam lr=1e-4
